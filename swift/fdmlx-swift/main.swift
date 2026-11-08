// fdmlx-swift: the command-line tool of the Swift version, with the same options as the Python `fdmlx`.

import Foundation
import FontDiffuserMLX
import MLX

let usage = """
fdmlx-swift: FontDiffuser on Apple silicon, in MLX Swift.

  fdmlx-swift generate   draw characters in the style of one reference glyph
  fdmlx-swift finetune   teach the model one font from a few dozen of its glyphs

Weights are .safetensors files: the official weights converted with the Python package (`fdmlx convert`), or
weights fine-tuned with either version. Run a command with --help for its options.
"""

struct Fail: Error, CustomStringConvertible {
    let description: String
    init(_ s: String) { description = s }
}

/// --name value pairs and --flag switches.
struct Options {
    var values = [String: String]()
    var flags = Set<String>()

    init(_ argv: ArraySlice<String>, flags known: Set<String>) throws {
        var it = argv.makeIterator()
        while let a = it.next() {
            guard a.hasPrefix("--") else { throw Fail("unexpected argument \(a)") }
            let key = String(a.dropFirst(2))
            if known.contains(key) { flags.insert(key); continue }
            guard let v = it.next() else { throw Fail("\(a) needs a value") }
            values[key] = v
        }
    }

    func string(_ k: String) -> String? { values[k] }
    func require(_ k: String) throws -> String {
        guard let v = values[k] else { throw Fail("missing --\(k)") }
        return v
    }
    func url(_ k: String) throws -> URL { URL(fileURLWithPath: try require(k)) }
    func int(_ k: String, _ d: Int) throws -> Int {
        guard let s = values[k] else { return d }
        guard let v = Int(s) else { throw Fail("--\(k) must be a whole number") }
        return v
    }
    func float(_ k: String, _ d: Float) throws -> Float {
        guard let s = values[k] else { return d }
        guard let v = Float(s) else { throw Fail("--\(k) must be a number") }
        return v
    }
}

func uniqueChars(_ s: String, excluding: String = "") -> String {
    var seen = Set<Character>(), out = ""
    for ch in s where !ch.isWhitespace && !excluding.contains(ch) && !seen.contains(ch) {
        seen.insert(ch)
        out.append(ch)
    }
    return out
}

func hexName(_ ch: Character) -> String {
    String(format: "U+%04X.png", ch.unicodeScalars.first!.value)
}

// MARK: - generate

let generateHelp = """
fdmlx-swift generate: draw characters in the style of a reference glyph

  --weights PATH        .safetensors weights
  --content-font PATH   plain font for the skeletons (KaiXinSongA.ttf from the official repo)
  --chars TEXT          characters to draw, or --chars-file PATH
  --font PATH --ref C   the reference glyph: character C drawn in this font
  --ref-image PATH      or an image of one reference glyph (dark ink on light background)
  --out DIR             output folder (one PNG per character plus sheet.png)
  --steps N             sampling steps (default 15; the official setting is 20)
  --batch N             characters per batch (default 8)
  --seed N              starting noise (default 0)
  --noise PATH          or the starting noise from a .npy file (N x 96 x 96 x 3), e.g. to compare runs
  --guidance X          classifier-free guidance scale (default 7.5)
"""

func generate(_ o: Options) throws {
    let chars = uniqueChars(try o.string("chars") ?? String(contentsOf: o.url("chars-file"), encoding: .utf8))
    guard !chars.isEmpty else { throw Fail("no characters to draw") }
    let contentFont = try GlyphFont(url: o.url("content-font"))
    let style: MLXArray
    if let img = o.string("ref-image") {
        style = try styleReference(fromImage: URL(fileURLWithPath: img))
    } else if let font = o.string("font"), let ref = o.string("ref")?.first {
        style = try GlyphFont(url: URL(fileURLWithPath: font)).glyphs(String(ref))
    } else {
        throw Fail("give --font and --ref, or --ref-image")
    }
    let model = FontDiffuser()
    try model.load(weights: o.url("weights"))
    model.train(false)
    let sampler = Sampler(model: model, guidanceScale: try o.float("guidance", 7.5))
    let out = try o.url("out")
    try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
    let n = chars.count
    let noise: MLXArray
    if let path = o.string("noise") {
        noise = try loadArray(url: URL(fileURLWithPath: path)).asType(.float32)
    } else {
        noise = MLXRandom.normal([n, glyphSize, glyphSize, 3], key: MLXRandom.key(UInt64(try o.int("seed", 0))))
    }
    let batch = try o.int("batch", 8), steps = try o.int("steps", 15)
    let list = Array(chars)
    var pixels = [UInt8]()
    let t0 = Date()
    for k in stride(from: 0, to: n, by: batch) {
        let part = String(list[k ..< min(k + batch, n)])
        let imgs = sampler(content: try contentFont.glyphs(part), style: style,
                           noise: noise[k ..< k + part.count], steps: steps)
        for (i, ch) in part.enumerated() { try imgs.writePNG(i, to: out.appendingPathComponent(hexName(ch))) }
        pixels += imgs.pixels
        let done = k + part.count
        print(String(format: "%d/%d  %.2f s per character", done, n, Date().timeIntervalSince(t0) / Double(done)))
        fflush(stdout)
    }
    try GlyphImages(count: n, size: glyphSize, pixels: pixels).writeSheet(to: out.appendingPathComponent("sheet.png"))
    print("wrote \(n) glyphs and sheet.png to \(out.path)")
}

// MARK: - finetune

let finetuneHelp = """
fdmlx-swift finetune: teach the model one font from a few dozen of its glyphs

  --weights PATH        start from these .safetensors weights
  --font PATH           the target font (.ttf / .otf)
  --train PATH          text file with the training characters
  --content-font PATH   plain font for the skeletons (KaiXinSongA.ttf from the official repo)
  --out DIR             output folder
  --test TEXT           held-out characters to draw at every evaluation (never trained on)
  --eval-refs TEXT      style references for the evaluation sheets (must be training characters)
  --steps N             default 800
  --eval-every N        default 200
  --batch N             default 16
  --accum N             split each batch into N passes: same update, less memory (default 4: about 7 GB)
  --lr X                default 2e-5
  --seed N              default 0
  --eval-steps N        sampling steps for the evaluation sheets (default 20)
  --eval-noise PATH     starting noise for the evaluation sheets (.npy, 1 x 96 x 96 x 3)
  --no-compile          run the training step without compile
"""

func finetune(_ o: Options) throws {
    let out = try o.url("out")
    try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
    let test = uniqueChars(o.string("test") ?? "")
    let train = uniqueChars(try String(contentsOf: o.url("train"), encoding: .utf8), excluding: test)
    let font = try GlyphFont(url: o.url("font")), contentFont = try GlyphFont(url: o.url("content-font"))
    for f in [font, contentFont] {
        let missing = (train + test).filter { !f.hasGlyph($0) }
        if !missing.isEmpty { throw Fail("\(f.url.lastPathComponent) has no glyph for: \(missing)") }
    }
    let refs = uniqueChars(o.string("eval-refs") ?? String(train.prefix(1)))
    guard refs.allSatisfy({ train.contains($0) }) else { throw Fail("eval refs must be training characters") }
    print("\(train.count) training characters, \(test.count) held-out test characters")

    var options = FineTuneOptions()
    options.batch = try o.int("batch", 16)
    options.accum = try o.int("accum", 4)
    options.learningRate = try o.float("lr", 2e-5)
    options.seed = UInt64(try o.int("seed", 0))
    options.compiled = !o.flags.contains("no-compile")
    guard options.batch % options.accum == 0 else { throw Fail("--batch must be divisible by --accum") }
    let steps = try o.int("steps", 800), evalEvery = try o.int("eval-every", 200)
    let evalSteps = try o.int("eval-steps", 20)

    let model = FontDiffuser()
    try model.load(weights: o.url("weights"))
    let tuner = FineTuner(model: model, target: try font.glyphs(train), content: try contentFont.glyphs(train),
                          options: options)
    let testContent = test.isEmpty ? nil : try contentFont.glyphs(test)
    var refGlyphs = [(Character, MLXArray)]()
    for r in refs { refGlyphs.append((r, try font.glyphs(String(r)))) }
    let evalNoise: MLXArray
    if let path = o.string("eval-noise") {
        evalNoise = try loadArray(url: URL(fileURLWithPath: path)).asType(.float32)
    } else {
        evalNoise = MLXRandom.normal([1, glyphSize, glyphSize, 3], key: MLXRandom.key(123))
    }

    func evaluate(_ n: Int) throws {
        guard let testContent else { return }
        model.train(false)
        let sampler = Sampler(model: model)      // a new compiled sampler: the weights have changed
        let noise = broadcast(evalNoise, to: testContent.shape)   // same start for every glyph and evaluation
        for (r, ref) in refGlyphs {
            let name = String(format: "grid-step%04d-ref", n) + String(r) + ".png"
            try sampler(content: testContent, style: ref, noise: noise, steps: evalSteps)
                .writeSheet(to: out.appendingPathComponent(name), cols: 4)
        }
        model.train(true)
    }

    var log = [[String: Any]]()
    try evaluate(0)
    let t0 = Date()
    var running: Float?
    for i in 1 ... steps {
        let loss = tuner.step()
        running = running.map { 0.95 * $0 + 0.05 * loss } ?? loss
        if i % 20 == 0 || i == steps {
            let el = Date().timeIntervalSince(t0)
            print(String(format: "step %d/%d loss %.4f  %.2fs/step  eta %.0f min  peak mem %.1f GB", i, steps,
                         running!, el / Double(i), Double(steps - i) * el / Double(i) / 60,
                         Double(Memory.peakMemory) / Double(1 << 30)))
            fflush(stdout)
            log.append(["step": i, "loss": Double(running!), "elapsed_s": (el * 10).rounded() / 10])
        }
        if i % evalEvery == 0 || i == steps {
            try model.save(weights: out.appendingPathComponent("weights.safetensors"))
            try evaluate(i)
            let info: [String: Any] = ["train_chars": train, "test_chars": test, "steps": steps,
                                       "batch": options.batch, "accum": options.accum,
                                       "lr": Double(options.learningRate), "seed": Int(options.seed), "log": log]
            try JSONSerialization.data(withJSONObject: info, options: [.prettyPrinted, .sortedKeys])
                .write(to: out.appendingPathComponent("log.json"))
        }
    }
    print(String(format: "done in %.1f min -> ", Date().timeIntervalSince(t0) / 60)
          + out.appendingPathComponent("weights.safetensors").path)
}

// MARK: - main

let argv = CommandLine.arguments
let commands: [String: (help: String, flags: Set<String>, run: (Options) throws -> Void)] = [
    "generate": (generateHelp, [], generate),
    "finetune": (finetuneHelp, ["no-compile"], finetune),
]
guard argv.count > 1, let cmd = commands[argv[1]] else {
    print(usage)
    exit(argv.count > 1 && ["-h", "--help"].contains(argv[1]) ? 0 : 1)
}
if argv.dropFirst(2).contains(where: { $0 == "-h" || $0 == "--help" }) {
    print(cmd.help)
    exit(0)
}
do {
    try cmd.run(try Options(argv.dropFirst(2), flags: cmd.flags))
} catch {
    FileHandle.standardError.write("error: \(error)\n".data(using: .utf8)!)
    exit(1)
}
