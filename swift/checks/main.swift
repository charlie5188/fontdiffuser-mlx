// fdmlx-swift-check: the Swift version against the Python version, on the reference outputs written by
// tests/swift_fixtures.py. For development; see the README's Swift section.
//
//   fdmlx-swift-check --fixtures fixtures.safetensors --weights W.safetensors --content-font KaiXinSongA.ttf \
//       --font F.ttf --image glyph.png      (the same files given to swift_fixtures.py)

import Foundation
import FontDiffuserMLX
import MLX
import MLXNN

var opts = [String: String]()
var it = CommandLine.arguments.dropFirst().makeIterator()
while let k = it.next(), let v = it.next() { opts[String(k.dropFirst(2))] = v }
func path(_ k: String) -> URL {
    guard let v = opts[k] else { fatalError("missing --\(k)") }
    return URL(fileURLWithPath: v)
}

let fx = try loadArrays(url: path("fixtures"))
let weightsURL = path("weights")
var failures = 0

func maxAbs(_ a: MLXArray) -> Float { abs(a).max().item(Float.self) }
func rel(_ a: MLXArray, _ b: MLXArray) -> Float { maxAbs(a - b) / max(maxAbs(b), 1e-12) }
func report(_ name: String, _ value: Float, limit: Float, _ unit: String = "max relative difference") {
    let ok = value <= limit
    if !ok { failures += 1 }
    print(String(format: "%@  %-44@ %@ %.2g (limit %.0g)", ok ? "ok  " : "FAIL", name, unit, value, limit))
}
func freshModel(training: Bool) throws -> FontDiffuser {
    let m = FontDiffuser()
    try m.load(weights: weightsURL)
    m.train(training)
    return m
}
let chars = "我是道谢", ref: Character = "永"

// 1. the 128 -> 96 resize alone, on the Python version's 128 px renders: must be identical
let r128 = fx["render128"]!.asArray(UInt8.self)
let expected96 = concatenated([fx["content"]!, fx["style"]!]).asArray(Float.self)
var resizeDiff: Float = 0
for i in 0 ..< 5 {
    let img = GrayImage(width: 128, height: 128, pixels: Array(r128[(i * 128 * 128) ..< ((i + 1) * 128 * 128)]))
    let mine = img.modelInput()
    let n = mine.count
    for j in 0 ..< n { resizeDiff = max(resizeDiff, abs(mine[j] - expected96[i * n + j])) }
}
report("resize 128 -> 96 (Pillow bilinear)", resizeDiff, limit: 0, "max difference")

// 2. rendering with Core Text instead of FreeType: close, not identical
let fonts = [try GlyphFont(url: path("content-font")), try GlyphFont(url: path("font"))]
var greyDiff = 0.0, inkUnion = 0, inkBoth = 0
for (i, (fi, ch)) in (chars.map { (0, $0) } + [(1, ref)]).enumerated() {
    let mine = fonts[fi].render(ch).pixels
    for j in 0 ..< mine.count {
        let theirs = r128[i * 128 * 128 + j]
        greyDiff += abs(Double(mine[j]) - Double(theirs))
        let (a, b) = (mine[j] < 128, theirs < 128)
        if a || b { inkUnion += 1 }
        if a && b { inkBoth += 1 }
    }
}
print(String(format: "info  rendering vs Pillow: mean grey difference %.2f of 255, ink overlap %.3f",
             greyDiff / Double(5 * 128 * 128), Double(inkBoth) / Double(inkUnion)))

// 3. photo -> style reference (JPEG decoders differ slightly between Pillow and ImageIO)
let refimg = try styleReference(fromImage: path("image"))
print(String(format: "info  reference from image: mean difference %.4f, max %.3f (of 2)",
             abs(refimg - fx["refimg"]!).mean().item(Float.self), maxAbs(refimg - fx["refimg"]!)))

// 4-6. inference, on the Python version's inputs
do {
    let model = try freshModel(training: false)
    let content = fx["content"]!, style = fx["style"]!
    let cond = model.encode(content: content, style: broadcast(style, to: content.shape))
    report("style encoder", rel(cond.styleTokens, fx["enc.style_tokens"]!), limit: 1e-4)
    for (i, a) in cond.content.enumerated() {
        report("content encoder, content glyph, output \(i)", rel(a, fx["enc.content.\(i)"]!), limit: 1e-4)
    }
    for (i, a) in cond.styleStructure.enumerated() {
        report("content encoder, reference, output \(i)", rel(a, fx["enc.style_structure.\(i)"]!), limit: 1e-4)
    }
    let (eps, off) = model.denoise(fx["unet.x"]!, t: fx["unet.t"]!, cond: cond)
    report("denoiser: noise prediction", rel(eps, fx["unet.eps"]!), limit: 1e-4)
    report("denoiser: deformable offsets", rel(off, fx["unet.offset"]!), limit: 1e-4)
    let imgs = Sampler(model: model)(content: content, style: style, noise: fx["sample.noise"]!, steps: 15)
    let theirs = fx["sample"]!.asArray(UInt8.self)
    var maxd = 0, differ = 0
    for (a, b) in zip(imgs.pixels, theirs) {
        let d = abs(Int(a) - Int(b))
        maxd = max(maxd, d)
        if d > 0 { differ += 1 }
    }
    report("sampling, 15 steps: pixels that differ", Float(differ) / Float(theirs.count), limit: 0.01, "share")
    print("      largest pixel difference \(maxd) of 255")
}

// 7. training loss and every gradient
let acp = MLXArray(NoiseSchedule.alphasCumprod())
let (tgt, con, sty, tn, tt) = (fx["train.tgt"]!, fx["train.con"]!, fx["train.sty"]!, fx["train.noise"]!,
                               fx["train.t"]!)
do {
    let model = try freshModel(training: true)
    let lg = valueAndGrad(model: model) { (m: FontDiffuser, a: [MLXArray]) -> [MLXArray] in
        let (pred, off) = m(a[0], t: a[1], content: a[2], style: a[3])
        return [((pred - a[4]) ** 2).mean() + 0.5 * (off / 2)]
    }
    let alpha = acp[tt].reshaped(-1, 1, 1, 1)
    let (loss, grads) = lg(model, [sqrt(alpha) * tgt + sqrt(1 - alpha) * tn, tt, con, sty, tn])
    report("training loss", rel(loss[0], fx["train.loss"]!), limit: 1e-4)
    let flat = grads.flattened()
    let scales = flat.map { maxAbs(fx["grad.\($0.0)"]!) }.sorted()
    let median = scales[scales.count / 2]
    var errs = [(Float, String)]()
    for (k, g) in flat {
        let ref = fx["grad.\(k)"]!
        errs.append((maxAbs(g - ref) / max(maxAbs(ref), 1e-3 * median), k))
    }
    errs.sort { $0.0 < $1.0 }
    report("gradients (\(errs.count)): median", errs[errs.count / 2].0, limit: 1e-4)
    report("gradients: worst", errs.last!.0, limit: 0.1)
    print("      worst: \(errs.last!.1)")
}

// 8. one training step through FineTuner: 2 passes, clipping, AdamW, spectral-norm updates
do {
    let before = try loadArrays(url: weightsURL)
    let model = try freshModel(training: true)
    var o = FineTuneOptions()
    o.batch = 4
    o.accum = 2
    let tuner = FineTuner(model: model, target: tgt, content: con, options: o)
    let loss = tuner.step(target: tgt, content: con, style: sty, noise: tn, t: tt)
    report("training step: loss", rel(loss, fx["step.loss"]!), limit: 1e-4)
    let lr = o.learningRate
    var worst: Float = 0, flips = 0, total = 0, snWorst: Float = 0
    for (k, after) in model.parameters().flattened() {
        let theirs = fx["step.after.\(k)"]!
        if k.hasSuffix(".u0") || k.hasSuffix(".sv0") {
            snWorst = max(snWorst, rel(after, theirs))
            continue
        }
        // the first Adam step moves each weight by about lr; compare the moves
        let d = abs((after - before[k]!) - (theirs - before[k]!))
        worst = max(worst, d.max().item(Float.self) / lr)
        flips += (d .> 0.5 * lr).sum().item(Int.self)
        total += d.size
    }
    report("training step: spectral-norm vectors", snWorst, limit: 1e-3)
    report("training step: weights that moved differently", Float(flips) / Float(total), limit: 1e-3, "share")
    print(String(format: "      largest difference in a move: %.2f x the learning rate", worst))
}

print(failures == 0 ? "ALL OK" : "\(failures) check(s) failed")
exit(failures == 0 ? 0 : 1)
