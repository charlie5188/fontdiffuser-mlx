// Fine-tune FontDiffuser on one font's glyphs, translated from fdmlx/finetune.py.
//
// Recipe (the official phase-1 objective, at fine-tuning scale):
//   sample = (target glyph X in the target font, X in the plain content font, another training glyph Y in the
//   target font as the style reference); loss = noise MSE + 0.5 * mean |deformable offset| / 2;
//   10 % of samples get blank content and style images, so classifier-free guidance keeps working;
//   AdamW (lr 2e-5, weight decay 0.01), gradient norm clipped to 1.
//
// Random draws use a seeded generator of their own (Swift cannot reproduce numpy's), so a Swift run trains on a
// different random order than a Python run with the same seed. Each step is the same computation.

import Foundation
import MLX
import MLXNN
import MLXOptimizers

public struct FineTuneOptions {
    public var batch = 16
    /// split each batch into this many passes: same update, less memory (batch 16: about 19 GB in one pass,
    /// 10 GB in 2, 7 GB in 4)
    public var accum = 4
    public var learningRate: Float = 2e-5
    public var seed: UInt64 = 0
    /// run the training step without compile (slower; for debugging)
    public var compiled = true

    public init() {}
}

/// SplitMix64: small, fast, seedable.
struct SplitMix64: RandomNumberGenerator {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

public final class FineTuner {
    public let model: FontDiffuser
    public let options: FineTuneOptions
    let target: MLXArray
    let content: MLXArray
    let optimizer: AdamW
    let acp: MLXArray
    let gradKeys: [String]
    var rng: SplitMix64
    var micro: ([MLXArray]) -> [MLXArray] = { _ in [] }
    var update: ([MLXArray]) -> [MLXArray] = { _ in [] }

    /// target: the training glyphs in the target font, content: the same characters in the content font,
    /// both (N, 96, 96, 3). The model is put in training mode.
    public init(model: FontDiffuser, target: MLXArray, content: MLXArray, options: FineTuneOptions = .init()) {
        precondition(options.batch % options.accum == 0, "batch must be divisible by accum")
        precondition(target.dim(0) >= 2, "need at least 2 training glyphs")
        self.model = model
        self.options = options
        self.target = target
        self.content = content
        optimizer = AdamW(learningRate: options.learningRate, betas: (0.9, 0.999), eps: 1e-8, weightDecay: 1e-2,
                          biasCorrection: true)
        acp = MLXArray(NoiseSchedule.alphasCumprod())
        rng = SplitMix64(state: options.seed)
        model.train(true)
        gradKeys = model.trainableParameters().flattened().map { $0.0 }   // sorted, so the order is stable

        let lossAndGrad = valueAndGrad(model: model) { (m: FontDiffuser, a: [MLXArray]) -> [MLXArray] in
            let (pred, off) = m(a[0], t: a[1], content: a[2], style: a[3])
            return [((pred - a[4]) ** 2).mean() + 0.5 * (off / 2)]
        }
        let keys = gradKeys, acp = acp, model = model, optimizer = optimizer
        // one pass: (target, content, style, noise, t) -> [loss] + gradients in gradKeys order
        func microFn(_ a: [MLXArray]) -> [MLXArray] {
            let (tgt, con, sty, noise, t) = (a[0], a[1], a[2], a[3], a[4])
            let alpha = acp[t].reshaped(-1, 1, 1, 1)
            let noisy = sqrt(alpha) * tgt + sqrt(1 - alpha) * noise
            let (loss, grads) = lossAndGrad(model, [noisy, t, con, sty, noise])
            let flat = Dictionary(uniqueKeysWithValues: grads.flattened())
            return [loss[0]] + keys.map { flat[$0]! }
        }
        func updateFn(_ g: [MLXArray]) -> [MLXArray] {
            let grads = ModuleParameters.unflattened(Array(zip(keys, g)))
            let (clipped, _) = clipGradNorm(gradients: grads, maxNorm: 1.0)
            optimizer.update(model: model, gradients: clipped)
            return []
        }
        // Each pass is evaluated before the next one starts, so its activations are freed: peak memory follows
        // the pass size, not the batch size. The model is an output of the pass too: the spectral-norm vectors
        // update in training mode.
        if options.compiled {
            micro = compile(inputs: [model], outputs: [model], microFn)
            update = compile(inputs: [model, optimizer], outputs: [model, optimizer], updateFn)
        } else {
            micro = microFn
            update = updateFn
        }
    }

    /// One update on explicit inputs (target, content, style, noise: (batch, 96, 96, 3); t: (batch,) int32).
    /// Returns the loss. Exposed for checks against the Python version.
    public func step(target tgt: MLXArray, content con: MLXArray, style sty: MLXArray, noise: MLXArray,
                     t: MLXArray) -> MLXArray {
        let parts = options.batch / options.accum
        var loss = MLXArray(Float(0))
        var grads: [MLXArray]?
        for k in stride(from: 0, to: options.batch, by: parts) {
            let r = k ..< k + parts
            let out = micro([tgt[r], con[r], sty[r], noise[r], t[r]])
            loss = loss + out[0] / Float(options.accum)
            let g = Array(out.dropFirst())
            grads = grads.map { zip($0, g).map { $0 + $1 } } ?? g
            eval([loss] + grads!)
            eval(model)
        }
        if options.accum > 1 { grads = grads!.map { $0 / Float(options.accum) } }
        _ = update(grads!)
        eval(model, optimizer, loss)
        return loss
    }

    /// One update on a random batch of training glyphs. Returns the loss.
    public func step() -> Float {
        let n = target.dim(0), b = options.batch
        let idx = (0 ..< b).map { _ in Int.random(in: 0 ..< n, using: &rng) }
        let sidx = idx.map { k -> Int in                              // another training glyph as the reference
            let j = Int.random(in: 0 ..< n - 1, using: &rng)
            return j >= k ? j + 1 : j
        }
        let drop = (0 ..< b).map { _ in Double.random(in: 0 ..< 1, using: &rng) < 0.1 }   // condition dropout
        let t = (0 ..< b).map { _ in Int32.random(in: 0 ..< Int32(NoiseSchedule.T), using: &rng) }
        let noise = MLXRandom.normal([b, glyphSize, glyphSize, 3], key: MLXRandom.key(rng.next()))
        let d = MLXArray(drop).reshaped(b, 1, 1, 1)
        let i = MLXArray(idx.map { Int32($0) }), s = MLXArray(sidx.map { Int32($0) })
        let con = which(d, Float(1), content[i]), sty = which(d, Float(1), target[s])
        return step(target: target[i], content: con, style: sty, noise: noise, t: MLXArray(t)).item(Float.self)
    }
}
