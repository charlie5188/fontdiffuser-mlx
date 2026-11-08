// Noise schedule and DPM-Solver++ sampling, translated from fdmlx/sampler.py.
//
// Same algorithm and settings as the official pipeline (DPM-Solver++, order 2, multistep, time-uniform steps,
// classifier-free guidance 7.5). The per-step coefficients are scalars, computed in Double on the CPU with the
// same formulas numpy uses in the Python version, so both versions give the same glyphs from the same noise.
// The encoders run once per batch: their inputs do not change between steps.

import Foundation
import MLX

public enum NoiseSchedule {
    /// training diffusion steps
    public static let T = 1000

    /// diffusers DDPMScheduler(beta_schedule="scaled_linear", 1e-4 ... 0.02) in float32, like diffusers:
    /// numpy linspace in double, cast to float32, squared in float32.
    public static func betas() -> [Float] {
        let start = pow(1e-4, 0.5), stop = pow(0.02, 0.5)
        let step = (stop - start) / Double(T - 1)
        return (0 ..< T).map { i in
            let v = Float(i == T - 1 ? stop : Double(i) * step + start)
            return v * v
        }
    }

    /// cumulative product of (1 - beta) in float32, as np.cumprod(..., dtype=float32)
    public static func alphasCumprod() -> [Float] {
        var p: Float = 1
        return betas().map { b in
            p *= 1 - b
            return p
        }
    }
}

/// DPM-Solver's discrete-time VP schedule (NoiseScheduleVP('discrete')) as Double scalar functions.
struct VPSchedule {
    let logAlpha: [Double]
    let tGrid: [Double]

    init() {
        var acc = 0.0
        logAlpha = NoiseSchedule.betas().map { b in
            acc += log(1.0 - Double(b))
            return 0.5 * acc
        }
        let T = NoiseSchedule.T
        tGrid = (1 ... T).map { $0 == T ? 1.0 : Double($0) * (1.0 / Double(T)) }
    }

    /// np.interp(t, tGrid, logAlpha)
    func logAlphaAt(_ t: Double) -> Double {
        let n = tGrid.count
        if t < tGrid[0] { return logAlpha[0] }
        if t >= tGrid[n - 1] { return logAlpha[n - 1] }
        var lo = 0, hi = n - 1                  // largest j with tGrid[j] <= t
        while hi - lo > 1 {
            let mid = (lo + hi) / 2
            if tGrid[mid] <= t { lo = mid } else { hi = mid }
        }
        if tGrid[lo] == t { return logAlpha[lo] }
        let slope = (logAlpha[lo + 1] - logAlpha[lo]) / (tGrid[lo + 1] - tGrid[lo])
        return slope * (t - tGrid[lo]) + logAlpha[lo]
    }

    /// (alpha, sigma, lambda)
    func coeffs(_ t: Double) -> (Double, Double, Double) {
        let la = logAlphaAt(t)
        let sigma = (1.0 - exp(2 * la)).squareRoot()
        return (exp(la), sigma, la - log(sigma))
    }
}

/// Generated glyphs: `count` images of 96 x 96 RGB, uint8, row-major.
public struct GlyphImages {
    public let count: Int
    public let size: Int
    public let pixels: [UInt8]

    public init(count: Int, size: Int, pixels: [UInt8]) {
        self.count = count
        self.size = size
        self.pixels = pixels
    }

    /// One glyph's RGB bytes.
    public func image(_ i: Int) -> ArraySlice<UInt8> {
        let n = size * size * 3
        return pixels[(i * n) ..< ((i + 1) * n)]
    }
}

public final class Sampler {
    /// A replacement for the denoiser call (for example a Core ML model):
    /// (x, t, cond) -> noise prediction, for the doubled [unconditional, conditional] batch.
    public typealias Denoiser = (_ x: MLXArray, _ t: MLXArray, _ cond: Conditions) -> MLXArray

    let model: FontDiffuser
    public var guidanceScale: Float
    let denoiser: Denoiser
    let schedule = VPSchedule()

    public init(model: FontDiffuser, guidanceScale: Float = 7.5, denoiser: Denoiser? = nil, compiled: Bool = true) {
        self.model = model
        self.guidanceScale = guidanceScale
        if let denoiser {
            self.denoiser = denoiser
        } else if compiled {
            // compile fuses the many small ops (about 13 % faster per call in Python on an M4 Pro). The weights are
            // fixed while sampling, so they are captured as constants.
            let f = compile { (a: [MLXArray]) -> [MLXArray] in
                [model.denoise(a[0], t: a[1], cond: Conditions(arrays: Array(a[2...]))).0]
            }
            self.denoiser = { x, t, cond in f([x, t] + cond.arrays)[0] }
        } else {
            self.denoiser = { x, t, cond in model.denoise(x, t: t, cond: cond).0 }
        }
    }

    /// Classifier-free guidance: [unconditional (all-white images), conditional] stacked along the batch.
    func encode(content: MLXArray, style: MLXArray) -> Conditions {
        let white = MLXArray.ones(like: content)
        return model.encode(content: concatenated([white, content]), style: concatenated([white, style]))
    }

    /// content, style: (N, 96, 96, 3) in [-1, 1] (style may have N = 1, then it is shared); noise: (N, 96, 96, 3).
    public func callAsFunction(content: MLXArray, style: MLXArray, noise: MLXArray, steps: Int = 20,
                               order: Int = 2) -> GlyphImages {
        let N = content.dim(0)
        let T = Double(NoiseSchedule.T)
        let style = style.dim(0) == 1 && N > 1 ? broadcast(style, to: content.shape) : style
        let cond = encode(content: content, style: style)
        eval(cond.arrays)
        var x = noise

        func x0Pred(_ x: MLXArray, _ t: Double) -> MLXArray {
            let (alpha, sigma, _) = schedule.coeffs(t)
            let tIn = MLXArray.full([2 * N], values: MLXArray(Float((t - 1.0 / T) * 1000.0)))  // as model_wrapper
            let eps = denoiser(concatenated([x, x]), tIn, cond)
            let eu = eps[0 ..< N], ec = eps[N...]
            let guided = eu + guidanceScale * (ec - eu)
            return (x - Float(sigma) * guided) / Float(alpha)     // DPM-Solver++ uses the data prediction
        }

        // np.linspace(1, 1/T, steps + 1)
        let tStep = (1.0 / T - 1.0) / Double(steps)
        let ts = (0 ... steps).map { $0 == steps ? 1.0 / T : Double($0) * tStep + 1.0 }
        var prevM = [x0Pred(x, ts[0])], prevT = [ts[0]]
        for step in 1 ... steps {
            let t = ts[step]
            let (alphaT, sigmaT, lamT) = schedule.coeffs(t)
            let (_, sigmaP, lamP) = schedule.coeffs(prevT[prevT.count - 1])
            let h = lamT - lamP
            let phi = expm1(-h)
            var useOrder = min(order, step)
            if steps < 10 { useOrder = min(useOrder, steps + 1 - step) }   // lower_order_final, as in the original
            let last = prevM[prevM.count - 1]
            if useOrder == 1 {
                x = Float(sigmaT / sigmaP) * x - Float(alphaT * phi) * last
            } else {
                let (_, _, lamPP) = schedule.coeffs(prevT[prevT.count - 2])
                let r0 = (lamP - lamPP) / h
                let d1 = (last - prevM[prevM.count - 2]) * Float(1.0 / r0)
                x = Float(sigmaT / sigmaP) * x - Float(alphaT * phi) * last - Float(0.5 * (alphaT * phi)) * d1
            }
            if step < steps {
                prevM = Array((prevM + [x0Pred(x, t)]).suffix(order))
                prevT = Array((prevT + [t]).suffix(order))
            }
            eval(x)
        }
        let img = clip(x / 2 + 0.5, min: 0, max: 1).asArray(Float.self)
        return GlyphImages(count: N, size: x.dim(1),
                           pixels: img.map { UInt8(($0 * 255).rounded(.toNearestOrEven)) })
    }
}
