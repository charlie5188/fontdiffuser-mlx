"""Noise schedule and DPM-Solver++ sampling for FontDiffuser, in MLX.

Same algorithm and settings as the official pipeline (DPM-Solver++, order 2, multistep, time-uniform steps,
classifier-free guidance 7.5, "noise" model type), so with the same starting noise the output matches the
PyTorch pipeline. Two things are done differently, for speed only:
- The per-step coefficients are scalars, so they are computed once on the CPU in float64 (numpy).
- The style and content encoders run once per batch instead of at every step: their inputs never change.
"""
import mlx.core as mx
import numpy as np

T = 1000                                    # training diffusion steps


def betas():
    # diffusers DDPMScheduler(beta_schedule="scaled_linear", 1e-4 .. 0.02), computed in float32 like diffusers
    return (np.linspace(1e-4 ** 0.5, 0.02 ** 0.5, T, dtype=np.float32) ** 2).astype(np.float32)


def alphas_cumprod():
    return np.cumprod(1.0 - betas(), dtype=np.float32)


class Schedule:
    """DPM-Solver's discrete-time VP schedule (NoiseScheduleVP('discrete')), as float64 scalar functions."""

    def __init__(self):
        self.log_alpha = 0.5 * np.cumsum(np.log(1.0 - betas().astype(np.float64)))
        self.t_grid = np.linspace(0.0, 1.0, T + 1)[1:]

    def log_alpha_t(self, t):
        return np.interp(t, self.t_grid, self.log_alpha)

    def coeffs(self, t):
        la = self.log_alpha_t(t)
        sigma = np.sqrt(1.0 - np.exp(2 * la))
        # plain Python floats: numpy scalars on the left of an mx.array would turn the result into a numpy array
        return float(np.exp(la)), float(sigma), float(la - np.log(sigma))          # alpha, sigma, lambda


class Sampler:
    """Generate glyphs. `unet` is an optional replacement for the denoiser call (e.g. a Core ML model):
    unet(x, t, cond) -> noise prediction, for the doubled [unconditional, conditional] batch."""

    def __init__(self, model, guidance_scale=7.5, unet=None):
        self.model, self.gs = model, guidance_scale
        # mx.compile fuses the many small ops: about 13 % faster per denoiser call on an M4 Pro. Half precision
        # gave no further speed-up there, so the model stays in fp32. The weights are fixed while sampling.
        self.unet = unet or mx.compile(lambda x, t, cond: model.unet(x, t, cond)[0])
        self.sched = Schedule()

    def encode(self, content, style):
        # classifier-free guidance: [unconditional (all-white images), conditional] stacked along the batch
        white = mx.ones_like(content)
        return self.model.encode(mx.concatenate([white, content]), mx.concatenate([white, style]))

    def __call__(self, content, style, noise, steps=20, order=2):
        """content, style: (N, 96, 96, 3) in [-1, 1] (style may have N=1, then it is shared); noise: (N, 96, 96, 3).
        Returns uint8 images (N, 96, 96, 3)."""
        N = content.shape[0]
        if style.shape[0] == 1 and N > 1:
            style = mx.broadcast_to(style, content.shape)
        cond = self.encode(content, style)
        mx.eval(cond)
        x = noise

        def x0_pred(x, t):
            alpha, sigma, _ = self.sched.coeffs(t)
            t_in = mx.full((2 * N,), (t - 1.0 / T) * 1000.0)            # model time label, as model_wrapper does
            eps = self.unet(mx.concatenate([x, x]), t_in, cond)
            eu, ec = eps[:N], eps[N:]
            eps = eu + self.gs * (ec - eu)
            return (x - sigma * eps) / alpha                            # DPM-Solver++ uses the data prediction

        ts = [float(t) for t in np.linspace(1.0, 1.0 / T, steps + 1)]
        prev_m, prev_t = [x0_pred(x, ts[0])], [ts[0]]
        for step in range(1, steps + 1):
            t = ts[step]
            alpha_t, sigma_t, lam_t = self.sched.coeffs(t)
            _, sigma_p, lam_p = self.sched.coeffs(prev_t[-1])
            h = lam_t - lam_p
            phi = float(np.expm1(-h))
            use_order = min(order, step)
            if steps < 10:                                               # lower_order_final, as in the original
                use_order = min(use_order, steps + 1 - step)
            if use_order == 1:
                x = (sigma_t / sigma_p) * x - (alpha_t * phi) * prev_m[-1]
            else:
                _, _, lam_pp = self.sched.coeffs(prev_t[-2])
                r0 = (lam_p - lam_pp) / h
                d1 = (prev_m[-1] - prev_m[-2]) * (1.0 / r0)
                x = (sigma_t / sigma_p) * x - (alpha_t * phi) * prev_m[-1] - 0.5 * (alpha_t * phi) * d1
            if step < steps:
                prev_m = (prev_m + [x0_pred(x, t)])[-order:]
                prev_t = (prev_t + [t])[-order:]
            mx.eval(x)
        img = mx.clip(x / 2 + 0.5, 0, 1)
        return np.round(np.array(img) * 255).astype(np.uint8)
