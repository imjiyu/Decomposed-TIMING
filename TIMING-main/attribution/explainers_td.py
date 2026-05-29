"""
Trend-Residual decomposition extension of TIMING. (alpha-chunked, OOM-safe)
"""

import torch
import numpy as np
from scipy.interpolate import CubicSpline, UnivariateSpline


def compute_trend_global(inputs):
    B, T, D = inputs.shape
    x_np = inputs.detach().cpu().numpy()
    t = np.arange(T)
    trend_np = np.zeros((B, T, D), dtype=np.float32)
    for b in range(B):
        for d in range(D):
            spl = UnivariateSpline(t, x_np[b, :, d], s=max(T, 1.0))
            trend_np[b, :, d] = spl(t)
    return torch.from_numpy(trend_np).to(inputs.device).float()


def compute_trend_per_mask(inputs, time_mask):
    n_s, B, T, D = time_mask.shape
    x_np = inputs.detach().cpu().numpy()
    m_np = time_mask.detach().cpu().numpy()
    t = np.arange(T)
    trend_np = np.tile(x_np[None], (n_s, 1, 1, 1)).astype(np.float32)
    for s in range(n_s):
        for b in range(B):
            for d in range(D):
                anchor = (m_np[s, b, :, d] == 0)
                if anchor.sum() >= 4:
                    cs = CubicSpline(t[anchor], x_np[b, anchor, d], extrapolate=True)
                    trend_np[s, b, :, d] = cs(t)
    return torch.from_numpy(trend_np).to(inputs.device).float()


def compute_trend_per_mask_kalman(inputs, time_mask, q_level=1e-4, q_slope=1e-2, r=1.0):
    device = inputs.device
    dtype = torch.float32
    n_s, B, T, D = time_mask.shape

    x = inputs.unsqueeze(0).expand(n_s, B, T, D)
    x = x.permute(0, 1, 3, 2).reshape(-1, T).to(dtype)
    obs = (time_mask == 0).permute(0, 1, 3, 2).reshape(-1, T).to(dtype)
    N = x.shape[0]

    cnt = obs.sum(dim=1).clamp(min=1.0)
    mean = (x * obs).sum(dim=1) / cnt
    var = (((x - mean[:, None]) ** 2) * obs).sum(dim=1) / cnt
    std = var.sqrt().clamp(min=1e-6)
    xn = (x - mean[:, None]) / std[:, None]

    F = torch.tensor([[1., 1.], [0., 1.]], device=device, dtype=dtype)
    Q = torch.diag(torch.tensor([q_level, q_slope], device=device, dtype=dtype))
    Fb = F.unsqueeze(0).expand(N, 2, 2)
    eye2 = torch.eye(2, device=device, dtype=dtype).unsqueeze(0).expand(N, 2, 2)

    m_filt = torch.zeros(N, T, 2, device=device, dtype=dtype)
    P_filt = torch.zeros(N, T, 2, 2, device=device, dtype=dtype)

    m = torch.zeros(N, 2, device=device, dtype=dtype)
    P = eye2 * 1e4

    for t in range(T):
        m_pred = (Fb @ m.unsqueeze(-1)).squeeze(-1)
        P_pred = Fb @ P @ Fb.transpose(-1, -2) + Q

        innov = xn[:, t] - m_pred[:, 0]
        S = P_pred[:, 0, 0] + r
        K = P_pred[:, :, 0] / S[:, None]

        m_upd = m_pred + K * innov[:, None]
        KH = torch.zeros(N, 2, 2, device=device, dtype=dtype)
        KH[:, :, 0] = K
        P_upd = (eye2 - KH) @ P_pred
        P_upd = 0.5 * (P_upd + P_upd.transpose(-1, -2))

        o_bool = obs[:, t] > 0.5
        m = torch.where(o_bool[:, None],       m_upd, m_pred)
        P = torch.where(o_bool[:, None, None], P_upd, P_pred)

        m_filt[:, t] = m
        P_filt[:, t] = P

    m_smooth = torch.zeros_like(m_filt)
    m_smooth[:, T - 1] = m_filt[:, T - 1]
    m_prev = m_filt[:, T - 1]

    for t in range(T - 2, -1, -1):
        Pf = P_filt[:, t]
        mf = m_filt[:, t]
        P_pred_next = Fb @ Pf @ Fb.transpose(-1, -2) + Q
        m_pred_next = (Fb @ mf.unsqueeze(-1)).squeeze(-1)

        a = P_pred_next[:, 0, 0]; b = P_pred_next[:, 0, 1]
        c = P_pred_next[:, 1, 0]; d = P_pred_next[:, 1, 1]
        det = a * d - b * c
        det = torch.where(det.abs() < 1e-8, torch.full_like(det, 1e-8), det)
        inv = torch.empty_like(P_pred_next)
        inv[:, 0, 0] = d / det;  inv[:, 0, 1] = -b / det
        inv[:, 1, 0] = -c / det; inv[:, 1, 1] = a / det

        C = Pf @ Fb.transpose(-1, -2) @ inv
        m_s = mf + (C @ (m_prev - m_pred_next).unsqueeze(-1)).squeeze(-1)
        m_smooth[:, t] = m_s
        m_prev = m_s

    trend = m_smooth[:, :, 0] * std[:, None] + mean[:, None]
    no_obs = obs.sum(dim=1) == 0
    trend = torch.where(no_obs[:, None], x, trend)
    trend = trend.reshape(n_s, B, D, T).permute(0, 1, 3, 2)
    return trend.contiguous().float()


class OUR_TD:
    """TIMING + Trend-Residual decomposition."""

    def __init__(self, model):
        self.model = model

    def _ig_phase(self, start, end, time_mask, fixed_inputs,
                  alphas, targets, return_all,
                  n_samples, n_alphas, B, T, D, alpha_chunk):
        direction = end - start
        start4 = start.unsqueeze(1)
        dir4   = direction.unsqueeze(1)
        tm4    = time_mask.unsqueeze(1)
        fix4   = fixed_inputs.unsqueeze(1)

        attr_sum = torch.zeros(n_samples, B, T, D, device=start.device)

        for a0 in range(0, n_alphas, alpha_chunk):
            a1 = min(a0 + alpha_chunk, n_alphas)
            a_chunk = alphas[a0:a1].view(1, -1, 1, 1, 1)
            ck = a1 - a0

            interp = start4 + a_chunk * dir4
            path = tm4 * interp + (1 - tm4) * fix4
            path.requires_grad_(True)

            pred = self.model(
                path.reshape(-1, T, D),
                mask=None, timesteps=None, return_all=return_all,
            )
            if pred.dim() == 1:
                pred = pred.unsqueeze(-1)
            pred = pred.view(n_samples, ck, B, -1)
            tgt = targets.view(1, 1, B, 1).expand(n_samples, ck, B, 1)
            g = pred.gather(3, tgt).squeeze(-1)
            grad = torch.autograd.grad(g.sum(), path, retain_graph=False)[0]
            grad = grad * tm4

            attr_sum += (grad * dir4).sum(dim=1)

            del path, pred, g, grad, interp
            torch.cuda.empty_cache()

        # attr = attr_sum / n_alphas ### completeness 실험 시 이부분 주석 풀기
        # attr = attr.mean(dim=0)    ### completeness 실험 시 이부분 주석 풀기 (대신 아래 3줄 주석 처리)
        
        N_free = time_mask.sum(dim=0)                                    # [B, T, D] time_mask==1인 attribution 흐른 횟수
        attr = attr_sum.sum(dim=0) / (n_alphas * N_free.clamp_min(1))    # 위치별 정규화
        attr = torch.where(N_free > 0, attr, torch.zeros_like(attr))     # free 적 없는 위치는 0
        
        return attr

    def attribute_trend_residual_segments(
        self,
        inputs, baselines, targets, additional_forward_args,
        n_samples=50, num_segments=3,
        max_seg_len=None, min_seg_len=None,
        trend_method="global_spline",
        kalman_q_level=1e-4, kalman_q_slope=1e-2,
        n_alphas=None, alpha_chunk=10,
    ):
        if inputs.shape != baselines.shape:
            raise ValueError("Inputs and baselines must have the same shape.")

        B, T, D = inputs.shape
        device = inputs.device
        return_all = additional_forward_args[2]

        if max_seg_len is None: max_seg_len = T
        if min_seg_len is None: min_seg_len = 1
        if n_alphas is None: n_alphas = n_samples

        alphas = torch.linspace(0, 1 - 1/n_alphas, n_alphas, device=device)

        time_mask = torch.ones(n_samples, B, T, D, device=device)
        dims     = torch.randint(0, D, (n_samples, B, num_segments), device=device)
        seg_lens = torch.randint(min_seg_len, max_seg_len+1, (n_samples, B, num_segments), device=device)
        t_starts = (torch.rand(n_samples, B, num_segments, device=device) * (T - seg_lens)).long()

        batch_indices  = torch.arange(B, device=device)
        sample_indices = torch.arange(n_samples, device=device)
        for s in range(num_segments):
            mlen = seg_lens[:, :, s].max()
            base_range = torch.arange(mlen, device=device).unsqueeze(0).unsqueeze(0)
            indices = t_starts[:, :, s].unsqueeze(-1) + base_range
            end_points = (t_starts[:, :, s] + seg_lens[:, :, s]).unsqueeze(-1)
            valid = (indices < end_points) & (indices < T)
            time_mask[sample_indices.view(-1, 1, 1),
                      batch_indices.view(1, -1, 1),
                      indices * valid,
                      dims[:, :, s].unsqueeze(-1)] = 0

        if trend_method == "global_spline":
            trends = compute_trend_global(inputs).unsqueeze(0).expand(n_samples, B, T, D).contiguous()
        elif trend_method == "spline":
            trends = compute_trend_per_mask(inputs, time_mask)
        elif trend_method == "kalman":
            trends = compute_trend_per_mask_kalman(
                inputs, time_mask, q_level=kalman_q_level, q_slope=kalman_q_slope,
            )
        else:
            raise ValueError(f"Unknown trend_method: {trend_method}")

        baselines_s = baselines.unsqueeze(0).expand(n_samples, B, T, D).contiguous()
        inputs_s    = inputs.unsqueeze(0).expand(n_samples, B, T, D).contiguous()
        fixed_inputs = inputs_s.detach()

        trend_attr = self._ig_phase(
            baselines_s, trends, time_mask, fixed_inputs,
            alphas, targets, return_all,
            n_samples, n_alphas, B, T, D, alpha_chunk)

        resid_attr = self._ig_phase(
            trends, inputs_s, time_mask, fixed_inputs,
            alphas, targets, return_all,
            n_samples, n_alphas, B, T, D, alpha_chunk)

        with torch.no_grad():
            c_masked = time_mask * baselines_s + (1 - time_mask) * fixed_inputs
            fc = self.model(c_masked.reshape(-1, T, D),
                            mask=None, timesteps=None, return_all=return_all)
            fx = self.model(inputs,
                            mask=None, timesteps=None, return_all=return_all)
            if fc.dim() == 1: fc = fc.unsqueeze(-1)
            if fx.dim() == 1: fx = fx.unsqueeze(-1)
            fc = fc.view(n_samples, B, -1)
            fc = fc.gather(2, targets.view(1, B, 1).expand(n_samples, B, 1)).squeeze(-1)
            fx = fx.gather(1, targets.view(B, 1)).squeeze(-1)
            fxc = fx - fc.mean(dim=0)

        return trend_attr, resid_attr, fxc