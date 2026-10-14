#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_unmix_tiled_imageinput_revised.py
==============================================================================
分块版高光谱盲解混：以"影像为输入"进行端元/丰度估计，导出 E[bands,r] 与 A[H,W,r]。

本版本在原始脚本基础上，按审稿意见做了以下与代码相关的修改（不改变核心算法行为）：

  [R2.3 / R2.4] 预处理由"降维"更正为"低秩去噪(Low-rank denoising)"：
        SVD 仅做秩-r 子空间投影重建  Y_hat = U_r U_r^T Y，输出仍保留全部 p 个波段，
        网络在全波段上学习；新增可配置 `svd_rank`（默认= rmax），并以注释给出显式公式。
        `use_svd` 开关可直接用于"有/无低秩去噪"的消融实验。

  [R2.9] 训练过程可视化：逐迭代记录全部损失分量(L2/SAM/D2/VOL/TV/BAR)与总损失，
        保存为 history.csv / history.npz，并输出训练曲线图 training_curves.png/svg。

  [R2.10] 评估与统计：计算整图重建 RMSE/SAM；若提供真值(端元/丰度)则计算端元 SAM/RMSE
        与丰度 RMSE（贪心/匈牙利匹配）；提供多随机种子重复实验 `run_multi_seed`，
        报告 mean ± std（盲解混逐图优化，不适用 k 折交叉验证，故以重复种子代替）。

  [R1.5 / R2.10] 资源消耗定量分析：参数量、FLOPs(若安装 thop)、训练时间、推理时间、
        峰值显存，保存为 resource_report.txt / .json。

  [日志] 所有运行日志通过 logging 同时写入控制台与文件 run.log（带时间戳）。

  [R2.11] 说明：本方法为逐场景自监督盲解混，network 对每幅输入影像独立优化，
        无跨数据集训练 / 迁移 / 微调（见 process_hyperspectral_image 文档字符串）。
==============================================================================
"""

import os
import sys
import csv
import json
import time
import logging
import datetime

import numpy as np
import scipy.io
import scipy.linalg

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --------------------------
# 设备
# --------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.enabled = True if device.type == 'cuda' else False
torch.backends.cudnn.benchmark = True if device.type == 'cuda' else False


# ==========================================================================
# 日志：同时写入控制台与文件（带时间戳）
# ==========================================================================
def setup_logger(output_dir, name="unmix", log_name="run.log"):
    """创建一个同时输出到 stdout 和 output_dir/run.log 的 logger。"""
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()           # 避免重复 handler（多次调用时）
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(os.path.join(output_dir, log_name), mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.info("Logger initialized. Log file -> %s",
                os.path.join(output_dir, log_name))
    return logger


# --------------------------
# 数学与数值稳定
# --------------------------
def inv_sigmoid(x, eps=1e-6):
    x = np.clip(x, eps, 1 - eps)
    return np.log(x / (1 - x))

def dct_matrix(n):
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    mat = np.cos(np.pi * (i + 0.5) * k / n)
    mat[0, :] = mat[0, :] / np.sqrt(2)
    mat = mat * np.sqrt(2 / n)
    return mat

def hapke_forward(E, A, m=1.0, m0=1.0, eps=1e-8):
    E = E.clamp(eps, 1 - eps)
    denom_W1 = 1 + 4.0 * m * m0 * E
    inside = (m0 + m) ** 2 * (E ** 2) + denom_W1 * (1.0 - E)
    inside = torch.clamp(inside, min=eps)
    W  = torch.sqrt(inside) - (m0 + m) * E
    W1 = W / denom_W1
    End3 = (1.0 - W1 ** 2).clamp(eps, 1 - eps)        # [p, r]
    HR = torch.matmul(End3, A).clamp(eps, 1 - eps)     # [p, N]
    sqrt_term = torch.sqrt(torch.clamp(1.0 - HR, min=eps))
    Temp1 = 1.0 + 2.0 * m  * sqrt_term
    Temp2 = 1.0 + 2.0 * m0 * sqrt_term
    R = HR / (Temp1 * Temp2 + eps)
    return R.clamp(eps, 1 - eps)

def safe_neg_logdet(mat, base_jitter=1e-6, max_tries=6):
    I = torch.eye(mat.shape[0], device=mat.device, dtype=mat.dtype)
    jitter = base_jitter
    for _ in range(max_tries):
        sym = 0.5 * (mat + mat.transpose(0,1)) + jitter * I
        sign, logabsdet = torch.linalg.slogdet(sym)
        if torch.isfinite(logabsdet) and (sign > 0):
            return -logabsdet
        jitter *= 10
    return torch.tensor(0.0, device=mat.device)

def loss_total(target, A_maps, endmember_fn, weights, m=1.0, m0=1.0):
    """
    target: [1,p,H,W]; A_maps: [1,r,H,W]; endmember_fn()-> E:[p,r]
    """
    _, p, H, W = target.shape
    r = A_maps.shape[1]
    N = H * W

    E = endmember_fn()
    E = torch.nan_to_num(E, nan=0.5, posinf=0.999, neginf=0.001)

    Y = target.reshape(p, N).clamp(0.0, 1.0)
    A = A_maps.reshape(r, N)
    A = torch.nan_to_num(A, nan=1.0/r, posinf=1.0, neginf=0.0)
    A = A.clamp_min(1e-8); A = A / A.sum(0, keepdim=True).clamp_min(1e-8)

    Y_pred = hapke_forward(E, A, m=m, m0=m0)
    recon_l2 = torch.mean((Y_pred - Y).pow(2))

    num = (Y_pred * Y).sum(0)
    den = Y_pred.norm(dim=0) * Y.norm(dim=0) + 1e-8
    cos = torch.clamp(num / den, -1+1e-7, 1-1e-7)
    sam = torch.acos(cos[torch.isfinite(cos)]).mean() if torch.isfinite(cos).any() \
          else torch.zeros((), device=Y.device)

    d2 = (E[2:, :] - 2 * E[1:-1, :] + E[:-2, :]).pow(2).mean() if p >= 3 \
         else torch.zeros((), device=E.device)

    C = (E - E.mean(0, keepdim=True)).T @ (E - E.mean(0, keepdim=True)) / max(p-1, 1)
    vol_reg = safe_neg_logdet(C, base_jitter=1e-6, max_tries=6)

    A_map = A_maps.clamp_min(0.0)
    tv = (A_map[:, :, 1:, :] - A_map[:, :, :-1, :]).abs().mean() + \
         (A_map[:, :, :, 1:] - A_map[:, :, :, :-1]).abs().mean()

    barrier = - (E.clamp(1e-6, 1-1e-6).log() + (1 - E).clamp(1e-6,1-1e-6).log()).mean()

    w = weights
    total = (w['l2']*recon_l2 + w['sam']*sam + w['d2']*d2 +
             w['vol']*vol_reg + w['tv']*tv + w['bar']*barrier)
    return total, {'l2':recon_l2, 'sam':sam, 'd2':d2, 'vol':vol_reg, 'tv':tv, 'bar':barrier}

def get_weights(iter_idx, num_iter_total):
    if iter_idx < 2000:
        return {'l2':1.0,'sam':0.10,'d2':0.02,'vol':0.003,'tv':0.0005,'bar':0.05}
    elif iter_idx < 5000:
        return {'l2':1.0,'sam':0.15,'d2':0.03,'vol':0.006,'tv':0.0005,'bar':0.04}
    else:
        return {'l2':1.0,'sam':0.20,'d2':0.04,'vol':0.010,'tv':0.0005,'bar':0.03}

def scale_cube_01(cube_bhw, per_band=False):
    x = cube_bhw.astype(np.float32, copy=False)
    if per_band:
        b, h, w = x.shape
        x2 = x.reshape(b, -1)
        mn = np.nanmin(x2, axis=1, keepdims=True)
        mx = np.nanmax(x2, axis=1, keepdims=True)
        x2 = (x2 - mn) / np.maximum(mx - mn, 1e-8)
        return np.nan_to_num(x2.reshape(b, h, w))
    mn, mx = float(np.nanmin(x)), float(np.nanmax(x))
    return np.nan_to_num((x - mn) / max(mx - mn, 1e-8))


# ==========================================================================
# [R2.3 / R2.4] 低秩去噪（Low-rank denoising，非降维）
# ==========================================================================
def low_rank_denoise(img_np, svd_rank):
    """
    秩-r 低秩去噪（不是降维）。

    展平 Y = [p, N]，做 SVD：  Y = U Σ V^T
    取前 r 个左奇异向量 U_r，子空间投影重建：
                    Y_hat = U_r U_r^T Y   ∈  R^{p x N}
    重建结果仍保留全部 p 个波段，仅滤除落在小奇异值（噪声主导）子空间的分量。

    img_np : [p, H, W]
    返回    : 去噪后的 [p, H, W]（值域裁剪到 [0,1]）
    """
    p1, H, W = img_np.shape
    Y = img_np.reshape(p1, -1)                               # [p, N]
    U, s, Vt = scipy.linalg.svd(Y, full_matrices=False)      # U:[p,p]
    r = int(max(1, min(svd_rank, U.shape[1])))
    Ur = U[:, :r]
    Y_hat = (Ur @ Ur.T) @ Y                                  # [p, N]，保留全部 p 波段
    img_dn = np.clip(Y_hat, 0, 1).reshape(p1, H, W)
    # 估计去噪强度（被滤除能量占比），便于日志记录
    energy_total = float(np.sum(s ** 2)) + 1e-12
    energy_kept  = float(np.sum(s[:r] ** 2))
    removed_ratio = 1.0 - energy_kept / energy_total
    return img_dn, dict(svd_rank=r, removed_energy_ratio=removed_ratio,
                        singular_values_head=[float(v) for v in s[:min(10, len(s))]])


# --------------------------
# 模型：丰度 CAE + DCT 端元（输入=影像）
# --------------------------
class CAE_EndEst(nn.Module):
    def __init__(self, input_depth, p1, rmax, K_basis=24):
        super().__init__()
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(input_depth, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.spectral_branch = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(256, 128, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, rmax, 1)
        )

        self.p1 = p1
        self.rmax = rmax
        self.K = min(K_basis, p1)

        Phi_full = dct_matrix(p1)
        Phi = Phi_full[:self.K, :].T
        self.register_buffer('Phi', torch.from_numpy(Phi).float())
        self.register_buffer('freq_w', torch.linspace(0, 1, self.K).view(self.K, 1))
        self.C = nn.Parameter(torch.randn(self.K, rmax) * 0.01)

        self.register_buffer('temperature', torch.tensor(1.5))
        self.register_buffer('E_eps', torch.tensor(0.02))
        self.register_buffer('E_temp', torch.tensor(1.5))

    def endmembers_raw(self):
        return torch.matmul(self.Phi, self.C)

    def endmembers(self):
        Z = self.endmembers_raw()
        E = torch.sigmoid(Z / self.E_temp.clamp_min(1.0))
        E = E * (1.0 - 2.0 * self.E_eps) + self.E_eps
        return E

    def forward(self, x):
        B, C, H, W = x.shape
        xs = self.spatial_branch(x)
        xr = x.permute(0,2,3,1).reshape(B*H*W,1,C)
        xr = self.spectral_branch(xr).mean(-1).view(B,H,W,128).permute(0,3,1,2)
        logits = self.fusion_conv(torch.cat([xs, xr], dim=1))
        A_maps = F.softmax(logits / self.temperature.clamp_min(1.2), dim=1)
        return A_maps


# --------------------------
# 端元初始化（PCA 主方向 + 极值 + 随机投影补齐）
# --------------------------
def init_endmembers_via_pca_extrema(Y_pN, r, trials=64, seed=0):
    rng = np.random.default_rng(seed)
    p, N = Y_pN.shape
    C = (Y_pN @ Y_pN.T) / max(N - 1, 1)
    w, V = np.linalg.eigh(C)
    V = V[:, ::-1][:, :min(r, p)]
    proj = V.T @ Y_pN
    idx = list(np.argmax(proj, axis=1))
    counts = {}
    for _ in range(trials):
        u = rng.normal(size=(p,)); u /= (np.linalg.norm(u)+1e-12)
        k = int(np.argmax(u @ Y_pN))
        counts[k] = counts.get(k, 0) + 1
    extra = [k for k,_ in sorted(counts.items(), key=lambda x:-x[1]) if k not in idx]
    idx += extra
    if len(idx) < r:
        pool = [j for j in range(N) if j not in idx]
        if len(pool) >= r - len(idx):
            idx += rng.choice(pool, size=r-len(idx), replace=False).tolist()
        else:
            idx += rng.choice(N, size=r-len(idx), replace=False).tolist()
    idx = idx[:r]
    E = Y_pN[:, idx]
    return np.clip(E, 1e-6, 1-1e-6)


# --------------------------
# Patch/滑窗工具
# --------------------------
def make_patch_grid(H, W, ph, pw, stride):
    ys = list(range(0, max(H - ph, 0) + 1, max(stride, 1)))
    xs = list(range(0, max(W - pw, 0) + 1, max(stride, 1)))
    if len(ys) == 0 or ys[-1] != H - ph: ys.append(max(H - ph, 0))
    if len(xs) == 0 or xs[-1] != W - pw: xs.append(max(W - pw, 0))
    return ys, xs

def sample_random_patch_coords(H, W, ph, pw):
    y = np.random.randint(0, max(H - ph, 1))
    x = np.random.randint(0, max(W - pw, 1))
    y = min(y, H - ph); x = min(x, W - pw)
    return y, x

def sliding_window_infer_A_image(net, r, img_cpu, ph, pw, stride):
    _, _, H, W = img_cpu.shape
    A_accum = torch.zeros(r, H, W, dtype=torch.float32)
    W_accum = torch.zeros(1, H, W, dtype=torch.float32)

    ph = max(int(ph), 1); pw = max(int(pw), 1)
    w_h = torch.hann_window(ph, periodic=False, dtype=torch.float32)
    w_w = torch.hann_window(pw, periodic=False, dtype=torch.float32)
    w2d = (w_h[:, None] * w_w[None, :])
    if float(w2d.max()) == 0.0:
        w2d = torch.ones_like(w2d)
    w2d = (w2d / (w2d.max() + 1e-8)).unsqueeze(0)

    ys, xs = make_patch_grid(H, W, ph, pw, stride)
    net.eval()
    with torch.no_grad():
        for y in ys:
            for x in xs:
                x_patch = img_cpu[..., y:y+ph, x:x+pw].to(device, non_blocking=True)
                A_patch = net(x_patch)[0].detach().cpu()
                A_accum[:, y:y+ph, x:x+pw] += A_patch * w2d
                W_accum[:, y:y+ph, x:x+pw] += w2d

    A_accum /= W_accum.clamp_min(1e-8)
    return A_accum.numpy().clip(0, 1)


# ==========================================================================
# [R2.10] 评估指标
# ==========================================================================
def _sam_columns(X, Y, eps=1e-8):
    """逐列光谱角(弧度)。X,Y: [p, k] -> [k]"""
    num = (X * Y).sum(0)
    den = np.linalg.norm(X, axis=0) * np.linalg.norm(Y, axis=0) + eps
    cos = np.clip(num / den, -1 + 1e-7, 1 - 1e-7)
    return np.arccos(cos)

def compute_recon_metrics(E, A_rHW, Y_pHW, m=1.0, m0=1.0):
    """整图重建 RMSE / SAM（与去噪后输入比较）。"""
    p, H, W = Y_pHW.shape
    r = A_rHW.shape[0]
    Et = torch.from_numpy(E.astype(np.float32))
    At = torch.from_numpy(A_rHW.reshape(r, -1).astype(np.float32))
    At = At.clamp_min(1e-8); At = At / At.sum(0, keepdim=True).clamp_min(1e-8)
    with torch.no_grad():
        Ypred = hapke_forward(Et, At, m=m, m0=m0).numpy()       # [p, N]
    Y = Y_pHW.reshape(p, -1)
    rmse = float(np.sqrt(np.mean((Ypred - Y) ** 2)))
    sam = float(np.degrees(np.mean(_sam_columns(Ypred, Y))))
    return dict(recon_rmse=rmse, recon_sam_deg=sam)

def _match_columns(E_est, E_gt):
    """按 SAM 最小做端元匹配；优先匈牙利，缺 scipy.optimize 时回退贪心。"""
    r = E_gt.shape[1]
    cost = np.zeros((r, r))
    for i in range(r):
        for j in range(r):
            cost[i, j] = _sam_columns(E_est[:, [i]], E_gt[:, [j]])[0]
    try:
        from scipy.optimize import linear_sum_assignment
        ri, cj = linear_sum_assignment(cost)
        return list(zip(ri.tolist(), cj.tolist()))
    except Exception:
        pairs, used = [], set()
        for i in range(r):
            order = np.argsort(cost[i])
            for j in order:
                if j not in used:
                    pairs.append((i, j)); used.add(j); break
        return pairs

def compute_gt_metrics(E_est, A_est_rHW, E_gt=None, A_gt_rHW=None):
    """若提供真值，计算端元 SAM/RMSE 与丰度 RMSE（匹配后）。"""
    out = {}
    if E_gt is not None:
        pairs = _match_columns(E_est, E_gt)
        sams, rmses = [], []
        for i, j in pairs:
            sams.append(np.degrees(_sam_columns(E_est[:, [i]], E_gt[:, [j]])[0]))
            rmses.append(np.sqrt(np.mean((E_est[:, i] - E_gt[:, j]) ** 2)))
        out['endmember_sam_deg_mean'] = float(np.mean(sams))
        out['endmember_rmse_mean'] = float(np.mean(rmses))
        out['endmember_match'] = pairs
        if A_gt_rHW is not None:
            r = E_gt.shape[1]
            a_est = A_est_rHW.reshape(r, -1)
            a_gt = A_gt_rHW.reshape(r, -1)
            perm = [j for _, j in sorted(pairs, key=lambda x: x[0])]
            a_est_p = a_est[[i for i, _ in sorted(pairs, key=lambda x: x[0])]]
            a_gt_p = a_gt[perm]
            out['abundance_rmse_mean'] = float(np.sqrt(np.mean((a_est_p - a_gt_p) ** 2)))
    return out


# ==========================================================================
# [R1.5 / R2.10] 资源消耗分析
# ==========================================================================
def count_parameters(net):
    total = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total, trainable

def estimate_flops(net, sample):
    """优先用 thop 统计单次前向 FLOPs；未安装则返回 None。"""
    try:
        from thop import profile
        macs, _ = profile(net, inputs=(sample,), verbose=False)
        return float(macs) * 2.0       # FLOPs ≈ 2 × MACs
    except Exception:
        return None


# ==========================================================================
# [R2.9] 训练曲线绘制
# ==========================================================================
def plot_training_curves(history, output_dir):
    """history: dict of lists，至少包含 iter/total/l2/sam/d2/vol/tv/bar。"""
    it = np.asarray(history['iter'])
    keys = ['total', 'l2', 'sam', 'd2', 'vol', 'tv', 'bar']
    colors = {'total':'#2B2B2B', 'l2':'#0F4D92', 'sam':'#B64342', 'd2':'#2E7D52',
              'vol':'#9A4D8E', 'tv':'#C8922A', 'bar':'#42949E'}

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['svg.fonttype'] = 'none'

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    # 左：总损失 + L2 + SAM（主指标）
    for k in ['total', 'l2', 'sam']:
        v = np.asarray(history[k], dtype=float)
        axes[0].plot(it, np.clip(v, 1e-8, None), color=colors[k], lw=1.6, label=k.upper())
    axes[0].set_yscale('log'); axes[0].set_xlabel('Iteration'); axes[0].set_ylabel('Loss (log)')
    axes[0].set_title('Total / reconstruction (L2) / SAM')
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)
    # 右：各正则分量
    for k in ['d2', 'vol', 'tv', 'bar']:
        v = np.asarray(history[k], dtype=float)
        axes[1].plot(it, np.clip(np.abs(v), 1e-10, None), color=colors[k], lw=1.4, label=k.upper())
    axes[1].set_yscale('log'); axes[1].set_xlabel('Iteration'); axes[1].set_ylabel('|term| (log)')
    axes[1].set_title('Regularization terms')
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)

    fig.tight_layout(pad=1.5)
    fig.savefig(os.path.join(output_dir, "training_curves.png"), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, "training_curves.svg"), bbox_inches='tight')
    plt.close(fig)

def save_history(history, output_dir):
    keys = ['iter', 'phase', 'total', 'l2', 'sam', 'd2', 'vol', 'tv', 'bar', 'temperature']
    with open(os.path.join(output_dir, "history.csv"), "w", newline="", encoding="utf-8") as f:
        wri = csv.writer(f); wri.writerow(keys)
        for i in range(len(history['iter'])):
            wri.writerow([history[k][i] for k in keys])
    np.savez(os.path.join(output_dir, "history.npz"),
             **{k: np.asarray(v) for k, v in history.items()
                if k != 'phase'})


# --------------------------
# 核心流程（以影像为输入）
# --------------------------
def process_hyperspectral_image(
    img_np, rmax, output_dir="results",
    num_iter=8000, show_every=800, exp_weight=0.99,
    lr_A=1e-4, lr_E=2e-5, m=1.0, m0=1.0, K_basis=24, ao_period=6,
    train_patch=(128,128), infer_patch=(192,192), stride=128,
    seed=0, use_svd=True, svd_rank=None,
    gt_endmembers=None, gt_abundances=None,
    logger=None
):
    """
    逐场景自监督盲解混（per-scene self-supervised blind unmixing）。
    network 仅在该输入影像上优化并对其推理，无跨数据集训练/迁移/微调 [R2.11]。

    img_np: [bands,H,W]（若不在 [0,1] 将自动缩放）
    svd_rank: 低秩去噪保留的秩，默认 None -> 取 rmax [R2.3]
    gt_endmembers/gt_abundances: 可选真值，用于精度评估 [R2.10]
    return: E_final [p,r], A_full [r,H,W], report(dict)
    """
    os.makedirs(output_dir, exist_ok=True)
    if logger is None:
        logger = setup_logger(output_dir)

    # 统一缩放到 [0,1]
    img_np = scale_cube_01(img_np, per_band=False)
    p1, H, W = img_np.shape
    if svd_rank is None:
        svd_rank = rmax

    logger.info("=" * 70)
    logger.info("Per-scene self-supervised blind unmixing [R2.11]")
    logger.info("device=%s | p=%d | HxW=%dx%d | r=%d | K=%d", device.type, p1, H, W, rmax, K_basis)
    logger.info("train_patch=%s infer_patch=%s stride=%d num_iter=%d seed=%d",
                train_patch, infer_patch, stride, num_iter, seed)
    logger.info("use_svd(low-rank denoise)=%s svd_rank=%s [R2.3/R2.4]", use_svd, svd_rank)

    # ---- [R2.3/R2.4] 低秩去噪（非降维），输出仍为全部 p 波段 ----
    if use_svd:
        img_dn, dn_info = low_rank_denoise(img_np, svd_rank=svd_rank)
        logger.info("Low-rank denoising: kept rank=%d, removed noise-energy ratio=%.4f",
                    dn_info['svd_rank'], dn_info['removed_energy_ratio'])
    else:
        img_dn = img_np
        dn_info = dict(svd_rank=0, removed_energy_ratio=0.0, note="ablation: no denoising")
        logger.info("Low-rank denoising DISABLED (ablation for R2.3).")

    img_cpu = torch.from_numpy(img_dn).float().unsqueeze(0)   # [1,p,H,W] CPU

    # 端元初始化
    YpN = img_dn.reshape(p1, -1)
    E0 = init_endmembers_via_pca_extrema(YpN, r=rmax, trials=64, seed=seed)

    # 模型
    net = CAE_EndEst(input_depth=p1, p1=p1, rmax=rmax, K_basis=K_basis).to(device)

    # 初始化 C：岭回归
    with torch.no_grad():
        Phi = net.Phi.cpu().numpy()
        Z = inv_sigmoid(E0)
        lam = 1e-2
        C0 = np.linalg.solve(Phi.T @ Phi + lam * np.eye(Phi.shape[1]), Phi.T @ Z)
        net.C.copy_(torch.from_numpy(C0).float().to(device))

    # ---- [R1.5] 资源：参数量 + FLOPs ----
    total_params, trainable_params = count_parameters(net)
    ph0, pw0 = (min(train_patch[0], H), min(train_patch[1], W))
    flops = estimate_flops(net, torch.zeros(1, p1, ph0, pw0, device=device))
    logger.info("Model params: total=%d (%.3f M), trainable=%d | FLOPs/forward=%s",
                total_params, total_params/1e6, trainable_params,
                ("%.3e" % flops) if flops else "N/A (install thop)")

    params_A = [p for n,p in net.named_parameters() if n != 'C']
    opt_A = torch.optim.Adam(params_A, lr=lr_A, betas=(0.9,0.999), eps=1e-8)
    opt_E = torch.optim.Adam([net.C], lr=lr_E, betas=(0.9,0.999), eps=1e-8)

    ph, pw = train_patch
    ph = min(int(ph), H); pw = min(int(pw), W)

    # ---- [R2.9] 训练历史容器 ----
    history = {k: [] for k in ['iter','phase','total','l2','sam','d2','vol','tv','bar','temperature']}
    def _rec(i, phase, total, parts, T):
        history['iter'].append(int(i)); history['phase'].append(phase)
        history['total'].append(float(total.detach().cpu()))
        for k in ['l2','sam','d2','vol','tv','bar']:
            history[k].append(float(parts[k].detach().cpu()))
        history['temperature'].append(float(T))

    out_avg = None
    i = 0; t = 0
    torch.manual_seed(seed); np.random.seed(seed)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    t_train0 = time.perf_counter()
    logger.info("Start training (patch sampling, image as input) ...")

    while t < num_iter:
        for _ in range(ao_period-1):           # 更新 A 分支
            if t >= num_iter: break
            y, x = sample_random_patch_coords(H, W, ph, pw)
            target = img_cpu[..., y:y+ph, x:x+pw].to(device, non_blocking=True)
            x_in = target

            opt_E.zero_grad(set_to_none=True)
            A_maps = net(x_in)

            with torch.no_grad():
                if out_avg is None: out_avg = A_maps.detach()
                else: out_avg = out_avg * exp_weight + A_maps.detach() * (1 - exp_weight)
                T = 1.5 if i < 2000 else 1.5 - 0.3 * min((i-2000)/max(num_iter-2000,1), 1.0)
                net.temperature.copy_(torch.tensor(T, device=device))
                net.E_temp.copy_(torch.tensor(T, device=device))

            weights = get_weights(i, num_iter)
            total, parts = loss_total(target, A_maps, net.endmembers, weights, m=m, m0=m0)
            if not torch.isfinite(total):
                weights = {'l2':1.0,'sam':0.,'d2':0.,'vol':0.,'tv':0.,'bar':0.}
                total, parts = loss_total(target, A_maps, net.endmembers, weights, m=m, m0=m0)

            lam_c, lam_d2z, lam_z2 = 5e-4, 5e-2, 1e-3
            c_reg = lam_c * (net.C * (1.0 + 10.0 * net.freq_w)).pow(2).mean()
            Z = net.endmembers_raw()
            d2z = (Z[2:, :] - 2*Z[1:-1, :] + Z[:-2, :]).pow(2).mean() if Z.shape[0] >= 3 else torch.zeros((), device=Z.device)
            total = total + c_reg + lam_d2z * d2z + lam_z2 * Z.pow(2).mean()

            total.backward()
            torch.nn.utils.clip_grad_norm_(params_A, 1.0)
            opt_A.step()

            _rec(i, 'A', total, parts, T)
            if (i % show_every) == 0:
                def sf(x): return float(x.detach().cpu()) if torch.isfinite(x) else float('nan')
                logger.info("Iter %05d [A] | Loss=%.6f L2=%.4e SAM=%.4e VOL=%.4e",
                            i, sf(total), sf(parts['l2']), sf(parts['sam']), sf(parts['vol']))
            i += 1; t += 1
            if t >= num_iter: break

        if t >= num_iter: break
        # 更新端元 C（1 次）
        y, x = sample_random_patch_coords(H, W, ph, pw)
        target = img_cpu[..., y:y+ph, x:x+pw].to(device, non_blocking=True)
        x_in = target

        opt_A.zero_grad(set_to_none=True)
        A_maps = net(x_in)
        weights = get_weights(i, num_iter)
        total, parts = loss_total(target, A_maps, net.endmembers, weights, m=m, m0=m0)
        if not torch.isfinite(total):
            weights = {'l2':1.0,'sam':0.,'d2':0.,'vol':0.,'tv':0.,'bar':0.}
            total, parts = loss_total(target, A_maps, net.endmembers, weights, m=m, m0=m0)

        lam_c, lam_d2z, lam_z2 = 5e-4, 5e-2, 1e-3
        c_reg = lam_c * (net.C * (1.0 + 10.0 * net.freq_w)).pow(2).mean()
        Z = net.endmembers_raw()
        d2z = (Z[2:, :] - 2*Z[1:-1, :] + Z[:-2, :]).pow(2).mean() if Z.shape[0] >= 3 else torch.zeros((), device=Z.device)
        total = total + c_reg + lam_d2z * d2z + lam_z2 * Z.pow(2).mean()

        total.backward()
        torch.nn.utils.clip_grad_norm_([net.C], 1.0)
        opt_E.step()

        _rec(i, 'E', total, parts, float(net.temperature.detach().cpu()))
        if (i % show_every) == 0:
            def sf(x): return float(x.detach().cpu()) if torch.isfinite(x) else float('nan')
            logger.info("Iter %05d [E] | Loss=%.6f L2=%.4e SAM=%.4e VOL=%.4e",
                        i, sf(total), sf(parts['l2']), sf(parts['sam']), sf(parts['vol']))
        i += 1; t += 1

    train_time = time.perf_counter() - t_train0
    logger.info("Training done in %.2f s (%.2f iters/s)", train_time, num_iter/max(train_time,1e-9))

    # 训练完成：端元
    E_final = net.endmembers().detach().cpu().numpy().clip(1e-6, 1-1e-6)

    # ---- 分块推理（整图） + 计时 ----
    iph, ipw = infer_patch
    iph = min(int(iph), H); ipw = min(int(ipw), W)
    logger.info("Sliding-window inference (image as input) ...")
    t_inf0 = time.perf_counter()
    A_full = sliding_window_infer_A_image(net, r=rmax, img_cpu=img_cpu, ph=iph, pw=ipw, stride=int(stride))
    infer_time = time.perf_counter() - t_inf0
    logger.info("Inference done in %.2f s", infer_time)

    peak_mem_mb = (torch.cuda.max_memory_allocated() / 1024**2) if device.type == 'cuda' else None

    # ---- [R2.9] 保存历史 + 训练曲线 ----
    save_history(history, output_dir)
    plot_training_curves(history, output_dir)
    logger.info("Saved training history (history.csv/.npz) and curves (training_curves.png/svg) [R2.9]")

    # ---- [R2.10] 指标 ----
    recon = compute_recon_metrics(E_final, A_full, img_dn, m=m, m0=m0)
    logger.info("Reconstruction metrics: RMSE=%.6f  SAM=%.4f deg", recon['recon_rmse'], recon['recon_sam_deg'])
    gt_metrics = compute_gt_metrics(E_final, A_full, E_gt=gt_endmembers, A_gt_rHW=gt_abundances)
    if gt_metrics:
        logger.info("GT metrics: %s", json.dumps({k:v for k,v in gt_metrics.items() if k!='endmember_match'}))

    # ---- 保存结果 ----
    scipy.io.savemat(os.path.join(output_dir, "endmembers.mat"), {'endmembers': E_final})
    scipy.io.savemat(os.path.join(output_dir, "abundances.mat"), {'abundances': A_full.transpose(1,2,0)})
    np.save(os.path.join(output_dir, "E_endmembers.npy"), E_final)
    np.save(os.path.join(output_dir, "A_abundances.npy"), A_full.transpose(1,2,0))

    # 端元谱线 + 丰度图
    plt.figure(figsize=(10,4)); plt.plot(E_final); plt.title('Final endmember spectra'); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "final_endmembers.png")); plt.close()
    for k in range(rmax):
        plt.figure(figsize=(6,6)); plt.imshow(A_full[k], cmap='rainbow', vmin=0.0, vmax=1.0)
        plt.title(f'Abundance map {k+1}'); plt.colorbar(); plt.axis('off'); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"abundance_map_{k+1}.png")); plt.close()

    # ---- [R1.5/R2.10] 资源报告 ----
    report = dict(
        device=device.type, bands=p1, height=H, width=W, rmax=rmax, K_basis=K_basis,
        num_iter=num_iter, seed=seed, use_svd=use_svd, svd_rank=int(dn_info['svd_rank']),
        removed_energy_ratio=dn_info.get('removed_energy_ratio'),
        params_total=int(total_params), params_trainable=int(trainable_params),
        params_million=round(total_params/1e6, 4),
        flops_per_forward=flops, train_time_sec=round(train_time, 3),
        infer_time_sec=round(infer_time, 3), peak_gpu_mem_mb=peak_mem_mb,
        recon_rmse=recon['recon_rmse'], recon_sam_deg=recon['recon_sam_deg'],
    )
    report.update({k: v for k, v in gt_metrics.items() if k != 'endmember_match'})
    with open(os.path.join(output_dir, "resource_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "resource_report.txt"), "w", encoding="utf-8") as f:
        f.write("Resource & accuracy report  [R1.5 / R2.10]\n")
        f.write("-" * 50 + "\n")
        for k, v in report.items():
            f.write(f"{k:24s}: {v}\n")
    logger.info("Saved resource_report.json/.txt [R1.5/R2.10]")
    logger.info("[Done] Saved to: %s", output_dir)

    return E_final, A_full, report


# ==========================================================================
# [R2.10] 多随机种子重复实验（盲解混不适用 k 折交叉验证，以重复种子代替）
# ==========================================================================
def run_multi_seed(img_np, seeds=(0, 1, 2), base_output_dir="results_multiseed",
                   gt_endmembers=None, gt_abundances=None, **cfg):
    os.makedirs(base_output_dir, exist_ok=True)
    logger = setup_logger(base_output_dir, name="multiseed", log_name="multiseed.log")
    logger.info("Multi-seed repeated runs: seeds=%s [R2.10]", list(seeds))

    rows = []
    for s in seeds:
        od = os.path.join(base_output_dir, f"seed_{s}")
        sub_logger = setup_logger(od, name=f"seed{s}")
        cfg_s = dict(cfg); cfg_s.update(output_dir=od, seed=s)
        _, _, rep = process_hyperspectral_image(
            img_np, gt_endmembers=gt_endmembers, gt_abundances=gt_abundances,
            logger=sub_logger, **cfg_s)
        rows.append(rep)

    def _agg(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        if not vals: return None, None
        return float(np.mean(vals)), float(np.std(vals))

    summary = {}
    for key in ['recon_rmse', 'recon_sam_deg', 'endmember_sam_deg_mean',
                'endmember_rmse_mean', 'abundance_rmse_mean',
                'train_time_sec', 'infer_time_sec', 'params_million']:
        mu, sd = _agg(key)
        if mu is not None:
            summary[key] = dict(mean=mu, std=sd)
            logger.info("%-26s : %.6f ± %.6f", key, mu, sd)

    with open(os.path.join(base_output_dir, "summary_mean_std.json"), "w", encoding="utf-8") as f:
        json.dump(dict(seeds=list(seeds), per_seed=rows, summary=summary),
                  f, indent=2, ensure_ascii=False)
    logger.info("Saved summary_mean_std.json [R2.10]")
    return summary


# --------------------------
# .mat 自动取三维数组工具
# --------------------------
def auto_find_cube_in_mat(matdict, prefer_key=None):
    if prefer_key and prefer_key in matdict and isinstance(matdict[prefer_key], np.ndarray):
        arr = matdict[prefer_key]
        if arr.ndim == 3:
            return arr
    candidates = []
    for k, v in matdict.items():
        if isinstance(v, np.ndarray) and v.ndim == 3 and np.issubdtype(v.dtype, np.number):
            candidates.append((k, v.size, v))
    if not candidates:
        raise ValueError("No 3D array found in .mat file. Use prefer_key or check data.")
    candidates.sort(key=lambda x: -x[1])
    return candidates[0][2]


# --------------------------
# main（示例）
# --------------------------
def main():
    cfg = dict(
        rmax=4, output_dir='results_tiled_imageinput0614',
        num_iter=6000, show_every=800,
        lr_A=1e-4, lr_E=2e-5, m=1.0, m0=1.0, K_basis=24, ao_period=6,
        train_patch=(128,128), infer_patch=(192,192), stride=128,
        seed=0, use_svd=True, svd_rank=None,   # svd_rank=None -> 取 rmax [R2.3]
    )

    logger = setup_logger(cfg['output_dir'])
    logger.info("Loading HSI from MAT ...")
    mat_path = 'tunnel_data/2wjz0527.mat'   # TODO: 改成你的路径
    md = scipy.io.loadmat(mat_path)
    arr = auto_find_cube_in_mat(md, prefer_key='data')
    if arr.ndim != 3:
        raise ValueError("Input data must be a 3D array")
    if arr.shape[0] <= arr.shape[1] and arr.shape[0] <= arr.shape[2]:
        img_np = arr.astype(np.float32, copy=False)
    else:
        img_np = arr.transpose(2,0,1).astype(np.float32, copy=False)

    # 单次运行
    E, A, report = process_hyperspectral_image(img_np, logger=logger, **cfg)

    # 如需多种子统计（R2.10），取消下面注释：
    # run_multi_seed(img_np, seeds=(0,1,2),
    #                base_output_dir='results_multiseed_0614',
    #                rmax=cfg['rmax'], num_iter=cfg['num_iter'], show_every=cfg['show_every'],
    #                lr_A=cfg['lr_A'], lr_E=cfg['lr_E'], m=cfg['m'], m0=cfg['m0'],
    #                K_basis=cfg['K_basis'], ao_period=cfg['ao_period'],
    #                train_patch=cfg['train_patch'], infer_patch=cfg['infer_patch'],
    #                stride=cfg['stride'], use_svd=cfg['use_svd'], svd_rank=cfg['svd_rank'])


if __name__ == "__main__":
    main()