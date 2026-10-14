#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_unmix_relab_dctAO_revised.py
==============================================================================
Relab 基准数据（含真值）盲解混：DCT 低频端元 + 交替优化(AO)，Hapke 物理约束。

本版本在原始脚本基础上，按审稿意见做了与代码相关的修改（不改变核心算法行为）：

  [R2.3 / R2.4] 预处理由"降维"更正为"低秩去噪(Low-rank denoising)"：
        SVD 仅做秩-r 子空间投影重建 Y_hat = U_r U_r^T Y，输出仍保留全部 p 个波段；
        新增可配置 `svd_rank`（默认= rmax），`use_svd=False` 可做有/无去噪消融。

  [R2.9] 训练过程可视化：逐迭代记录全部损失分量(L2/SAM/D2/VOL/TV/BAR)与总损失，
        保存 history.csv/.npz，输出 training_curves.png/svg；若有真值，另输出
        端元 SAM/RMSE 随迭代的"验证曲线"(validation_curve)。

  [R2.10] 因 Relab 含真值，本脚本计算端元 SAM/RMSE 与丰度 RMSE（�.hungarian 匹配），
        作为定量验证；并提供多随机种子重复实验 run_multi_seed，报告 mean ± std
        （盲解混逐场景优化，不适用 k 折交叉验证，故以重复种子代替）。

  [R1.5 / R2.10] 资源消耗：参数量、FLOPs(若装 thop)、训练/推理时间、峰值显存，
        保存 resource_report.json/.txt。

  [日志] 所有运行信息经 logging 同时写入控制台与 run.log（带时间戳）。

  [R2.11] 数据集关系：Relab 含真值，用于"定量精度验证"；隧道数据无真值，用于
        "真实工程演示"。本方法逐场景自监督，network 对每个场景独立优化，
        无跨数据集训练/迁移/微调。

  注意：本脚本的网络输入为"噪声输入(DIP 风格)"，与隧道脚本的"影像输入"不同。
        若论文统一声称"以影像为输入"，请二者保持一致（详见交付说明）。该差异
        不在本次按审稿意见的改动范围内，未擅自更改算法行为。
==============================================================================
"""

import os
import sys
import csv
import json
import time
import logging

import numpy as np
import scipy.io
import scipy.linalg

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

# === project utils（用户环境若有则用其实现；缺失时使用等价回退，保证可独立运行）===
_HAS_PROJECT_UTILS = True
try:
    from common import *                 # get_noise, get_params, etc.
    from UtilityMine import *            # Endmember_extract, etc.
except Exception:
    _HAS_PROJECT_UTILS = False

if 'get_noise' not in globals():
    def get_noise(input_depth, method, spatial_size, var=0.1):
        """DIP 噪声输入回退实现：返回 [1, C, H, W] 的均匀噪声。"""
        H, W = spatial_size
        return torch.rand(1, input_depth, H, W) * var

if 'Endmember_extract' not in globals():
    def Endmember_extract(Y, r, trials=64, seed=0):
        """端元像元索引提取回退实现：PCA 主方向极值 + 随机投影补齐。
        Y: [p, N]；返回 (indices[list,len=r], None)。"""
        rng = np.random.default_rng(seed)
        p, N = Y.shape
        Cov = (Y @ Y.T) / max(N - 1, 1)
        _, V = np.linalg.eigh(Cov)
        V = V[:, ::-1][:, :min(r, p)]
        proj = V.T @ Y
        idx = list(np.argmax(proj, axis=1))
        counts = {}
        for _ in range(trials):
            u = rng.normal(size=(p,)); u /= (np.linalg.norm(u) + 1e-12)
            k = int(np.argmax(u @ Y)); counts[k] = counts.get(k, 0) + 1
        for k, _ in sorted(counts.items(), key=lambda z: -z[1]):
            if k not in idx: idx.append(k)
        while len(idx) < r:
            idx.append(int(rng.integers(0, N)))
        return idx[:r], None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.enabled = True if device.type == 'cuda' else False
torch.backends.cudnn.benchmark = True if device.type == 'cuda' else False
dtype = torch.cuda.FloatTensor if device.type == 'cuda' else torch.FloatTensor

PLOT = False          # 训练过程内联绘图（headless 下建议 False）
save_result = True


# ==========================================================================
# 日志
# ==========================================================================
def setup_logger(output_dir, name="relab", log_name="run.log"):
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(os.path.join(output_dir, log_name), mode="w", encoding="utf-8")
    fh.setFormatter(fmt); logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    logger.info("Logger initialized. Log file -> %s", os.path.join(output_dir, log_name))
    if not _HAS_PROJECT_UTILS:
        logger.warning("common/UtilityMine not found -> using fallback get_noise/Endmember_extract.")
    return logger


# --------------------------
# helpers
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
    End3 = (1.0 - W1 ** 2).clamp(eps, 1 - eps)
    HR = torch.matmul(End3, A).clamp(eps, 1 - eps)
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
    B, p, H, W = target.shape
    r = A_maps.shape[1]
    N = H * W

    E = endmember_fn()
    E = torch.nan_to_num(E, nan=0.5, posinf=0.999, neginf=0.001)

    Y = target.reshape(p, N).clamp(0.0, 1.0)
    A = A_maps.reshape(r, N)
    A = torch.nan_to_num(A, nan=1.0/r, posinf=1.0, neginf=0.0)
    A = (A.clamp_min(1e-8) / A.clamp_min(1e-8).sum(0, keepdim=True))

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
        return {'l2':1.0,'sam':0.10,'d2':0.010,'vol':0.006,'tv':0.0005,'bar':0.010}
    elif iter_idx < 5000:
        return {'l2':1.0,'sam':0.15,'d2':0.015,'vol':0.010,'tv':0.0005,'bar':0.010}
    else:
        return {'l2':1.0,'sam':0.20,'d2':0.020,'vol':0.015,'tv':0.0005,'bar':0.010}


# ==========================================================================
# [R2.3 / R2.4] 低秩去噪
# ==========================================================================
def low_rank_denoise(img_resh, svd_rank):
    """
    秩-r 低秩去噪（非降维）。  Y = U Σ V^T,  Y_hat = U_r U_r^T Y ∈ R^{p×N}
    img_resh: [p, N]；返回 (Y_hat[p,N], info)
    """
    U, s, Vt = scipy.linalg.svd(img_resh, full_matrices=False)   # U:[p,p]
    r = int(max(1, min(svd_rank, U.shape[1])))
    Ur = U[:, :r]
    Y_hat = (Ur @ Ur.T) @ img_resh
    energy_total = float(np.sum(s ** 2)) + 1e-12
    removed = 1.0 - float(np.sum(s[:r] ** 2)) / energy_total
    return Y_hat, dict(svd_rank=r, removed_energy_ratio=removed)


# ==========================================================================
# [R2.10] 指标
# ==========================================================================
def _sam_columns(X, Y, eps=1e-8):
    num = (X * Y).sum(0)
    den = np.linalg.norm(X, axis=0) * np.linalg.norm(Y, axis=0) + eps
    cos = np.clip(num / den, -1 + 1e-7, 1 - 1e-7)
    return np.arccos(cos)

def compute_recon_metrics(E, A_rHW, Y_pHW, m=1.0, m0=1.0):
    p, H, W = Y_pHW.shape
    r = A_rHW.shape[0]
    Et = torch.from_numpy(E.astype(np.float32))
    At = torch.from_numpy(A_rHW.reshape(r, -1).astype(np.float32))
    At = At.clamp_min(1e-8); At = At / At.sum(0, keepdim=True).clamp_min(1e-8)
    with torch.no_grad():
        Ypred = hapke_forward(Et, At, m=m, m0=m0).numpy()
    Y = Y_pHW.reshape(p, -1)
    return dict(recon_rmse=float(np.sqrt(np.mean((Ypred - Y) ** 2))),
                recon_sam_deg=float(np.degrees(np.mean(_sam_columns(Ypred, Y)))))

def match_columns(E_est, E_gt):
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
            for j in np.argsort(cost[i]):
                if j not in used: pairs.append((i, int(j))); used.add(int(j)); break
        return pairs

def endmember_gt_metrics(E_est, E_gt):
    pairs = match_columns(E_est, E_gt)
    sams = [np.degrees(_sam_columns(E_est[:, [i]], E_gt[:, [j]])[0]) for i, j in pairs]
    rmses = [np.sqrt(np.mean((E_est[:, i] - E_gt[:, j]) ** 2)) for i, j in pairs]
    return float(np.mean(sams)), float(np.mean(rmses)), pairs

def compute_gt_metrics(E_est, A_est_rHW, E_gt=None, A_gt_rHW=None):
    out = {}
    if E_gt is not None:
        sam, rmse, pairs = endmember_gt_metrics(E_est, E_gt)
        out['endmember_sam_deg_mean'] = sam
        out['endmember_rmse_mean'] = rmse
        out['endmember_match'] = pairs
        if A_gt_rHW is not None:
            r = E_gt.shape[1]
            est_order = [i for i, _ in sorted(pairs, key=lambda x: x[0])]
            gt_order = [j for _, j in sorted(pairs, key=lambda x: x[0])]
            a_est = A_est_rHW.reshape(r, -1)[est_order]
            a_gt = A_gt_rHW.reshape(r, -1)[gt_order]
            out['abundance_rmse_mean'] = float(np.sqrt(np.mean((a_est - a_gt) ** 2)))
    return out


# ==========================================================================
# [R1.5] 资源
# ==========================================================================
def count_parameters(net):
    total = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total, trainable

def estimate_flops(net, sample):
    try:
        from thop import profile
        macs, _ = profile(net, inputs=(sample,), verbose=False)
        return float(macs) * 2.0
    except Exception:
        return None


# ==========================================================================
# [R2.9] 历史保存与绘图
# ==========================================================================
def save_history(history, output_dir):
    keys = ['iter','phase','total','l2','sam','d2','vol','tv','bar','temperature']
    with open(os.path.join(output_dir, "history.csv"), "w", newline="", encoding="utf-8") as f:
        wri = csv.writer(f); wri.writerow(keys)
        for i in range(len(history['iter'])):
            wri.writerow([history[k][i] for k in keys])
    np.savez(os.path.join(output_dir, "history.npz"),
             **{k: np.asarray(v) for k, v in history.items() if k != 'phase'})

def plot_training_curves(history, output_dir, val_history=None):
    it = np.asarray(history['iter'])
    colors = {'total':'#2B2B2B','l2':'#33567E','sam':'#9E4D49','d2':'#3C6B50',
              'vol':'#6E4F77','tv':'#9A7B33','bar':'#3E6F77'}
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['svg.fonttype'] = 'none'

    ncol = 3 if val_history else 2
    fig, axes = plt.subplots(1, ncol, figsize=(4.0*ncol, 3.6))
    for k in ['total','l2','sam']:
        axes[0].plot(it, np.clip(np.asarray(history[k],float),1e-8,None), color=colors[k], lw=1.3, label=k.upper())
    axes[0].set_yscale('log'); axes[0].set_xlabel('Iteration'); axes[0].set_ylabel('Loss (log)')
    axes[0].set_title('Total / L2 / SAM'); axes[0].legend(frameon=False, fontsize=7)
    for sp in ['top','right']: axes[0].spines[sp].set_visible(False)

    for k in ['d2','vol','tv','bar']:
        axes[1].plot(it, np.clip(np.abs(np.asarray(history[k],float)),1e-10,None), color=colors[k], lw=1.2, label=k.upper())
    axes[1].set_yscale('log'); axes[1].set_xlabel('Iteration'); axes[1].set_ylabel('|term| (log)')
    axes[1].set_title('Regularization terms'); axes[1].legend(frameon=False, fontsize=7)
    for sp in ['top','right']: axes[1].spines[sp].set_visible(False)

    if val_history:
        vit = np.asarray(val_history['iter'])
        ax = axes[2]; ax.plot(vit, val_history['em_sam_deg'], color='#9E4D49', lw=1.6, marker='o', ms=3, label='Endmember SAM (deg)')
        ax.set_xlabel('Iteration'); ax.set_ylabel('Endmember SAM (deg)'); ax.set_title('Validation vs ground truth')
        ax2 = ax.twinx(); ax2.plot(vit, val_history['em_rmse'], color='#33567E', lw=1.6, marker='s', ms=3, label='Endmember RMSE')
        ax2.set_ylabel('Endmember RMSE')
        for sp in ['top']: ax.spines[sp].set_visible(False); ax2.spines[sp].set_visible(False)
        l1,la1 = ax.get_legend_handles_labels(); l2,la2 = ax2.get_legend_handles_labels()
        ax.legend(l1+l2, la1+la2, frameon=False, fontsize=7, loc='upper right')

    fig.tight_layout(pad=1.4)
    fig.savefig(os.path.join(output_dir, "training_curves.png"), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, "training_curves.svg"), bbox_inches='tight')
    plt.close(fig)


# --------------------------
# Model
# --------------------------
class CAE_EndEst(nn.Module):
    def __init__(self, input_depth, p1, rmax, K_basis=64):
        super(CAE_EndEst, self).__init__()
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(input_depth, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.spectral_branch = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(256, 128, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, rmax, 1)
        )
        self.p1 = p1; self.rmax = rmax; self.K = min(K_basis, p1)
        Phi_full = dct_matrix(p1)
        Phi = Phi_full[:self.K, :].T
        self.register_buffer('Phi', torch.from_numpy(Phi).float())
        self.C = nn.Parameter(torch.randn(self.K, rmax) * 0.01)
        self.register_buffer('temperature', torch.tensor(1.2))

    def endmembers(self):
        E = torch.matmul(self.Phi, self.C)
        return torch.sigmoid(E)

    def forward(self, x):
        B, C, H, W = x.shape
        xs = self.spatial_branch(x)
        xr = x.permute(0,2,3,1).reshape(B*H*W,1,C)
        xr = self.spectral_branch(xr).mean(-1).view(B,H,W,128).permute(0,3,1,2)
        logits = self.fusion_conv(torch.cat([xs, xr], dim=1))
        A_maps = F.softmax(logits / self.temperature.clamp_min(0.9), dim=1)
        E = self.endmembers().view(self.p1, self.rmax, 1, 1)
        preview = F.conv2d(A_maps, E, bias=None, stride=1, padding=0)
        return A_maps, preview


# --------------------------
# Training (AO)
# --------------------------
def train_one(img_noisy_np, rmax, output_dir="ResultsRelab6", show_every=500,
              num_iter1=8000, exp_weight=0.99, lr_A=1e-4, lr_E=2e-5,
              m=1.0, m0=1.0, K_basis=64, ao_period=6, seed=0,
              use_svd=True, svd_rank=None,
              gt_endmembers=None, gt_abundances=None, logger=None):
    """
    逐场景自监督盲解混（噪声输入 / DIP 风格）。
    含真值时计算端元 SAM/RMSE、丰度 RMSE 作为定量验证 [R2.10]。
    返回 (E_final[p,r], Aavg[r,H,W], report)
    """
    os.makedirs(output_dir, exist_ok=True)
    if logger is None:
        logger = setup_logger(output_dir)
    if svd_rank is None:
        svd_rank = rmax

    torch.manual_seed(seed); np.random.seed(seed)
    p1, nr1, nc1 = img_noisy_np.shape
    logger.info("=" * 70)
    logger.info("Relab blind unmixing (noise-input DIP) | device=%s p=%d HxW=%dx%d r=%d K=%d seed=%d",
                device.type, p1, nr1, nc1, rmax, K_basis, seed)
    logger.info("use_svd(low-rank denoise)=%s svd_rank=%s [R2.3/R2.4]", use_svd, svd_rank)

    # ---- [R2.3/R2.4] 低秩去噪 ----
    img_resh = np.reshape(img_noisy_np, (p1, nr1 * nc1))
    if use_svd:
        img_resh_DN, dn_info = low_rank_denoise(img_resh, svd_rank=svd_rank)
        logger.info("Low-rank denoising: rank=%d removed-energy=%.4f",
                    dn_info['svd_rank'], dn_info['removed_energy_ratio'])
    else:
        img_resh_DN = img_resh.copy()
        dn_info = dict(svd_rank=0, removed_energy_ratio=0.0)
        logger.info("Low-rank denoising DISABLED (ablation).")
    img_resh_np_clip = np.clip(img_resh_DN, 0, 1)
    img_dn_pHW = img_resh_DN.reshape(p1, nr1, nc1)

    # 初始端元 -> 初始化 C
    II, _ = Endmember_extract(img_resh_np_clip, rmax)
    E_np1 = img_resh_np_clip[:, II].clip(1e-6, 1-1e-6)

    net = CAE_EndEst(input_depth=p1, p1=p1, rmax=rmax, K_basis=K_basis).to(device)
    with torch.no_grad():
        Phi = net.Phi.cpu().numpy()
        Z = inv_sigmoid(E_np1)
        C0 = np.linalg.lstsq(Phi, Z, rcond=None)[0]
        net.C.copy_(torch.from_numpy(C0).float().to(net.C.device))

    # ---- [R1.5] 资源：参数量 + FLOPs ----
    total_params, trainable_params = count_parameters(net)
    flops = estimate_flops(net, torch.zeros(1, p1, nr1, nc1, device=device))
    logger.info("Params: total=%d (%.3f M) trainable=%d | FLOPs/forward=%s",
                total_params, total_params/1e6, trainable_params,
                ("%.3e" % flops) if flops else "N/A (install thop)")

    img_noisy_torch = torch.from_numpy(img_resh_DN).view(1, p1, nr1, nc1).type(dtype)
    net_input = get_noise(p1, 'noise', (nr1, nc1)).type(dtype).detach()

    opt_A = torch.optim.Adam([{'params': [p for n,p in net.named_parameters() if n != 'C']}],
                             lr=lr_A, betas=(0.9,0.999), eps=1e-8)
    opt_E = torch.optim.Adam([{'params': [net.C]}], lr=lr_E, betas=(0.9,0.999), eps=1e-8)

    history = {k: [] for k in ['iter','phase','total','l2','sam','d2','vol','tv','bar','temperature']}
    val_history = {'iter': [], 'em_sam_deg': [], 'em_rmse': []} if gt_endmembers is not None else None
    out_avg = None
    i = 0

    def step(update_E=False):
        nonlocal i, out_avg
        if update_E: opt = opt_E; opt_A.zero_grad(set_to_none=True)
        else:        opt = opt_A; opt_E.zero_grad(set_to_none=True)

        A_maps, _ = net(net_input)
        if out_avg is None: out_avg = A_maps.detach()
        else: out_avg = out_avg * exp_weight + A_maps.detach() * (1 - exp_weight)

        with torch.no_grad():
            T = 1.2 if i < 1500 else 1.2 - 0.3 * min((i-1500)/max(num_iter1-1500,1), 1.0)
            net.temperature.copy_(torch.tensor(T, device=net.temperature.device))

        weights = get_weights(i, num_iter1)
        total, parts = loss_total(img_noisy_torch, A_maps, net.endmembers, weights, m=m, m0=m0)
        if not torch.isfinite(total):
            weights = {'l2':1.0,'sam':0.,'d2':0.,'vol':0.,'tv':0.,'bar':0.}
            total, parts = loss_total(img_noisy_torch, A_maps, net.endmembers, weights, m=m, m0=m0)

        total.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        opt.step()

        # 记录历史
        history['iter'].append(int(i)); history['phase'].append('E' if update_E else 'A')
        history['total'].append(float(total.detach().cpu()))
        for k in ['l2','sam','d2','vol','tv','bar']:
            history[k].append(float(parts[k].detach().cpu()))
        history['temperature'].append(float(net.temperature.item()))

        if i % show_every == 0:
            def sf(x): return float(x.detach().cpu()) if torch.isfinite(x) else float('nan')
            msg = (f"Iter {i:05d} ({'E' if update_E else 'A'}) | Loss={sf(total):.6f} "
                   f"L2={sf(parts['l2']):.4e} SAM={sf(parts['sam']):.4e} VOL={sf(parts['vol']):.4e} "
                   f"T={net.temperature.item():.2f}")
            # 验证（含真值）：端元 SAM/RMSE 随迭代
            if val_history is not None:
                E_now = net.endmembers().detach().cpu().numpy().clip(1e-6, 1-1e-6)
                vsam, vrmse, _ = endmember_gt_metrics(E_now, gt_endmembers)
                val_history['iter'].append(int(i))
                val_history['em_sam_deg'].append(vsam); val_history['em_rmse'].append(vrmse)
                msg += f" | [val] EM-SAM={vsam:.3f}deg EM-RMSE={vrmse:.4f}"
            logger.info(msg)
        i += 1

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    t_train0 = time.perf_counter()
    logger.info("Start training (AO, noise input) ...")
    t = 0
    while t < num_iter1:
        for _ in range(ao_period-1):
            if t >= num_iter1: break
            step(update_E=False); t += 1
        if t >= num_iter1: break
        step(update_E=True); t += 1
    train_time = time.perf_counter() - t_train0
    logger.info("Training done in %.2f s", train_time)

    # 推理计时（单次前向整图）
    net.eval()
    t_inf0 = time.perf_counter()
    with torch.no_grad():
        A_maps, _ = net(net_input)
    infer_time = time.perf_counter() - t_inf0
    peak_mem_mb = (torch.cuda.max_memory_allocated() / 1024**2) if device.type == 'cuda' else None

    out_avg_np = out_avg.detach().cpu().squeeze().numpy().clip(0, 1)   # [r,H,W]
    E_final = net.endmembers().detach().cpu().numpy().clip(1e-6, 1-1e-6)

    # ---- 保存历史 + 曲线 ----
    save_history(history, output_dir)
    plot_training_curves(history, output_dir, val_history=val_history)
    logger.info("Saved history + training curves [R2.9]")

    # ---- 指标 ----
    recon = compute_recon_metrics(E_final, out_avg_np, img_dn_pHW, m=m, m0=m0)
    logger.info("Reconstruction: RMSE=%.6f SAM=%.4f deg", recon['recon_rmse'], recon['recon_sam_deg'])
    gt_metrics = compute_gt_metrics(E_final, out_avg_np, E_gt=gt_endmembers, A_gt_rHW=gt_abundances)
    if gt_metrics:
        logger.info("GT metrics: %s", json.dumps({k:v for k,v in gt_metrics.items() if k!='endmember_match'}))

    # ---- 保存结果 ----
    if save_result:
        scipy.io.savemat(os.path.join(output_dir, "Eest_dctAO.mat"), {'Eest': E_final})
        scipy.io.savemat(os.path.join(output_dir, "Aavg_dctAO.mat"), {'Aavg': out_avg_np.transpose(1,2,0)})
        np.save(os.path.join(output_dir, "E_endmembers.npy"), E_final)
        np.save(os.path.join(output_dir, "A_abundances.npy"), out_avg_np.transpose(1,2,0))
        plt.figure(figsize=(6,3)); plt.plot(E_final); plt.title('Final endmembers (DCT-AO)')
        plt.tight_layout(); plt.savefig(os.path.join(output_dir, "final_endmembers.png")); plt.close()

    # ---- 资源报告 ----
    report = dict(device=device.type, bands=p1, height=nr1, width=nc1, rmax=rmax, K_basis=K_basis,
                  num_iter=num_iter1, seed=seed, use_svd=use_svd, svd_rank=int(dn_info['svd_rank']),
                  removed_energy_ratio=dn_info.get('removed_energy_ratio'),
                  params_total=int(total_params), params_trainable=int(trainable_params),
                  params_million=round(total_params/1e6, 4), flops_per_forward=flops,
                  train_time_sec=round(train_time, 3), infer_time_sec=round(infer_time, 4),
                  peak_gpu_mem_mb=peak_mem_mb,
                  recon_rmse=recon['recon_rmse'], recon_sam_deg=recon['recon_sam_deg'])
    report.update({k: v for k, v in gt_metrics.items() if k != 'endmember_match'})
    with open(os.path.join(output_dir, "resource_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "resource_report.txt"), "w", encoding="utf-8") as f:
        f.write("Resource & accuracy report  [R1.5 / R2.10]\n" + "-"*50 + "\n")
        for k, v in report.items(): f.write(f"{k:24s}: {v}\n")
    logger.info("Saved resource_report.* | [Done] -> %s", output_dir)
    return E_final, out_avg_np, report


# ==========================================================================
# [R2.10] 多随机种子重复实验（mean ± std）
# ==========================================================================
def run_multi_seed(img_noisy_np, seeds=(0,1,2), base_output_dir="ResultsRelab_multiseed",
                   gt_endmembers=None, gt_abundances=None, **cfg):
    os.makedirs(base_output_dir, exist_ok=True)
    logger = setup_logger(base_output_dir, name="relab_multiseed", log_name="multiseed.log")
    logger.info("Multi-seed repeated runs: seeds=%s [R2.10]", list(seeds))
    rows = []
    for s in seeds:
        od = os.path.join(base_output_dir, f"seed_{s}")
        sub = setup_logger(od, name=f"relab_seed{s}")
        cfg_s = dict(cfg); cfg_s.update(output_dir=od, seed=s)
        _, _, rep = train_one(img_noisy_np, gt_endmembers=gt_endmembers,
                              gt_abundances=gt_abundances, logger=sub, **cfg_s)
        rows.append(rep)
    summary = {}
    for key in ['endmember_sam_deg_mean','endmember_rmse_mean','abundance_rmse_mean',
                'recon_rmse','recon_sam_deg','train_time_sec','infer_time_sec','params_million']:
        vals = [r[key] for r in rows if r.get(key) is not None]
        if vals:
            summary[key] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)))
            logger.info("%-26s : %.6f ± %.6f", key, summary[key]['mean'], summary[key]['std'])
    with open(os.path.join(base_output_dir, "summary_mean_std.json"), "w", encoding="utf-8") as f:
        json.dump(dict(seeds=list(seeds), per_seed=rows, summary=summary), f, indent=2, ensure_ascii=False)
    logger.info("Saved summary_mean_std.json [R2.10]")
    return summary


# ==========================================================================
# 真值加载（按你的 Relab 文件结构填写 key/path）
# ==========================================================================
def load_ground_truth(em_path=None, em_key=None, ab_path=None, ab_key=None,
                      p=None, r=None, HW=None):
    """
    返回 (E_gt[p,r] 或 None, A_gt[r,H,W] 或 None)。
    Relab ABO 三元混合：端元为 3 个纯矿物谱，丰度为合成时的比例图。
    请按实际 .mat 变量名/形状填写；形状会自动尝试规整。
    """
    E_gt = A_gt = None
    if em_path and em_key:
        E = scipy.io.loadmat(em_path)[em_key]
        E = np.asarray(E, dtype=np.float32)
        if p is not None and E.shape[0] != p and E.shape[-1] == p:
            E = E.T
        E_gt = E
    if ab_path and ab_key:
        A = scipy.io.loadmat(ab_path)[ab_key]
        A = np.asarray(A, dtype=np.float32)
        if A.ndim == 3 and r is not None:
            if A.shape[-1] == r:        # [H,W,r] -> [r,H,W]
                A = A.transpose(2, 0, 1)
        A_gt = A
    return E_gt, A_gt


# --------------------------
# main
# --------------------------
def main():
    rmax       = 3
    show_every = 500
    num_iter1  = 8000
    exp_weight = 0.99
    lr_A, lr_E = 1e-4, 2e-5
    m, m0      = 1.0, 1.0
    K_basis    = 64
    ao_period  = 6
    out_base   = "ResultsRelab6_0614_1"

    logger = setup_logger(out_base)
    logger.info("Loading Relab data ...")

    # 干净参考（可用于额外检查）
    fname2 = "Git_Data/RELAB/Y_clean.mat"
    img_np_gt = scipy.io.loadmat(fname2)["Y_clean"].transpose(2, 0, 1)   # [bands,H,W]
    p1, nr1, nc1 = img_np_gt.shape

    data = scipy.io.loadmat('Git_Data/Noisy/ternary_mixtures_ABO_reflectance.mat')
    YY = data['YY']

    # ---- 真值：与噪声同在 ternary_mixtures_ABO_reflectance.mat 内 ----
    #   'E'      端元真值, 形状 (461, 3) = [p, r]，直接用
    #   'A_true' 丰度真值, 形状 (60, 60, 3) = [H, W, r] -> 转为 [r, H, W]
    E_gt = np.asarray(data['E'], dtype=np.float32)                       # [p, r]
    if E_gt.shape[0] != p1 and E_gt.shape[-1] == p1:                     # 万一是 [r, p]
        E_gt = E_gt.T
    A_gt = np.asarray(data['A_true'], dtype=np.float32).transpose(2, 0, 1)  # [r, H, W]
    logger.info("Ground truth loaded: E_gt=%s, A_gt=%s [R2.10]", E_gt.shape, A_gt.shape)

    tol1, tol2 = 1, 1
    for fi in range(tol1):
        for fj in range(tol2):
            img_noisy_np = YY[fj][fi].transpose(2, 0, 1)
            od = os.path.join(out_base, f"case_{fi+1:02d}_{fj+1:02d}")
            sub = setup_logger(od, name=f"relab_{fi}_{fj}")
            Eest, Aavg, rep = train_one(
                img_noisy_np=img_noisy_np, rmax=rmax, output_dir=od,
                show_every=show_every, num_iter1=num_iter1, exp_weight=exp_weight,
                lr_A=lr_A, lr_E=lr_E, m=m, m0=m0, K_basis=K_basis, ao_period=ao_period,
                seed=0, use_svd=True, svd_rank=None,
                gt_endmembers=E_gt, gt_abundances=A_gt, logger=sub)

    # 多种子统计（R2.10）：取消注释
    run_multi_seed(YY[0][0].transpose(2, 0, 1), seeds=(0, 1, 2, 3, 4),
                   base_output_dir=os.path.join(out_base, 'multiseed'),
                   rmax=rmax, num_iter1=num_iter1, show_every=show_every,
                   lr_A=lr_A, lr_E=lr_E, m=m, m0=m0, K_basis=K_basis, ao_period=ao_period,
                   use_svd=True, svd_rank=None, gt_endmembers=E_gt, gt_abundances=A_gt)

    logger.info("Done.")


if __name__ == "__main__":
    main()