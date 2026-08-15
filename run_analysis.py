#!/usr/bin/env python3
"""邪馬台国・卑弥呼時代の「皆既日食可能性マップ」解析ランナー.

パイプライン:
  1. NASA Bessel要素エンジン(bessel_engine)で、基準ΔTにおける
     「可視皆既マスク」を拡張経度グリッド上に1回だけ計算する。
  2. ΔTの変更は観測点の経度シフトと厳密に等価 (H = μ + λ - kΔT) なので、
     モンテカルロΔT分布の確率地図は、基準マスクとΔT分布(経度換算)の
     畳み込みで一気に得られる。
  3. 検証として、ランダムに選んだΔTサンプルでエンジン直接計算と比較。

出力: figures/*.png, output/*.csv
"""

from __future__ import annotations

import csv
import math
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from bessel_engine import (
    ECLIPSES, SITES, K_DEG_PER_SEC, REFRACTION_HORIZON_DEG,
    local_circumstances, find_totality_delta_t_window,
)
from deltat_ensemble import (
    DeltaTEnsemble, luoyang_weights, center_delta_t, sigma_nasa, NASA_TABLE,
)

# ------------------------------------------------------------------ 設定
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

FIG_DIR = "figures"
OUT_DIR = "output"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# 表示領域 (九州〜畿内)
LON_MIN, LON_MAX = 128.5, 137.5
LAT_MIN, LAT_MAX = 30.5, 36.5
GRID_STEP = 0.02  # deg

MIN_ALT = REFRACTION_HORIZON_DEG  # 可視判定: 見かけの地平線

SITE_LABELS = {
    "dazaifu": "大宰府", "asuka": "飛鳥", "makimuku": "纒向",
    "yoshinogari": "吉野ヶ里", "usa": "宇佐", "izumo": "出雲",
}

# 単一色相sequentialランプ (確率=大きさ表現; datavizルール)
PROB_CMAP = plt.cm.Blues

try:
    from global_land_mask import globe
    HAVE_LANDMASK = True
except ImportError:
    HAVE_LANDMASK = False


# ------------------------------------------------- 基準マスクと畳み込み
def reference_masks(ecl, dt_ref, lon_lo, lon_hi, step=GRID_STEP):
    """基準ΔTでの可視皆既マスクと食分0.98以上マスク(拡張経度グリッド)."""
    lons = np.arange(lon_lo, lon_hi + step, step)
    lats = np.arange(LAT_MIN, LAT_MAX + step, step)
    LON, LAT = np.meshgrid(lons, lats)
    lc = local_circumstances(ecl, dt_ref, LON, LAT, min_alt_deg=MIN_ALT)
    mask_total = lc.is_total.astype(float)
    mask_98 = ((lc.magnitude >= 0.98) | lc.is_total).astype(float)
    return lons, lats, mask_total, mask_98


def probability_map(mask, lons, dt_samples, weights, dt_ref, step=GRID_STEP):
    """ΔTサンプル分布との畳み込みで P(条件成立) の地図を作る.

    地点(λ,φ)のΔT=dtでの条件は、基準ΔTでの地点(λ - k(dt - dt_ref), φ)の
    条件と厳密に等価。したがって
        P(λ) = Σ_s w_s · mask(λ - shift_s) / Σ_s w_s
    これはmask行とシフト分布ヒストグラムの畳み込みになる。
    """
    if weights.sum() == 0:
        return np.zeros_like(mask)
    shifts = K_DEG_PER_SEC * (dt_samples - dt_ref)  # [deg]
    m_idx = np.round(shifts / step).astype(int)     # 格子単位のシフト
    m_min, m_max = int(m_idx.min()), int(m_idx.max())
    kernel = np.bincount(m_idx - m_min, weights=weights,
                         minlength=m_max - m_min + 1)
    kernel = kernel / kernel.sum()

    n_lon = mask.shape[1]
    prob = np.zeros_like(mask)
    # P[:, j] = Σ_m kernel[m] * mask[:, j - (m + m_min)]
    for k, w in enumerate(kernel):
        if w == 0.0:
            continue
        shift_cols = k + m_min
        j_lo = max(0, shift_cols)
        j_hi = min(n_lon, n_lon + shift_cols)
        prob[:, j_lo:j_hi] += w * mask[:, j_lo - shift_cols:j_hi - shift_cols]
    return prob


def crop_to_display(lons, field):
    """拡張グリッドから表示領域だけ切り出す."""
    sel = (lons >= LON_MIN - 1e-9) & (lons <= LON_MAX + 1e-9)
    return lons[sel], field[:, sel]


def verify_convolution(ecl, dt_ref, lons, lats, mask, dt_samples, n_check=60,
                       seed=7):
    """畳み込み確率地図をエンジン直接計算と突き合わせて検証する."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(dt_samples), size=min(n_check, len(dt_samples)),
                      replace=False)
    lons_d, _ = crop_to_display(lons, mask)
    LON, LAT = np.meshgrid(lons_d[::10], lats[::10])
    acc = np.zeros_like(LON, dtype=float)
    for i in pick:
        lc = local_circumstances(ecl, float(dt_samples[i]), LON, LAT,
                                 min_alt_deg=MIN_ALT)
        acc += lc.is_total.astype(float)
    direct = acc / len(pick)
    conv = probability_map(mask, lons, dt_samples[pick],
                           np.ones(len(pick)), dt_ref)
    _, conv_d = crop_to_display(lons, conv)
    conv_sub = conv_d[::10, ::10]
    err = np.abs(direct - conv_sub).max()
    return err


# ------------------------------------------------------------- 地図描画
def draw_japan(ax, lons, lats, extra=""):
    """海岸線(land maskの等高線)と代表地点."""
    if HAVE_LANDMASK:
        LON, LAT = np.meshgrid(lons, lats)
        land = globe.is_land(LAT, LON).astype(float)
        ax.contour(LON, LAT, land, levels=[0.5], colors="#5f6b7a",
                   linewidths=0.7, zorder=3)
    for name in ("dazaifu", "asuka", "makimuku", "yoshinogari"):
        lon, lat = SITES[name]
        ax.plot(lon, lat, "o", ms=5, mfc="#1a1a1a", mec="white", mew=0.8,
                zorder=6)
        dy = -0.16 if name == "makimuku" else 0.12
        ha = "left" if name != "makimuku" else "left"
        ax.text(lon + 0.09, lat + dy, SITE_LABELS[name], fontsize=8.5,
                color="#1a1a1a", zorder=6, ha=ha,
                path_effects=None)
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_aspect(1.0 / math.cos(math.radians(33.5)))
    ax.grid(True, color="#e8e8e8", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)


# ================================================================ 図1
def _violin_profile(samples, y_grid, smooth_sigma=3):
    """サンプル群から滑らかな密度プロファイルを作る (scipy不要)."""
    hist, edges = np.histogram(samples, bins=y_grid, density=True)
    k = np.exp(-0.5 * (np.arange(-3 * smooth_sigma, 3 * smooth_sigma + 1)
                       / smooth_sigma) ** 2)
    k /= k.sum()
    dens = np.convolve(hist, k, mode="same")
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, dens


def figure_deltat_ensemble(ens_a, ens_b, ecl_key="0247-03-24"):
    """1枚構成: 軌道の束 + 247年時点の分布(バイオリン) + 皆既条件."""
    ecl = ECLIPSES[ecl_key]
    year_ev = ecl["year"]
    fig, ax = plt.subplots(figsize=(9.2, 5.6), dpi=200)

    yrs = ens_b.years
    show = yrs <= 900
    idx = np.linspace(0, ens_b.paths.shape[0] - 1, 120).astype(int)
    for i in idx:
        ax.plot(yrs[show], ens_b.paths[i, show], color="#9db8d9", lw=0.4,
                alpha=0.32, zorder=1)
    for i in idx[:60]:
        ax.plot(yrs[show], ens_a.paths[i, show], color="#2f6db3", lw=0.4,
                alpha=0.32, zorder=2)
    ax.plot(yrs[show], center_delta_t(yrs[show]), color="#1a1a1a", lw=1.6,
            zorder=4, label="中心曲線 (NASA/Espenak-Meeus多項式)")
    tb_y = sorted(NASA_TABLE)
    tb_v = [NASA_TABLE[y][0] for y in tb_y]
    tb_s = [NASA_TABLE[y][1] for y in tb_y]
    ax.errorbar(tb_y[:10], tb_v[:10], yerr=tb_s[:10], fmt="s", ms=4,
                color="#c25b28", lw=0, elinewidth=1.2, capsize=2, zorder=5,
                label="NASA歴史表 (M&S2004系, ±1σ)")

    # ---- 247年時点の分布 (バイオリン: 左=公式σ, 右=揺らぎ3倍)
    s_a = ens_a.sample_at(year_ev)
    s_b = ens_b.sample_at(year_ev)
    y_grid = np.arange(5600, 11800, 60)
    half_w = 90.0  # 年方向の最大半幅
    cy, da = _violin_profile(s_a, y_grid)
    _, db = _violin_profile(s_b, y_grid)
    scale = half_w / max(da.max(), db.max())
    ax.fill_betweenx(cy, year_ev - da * scale, year_ev, fc="#2f6db3",
                     alpha=0.85, ec="none", zorder=6,
                     label="247年のΔT分布 (左: 公式σ)")
    ax.fill_betweenx(cy, year_ev, year_ev + db * scale, fc="#9db8d9",
                     alpha=0.9, ec="none", zorder=6,
                     label="同 (右: 揺らぎ3倍シナリオ)")
    ax.axvline(year_ev, color="#666666", lw=0.8, ls="--", zorder=5)
    ax.text(year_ev - 105, 12750, f"{year_ev}年\n卑弥呼「以死」", fontsize=9,
            color="#444444", ha="center")

    # ---- 皆既の条件 (水平線) と中国史料の整合帯
    x_line0 = year_ev + 115
    ax.axhspan(7750, 8900, xmin=(x_line0 - 0) / 900, xmax=1.0,
               color="#e8c07a", alpha=0.30, zorder=0)
    ax.text(560, 8290, "中国史料と整合するΔT範囲 (7750–8900秒)\n"
            "(『三国志』: 洛陽では皆既でなかった)", fontsize=8.5,
            color="#8a4a1f", ha="center", va="center")
    marks = [
        (9656, "飛鳥が皆既となる下限 ΔT=9656秒"),
        (11963, "大宰府が皆既となる下限 ΔT=11963秒"),
    ]
    for y, label in marks:
        ax.hlines(y, x_line0, 895, color="#8a4a1f", lw=1.1, ls=":", zorder=3)
        ax.text(560, y + 240, label, fontsize=8.5, color="#8a4a1f",
                ha="center")

    ax.set_xlabel("西暦 [年]")
    ax.set_ylabel("ΔT [秒]")
    ax.set_title("ΔT軌道アンサンブル (濃青: 公式σ / 淡青: 揺らぎ3倍、各120本表示) と\n"
                 "247年時点のΔT分布・皆既の条件", fontsize=10.5)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_xlim(0, 900)
    ax.set_ylim(0, 13600)
    ax.grid(True, color="#eeeeee", lw=0.5)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig1_deltaT_ensemble.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ================================================================ 図2
def figure_band_sweep(ecl_key="0247-03-24"):
    """ΔTを変えると皆既帯が日本列島上を東へ滑る様子 (決定論的スイープ)."""
    ecl = ECLIPSES[ecl_key]
    sweep = [7300, 7750, 8152, 8500, 9000, 9656, 10500, 11963]
    step = 0.02
    lons = np.arange(LON_MIN - 0.3, LON_MAX + 0.3, step)
    lats = np.arange(LAT_MIN - 0.3, LAT_MAX + 0.3, step)
    LON, LAT = np.meshgrid(lons, lats)

    fig, ax = plt.subplots(figsize=(9.6, 7.2), dpi=200)
    # 単一色相・明→暗 (ΔTの大きさを表すsequential)
    colors = [PROB_CMAP(0.25 + 0.7 * i / (len(sweep) - 1))
              for i in range(len(sweep))]
    label_at = {7300: (0.30, "7300s\n(2010年論文)"),
                7750: (0.40, "7750s\n(洛陽下限)"),
                8152: (0.52, "8152s\n(NASA採用値)"),
                8500: (0.62, None), 9000: (0.72, None),
                9656: (0.80, "9656s\n(飛鳥皆既)"),
                10500: (0.88, None),
                11963: (0.955, "11963s\n(大宰府皆既)")}
    lat_cycle = (35.55, 35.9, 35.2)  # ラベルは帯の上の余白に段違いで置く
    for k, (dt, col) in enumerate(zip(sweep, colors)):
        lc = local_circumstances(ecl, float(dt), LON, LAT, min_alt_deg=MIN_ALT)
        mask = lc.is_total.astype(float)
        if mask.sum() == 0:
            continue
        ax.contourf(LON, LAT, mask, levels=[0.5, 1.5], colors=[col],
                    alpha=0.35, zorder=2)
        ax.contour(LON, LAT, mask, levels=[0.5], colors=[col], linewidths=1.2,
                   zorder=2)
        # 帯の「東端」(可視皆既の限界=日没線側)に引き出し線でラベル
        rows_i, cols_i = np.where(mask > 0.5)
        j_east = int(cols_i.max())
        i_east = int(np.median(rows_i[cols_i == j_east]))
        e_lon = min(float(lons[j_east]), LON_MAX - 0.05)
        e_lat = float(lats[i_east])
        note = (label_at[dt][1] or f"{dt}s").replace("\n", " ")
        t_lon = min(max(e_lon - 0.3, LON_MIN + 0.8), LON_MAX - 0.8)
        t_lat = lat_cycle[k % len(lat_cycle)]
        ax.annotate(note, xy=(e_lon, e_lat), xytext=(t_lon, t_lat),
                    fontsize=7.5, color="#123a5f", ha="center",
                    zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec="#9db8d9", lw=0.5, alpha=0.9),
                    arrowprops=dict(arrowstyle="-", color="#5a7fa8", lw=0.7))
    draw_japan(ax, lons, lats)
    ax.set_xlabel("東経 [度]")
    ax.set_ylabel("北緯 [度]")
    ax.set_title(f"{ecl['label']}: ΔTを変えると「可視皆既帯」が東へ動く\n"
                 f"(ΔT 1秒 = 東西約{111.32*math.cos(math.radians(33.5))*K_DEG_PER_SEC:.2f} km。"
                 "日没後に皆既となる領域は帯から除外済み)", fontsize=10)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig2_band_sweep_247.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ================================================================ 図3・図4
def figure_probability_maps(ecl_key, scenarios, fname, suptitle,
                            overlay_row=False):
    """シナリオ別の可視皆既確率マップ.

    overlay_row=True: 1段構成。深食確率を青、可視皆既確率を赤で
    同じパネルに重ね描きする (誌面スペース節約用)。
    """
    ecl = ECLIPSES[ecl_key]
    dt_ref = ecl["dt_nasa"]

    # 拡張経度範囲はサンプルの実範囲から決める
    all_dt = np.concatenate([s for s, _, _ in scenarios])
    sh_lo = K_DEG_PER_SEC * (all_dt.min() - dt_ref)
    sh_hi = K_DEG_PER_SEC * (all_dt.max() - dt_ref)
    lon_lo = LON_MIN - max(0.0, sh_hi) - 0.5
    lon_hi = LON_MAX - min(0.0, sh_lo) + 0.5
    lons, lats, mask_total, mask_98 = reference_masks(ecl, dt_ref,
                                                      lon_lo, lon_hi)

    n = len(scenarios)
    if overlay_row:
        fig, axes = plt.subplots(1, n, figsize=(5.0 * n + 1.6, 5.2), dpi=200,
                                 squeeze=False)
        # 各測度ごとにスケールをパネル間で統一
        probs_tot, probs_98 = [], []
        for dt_s, w, _ in scenarios:
            probs_tot.append(crop_to_display(
                lons, probability_map(mask_total, lons, dt_s, w, dt_ref)))
            probs_98.append(crop_to_display(
                lons, probability_map(mask_98, lons, dt_s, w, dt_ref)))
        vmax_98 = max(0.25, max(p.max() for _, p in probs_98))
        vmax_tot = max(0.25, max(p.max() for _, p in probs_tot))
        pc_98 = pc_tot = None
        for ax, (dt_s, w, title), (lons_d, p98), (_, ptot) in zip(
                axes[0], scenarios, probs_98, probs_tot):
            LON, LAT = np.meshgrid(lons_d, lats)
            pc_98 = ax.pcolormesh(LON, LAT, p98, cmap=plt.cm.Blues,
                                  vmin=0.0, vmax=vmax_98, shading="auto",
                                  zorder=1)
            ptot_m = np.ma.masked_less(ptot, 0.05)
            pc_tot = ax.pcolormesh(LON, LAT, ptot_m, cmap=plt.cm.Reds,
                                   vmin=0.0, vmax=vmax_tot, shading="auto",
                                   zorder=2)
            draw_japan(ax, lons_d, lats)
            ax.set_title(title, fontsize=9.5)
            ax.set_xlabel("東経 [度]", fontsize=8.5)
            ax.set_ylabel("北緯 [度]", fontsize=8.5)
        cb1 = fig.colorbar(pc_tot, ax=axes[0].tolist(), shrink=0.82,
                           pad=0.012, fraction=0.035)
        cb1.set_label("可視皆既の確率 (赤系)", fontsize=8.5)
        cb1.ax.tick_params(labelsize=7)
        cb2 = fig.colorbar(pc_98, ax=axes[0].tolist(), shrink=0.82,
                           pad=0.012, fraction=0.035)
        cb2.set_label("可視深食(食分0.98以上)の確率 (青系)", fontsize=8.5)
        cb2.ax.tick_params(labelsize=7)
        fig.suptitle(suptitle, fontsize=11, y=1.02)
        path = os.path.join(FIG_DIR, fname)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path, (lons, lats, mask_total, mask_98)

    fig, axes = plt.subplots(2, n, figsize=(5.1 * n, 10.2), dpi=200,
                             squeeze=False)
    row_specs = [(mask_total, "可視皆既の確率"),
                 (mask_98, "可視深食(食分0.98以上)の確率")]
    # 行ごとにスケールを揃える (パネル間比較を正確に)
    for r, (mask, cb_label) in enumerate(row_specs):
        probs = []
        for dt_s, w, _ in scenarios:
            prob = probability_map(mask, lons, dt_s, w, dt_ref)
            probs.append(crop_to_display(lons, prob))
        vmax = max(0.25, max(p.max() for _, p in probs))
        for ax, (dt_s, w, title), (lons_d, prob_d) in zip(
                axes[r], scenarios, probs):
            LON, LAT = np.meshgrid(lons_d, lats)
            pc = ax.pcolormesh(LON, LAT, prob_d, cmap=PROB_CMAP, vmin=0.0,
                               vmax=vmax, shading="auto", zorder=1)
            cb = fig.colorbar(pc, ax=ax, shrink=0.75, pad=0.02)
            cb.set_label(cb_label, fontsize=8)
            cb.ax.tick_params(labelsize=7)
            draw_japan(ax, lons_d, lats)
            ax.set_title(title, fontsize=9.5)
            ax.set_xlabel("東経 [度]", fontsize=8.5)
            ax.set_ylabel("北緯 [度]", fontsize=8.5)
    fig.suptitle(suptitle, fontsize=11, y=1.00)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, fname)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path, (lons, lats, mask_total, mask_98)


# ================================================================ CSV
def site_probability_rows(ecl_key, masks, scenarios):
    ecl = ECLIPSES[ecl_key]
    dt_ref = ecl["dt_nasa"]
    lons, lats, mask_total, mask_98 = masks
    rows = []
    for dt_s, w, scen_name in scenarios:
        p_tot = probability_map(mask_total, lons, dt_s, w, dt_ref)
        p_98 = probability_map(mask_98, lons, dt_s, w, dt_ref)
        for site, (slon, slat) in SITES.items():
            i = np.argmin(np.abs(lats - slat))
            j = np.argmin(np.abs(lons - slon))
            rows.append(dict(
                eclipse=ecl_key, scenario=scen_name, site=site,
                p_visible_totality=round(float(p_tot[i, j]), 4),
                p_visible_mag_ge_098=round(float(p_98[i, j]), 4),
            ))
    return rows


def main():
    print("== ΔTアンサンブル生成 ==")
    ens_a = DeltaTEnsemble(inflation=1.0, seed=42)
    ens_b = DeltaTEnsemble(inflation=3.0, seed=43)

    print("== 図1: ΔT軌道と247年分布 ==")
    p1 = figure_deltat_ensemble(ens_a, ens_b)
    print("  ->", p1)

    print("== 図2: 247年 皆既帯スイープ ==")
    p2 = figure_band_sweep()
    print("  ->", p2)

    all_rows = []

    print("== 図3: 247年 可視皆既確率マップ (3シナリオ) ==")
    s_a = ens_a.sample_at(247)
    s_b = ens_b.sample_at(247)
    ones_a = np.ones(len(s_a))
    ones_b = np.ones(len(s_b))
    w_luo = luoyang_weights(s_b, 7750.0, 8900.0)
    scen_247 = [
        (s_a, ones_a, "(a) 公式σ (M&S2004系, σ≈198s)"),
        (s_b, ones_b, "(b) 揺らぎ3倍シナリオ (σ≈594s)"),
        (s_b, w_luo, "(c) (b)+中国史料制約 (7750<ΔT<8900s)"),
    ]
    p3, masks_247 = figure_probability_maps(
        "0247-03-24", scen_247, "fig3_probability_247.png",
        "247年3月24日の日食: 「可視皆既」を経験した確率 (ΔTモンテカルロ)")
    print("  ->", p3)
    all_rows += site_probability_rows(
        "0247-03-24", masks_247,
        [(s_a, ones_a, "official"), (s_b, ones_b, "inflated3x"),
         (s_b, w_luo, "inflated3x+luoyang")])

    print("== 検証: 畳み込み確率地図 vs エンジン直接計算 ==")
    ecl = ECLIPSES["0247-03-24"]
    lons, lats, mask_total, _ = masks_247
    err = verify_convolution(ecl, ecl["dt_nasa"], lons, lats, mask_total, s_b)
    print(f"  最大絶対誤差 = {err:.4f} (格子離散化誤差の範囲なら合格)")

    print("== 図4: 他候補 (53年/158年/248年) の確率マップ ==")
    for key, fname in [("0053-03-09", "fig4a_probability_053.png"),
                       ("0158-07-13", "fig4b_probability_158.png"),
                       ("0248-09-05", "fig4c_probability_248.png")]:
        year = ECLIPSES[key]["year"]
        sa = ens_a.sample_at(year)
        sb = ens_b.sample_at(year)
        scen = [
            (sa, np.ones(len(sa)), "(a) 公式σ"),
            (sb, np.ones(len(sb)), "(b) 揺らぎ3倍シナリオ"),
        ]
        p, masks = figure_probability_maps(
            key, scen, fname,
            f"{ECLIPSES[key]['label']}: 「可視皆既」を経験した確率")
        print("  ->", p)
        all_rows += site_probability_rows(
            key, masks, [(sa, np.ones(len(sa)), "official"),
                         (sb, np.ones(len(sb)), "inflated3x")])

    print("== CSV出力 ==")
    csv_path = os.path.join(OUT_DIR, "site_probabilities.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(all_rows)
    print("  ->", csv_path)

    win_path = os.path.join(OUT_DIR, "totality_delta_t_windows.csv")
    with open(win_path, "w", newline="", encoding="utf-8") as f:
        wtr = csv.writer(f)
        wtr.writerow(["eclipse", "site", "dt_lo_sec", "dt_hi_sec",
                      "window_width_sec"])
        for key in ECLIPSES:
            for site, (slon, slat) in SITES.items():
                win = find_totality_delta_t_window(
                    ECLIPSES[key], slon, slat, dt_lo=4000, dt_hi=16000,
                    min_alt_deg=MIN_ALT)
                if win:
                    wtr.writerow([key, site, f"{win[0]:.0f}", f"{win[1]:.0f}",
                                  f"{win[1]-win[0]:.0f}"])
                else:
                    wtr.writerow([key, site, "", "", ""])
    print("  ->", win_path)
    print("完了")


if __name__ == "__main__":
    main()
