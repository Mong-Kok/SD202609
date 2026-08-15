#!/usr/bin/env python3
"""NASA Besselian elements engine for ancient solar eclipses.

邪馬台国・卑弥呼時代の日食を、NASA公開のBessel要素多項式から自前計算する。
天体暦ファイルへの依存なし(numpyのみ)。ΔTは「地球の自転位相=経度方向の
平行移動」として陽に現れる。具体的には観測点の時角が

    H = mu + lon_east - 0.00417807 * delta_t_sec   [deg]

となるため、ΔTを (dT) 秒変えることは、観測点を東経方向に
0.00417807 * dT 度だけ西へ動かすことと厳密に等価である。
この性質が、後段のモンテカルロ確率地図を高速化する鍵になる。

座標系・アルゴリズムは Explanatory Supplement / Meeus,
"Elements of Solar Eclipses" の標準的なBessel要素法に従う。

Bessel要素の出典: NASA Eclipse Web Site (Espenak & Meeus,
Five Millennium Canon; Sun=VSOP87, Moon=ELP-2000/82)
  https://eclipse.gsfc.nasa.gov/SEsearch/SEdata.php?Ecl=00530309
  https://eclipse.gsfc.nasa.gov/SEsearch/SEdata.php?Ecl=01580713
  https://eclipse.gsfc.nasa.gov/SEsearch/SEdata.php?Ecl=02470324
  https://eclipse.gsfc.nasa.gov/SEsearch/SEdata.php?Ecl=02480905

注意: これらの要素はNASAの採用幾何(VSOP87/ELP-2000/82)に基づく。
ΔTの値とどの月理論・暦表を組にしたかは常にペアで記録すること。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# ΔT 1秒あたりの経度換算係数 [deg/s]。恒星時レート(1.002738)込み。
K_DEG_PER_SEC = 0.00417807

# 「見かけの地平線」: 大気差(約34')により、太陽中心の真高度が-0.57°でも
# 見かけ高度は0°。Swiss Ephemeris系の可視判定(見かけ高度>=0)と合わせる
# にはこの値を min_alt_deg に使う。幾何学的地平線なら0.0。
REFRACTION_HORIZON_DEG = -0.57

# 地球楕円体 (Bessel要素法の標準値)
FLATTENING = 1.0 / 298.257
EARTH_RADIUS_M = 6378137.0


ECLIPSES: Dict[str, dict] = {
    # ユリウス暦の日付ラベル。t0はTDT(=TT)時。ΔTはNASA採用値[s]。
    "0053-03-09": dict(
        label="53年3月9日 (hybrid)", jtype="H",
        t0=4.0, dt_nasa=10005.9,
        x=[-0.4926130, 0.4834910, 0.0000244, -0.0000068],
        y=[0.5289480, 0.2650959, 0.0000087, -0.0000040],
        d=[-5.2501302, 0.0154340, 0.0000010, 0.0],
        l1=[0.5477330, -0.0001143, -0.0000116, 0.0],
        l2=[0.0015870, -0.0001137, -0.0000115, 0.0],
        mu=[236.963120, 15.004460, 0.0, 0.0],
        tanf1=0.0046669, tanf2=0.0046436,
        greatest=dict(tdt_h=4.0 + 19/60 + 20/3600, lat=33.4, lon=136.3,
                      mag=1.0065, width_km=30.7),
        year=53, month=3,
    ),
    "0158-07-13": dict(
        label="158年7月13日 (total)", jtype="T",
        t0=11.0, dt_nasa=8994.1,
        x=[0.0979340, 0.5526497, -0.0000278, -0.0000080],
        y=[0.6220060, -0.0224439, -0.0001802, 0.0000002],
        d=[22.3602791, -0.0054950, -0.0000050, 0.0],
        l1=[0.5442010, 0.0001158, -0.0000115, 0.0],
        l2=[-0.0019260, 0.0001152, -0.0000114, 0.0],
        mu=[344.665314, 14.999710, 0.0, 0.0],
        tanf1=0.0046119, tanf2=0.0045889,
        greatest=dict(tdt_h=10.0 + 52/60 + 7/3600, lat=61.3, lon=57.9,
                      mag=1.0206, width_km=90.4),
        year=158, month=7,
    ),
    "0247-03-24": dict(
        label="247年3月24日 (total)", jtype="T",
        t0=10.0, dt_nasa=8151.9,
        x=[-0.1646930, 0.5468715, 0.0000213, -0.0000089],
        y=[0.2405040, 0.1774078, -0.0000154, -0.0000028],
        d=[1.0995800, 0.0158540, 0.0, 0.0],
        l1=[0.5365960, -0.0000747, -0.0000125, 0.0],
        l2=[-0.0094940, -0.0000744, -0.0000125, 0.0],
        mu=[328.370117, 15.005180, 0.0, 0.0],
        tanf1=0.0046490, tanf2=0.0046258,
        greatest=dict(tdt_h=10.0 + 8/60 + 36/3600, lat=16.6, lon=58.4,
                      mag=1.0538, width_km=184.9),
        year=247, month=3,
    ),
    "0248-09-05": dict(
        label="248年9月5日 (total)", jtype="T",
        t0=1.0, dt_nasa=8138.0,
        x=[0.2301060, 0.5376427, -0.0000394, -0.0000082],
        y=[0.2824500, -0.1636429, -0.0000520, 0.0000024],
        d=[7.1743398, -0.0153900, -0.0000020, 0.0],
        l1=[0.5435250, 0.0001043, -0.0000120, 0.0],
        l2=[-0.0026000, 0.0001037, -0.0000120, 0.0],
        mu=[195.585815, 15.004170, 0.0, 0.0],
        tanf1=0.0046755, tanf2=0.0046522,
        greatest=dict(tdt_h=0.0 + 45/60 + 17/3600, lat=26.1, lon=-151.6,
                      mag=1.0263, width_km=94.7),
        year=248, month=9,
    ),
}

# 代表地点 (先行研究との照合用; 東経・北緯)
SITES = {
    "dazaifu": (130.5239, 33.5128),      # 大宰府
    "asuka": (135.8206, 34.4711),        # 飛鳥
    "makimuku": (135.8380, 34.5440),     # 纒向
    "yoshinogari": (130.3989, 33.3239),  # 吉野ヶ里
    "usa": (131.3750, 33.5320),          # 宇佐
    "izumo": (132.6860, 35.4020),        # 出雲
}


def _poly(coeffs: List[float], t: np.ndarray | float):
    """3次多項式評価とその導関数 (tはt0からの時間差[時])."""
    c0, c1, c2, c3 = coeffs
    val = c0 + t * (c1 + t * (c2 + t * c3))
    der = c1 + t * (2.0 * c2 + t * 3.0 * c3)
    return val, der


def observer_geocentric(lat_deg, height_m=0.0):
    """測地緯度 -> 地心量 rho*sin(phi'), rho*cos(phi')."""
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    b_a = 1.0 - FLATTENING  # 極半径/赤道半径
    u = np.arctan(b_a * np.tan(lat))
    h = height_m / EARTH_RADIUS_M
    rho_sin = b_a * np.sin(u) + h * np.sin(lat)
    rho_cos = np.cos(u) + h * np.cos(lat)
    return rho_sin, rho_cos


@dataclass
class LocalCircumstances:
    """グリッド(または単一点)の局地条件.

    「幾何学的最大」(geom_*) と「可視最大」(magnitude, is_total 等) を
    分離して持つ。247年のような日没際の日食では、幾何学的最大が
    地平線下に沈むことがあり、その場合の可視最大は日没(ζ=0)時点で
    評価する (04レポートの設計方針)。
    """
    t_max_tt: np.ndarray          # 幾何学的食甚時刻 (TDT時)
    t_eval_tt: np.ndarray         # 可視最大の評価時刻 (TDT時; 日没時にクリップ)
    geom_magnitude: np.ndarray    # 幾何学的最大の食分 (地平線下でも計算)
    geom_sun_alt_deg: np.ndarray  # 幾何学的食甚時の太陽高度
    magnitude: np.ndarray         # 可視最大の食分 (太陽が地平線上の時刻で評価)
    m_dist: np.ndarray            # 可視評価時刻での影軸距離
    L1p: np.ndarray               # 同・半影半径
    L2p: np.ndarray               # 同・本影半径 (皆既なら負)
    sun_alt_deg: np.ndarray       # 可視評価時刻での太陽高度
    is_total: np.ndarray          # 可視皆既か (bool)
    is_annular: np.ndarray        # 可視金環か (bool)
    is_visible: np.ndarray        # 可視最大時に食が進行中か (bool)


def _geometry_at(ecl, t, lon, lat, rho_sin, rho_cos, delta_t_sec):
    """時刻t(t0起点,時)における局地幾何量一式を返す."""
    x, xp = _poly(ecl["x"], t)
    y, yp = _poly(ecl["y"], t)
    d, dp = _poly(ecl["d"], t)
    mu, mup = _poly(ecl["mu"], t)
    l1, _ = _poly(ecl["l1"], t)
    l2, _ = _poly(ecl["l2"], t)
    d_r = np.radians(d)
    # 時角: ΔTの効果はここに集約される (経度シフトと厳密に等価)
    H = np.radians(mu + lon - K_DEG_PER_SEC * delta_t_sec)
    sinH, cosH = np.sin(H), np.cos(H)
    sind, cosd = np.sin(d_r), np.cos(d_r)
    xi = rho_cos * sinH
    eta = rho_sin * cosd - rho_cos * cosH * sind
    zeta = rho_sin * sind + rho_cos * cosH * cosd
    mup_r = np.radians(mup)
    dp_r = np.radians(dp)
    xip = mup_r * rho_cos * cosH
    etap = mup_r * xi * sind - dp_r * zeta
    zetap = (rho_sin * cosd * dp_r
             - rho_cos * (sinH * mup_r * cosd + cosH * sind * dp_r))
    return dict(x=x, y=y, xp=xp, yp=yp, xi=xi, eta=eta, zeta=zeta,
                xip=xip, etap=etap, zetap=zetap, l1=l1, l2=l2)


def local_circumstances(
    ecl: dict,
    delta_t_sec: float,
    lon_east_deg,
    lat_deg,
    height_m: float = 0.0,
    min_alt_deg: float = 0.0,
    n_iter: int = 8,
) -> LocalCircumstances:
    """指定ΔTでの各観測点の食甚と可視皆既判定 (numpyベクトル化).

    lon_east_deg, lat_deg: スカラーまたは同形状のndarray。
    min_alt_deg: 「見えた」と判定する最低太陽高度 [deg]。
    """
    lon = np.asarray(lon_east_deg, dtype=float)
    lat = np.asarray(lat_deg, dtype=float)
    lon, lat = np.broadcast_arrays(lon, lat)
    rho_sin, rho_cos = observer_geocentric(lat, height_m)

    # --- (1) 幾何学的食甚: 影軸距離最小へのニュートン反復
    t = np.zeros_like(lon, dtype=float)
    for _ in range(n_iter):
        g = _geometry_at(ecl, t, lon, lat, rho_sin, rho_cos, delta_t_sec)
        u = g["x"] - g["xi"]
        v = g["y"] - g["eta"]
        up = g["xp"] - g["xip"]
        vp = g["yp"] - g["etap"]
        denom = up * up + vp * vp
        dt = -(u * up + v * vp) / np.where(denom == 0.0, 1e-12, denom)
        t = t + np.clip(dt, -6.0, 6.0)
    t_geom = t

    g = _geometry_at(ecl, t_geom, lon, lat, rho_sin, rho_cos, delta_t_sec)
    zeta_geom = g["zeta"]
    geom_alt = np.degrees(np.arcsin(np.clip(zeta_geom, -1.0, 1.0)))
    m_geom = np.hypot(g["x"] - g["xi"], g["y"] - g["eta"])
    L1p_g = g["l1"] - zeta_geom * ecl["tanf1"]
    L2p_g = g["l2"] - zeta_geom * ecl["tanf2"]
    with np.errstate(invalid="ignore", divide="ignore"):
        mag_geom = (L1p_g - m_geom) / (L1p_g + L2p_g)
    mag_geom = np.maximum(0.0, mag_geom)

    # --- (2) 幾何学的最大が地平線下の点: 地平線通過(ζ=sin(min_alt))へ
    #         ニュートン収束し、そこで可視最大を評価する (日没食の処理)
    sin_min = math.sin(math.radians(min_alt_deg))
    below = zeta_geom < sin_min
    t_eval = t_geom.copy()
    if np.any(below):
        th = t_geom.copy()
        for _ in range(12):
            gh = _geometry_at(ecl, th, lon, lat, rho_sin, rho_cos, delta_t_sec)
            f = gh["zeta"] - sin_min
            fp = np.where(np.abs(gh["zetap"]) < 1e-9,
                          np.sign(gh["zetap"]) * 1e-9 + 1e-12, gh["zetap"])
            step = np.clip(-f / fp, -3.0, 3.0)
            th = np.where(below, th + step, th)
        t_eval = np.where(below, th, t_geom)

    # --- (3) 可視最大の評価
    ge = _geometry_at(ecl, t_eval, lon, lat, rho_sin, rho_cos, delta_t_sec)
    zeta = ge["zeta"]
    m = np.hypot(ge["x"] - ge["xi"], ge["y"] - ge["eta"])
    L1p = ge["l1"] - zeta * ecl["tanf1"]
    L2p = ge["l2"] - zeta * ecl["tanf2"]
    with np.errstate(invalid="ignore", divide="ignore"):
        mag = (L1p - m) / (L1p + L2p)
    mag = np.maximum(0.0, mag)

    sun_alt = np.degrees(np.arcsin(np.clip(zeta, -1.0, 1.0)))
    on_horizon_ok = zeta >= sin_min - 1e-9   # 地平線(または指定高度)以上
    in_eclipse = m < L1p                     # その時刻に食が進行中
    is_visible = on_horizon_ok & in_eclipse
    is_total = (m < -L2p) & (L2p < 0.0) & on_horizon_ok
    is_annular = (L2p > 0.0) & (m < L2p) & on_horizon_ok
    mag = np.where(is_visible, mag, 0.0)

    return LocalCircumstances(
        t_max_tt=ecl["t0"] + t_geom,
        t_eval_tt=ecl["t0"] + t_eval,
        geom_magnitude=mag_geom,
        geom_sun_alt_deg=geom_alt,
        magnitude=mag,
        m_dist=m,
        L1p=L1p,
        L2p=L2p,
        sun_alt_deg=sun_alt,
        is_total=is_total,
        is_annular=is_annular,
        is_visible=is_visible,
    )


def totality_duration_sec(ecl, delta_t_sec, lon_east_deg, lat_deg,
                          height_m=0.0, n_scan=241, half_window_h=0.05):
    """単一地点の皆既継続時間[秒] (皆既でなければ0)."""
    lc = local_circumstances(ecl, delta_t_sec, lon_east_deg, lat_deg, height_m)
    if not bool(np.ravel(lc.is_total)[0]):
        return 0.0
    t_mid = float(np.ravel(lc.t_max_tt)[0]) - ecl["t0"]
    ts = np.linspace(t_mid - half_window_h, t_mid + half_window_h, n_scan)
    inside = []
    rho_sin, rho_cos = observer_geocentric(lat_deg, height_m)
    for t in ts:
        x, _ = _poly(ecl["x"], t)
        y, _ = _poly(ecl["y"], t)
        d, _ = _poly(ecl["d"], t)
        mu, _ = _poly(ecl["mu"], t)
        l2, _ = _poly(ecl["l2"], t)
        d_r = math.radians(d)
        H = math.radians(mu + lon_east_deg - K_DEG_PER_SEC * delta_t_sec)
        xi = rho_cos * math.sin(H)
        eta = rho_sin * math.cos(d_r) - rho_cos * math.cos(H) * math.sin(d_r)
        zeta = rho_sin * math.sin(d_r) + rho_cos * math.cos(H) * math.cos(d_r)
        m = math.hypot(x - xi, y - eta)
        L2p = l2 - zeta * ecl["tanf2"]
        inside.append(m < -L2p and L2p < 0.0)
    frac = sum(inside) / len(inside)
    return frac * 2.0 * half_window_h * 3600.0


def find_totality_delta_t_window(
    ecl, lon_east_deg, lat_deg,
    dt_lo=5000.0, dt_hi=14000.0, step=25.0, refine=0.5,
    require_visible=True, min_alt_deg=0.0,
) -> Optional[tuple]:
    """指定地点が(可視)皆既になるΔT区間 [s] を走査+二分探索で求める."""
    dts = np.arange(dt_lo, dt_hi + step, step)
    ok = []
    for dt in dts:
        lc = local_circumstances(ecl, float(dt), lon_east_deg, lat_deg,
                                 min_alt_deg=min_alt_deg)
        t = bool(np.ravel(lc.is_total)[0])
        if require_visible:
            t = t and bool(np.ravel(lc.is_visible)[0])
        ok.append(t)
    ok = np.array(ok)
    if not ok.any():
        return None

    def bisect(lo, hi, want_true_at_hi):
        while hi - lo > refine:
            mid = 0.5 * (lo + hi)
            lc = local_circumstances(ecl, mid, lon_east_deg, lat_deg,
                                     min_alt_deg=min_alt_deg)
            t = bool(np.ravel(lc.is_total)[0])
            if require_visible:
                t = t and bool(np.ravel(lc.is_visible)[0])
            if t == want_true_at_hi:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    i_first = int(np.argmax(ok))
    i_last = len(ok) - 1 - int(np.argmax(ok[::-1]))
    lo_edge = (bisect(dts[i_first] - step, dts[i_first], True)
               if i_first > 0 else dts[0])
    hi_edge = (bisect(dts[i_last], dts[i_last] + step, False)
               if i_last < len(dts) - 1 else dts[-1])
    return (lo_edge, hi_edge)


# ---------------------------------------------------------------- 検算
def validate() -> bool:
    """NASA公表値と02レポートの閾値に対する検算。すべて通ればTrue."""
    ok = True
    print("=" * 72)
    print("検算1: NASA採用ΔTで、NASA公表の最大食地点が皆既/hybridになるか")
    print("-" * 72)
    for key, ecl in ECLIPSES.items():
        g = ecl["greatest"]
        lc = local_circumstances(ecl, ecl["dt_nasa"], g["lon"], g["lat"])
        total = bool(lc.is_total or lc.is_annular)  # hybridは境界注意
        t_err_min = (float(lc.t_max_tt) - g["tdt_h"]) * 60.0
        stat = "OK" if (total and abs(t_err_min) < 2.0) else "NG"
        if stat == "NG":
            ok = False
        print(f"  {key}: central={total}  食甚時刻差={t_err_min:+.2f}分  "
              f"mag={float(lc.magnitude):.4f}  [{stat}]")

    print()
    print("検算2: 247年の皆既ΔT閾値 (02レポート: 飛鳥≈9500-9700s, 大宰府≈1.19万s)")
    print("-" * 72)
    ecl = ECLIPSES["0247-03-24"]
    for name, expect in [("asuka", (9400, 9800)), ("dazaifu", (11500, 12300))]:
        lon, lat = SITES[name]
        win = find_totality_delta_t_window(ecl, lon, lat)
        if win is None:
            print(f"  {name}: 皆既区間なし [NG]")
            ok = False
            continue
        lo, hi = win
        stat = "OK" if expect[0] <= lo <= expect[1] else "NG"
        if stat == "NG":
            ok = False
        print(f"  {name}: 皆既ΔT区間 = {lo:.0f} - {hi:.0f} s "
              f"(期待下限域 {expect[0]}-{expect[1]}) [{stat}]")

    print()
    print("検算3: 158年の飛鳥皆既窓 (02レポート: おおむね8500-8750s域)")
    print("-" * 72)
    ecl = ECLIPSES["0158-07-13"]
    lon, lat = SITES["asuka"]
    win = find_totality_delta_t_window(ecl, lon, lat)
    if win is None:
        print("  asuka: 皆既区間なし [NG]")
        ok = False
    else:
        lo, hi = win
        stat = "OK" if 8300 <= lo <= 8800 and hi <= 9100 else "NG"
        if stat == "NG":
            ok = False
        print(f"  asuka: 皆既ΔT区間 = {lo:.0f} - {hi:.0f} s [{stat}]")

    print()
    print("検算4: 247年・記事/2012論文の代表ΔTでの大宰府・飛鳥の食分再現")
    print("       (可視最大; 見かけの地平線=-0.57°で日没クリップ)")
    print("-" * 72)
    ecl = ECLIPSES["0247-03-24"]
    expect_tbl = {  # 02レポートの再計算値 (mag, is_total)
        7750.0: dict(dazaifu=(0.838, False), asuka=(0.467, False)),
        8500.0: dict(dazaifu=(0.991, False), asuka=(0.694, False)),
        8900.0: dict(dazaifu=(0.990, False), asuka=(0.815, False)),
        9700.0: dict(dazaifu=(0.990, False), asuka=(None, True)),
    }
    for dt, row in expect_tbl.items():
        outs = []
        for name, (mag_e, tot_e) in row.items():
            lon, lat = SITES[name]
            lc = local_circumstances(ecl, dt, lon, lat,
                                     min_alt_deg=REFRACTION_HORIZON_DEG)
            mag = float(lc.magnitude)
            tot = bool(lc.is_total)
            good = (tot == tot_e) and (mag_e is None or abs(mag - mag_e) < 0.03)
            if not good:
                ok = False
            outs.append(f"{name}: mag={mag:.3f} total={tot} "
                        f"[{'OK' if good else 'NG'}]")
        print(f"  ΔT={dt:6.0f}s  " + "   ".join(outs))

    print("=" * 72)
    print("総合判定:", "ALL OK" if ok else "FAILURES PRESENT")
    return ok


if __name__ == "__main__":
    validate()
