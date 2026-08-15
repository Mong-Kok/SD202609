#!/usr/bin/env python3
"""ΔT(地球自転の累積遅れ)のモンテカルロ軌道アンサンブル.

物理的背景 (11/11.5レポートの整理):
  - ΔT = TT - UT1。長期トレンドは潮汐摩擦 (月潮汐 ~+2.4 ms/cy) と
    氷床後退のGIA (~-0.8 ms/cy) の和。
  - しかし数年〜十年〜千年スケールでは、コア・マントル結合による
    非潮汐の揺らぎが乗る。これが数百年積分されると、ΔTの不確かさの
    裾野を大きくする (NASAのσ=0.8t²の背後にはブラウン運動+ドリフト型
    モデルがある)。
  - 628年の飛鳥皆既日食は、標準スプライン値4449sに対し2267-2959sを
    要求する (公称σの10倍以上の乖離)。つまり「公式σ」は局所的な
    揺らぎを過小評価している可能性がある。

実装:
  中心曲線 = NASA/Espenak-Meeus多項式 (NASAのBessel要素と同じ幾何・
  同じ月理論ELP-2000/82の永年加速度補正込み。「ΔT値と暦表はペアで
  扱う」の原則)。
  揺らぎ = LOD(1日の長さ)の偏差をランダムウォークとして年々積分した
  経路群を生成し、各年代の周辺分布の標準偏差が
      inflation × σ_NASA(year)
  に一致するよう経験的にスケールする。inflation=1が「公式の不確かさ」、
  inflation>1が「628年問題を踏まえた揺らぎ拡大シナリオ」。
"""

from __future__ import annotations

import numpy as np


def em2006_delta_t(year: float, month: float = 6.5) -> float:
    """NASA Five Millennium Canon の -500..+500 用 ΔT多項式 [s].

    ELP-2000/82の月の永年加速度差に対する補正項込み。
    NASAのBessel要素ページのΔT値(247年: 8151.9s)を再現する。
    """
    y = year + (month - 0.5) / 12.0
    u = y / 100.0
    dt = (10583.6 - 1014.41 * u + 33.78311 * u**2 - 5.952053 * u**3
          - 0.1798452 * u**4 + 0.022174192 * u**5 + 0.0090316521 * u**6)
    return dt - 0.000012932 * (y - 1955.0) ** 2


def em2006_delta_t_500_1600(year: float) -> float:
    """同・+500..+1600 用の多項式 [s]."""
    u = (year - 1000.0) / 100.0
    dt = (1574.2 - 556.01 * u + 71.23472 * u**2 + 0.319781 * u**3
          - 0.8503463 * u**4 - 0.005050998 * u**5 + 0.0083572073 * u**6)
    return dt - 0.000012932 * (year - 1955.0) ** 2


def center_delta_t(years: np.ndarray) -> np.ndarray:
    """中心曲線: 年代に応じてNASA多項式を切り替え."""
    years = np.asarray(years, dtype=float)
    out = np.empty_like(years)
    lo = years <= 500.0
    out[lo] = np.vectorize(em2006_delta_t)(years[lo])
    out[~lo] = np.vectorize(em2006_delta_t_500_1600)(years[~lo])
    return out


def sigma_nasa(years: np.ndarray) -> np.ndarray:
    """ΔT標準誤差の放物線近似 σ=0.8t² [s] (1000BCE-1200CE向け).

    提案者はMorrison & Stephenson (2004)。NASA Eclipse Web Siteの
    "Uncertainty in Delta T" ページに掲載されている:
      https://eclipse.gsfc.nasa.gov/SEcat5/uncertainty.html
    同ページには「年0でσ=265秒=経度1.10°」という換算例もある。
    """
    t = (np.asarray(years, dtype=float) - 1820.0) / 100.0
    return 0.8 * t * t


# NASA歴史表 (Morrison & Stephenson 2004系; 図の照合用)
NASA_TABLE = {
    0: (10580, 260), 100: (9600, 240), 200: (8640, 210), 300: (7680, 180),
    400: (6700, 160), 500: (5710, 140), 600: (4740, 120), 700: (3810, 100),
    800: (2960, 80), 900: (2200, 70), 1000: (1570, 55),
}


class DeltaTEnsemble:
    """ΔT(t)軌道のモンテカルロアンサンブル.

    anchor_year(既定1900年; ΔTがよく分かっている時代)から過去へ、
    LOD偏差のランダムウォークを積分した経路を生成する。
    """

    def __init__(self, n_paths: int = 4000, year_min: int = 0,
                 anchor_year: int = 1900, inflation: float = 1.0,
                 sigma_floor: float = 2.0, seed: int = 42):
        self.inflation = inflation
        rng = np.random.default_rng(seed)
        # 年配列は anchor_year から過去へ降順
        self.years = np.arange(anchor_year, year_min - 1, -1, dtype=float)
        n_steps = len(self.years) - 1

        # LOD偏差のランダムウォーク → その積分がΔT偏差 (二重積分構造)
        lod_dev = np.cumsum(rng.standard_normal((n_paths, n_steps)), axis=1)
        W = np.concatenate(
            [np.zeros((n_paths, 1)), np.cumsum(lod_dev, axis=1)], axis=1)

        # 各年代の周辺分布のstdを inflation×σ_NASA に経験的スケーリング
        raw_std = W.std(axis=0)
        raw_std[raw_std == 0.0] = 1.0
        target = np.maximum(self.inflation * sigma_nasa(self.years),
                            sigma_floor)
        target[0] = 0.0  # アンカー年は固定
        self.W = W * (target / raw_std)[None, :]
        self.paths = center_delta_t(self.years)[None, :] + self.W

    def sample_at(self, year: float) -> np.ndarray:
        """指定年のΔTサンプル群 [s] (N,)."""
        idx = np.argmin(np.abs(self.years - year))
        return self.paths[:, idx]


def luoyang_weights(samples: np.ndarray, lo: float = 7750.0,
                    hi: float | None = 8900.0) -> np.ndarray:
    """2012年国立天文台報論文の中国史料制約による重み.

    『三国志』『晋書』の247年3月24日日食記事から:
      - 洛陽が部分食 (皆既でない) → ΔT > 7750 s
      - 洛陽で食分0.99級の深食まで許すなら → ΔT < 8900 s
    hi=Noneで下限のみの制約になる。
    """
    w = (samples > lo).astype(float)
    if hi is not None:
        w *= (samples < hi).astype(float)
    return w


if __name__ == "__main__":
    for inflation in (1.0, 3.0):
        ens = DeltaTEnsemble(inflation=inflation)
        s = ens.sample_at(247)
        w = luoyang_weights(s)
        print(f"inflation={inflation}: ΔT(247) = {s.mean():.0f} ± {s.std():.0f} s "
              f"(5-95%: {np.percentile(s,5):.0f}-{np.percentile(s,95):.0f}) "
              f"洛陽制約通過率={w.mean()*100:.1f}%")
