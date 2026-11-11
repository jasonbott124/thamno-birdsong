"""
song_vs_geo_distance.py  --  Thamno-2026

Isolation by distance in song: for each species, how does song distance
(Euclidean distance in the song-feature space) grow with geographic distance?

Uses the per-recording features from genus_song_clusters.py, so run that first
for the genus you want.

Steps, per species:
  1. Pick the feature columns (FEATURES; MFCCs optional).
  2. Standardise each feature within the species (z-scores), so every
     feature counts equally in the distance.
  3. Average recordings at the same site (UNIT = "site"), so one recordist
     with many files at one spot doesn't count many times. Or UNIT = "recording".
  4. Pairwise song distance (Euclidean) and geographic distance (great-circle km).
  5. Mantel test: correlation between the two distance matrices, with a
     permutation p-value (rows/columns of one matrix shuffled together).

Inputs:  output/genus_clusters/<GENUS>/features.csv
Outputs: output/song_vs_geo/<GENUS>/
  pairs.csv          one row per pair of sites: species, both ids and positions,
                     geo_km, song_dist  (ready for adding a cross-river flag later)
  mantel.csv         per species: sites, pairs, Mantel r, p, slope per 1,000 km
  song_vs_geo.png    one panel per species: pairs (grey) + binned means, r and p
  mantel_summary.png Mantel r per species, filled = p < 0.05

Run from Thamno-2026/code:  python song_vs_geo_distance.py
Needs: numpy, pandas, matplotlib, scipy
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# SETTINGS
# =============================================================================
GENUS = "Thamnophilus"
UNIT = "site"                 # "site" (average recordings per site) or "recording"
MIN_UNITS = 8                 # species need at least this many sites/recordings
INCLUDE_MFCC = False          # MFCCs mostly reflect recording conditions; off by default
FEATURES = ["bout_s", "notes", "rate_hz", "rate_change", "ioi_cv", "peak_hz", "f10_hz",
            "f90_hz", "bw_hz", "slope_hz_s", "bouts_per_min"]
METHOD = "spearman"           # Mantel correlation: "spearman" (rank) or "pearson"
N_PERM = 999
SEED = 0
N_BINS = 8                    # distance bins for the binned means

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent if (_HERE.parent / "data").exists() else Path.home() / "Desktop" / "Thamno-2026"
IN = ROOT / "output" / "genus_clusters" / GENUS / "features.csv"
OUT = ROOT / "output" / "song_vs_geo" / GENUS

# =============================================================================


def haversine_km(lat, lon):
    """Condensed great-circle distance matrix (same order as pdist)."""
    la, lo = np.radians(lat), np.radians(lon)
    i, j = np.triu_indices(len(la), k=1)
    a = (np.sin((la[j] - la[i]) / 2) ** 2
         + np.cos(la[i]) * np.cos(la[j]) * np.sin((lo[j] - lo[i]) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def mantel(d_song, d_geo, method=METHOD, n_perm=N_PERM, seed=SEED):
    """Mantel r and one-sided permutation p (H1: positive correlation)."""
    if method == "spearman":
        x, y = rankdata(d_song), rankdata(d_geo)
    else:
        x, y = d_song.astype(float), d_geo.astype(float)
    r_obs = np.corrcoef(x, y)[0, 1]
    S = squareform(x)
    n = S.shape[0]
    iu = np.triu_indices(n, k=1)
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        p = rng.permutation(n)
        if np.corrcoef(S[np.ix_(p, p)][iu], y)[0, 1] >= r_obs:
            hits += 1
    return r_obs, (hits + 1) / (n_perm + 1)


def prepare(df, cols):
    """Within-species z-scores, then average per site if UNIT == 'site'."""
    X = df[cols].astype(float)
    X = X.fillna(X.median())
    sd = X.std(ddof=0).replace(0, np.nan)
    Z = ((X - X.mean()) / sd).dropna(axis=1, how="all").fillna(0)
    Z = Z.assign(lat=df.lat.values, lon=df.lon.values, site=df.site.values, id=df.id.astype(str).values)
    if UNIT == "site":
        Z = (Z.groupby("site")
               .agg({**{c: "mean" for c in Z.columns if c not in ("site", "id")},
                     "id": lambda s: ";".join(s)})
               .reset_index())
    else:
        Z = Z.reset_index(drop=True)
    feats = [c for c in Z.columns if c not in ("site", "id", "lat", "lon")]
    return Z, feats


def main():
    if not IN.exists():
        raise SystemExit(f"Missing {IN}. Run genus_song_clusters.py for {GENUS} first.")
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(IN)
    cols = FEATURES + ([c for c in df.columns if c.startswith("mfcc")] if INCLUDE_MFCC else [])
    cols = [c for c in cols if c in df.columns]
    print(f"{GENUS}: {len(df)} recordings, {df.species.nunique()} species; "
          f"{len(cols)} features ({'with' if INCLUDE_MFCC else 'without'} MFCCs); unit = {UNIT}")

    pairs, rows = [], []
    for sp, d in df.groupby("species"):
        Z, feats = prepare(d, cols)
        if len(Z) < MIN_UNITS:
            continue
        ds = pdist(Z[feats].values)                       # Euclidean song distance
        dg = haversine_km(Z.lat.values, Z.lon.values)
        keep = dg > 0                                     # same-site pairs (recording mode) carry no distance
        r, p = mantel(ds, dg)
        slope = np.polyfit(dg[keep] / 1000, ds[keep], 1)[0] if keep.sum() > 2 else np.nan
        rows.append(dict(species=sp, units=len(Z), pairs=len(ds), mantel_r=round(r, 3), p=round(p, 3),
                         slope_per_1000km=round(slope, 3), max_km=round(dg.max())))
        i, j = np.triu_indices(len(Z), k=1)
        pairs.append(pd.DataFrame(dict(
            species=sp, id_a=Z.id.values[i], id_b=Z.id.values[j],
            lat_a=Z.lat.values[i], lon_a=Z.lon.values[i], lat_b=Z.lat.values[j], lon_b=Z.lon.values[j],
            geo_km=dg.round(1), song_dist=ds.round(4))))
    if not rows:
        raise SystemExit(f"No species with >= {MIN_UNITS} {UNIT}s. Lower MIN_UNITS.")

    res = pd.DataFrame(rows).sort_values("mantel_r", ascending=False)
    pr = pd.concat(pairs, ignore_index=True)
    res.to_csv(OUT / "mantel.csv", index=False)
    pr.to_csv(OUT / "pairs.csv", index=False)
    print(res.to_string(index=False))

    # ---- per-species panels -------------------------------------------------
    n = len(res); ncol = 4; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.3 * nrow), squeeze=False)
    for ax, (_, r) in zip(axes.flat, res.iterrows()):
        d = pr[pr.species == r.species]
        ax.scatter(d.geo_km, d.song_dist, s=5, color="#bdbdbd", alpha=0.6, lw=0)
        edges = np.quantile(d.geo_km, np.linspace(0, 1, N_BINS + 1))
        b = pd.cut(d.geo_km, np.unique(edges), include_lowest=True)
        m = d.groupby(b, observed=True).agg(x=("geo_km", "median"), y=("song_dist", "mean"),
                                            se=("song_dist", lambda v: v.std() / np.sqrt(len(v))))
        ax.errorbar(m.x, m.y, yerr=m.se, color="#08519c", marker="o", ms=4, lw=1.4, capsize=2)
        sig = "" if r.p >= 0.05 else " *"
        ax.set_title(f"{r.species.split()[1]}  ({r.units} {UNIT}s)\nMantel r = {r.mantel_r:.2f}, "
                     f"p = {r.p:.3f}{sig}", fontsize=8.5, loc="left")
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Geographic distance (km)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("Song distance (Euclidean, z-scored)", fontsize=8)
    fig.suptitle(f"{GENUS}: song distance vs geographic distance, per species "
                 f"({'Spearman' if METHOD == 'spearman' else 'Pearson'} Mantel, {N_PERM} permutations; "
                 f"grey = pairs, blue = binned mean ± SE)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "song_vs_geo.png", dpi=180)
    plt.close(fig)

    # ---- summary: Mantel r per species ---------------------------------------
    r_ = res.sort_values("mantel_r")
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(r_) + 1.4))
    y = np.arange(len(r_))
    sig = r_.p < 0.05
    ax.hlines(y, 0, r_.mantel_r, color="#bdbdbd", lw=1.2)
    ax.scatter(r_.mantel_r[~sig], y[~sig], s=40, facecolor="white", edgecolor="#08519c", lw=1.4,
               zorder=3, label="p >= 0.05")
    ax.scatter(r_.mantel_r[sig], y[sig], s=40, color="#08519c", zorder=3, label="p < 0.05")
    ax.axvline(0, color="#666", lw=0.8)
    ax.set_yticks(y, [f"{s_.split()[1]} ({u})" for s_, u in zip(r_.species, r_.units)],
                  fontsize=8, style="italic")
    ax.set_xlabel(f"Mantel r (song distance vs geographic distance, {METHOD})")
    ax.set_title(f"{GENUS}: isolation by distance in song, per species ({UNIT}s in brackets)",
                 loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "mantel_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
