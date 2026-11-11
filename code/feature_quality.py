"""
feature_quality.py  --  Thamno-2026

How noisy is each song feature? Reads the per-recording features written by
genus_song_clusters.py (every genus found, or a chosen list) and scores each
feature on six checks, to decide which to keep in song_vs_geo_distance.py.

Checks, per feature:
  missing        share of recordings where the feature could not be measured (NaN)
  degenerate     share stuck at zero, i.e. the detector found nothing (e.g. notes = 0)
  outliers       share more than 5 robust SDs (MAD) from the median
  species_signal how much of the variation lies BETWEEN species rather than within them
                 (eta-squared on ranks, averaged over genera). A useful song feature
                 separates species; near 0 means it's mostly noise.
  site_noise     same species, same site, different recordings: how much they still
                 differ, relative to the species' overall spread (0 = perfectly repeatable,
                 1 = as variable as the whole species). High = measurement noise.
  artefact_rho   strongest rank correlation with recording conditions rather than the
                 bird: file length, quality grade (A vs B), sample rate, and number of
                 background species. High = the feature tracks the recording, not the song.
Plus a correlation matrix between features, to spot redundant pairs.

A feature is flagged when: missing > 20%, degenerate > 20%, species_signal < 0.10,
site_noise > 0.7, or |artefact_rho| > 0.3 (thresholds in SETTINGS).

Inputs:  output/genus_clusters/<genus>/features.csv
         data/xc_metadata/thamnophilidae_amazon.csv (recording metadata)
Outputs: output/feature_quality/
  feature_quality.csv      one row per feature, all checks + flags
  feature_quality.png      the checks as bar charts, flagged features in orange
  feature_correlation.png  feature-by-feature rank correlation

Run from Thamno-2026/code:  python feature_quality.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# SETTINGS
# =============================================================================
GENUS = "ALL"                 # one genus, a list, or "ALL" (every genus with features.csv)
INCLUDE_MFCC = True           # score the MFCCs too
MIN_PER_GROUP = 5             # recordings needed per species (species_signal) or site (site_noise)
FLAG = dict(missing=0.20, degenerate=0.20, outliers=0.05, species_signal=0.10,
            site_noise=0.70, artefact_rho=0.30)

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent if (_HERE.parent / "data").exists() else Path.home() / "Desktop" / "Thamno-2026"
FEAT_DIR = ROOT / "output" / "genus_clusters"
XC_CSV = ROOT / "data" / "xc_metadata" / "thamnophilidae_amazon.csv"
OUT = ROOT / "output" / "feature_quality"
META = ("id", "species", "lat", "lon", "site", "genus")

# =============================================================================


def load():
    if isinstance(GENUS, (list, tuple)):
        genera = list(GENUS)
    elif GENUS.upper() == "ALL":
        genera = sorted(p.parent.name for p in FEAT_DIR.glob("*/features.csv"))
    else:
        genera = [GENUS]
    frames = [pd.read_csv(FEAT_DIR / g / "features.csv").assign(genus=g)
              for g in genera if (FEAT_DIR / g / "features.csv").exists()]
    if not frames:
        raise SystemExit(f"No features.csv found in {FEAT_DIR}. Run genus_song_clusters.py first.")
    df = pd.concat(frames, ignore_index=True)
    cols = [c for c in df.columns if c not in META and (INCLUDE_MFCC or not c.startswith("mfcc"))]
    return df, cols


def recording_conditions(df):
    """Join per-recording metadata that describes the recording, not the bird."""
    if not XC_CSV.exists():
        return pd.DataFrame(index=df.index)
    x = pd.read_csv(XC_CSV, low_memory=False, usecols=lambda c: c in ("id", "q", "length", "smp", "also"))
    x["id"] = x["id"].astype(str)

    def secs(t):
        try:
            parts = [float(p) for p in str(t).split(":")]
            return sum(v * 60 ** i for i, v in enumerate(reversed(parts)))
        except ValueError:
            return np.nan

    cond = pd.DataFrame({"id": x["id"]})
    if "length" in x: cond["file_length_s"] = x["length"].map(secs)
    if "q" in x: cond["quality_B"] = (x["q"].astype(str).str.upper() == "B").astype(float)
    if "smp" in x: cond["sample_rate"] = pd.to_numeric(x["smp"], errors="coerce")
    if "also" in x:
        cond["n_background_spp"] = x["also"].fillna("").astype(str).map(
            lambda s: 0 if s.strip() in ("", "[]", "nan") else s.count(",") + 1)
    return df[["id"]].astype(str).merge(cond, on="id", how="left").drop(columns="id").set_index(df.index)


def eta2_ranks(values, groups):
    """Between-group share of variance, on ranks (robust to outliers)."""
    ok = values.notna()
    v, g = rankdata(values[ok]), groups[ok].values
    if len(np.unique(g)) < 2:
        return np.nan
    grand = v.mean()
    ss_tot = ((v - grand) ** 2).sum()
    ss_b = sum(len(v[g == k]) * (v[g == k].mean() - grand) ** 2 for k in np.unique(g))
    return ss_b / ss_tot if ss_tot > 0 else np.nan


def score(df, cols, cond):
    rows = []
    for c in cols:
        v = df[c].astype(float)
        med = v.median()
        mad = 1.4826 * (v - med).abs().median()
        r = dict(feature=c,
                 missing=v.isna().mean(),
                 degenerate=(v == 0).mean(),
                 outliers=((v - med).abs() > 5 * mad).mean() if mad > 0 else np.nan)

        # species signal: within each genus, species with enough recordings
        etas, w = [], []
        for _, d in df.groupby("genus"):
            keep = d.species.map(d.species.value_counts()) >= MIN_PER_GROUP
            d = d[keep]
            if d.species.nunique() >= 2:
                e = eta2_ranks(d[c].astype(float), d.species)
                if not np.isnan(e):
                    etas.append(e); w.append(len(d))
        r["species_signal"] = np.average(etas, weights=w) if etas else np.nan

        # site noise: spread among recordings of one species at one site / spread within the species
        ratios = []
        for _, d in df.groupby("species"):
            sd_sp = d[c].std()
            if not sd_sp or np.isnan(sd_sp):
                continue
            per_site = d.groupby("site")[c]
            sds = per_site.std()[per_site.count() >= 2].dropna()
            if len(sds):
                ratios.append(np.sqrt((sds ** 2).mean()) / sd_sp)
        r["site_noise"] = np.median(ratios) if ratios else np.nan

        # artefacts: correlation with recording conditions
        best, which = 0.0, ""
        for a in cond.columns:
            ok = v.notna() & cond[a].notna()
            if ok.sum() > 20 and cond[a][ok].nunique() > 1:
                rho = spearmanr(v[ok], cond[a][ok]).statistic
                if abs(rho) > abs(best):
                    best, which = rho, a
        r["artefact_rho"], r["artefact_with"] = best, which
        rows.append(r)

    q = pd.DataFrame(rows)
    q["flags"] = q.apply(lambda r: ", ".join(
        k for k, t in FLAG.items()
        if (k == "species_signal" and r[k] < t) or (k == "artefact_rho" and abs(r[k]) > t)
        or (k not in ("species_signal", "artefact_rho") and r[k] > t)), axis=1)
    q["n_flags"] = q["flags"].map(lambda s: 0 if not s else s.count(",") + 1)
    return q


def figures(q, df, cols):
    order = q.sort_values(["n_flags", "species_signal"], ascending=[True, False]).feature.tolist()
    q = q.set_index("feature").loc[order]
    checks = [("missing", "Missing (share)", False), ("degenerate", "Zero / not detected (share)", False),
              ("outliers", "Outliers > 5 MAD (share)", False), ("species_signal", "Species signal (eta², higher = better)", True),
              ("site_noise", "Same-site noise (lower = better)", False), ("artefact_rho", "Corr. with recording conditions", False)]
    fig, axes = plt.subplots(1, 6, figsize=(19, 0.28 * len(order) + 1.8), sharey=True)
    y = np.arange(len(order))
    for ax, (k, title, higher_good) in zip(axes, checks):
        vals = q[k].astype(float)
        t = FLAG[k]
        bad = (vals < t) if higher_good else (vals.abs() > t if k == "artefact_rho" else vals > t)
        ax.barh(y, vals, color=np.where(bad, "#e07b39", "#3a78c2"), height=0.7)
        for tt in ([t, -t] if k == "artefact_rho" else [t]):
            ax.axvline(tt, color="#555", lw=0.8, ls="--")
        if k == "artefact_rho":
            ax.axvline(0, color="#999", lw=0.6)
        ax.set_title(title, fontsize=9, loc="left")
        ax.tick_params(labelsize=7.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(y, order)
    axes[0].invert_yaxis()
    fig.suptitle(f"Song feature quality: {df.genus.nunique()} genera, {df.species.nunique()} species, "
                 f"{len(df)} recordings. Orange = past the dashed threshold. Sorted: fewest flags, "
                 f"then strongest species signal", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "feature_quality.png", dpi=170)
    plt.close(fig)

    C = df[order].astype(float).corr(method="spearman")
    fig, ax = plt.subplots(figsize=(0.32 * len(order) + 3, 0.32 * len(order) + 2))
    im = ax.imshow(C.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(order)), order, rotation=90, fontsize=7)
    ax.set_yticks(range(len(order)), order, fontsize=7)
    for i in range(len(order)):
        for j in range(len(order)):
            if i != j and abs(C.values[i, j]) >= 0.7:
                ax.text(j, i, f"{C.values[i, j]:.1f}", ha="center", va="center", fontsize=5.5,
                        color="white")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Spearman correlation")
    ax.set_title("Feature redundancy (|r| ≥ 0.7 labelled)", loc="left", fontsize=10)
    fig.savefig(OUT / "feature_correlation.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df, cols = load()
    cond = recording_conditions(df)
    q = score(df, cols, cond)
    q = q.sort_values(["n_flags", "species_signal"], ascending=[True, False])
    q.round(3).to_csv(OUT / "feature_quality.csv", index=False)
    figures(q, df, cols)
    pd.set_option("display.width", 200)
    print(q.round(2).drop(columns="n_flags").to_string(index=False))
    clean = q[q.n_flags == 0].feature.tolist()
    print(f"\nNo flags ({len(clean)}): {clean}")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
