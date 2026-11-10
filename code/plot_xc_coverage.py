"""
plot_xc_coverage.py  --  Thamno-2026

Plots Xeno-canto coverage for the top species in species_coverage_amazon.csv
(from xc_pull_metadata.py).

Figure 1 (top N species, shared rows):
  a  recordings -> songs -> usable songs (A/B quality, no playback), nested bars
  b  sites -> sites with a usable song, nested bars
  c  yield: % of recordings that are usable songs
Figure 2 (all species): usable songs vs sites with a usable song, top N labelled.

Outputs: Thamno-2026/output/xc_coverage/
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TOP_N = 20
NICE = {"songs_AB_no_playback": "usable songs (A/B quality, no playback)",
        "sites_with_good_song": "sites with a usable song"}
RANK_BY = "songs_AB_no_playback"          # or "sites_with_good_song"

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent if (_HERE.parent / "data").exists() else Path.home() / "Desktop" / "Thamno-2026"
CSV = ROOT / "data" / "xc_metadata" / "species_coverage_amazon.csv"
OUT = ROOT / "output" / "xc_coverage"

# One hue, light -> dark = all -> usable
LIGHT, MID, DARK = "#c6dbef", "#6baed6", "#08519c"
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e6"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False})


def style(ax):
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)


def main():
    df = pd.read_csv(CSV)
    df["yield_pct"] = 100 * df.songs_AB_no_playback / df.recordings
    top = df.sort_values(RANK_BY, ascending=False).head(TOP_N).iloc[::-1]   # best at top
    y = range(len(top))
    labels = [f"$\\it{{{s.replace(' ', '~')}}}$" for s in top.species]
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- Figure 1 --------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 0.32 * TOP_N + 1.6), sharey=True,
                             gridspec_kw={"width_ratios": [1.3, 1.1, 0.8], "wspace": 0.08})
    a, b, c = axes

    a.barh(y, top.recordings, color=LIGHT, height=0.78, label="All recordings")
    a.barh(y, top.songs, color=MID, height=0.56, label="Songs")
    a.barh(y, top.songs_AB_no_playback, color=DARK, height=0.34, label="Usable songs (A/B, no playback)")
    for i, v in zip(y, top.songs_AB_no_playback):
        a.text(v + 3, i, f"{v}", va="center", fontsize=7, color=INK)
    a.set_yticks(list(y), labels)
    a.set_xlabel("Recordings")
    a.set_title("a  Recordings", loc="left", fontsize=10, color=INK)
    a.legend(loc="lower right", fontsize=7.5, frameon=False)

    b.barh(y, top.sites, color=LIGHT, height=0.78, label="Sites (any recording)")
    b.barh(y, top.sites_with_good_song, color=DARK, height=0.34, label="Sites with a usable song")
    b.set_xlabel("Distinct sites (lat/lon to 0.01°)")
    b.set_title("b  Sites", loc="left", fontsize=10, color=INK)
    b.legend(loc="lower right", fontsize=7.5, frameon=False)

    c.hlines(y, 0, top.yield_pct, color=GRID, lw=1.5)
    c.plot(top.yield_pct, y, "o", color=DARK, ms=6)
    med = df.yield_pct.median()
    c.axvline(med, color=MUTED, lw=0.8, ls="--")
    c.text(med, len(top) - 0.3, f" median, all species ({med:.0f}%)", fontsize=7, color=MUTED, va="bottom")
    c.set_xlim(0, 100)
    c.set_xlabel("Usable songs / recordings (%)")
    c.set_title("c  Yield", loc="left", fontsize=10, color=INK)

    for ax in axes:
        style(ax)
    fig.suptitle(f"Xeno-canto coverage inside the Amazon window: top {TOP_N} antbird species "
                 f"by {NICE[RANK_BY]}", x=0.01, ha="left", fontsize=11, color=INK)
    fig.savefig(OUT / f"coverage_top{TOP_N}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 2 --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6.5))
    rest = df[~df.species.isin(top.species)]
    ax.scatter(rest.sites_with_good_song, rest.songs_AB_no_playback, s=14, color=LIGHT,
               edgecolor=MID, lw=0.5, label=f"Other species ({len(rest)})")
    ax.scatter(top.sites_with_good_song, top.songs_AB_no_playback, s=34, color=DARK,
               edgecolor="white", lw=1, label=f"Top {TOP_N}", zorder=3)
    # Labels: isolated points labelled in place; the crowded cluster goes to a
    # column of labels on the empty right side, in the same vertical order.
    ymax = df.songs_AB_no_playback.max()
    lone = top[(top.songs_AB_no_playback > 0.78 * ymax)]
    crowd = top.drop(lone.index).sort_values("songs_AB_no_playback")
    for _, r in lone.iterrows():
        ax.annotate(r.species, (r.sites_with_good_song, r.songs_AB_no_playback), xytext=(5, 0),
                    textcoords="offset points", fontsize=7, style="italic", color=INK, va="center")
    if len(crowd):
        col_x = df.sites_with_good_song.max() * 1.0
        ys = np.linspace(0.30 * ymax, 0.74 * ymax, len(crowd))
        for (_, r), ly in zip(crowd.iterrows(), ys):
            ax.annotate(r.species, (r.sites_with_good_song, r.songs_AB_no_playback),
                        xytext=(col_x, ly), textcoords="data", fontsize=7, style="italic",
                        color=INK, va="center",
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, alpha=0.7,
                                        shrinkA=0, shrinkB=3))
        ax.set_xlim(right=col_x * 1.55)
    lim = max(df.sites_with_good_song.max(), df.songs_AB_no_playback.max()) * 1.05
    ax.plot([0, lim], [0, lim], color=MUTED, lw=0.8, ls="--")
    ax.text(lim * 0.93, lim * 0.97, "1 usable\nsong per site", fontsize=7, color=MUTED, ha="right")
    ax.set_xlabel("Sites with a usable song")
    ax.set_ylabel("Usable songs (A/B quality, no playback)")
    ax.set_title("Spread vs depth: up and right = many songs from many places",
                 loc="left", fontsize=10, color=INK)
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.savefig(OUT / "songs_vs_sites_all_species.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
