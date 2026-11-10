"""
plot_archives_map.py  --  Thamno-2026

One river map (relief + SWORD rivers, drawn once) with every antbird
recording that has coordinates, by archive: Xeno-canto, Macaulay Library,
FNJV. Marker area is proportional to recordings at the site; recordings on
the Macaulay request list (combined_banks.py) are ringed.

Inputs: data/xc_metadata/thamnophilidae_amazon.csv
        data/other_metadata/other_archives_unified.csv
        output/combined_banks/request_macaulay.csv (optional)
        + base-map inputs of xc_banks_map.py
Output: output/combined_banks/recordings_by_archive_map.png

Run from Thamno-2026/code:  python plot_archives_map.py
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import xc_banks_map as xb
import combined_banks as cb

OUT = cb.OUT
SONGS_ONLY = True          # True: only recordings tagged purely as song
K = 3.5                    # marker area per recording

STYLE = {   # archive: (label, colour, marker)
    "xeno-canto": ("Xeno-canto", "#3a78c2", "o"),
    "macaulay":   ("Macaulay Library", "#e07b39", "s"),
    "fnjv":       ("FNJV (Unicamp)", "#222222", "^"),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    xc = cb.load_xc()
    other, _ = cb.harmonize_names(cb.load_other(), set(xc.species))
    rec = pd.concat([xc, other], ignore_index=True)
    w, s, e, n = xb.AMAZON_BBOX
    rec = rec[rec.lat.between(s, n) & rec.lon.between(w, e)]
    if SONGS_ONLY:
        rec = rec[xb.is_pure_song(rec.sound_type)]
    rec = rec.assign(site=rec.lat.round(2).astype(str) + "," + rec.lon.round(2).astype(str))
    site = (rec.groupby(["source", "site"])
               .agg(lat=("lat", "mean"), lon=("lon", "mean"), n=("catalog", "size"))
               .reset_index())

    sword = gpd.read_file(xb.GPKG, layer="sword_reaches")
    fig, ax = plt.subplots(figsize=(12, 9.4))
    xb.base_map(ax, sword)
    handles = []
    for src in ("xeno-canto", "macaulay", "fnjv"):        # densest first, sparsest on top
        lab, c, m = STYLE[src]
        d = site[site.source == src].sort_values("n", ascending=False)
        ax.scatter(d.lon, d.lat, s=K * d.n, c=c, marker=m, alpha=0.7,
                   edgecolor="white", lw=0.5, zorder=10)
        handles.append(Line2D([], [], ls="", marker=m, ms=8, mfc=c, mec="white",
                              label=f"{lab}: {int(d.n.sum()):,} recordings, {len(d):,} sites"))

    req_f = OUT / "request_macaulay.csv"
    if req_f.exists():
        req = pd.read_csv(req_f)
        if len(req):
            rs = req.groupby([req.lat.round(2), req.lon.round(2)]).size().reset_index(name="n")
            ax.scatter(rs.lon, rs.lat, s=K * rs.n + 90, facecolor="none", edgecolor="#b2182b",
                       lw=1.4, zorder=12)
            handles.append(Line2D([], [], ls="", marker="o", ms=11, mfc="none", mec="#b2182b",
                                  mew=1.4, label=f"On the Macaulay request list ({len(req)} recordings)"))

    leg = ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.92,
                    title="Archive", title_fontsize=8.5, borderpad=0.8)
    leg.set_zorder(30); ax.add_artist(leg)
    sizes = [5, 25, 100]
    sh = [plt.scatter([], [], s=K * v, color="#9e9e9e", edgecolor="white") for v in sizes]
    leg2 = ax.legend(sh, [f"{v} recording{'s' if v > 1 else ''}" for v in sizes], title="At one site",
                     loc="lower right", fontsize=8, title_fontsize=8.5, framealpha=0.92,
                     labelspacing=1.6, borderpad=1.1)
    leg2.set_zorder(30)
    what = "song recordings" if SONGS_ONLY else "recordings"
    ax.set_title(f"Antbird (Thamnophilidae) {what} with coordinates, by archive; "
                 "marker area = recordings per site", loc="left", fontsize=10)
    fig.savefig(OUT / "recordings_by_archive_map.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", OUT / "recordings_by_archive_map.png")


if __name__ == "__main__":
    main()
