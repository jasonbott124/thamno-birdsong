"""
xc_banks_map.py  --  Thamno-2026

Songs per species per riverbank, plus one map of the top species' sites.

1) FOCAL RIVERS. Each river in FOCAL_RIVERS is built from SWORD reaches in two
   ways, combined:
     - reaches carrying one of its SWORD names, kept only if connected to the
       river's seed point (drops mislabelled tributaries far away);
     - the main stem traced through the network from the seed point, down to
       where it joins a bigger river and up to its headwater (catches rivers
       SWORD leaves unnamed, e.g. Tapajós, Juruá, Japurá).
   Check rivers_traced.png; move a seed point or add a name if a river is off.
2) BANKS. Each recording within BAND_KM of a focal river gets a side, left or
   right bank looking downstream (also given as N/S/E/W), from the direction of
   the nearest reach. Recordings beyond either end, or with another focal river
   in between, are left out.
3) COUNTS of usable songs (QUALITIES, song, EXCLUDE_PLAYBACK) and sites per
   species per river per bank, and which species x rivers are testable.
4) ONE base map (relief + SWORD rivers) with the TOP_N_MAP species' sites;
   point area proportional to usable songs at the site.

Inputs (made by earlier scripts):
  output/amazon_maps/amazon_rivers.gpkg       (amazon_river_maps.py)
  output/amazon_maps/derived/dem_amazon_*.tif (amazon_river_maps.py)
  data/xc_metadata/thamnophilidae_amazon.csv  (xc_pull_metadata.py)
  data/xc_metadata/species_coverage_amazon.csv

Outputs: output/xc_banks/
  rivers_traced.png / .csv        check map and table of the focal rivers
  songs_per_species_per_bank.csv  species x river x bank
  testable_rivers_per_species.csv species x river where BOTH banks have data
  species_summary.csv             per species: number of testable rivers
  testable_heatmap_top20.png      top species x rivers
  top10_species_sites_map.png     the one combined map

Run from Thamno-2026/code:  python xc_banks_map.py
"""
from pathlib import Path
import glob
import time

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely.geometry import LineString, Point
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

# =============================================================================
# SETTINGS
# =============================================================================
_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent if (_HERE.parent / "data").exists() else Path.home() / "Desktop" / "Thamno-2026"
GPKG = ROOT / "output" / "amazon_maps" / "amazon_rivers.gpkg"
DEM_GLOB = str(ROOT / "output" / "amazon_maps" / "derived" / "dem_amazon_*.tif")
XC_CSV = ROOT / "data" / "xc_metadata" / "thamnophilidae_amazon.csv"
COVERAGE_CSV = ROOT / "data" / "xc_metadata" / "species_coverage_amazon.csv"
OUT = ROOT / "output" / "xc_banks"

AMAZON_BBOX = (-80.0, -20.0, -44.0, 6.5)
AEA = "+proj=aea +lat_1=-5 +lat_2=-42 +lat_0=-32 +lon_0=-60 +datum=WGS84 +units=m"

# Focal rivers: name -> (seed point(s) ON the river (lon, lat), [SWORD names]).
# Seed points are approximate; put them mid-river, away from confluences. A
# river can take several seeds, e.g. [(lon, lat), (lon, lat)], where braided
# stretches break the trace.
FOCAL_RIVERS = {
    "Amazon":     ((-55.50, -1.95), ["Amazon river", "Amazon River", "River Amazon"]),
    "Negro":      ([(-61.00, -2.60), (-62.93, -0.97), (-67.00, -0.15)],
                   ["Rio Negro", "Black River", "Negro"]),
    "Branco":     ((-60.70, 2.80),  ["Rio Branco"]),
    "Madeira":    ((-63.00, -7.50), ["Rio Madeira"]),
    "Purus":      ((-62.50, -5.00), ["Purus River", "Rio Purus", "Purus"]),
    "Juruá":      ((-66.90, -4.90), []),
    "Japurá":     ((-66.50, -1.70), []),
    "Putumayo/Içá": ((-70.00, -2.60), ["Putumayo", "Putumayo River"]),
    "Napo":       ((-73.70, -2.50), ["Rio Napo"]),
    "Javari":     ((-71.00, -5.00), ["Javary River"]),
    "Ucayali":    ((-74.50, -8.40), ["Ucayali River"]),
    "Marañón":    ((-75.50, -4.80), ["Maranon River"]),
    "Madre de Dios": ((-69.40, -12.55), ["Madre de Dios River", "Rio alto Madre de Dios"]),
    "Tapajós":    ((-55.30, -3.30), []),          # traces up the Teles Pires
    "Juruena":    ((-58.40, -9.50), ["Rio Juruena"]),
    "Xingu":      ((-52.20, -3.20), ["Rio Xingu"]),
    "Iriri":      ((-53.80, -5.00), ["Rio Iriri"]),
    "Tocantins":  ((-47.50, -7.30), ["Rio Tocantins", "Tocantins"]),
    "Trombetas":  ((-56.50, -1.30), ["Rio Trombetas"]),
}
SEED_MATCH_KM = 30        # seed must be this close to a major reach
LINK_KM = 40              # same-named reaches this close count as one river

FACC_MIN_KM2 = 30_000     # reaches draining at least this much count as "major"
JOIN_TOL_M = 1_500        # reach ends this close are treated as connected
BAND_KM = 100             # recordings this close to a river get a bank
END_BUFFER_KM = 5         # drop recordings that project onto a river's very ends

QUALITIES = ["A", "B"]    # Xeno-canto quality grades counted as usable
EXCLUDE_PLAYBACK = True   # drop songs recorded after playback
STRICT_SONG = True        # True: song only (no "call, song" mixes); False: any song tag
MIN_SONGS = 5             # a bank "has data" with at least this many usable songs...
MIN_SITES = 2             # ...from at least this many sites
TOP_N_MAP = 10
TOP_N_HEATMAP = 20

# =============================================================================


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def is_pure_song(types):
    """True when every comma-separated tag mentions song ("song", "dawn song"),
    so recordings that also hold calls are excluded."""
    t = types.fillna("").astype(str).str.lower().str.strip()
    return t.ne("") & t.str.split(",").map(lambda parts: all("song" in p for p in parts if p.strip()))


def usable_mask(df):
    t = df["type"].fillna("").str.lower()
    has_song = is_pure_song(df["type"]) if STRICT_SONG else t.str.contains("song")
    q = df["q"].fillna("")
    pb = (df["playback-used"].fillna("").str.lower().eq("yes")
          if ("playback-used" in df and EXCLUDE_PLAYBACK) else False)
    return has_song & q.isin(QUALITIES) & ~pb


# -----------------------------------------------------------------------------
# 1. Focal rivers
# -----------------------------------------------------------------------------
def trace_rivers():
    g = gpd.read_file(GPKG, layer="sword_reaches")
    g = g[(g.facc_km2 >= FACC_MIN_KM2) & g.geometry.notna()].to_crs(AEA).reset_index(drop=True)
    n = len(g)
    log(f"{n:,} SWORD reaches drain >= {FACC_MIN_KM2:,} km2")

    # connectivity from reach end points; direction from dist_out
    a = np.array([c.coords[0] for c in g.geometry])
    b = np.array([c.coords[-1] for c in g.geometry])
    tree = cKDTree(np.vstack([a, b]))
    nbrs = [set() for _ in range(n)]
    for i, j in tree.query_pairs(JOIN_TOL_M):
        if i % n != j % n:
            nbrs[i % n].add(j % n); nbrs[j % n].add(i % n)
    facc = g.facc_km2.values
    dout = np.nan_to_num(g.dist_out_m.values, nan=np.inf)
    down = np.full(n, -1)
    for i in range(n):
        lower = [j for j in nbrs[i] if dout[j] < dout[i]]
        if lower:
            down[i] = max(lower, key=lambda j: dout[j])
    ups = [[] for _ in range(n)]
    for i in range(n):
        if down[i] >= 0:
            ups[down[i]].append(i)
    # main stem upstream = branch with the longest path to a headwater (Hack);
    # SWORD's facc glitches on braided rivers, so it isn't used for this
    head_d = dout.copy()
    for i in np.argsort(-dout, kind="stable"):
        if down[i] >= 0:
            head_d[down[i]] = max(head_d[down[i]], head_d[i])
    main_up = lambda d: max(ups[d], key=lambda j: head_d[j])

    def oriented(i):        # coords upstream -> downstream
        c = np.asarray(g.geometry.iloc[i].coords)
        ref = down[i] if down[i] >= 0 else (ups[i][0] if ups[i] else -1)
        if ref < 0:
            return c
        t = np.asarray(g.geometry.iloc[ref].coords)
        d0 = np.min(np.hypot(*(t - c[0]).T)); d1 = np.min(np.hypot(*(t - c[-1]).T))
        if down[i] >= 0:
            return c if d1 <= d0 else c[::-1]
        return c if d0 <= d1 else c[::-1]
    g["coords"] = [oriented(i) for i in range(n)]
    lines = list(g.geometry.values)

    claimed = np.full(n, "", dtype=object)
    rows = []
    for name, (seeds_ll, sword_names) in FOCAL_RIVERS.items():
        if isinstance(seeds_ll[0], (int, float)):
            seeds_ll = [seeds_ll]
        members, offs = set(), []
        for lon, lat in seeds_ll:
            p = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(AEA).iloc[0]
            d = g.distance(p)
            seed = int(d.idxmin())
            if d[seed] > SEED_MATCH_KM * 1000:
                named_idx = g.index[g.river_name.isin(sword_names)]
                if len(named_idx):                  # fall back to nearest same-named reach
                    seed = int(d[named_idx].idxmin())
                    log(f"  {name}: seed ({lon}, {lat}) far from the network; using nearest "
                        f"'{g.river_name.iat[seed]}' reach ({d[seed]/1000:.0f} km away)")
                else:
                    log(f"  {name}: seed ({lon}, {lat}) is {d[seed]/1000:.0f} km from any major reach; ignored")
                    continue
            offs.append(d[seed] / 1000)
            # (a) traced main stem through the seed
            t = seed; members.add(t)
            while down[t] >= 0 and ups[down[t]] and main_up(down[t]) == t:
                t = down[t]; members.add(t)
            t = seed
            while ups[t]:
                t = main_up(t); members.add(t)
        if not members:
            continue
        # (b) same-named reaches connected (within LINK_KM) to the seed's cluster
        named = [i for i in range(n) if g.river_name.iat[i] in sword_names]
        if named:
            cluster = set(members)
            frontier = True
            while frontier:
                frontier = False
                cl = shapely.union_all([lines[i] for i in cluster])
                for i in named:
                    if i not in cluster and lines[i].distance(cl) < LINK_KM * 1000:
                        cluster.add(i); frontier = True
            members |= cluster
        members = [i for i in members if claimed[i] == ""]
        claimed[members] = name
        sub = g.loc[members]
        km = sub.length_m.sum() / 1000
        head_i = sub.index[np.argmax(sub.dist_out_m.values)]
        out_i = sub.index[np.argmin(sub.dist_out_m.values)]
        head, outlet = Point(g.at[head_i, "coords"][0]), Point(g.at[out_i, "coords"][-1])
        dx, dy = outlet.x - head.x, outlet.y - head.y
        if abs(dx) >= abs(dy):                      # flows E/W -> banks N/S (left, right)
            lb, rb_ = ("N", "S") if dx > 0 else ("S", "N")
        else:                                       # flows N/S -> banks E/W
            lb, rb_ = ("W", "E") if dy > 0 else ("E", "W")
        sw = sub.river_name[sub.river_name != ""].value_counts()
        log(f"  {name:13} {len(sub):4} reaches, {km:6.0f} km; seed(s) {', '.join(f'{o:.0f}' for o in offs)} km off; "
            f"SWORD names: {', '.join(f'{k} ({v})' for k, v in sw.head(3).items()) or '-'}")
        rows.append(dict(name=name, length_km=round(km), n_reaches=len(sub),
                         median_width_m=round(float(np.nanmedian(sub.width_m)), 0),
                         left_bank=lb, right_bank=rb_, head=head, outlet=outlet,
                         geometry=shapely.MultiLineString([LineString(c) for c in sub.coords])))
    rv = gpd.GeoDataFrame(rows, geometry="geometry", crs=AEA)
    g["river"] = claimed
    g = g[g.river != ""].copy()
    return rv, g


# -----------------------------------------------------------------------------
# 2. Banks for recordings
# -----------------------------------------------------------------------------
def assign_banks(rec, rv, g):
    pts = rec.to_crs(AEA)
    band, endb = BAND_KM * 1000, END_BUFFER_KM * 1000
    lines = [LineString(c) for c in g.coords]
    grp_of = g.river.values
    width = g.width_m.values
    tree = shapely.STRtree(lines)
    info = rv.set_index("name")
    out = []
    # nearest reach of EACH river within the band, per recording
    for name in rv.name:
        sel = np.where(grp_of == name)[0]
        sub = shapely.STRtree([lines[i] for i in sel])
        P = pts.geometry.values
        pi, li = sub.query(P, predicate="dwithin", distance=band)
        if not len(pi):
            continue
        cand = pd.DataFrame({"p": pi, "l": sel[li]})
        cand["d"] = shapely.distance(P[cand.p.values], np.array(lines, dtype=object)[cand.l.values])
        best = cand.sort_values("d").drop_duplicates("p")
        head, outlet = info.at[name, "head"], info.at[name, "outlet"]
        for p_i, l_i, d in best[["p", "l", "d"]].itertuples(index=False):
            p = P[p_i]; line = lines[l_i]
            s = line.project(p); q = line.interpolate(s)
            if q.distance(head) < endb or q.distance(outlet) < endb:
                continue                                    # beyond the river's ends
            seg = LineString([p, q])
            hits = tree.query(seg, predicate="intersects")
            if any(grp_of[h] != name for h in hits):
                continue                                    # another major river in between
            L = line.length
            t0 = line.interpolate(max(0, s - 1000)); t1 = line.interpolate(min(L, s + 1000))
            z = (t1.x - t0.x) * (p.y - q.y) - (t1.y - t0.y) * (p.x - q.x)
            side = "left" if z > 0 else "right"
            out.append(dict(rec_index=pts.index[p_i], river=name, side=side,
                            bank=info.at[name, "left_bank"] if side == "left" else info.at[name, "right_bank"],
                            dist_km=round(d / 1000, 1), width_m=width[l_i]))
    return pd.DataFrame(out)


# -----------------------------------------------------------------------------
# 3. Tables
# -----------------------------------------------------------------------------
def tables(rec, banks, rv):
    b = banks.merge(rec[["species", "site", "usable"]], left_on="rec_index", right_index=True)
    per = (b.groupby(["species", "river", "side", "bank"])
             .agg(recordings=("rec_index", "size"), usable_songs=("usable", "sum"),
                  sites_with_usable=("site", lambda s: s[b.loc[s.index, "usable"]].nunique()),
                  median_dist_km=("dist_km", "median"),
                  width_near_recordings_m=("width_m", "median"))
             .reset_index())
    per.to_csv(OUT / "songs_per_species_per_bank.csv", index=False)

    ok = per[(per.usable_songs >= MIN_SONGS) & (per.sites_with_usable >= MIN_SITES)]
    w = ok.pivot_table(index=["species", "river"], columns="side",
                       values=["usable_songs", "sites_with_usable"], aggfunc="sum").dropna()
    w.columns = [f"{a}_{c}" for a, c in w.columns]
    w = w.reset_index()
    if len(w):
        w = w.merge(rv[["name", "median_width_m", "left_bank", "right_bank"]],
                    left_on="river", right_on="name").drop(columns="name")
        w["min_bank_songs"] = w[["usable_songs_left", "usable_songs_right"]].min(axis=1)
        w = w.sort_values(["species", "min_bank_songs"], ascending=[True, False])
    w.to_csv(OUT / "testable_rivers_per_species.csv", index=False)

    summ = (w.groupby("species").agg(testable_rivers=("river", "nunique"),
                                     rivers=("river", lambda s: "; ".join(s)),
                                     songs_on_weaker_banks=("min_bank_songs", "sum"))
              .sort_values(["testable_rivers", "songs_on_weaker_banks"], ascending=False)
            if len(w) else pd.DataFrame(columns=["testable_rivers"]))
    summ.to_csv(OUT / "species_summary.csv")
    log(f"  {len(per):,} species x river x bank rows; {len(w)} testable species x river pairs; "
        f"{len(summ)} species testable on >= 1 river")
    return per, w, summ


# -----------------------------------------------------------------------------
# 4. Figures
# -----------------------------------------------------------------------------
def base_map(ax, sword4326, rb=AMAZON_BBOX):
    """Relief + rivers, drawn once per axes from cached inputs."""
    import amazon_river_maps as arm
    dems = sorted(glob.glob(DEM_GLOB))
    if dems:
        import rasterio
        with rasterio.open(dems[0]) as s:
            dem = s.read(1, masked=True).filled(np.nan).astype("float32")
            ext = (s.bounds.left, s.bounds.right, s.bounds.bottom, s.bounds.top)
        rgb = arm.dem_rgb(dem, ext)
        # lighten so coloured points stand out
        rgb = 0.55 * rgb + 0.45
        ax.imshow(rgb, extent=ext, origin="upper", interpolation="bilinear", zorder=0)
    sw = sword4326[~sword4326.is_lake_or_ghost]
    sw = sw.assign(width_class=sw.width_class.astype(int))
    arm.draw_rivers_by_class(ax, sw, lw_scale=0.8)
    arm.setup_axes(ax, rb)


def check_map(rv, sword4326):
    fig, ax = plt.subplots(figsize=(11, 8.6))
    base_map(ax, sword4326)
    r4 = rv.to_crs(4326)
    cols = plt.cm.tab20(np.arange(len(r4)) % 20)
    for (_, r), c in zip(r4.iterrows(), cols):
        for part in r.geometry.geoms:
            x, y = part.xy
            ax.plot(x, y, color=c, lw=2.2, zorder=8, solid_capstyle="round")
        longest = max(r.geometry.geoms, key=lambda gg: gg.length)
        m = longest.interpolate(0.5, normalized=True)
        ax.text(m.x, m.y, r["name"], fontsize=6.5, zorder=9, ha="center",
                path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    ax.set_title("Focal rivers used for banks. Check each; move its seed point or add a "
                 "SWORD name in FOCAL_RIVERS if one is wrong.", loc="left", fontsize=10)
    fig.savefig(OUT / "rivers_traced.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def species_map(rec, top, sword4326):
    fig, ax = plt.subplots(figsize=(12, 9.4))
    base_map(ax, sword4326)
    pal = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
           "#17becf", "#8c564b", "#e377c2", "#bcbd22", "#4d4d4d"]
    u = rec[rec.usable & rec.species.isin(top)]
    site = (u.groupby(["species", "site"]).agg(lat=("lat", "mean"), lon=("lon", "mean"),
                                                songs=("usable", "sum")).reset_index())
    k = 14                                     # marker area per song
    order = site.groupby("species").size().sort_values(ascending=False).index   # dense first
    handles = []
    for sp in top:
        c = pal[list(top).index(sp) % len(pal)]
        handles.append(Line2D([], [], ls="", marker="o", ms=7, mfc=c, mec="white",
                              label=f"{sp} ({int(site[site.species == sp].songs.sum())})"))
    for sp in order:
        c = pal[list(top).index(sp) % len(pal)]
        s = site[site.species == sp].sort_values("songs", ascending=False)
        ax.scatter(s.lon, s.lat, s=k * s.songs, color=c, alpha=0.75, edgecolor="white",
                   lw=0.6, zorder=10)
    leg1 = ax.legend(handles=handles, title="Species (usable songs)", loc="lower left",
                     fontsize=7.5, title_fontsize=8, framealpha=0.92, borderpad=0.8)
    leg1.set_zorder(30)
    ax.add_artist(leg1)
    sizes = [1, 5, 20]
    sh = [plt.scatter([], [], s=k * v, color="#9e9e9e", edgecolor="white") for v in sizes]
    leg2 = ax.legend(sh, [f"{v} song{'s' if v > 1 else ''}" for v in sizes], title="At one site",
                     loc="lower right", fontsize=7.5, title_fontsize=8, framealpha=0.92,
                     labelspacing=1.4, borderpad=1.0)
    leg2.set_zorder(30)
    ax.set_title(f"Top {len(top)} antbird species by usable Xeno-canto songs "
                 f"(quality {'/'.join(QUALITIES)}, song{', no playback' if EXCLUDE_PLAYBACK else ''}); "
                 "point area = songs per site",
                 loc="left", fontsize=10)
    fig.savefig(OUT / "top10_species_sites_map.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def heatmap(per, w, top20, rv):
    if not len(w):
        return
    rivers = (w[w.species.isin(top20)].groupby("river").min_bank_songs.sum()
                .sort_values(ascending=False).index.tolist())
    if not rivers:
        return
    m = (w.pivot_table(index="species", columns="river", values="min_bank_songs")
           .reindex(index=top20, columns=rivers))
    fig, ax = plt.subplots(figsize=(0.55 * len(rivers) + 4, 0.34 * len(top20) + 1.8))
    im = ax.imshow(m.values, cmap="Blues", aspect="auto", vmin=0)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, int(v), ha="center", va="center", fontsize=6.5,
                        color="white" if v > np.nanmax(m.values) * 0.55 else "#222")
    ax.set_xticks(range(len(rivers)), rivers, rotation=50, ha="right", fontsize=8)
    ax.set_yticks(range(len(top20)), top20, fontsize=8, style="italic")
    ax.set_xticks(np.arange(-.5, len(rivers)), minor=True)
    ax.set_yticks(np.arange(-.5, len(top20)), minor=True)
    ax.grid(which="minor", color="white", lw=1.5); ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Usable songs on the weaker bank", fontsize=8)
    ax.set_title(f"Testable rivers: both banks with >= {MIN_SONGS} usable songs from >= {MIN_SITES} "
                 f"sites within {BAND_KM} km (blank = not testable)", loc="left", fontsize=10)
    fig.savefig(OUT / "testable_heatmap_top20.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rv, reaches = trace_rivers()
    rv.drop(columns=["geometry", "head", "outlet"]).to_csv(OUT / "rivers_traced.csv", index=False)

    rec = pd.read_csv(XC_CSV, low_memory=False)
    rec = rec[rec.lat.notna() & rec.lon.notna()].copy()
    rec["species"] = (rec.gen.fillna("") + " " + rec.sp.fillna("")).str.strip()
    rec["usable"] = usable_mask(rec)
    rec["site"] = rec.lat.round(2).astype(str) + "," + rec.lon.round(2).astype(str)
    rec = gpd.GeoDataFrame(rec, geometry=gpd.points_from_xy(rec.lon, rec.lat), crs=4326)
    log(f"{len(rec):,} recordings with coordinates, {int(rec.usable.sum()):,} usable songs")

    log("Assigning banks")
    banks = assign_banks(rec, rv, reaches)
    per, w, summ = tables(rec, banks, rv)

    cov = pd.read_csv(COVERAGE_CSV)
    top10 = cov.sort_values("songs_AB_no_playback", ascending=False).species.head(TOP_N_MAP).tolist()
    top20 = cov.sort_values("songs_AB_no_playback", ascending=False).species.head(TOP_N_HEATMAP).tolist()

    log("Drawing figures")
    sword = gpd.read_file(GPKG, layer="sword_reaches")
    check_map(rv, sword)
    species_map(rec, top10, sword)
    heatmap(per, w, top20, rv)
    log(f"Done. Outputs in {OUT}")
    if len(summ):
        print(summ.head(15).to_string())


if __name__ == "__main__":
    main()
