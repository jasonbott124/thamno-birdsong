"""
amazon_river_maps.py  --  Thamno-2026

Maps every river in the Amazon from the datasets in Thamno-2026/data/geodata,
in the same spirit as the Accreted Ice radar maps: one overview, a set of
regional close-ups, a GeoPackage for QGIS and summary tables.

Layers
  * SWORD v17c (South America): river reaches drawn from the 200 m nodes,
    coloured by channel width and, on a second map, by number of channels.
  * GRWL summary stats: river segments with mean width (QGIS export, optional map layer).
  * GRWL water mask (30 m tiles): drawn on the regional maps only.
  * HydroSHEDS 3 arc-second conditioned DEM: shaded relief background.
    The DEM has no overviews, so the first run builds a reduced copy of the
    Amazon window and caches it (a few minutes, once).

Outputs (OUT_DIR)
  maps/amazon_overview_width.png/.pdf     rivers by width class
  maps/amazon_overview_channels.png/.pdf  rivers by number of channels (braiding)
  maps/regions/<region>.png               regional close-ups
  amazon_rivers.gpkg                      QGIS layers (EPSG:4326)
  tables/river_summary.csv                one row per named river
  tables/region_width_summary.csv         km of river per width class per region
  derived/dem_amazon_*.tif                cached reduced DEM

Run:  conda activate antbird
      python amazon_river_maps.py
Needs: numpy pandas geopandas pyogrio shapely rasterio netCDF4 matplotlib
"""

from pathlib import Path
import glob
import time
import warnings

import numpy as np
import pandas as pd

# =============================================================================
# SETTINGS
# =============================================================================
# Project folder. Uses the folder above this script when the script lives in
# Thamno-2026\code; otherwise falls back to the Desktop copy. Edit if you move it.
_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent if (_HERE.parent / "data" / "geodata").exists() \
    else Path.home() / "Desktop" / "Thamno-2026"
GEO = ROOT / "data" / "geodata"

SWORD_VERSION = "v17c"
SWORD_NC      = GEO / "netcdf" / f"sa_sword_{SWORD_VERSION}.nc"
GRWL_SUMMARY  = GEO / "GRWL_summaryStats" / "GRWL_summaryStats.shp"
GRWL_MASK_DIR = GEO / "GRWL_mask_V01.01"
DEM_TIF       = GEO / "sa_con_3s" / "sa_con_3s.tif"
# Optional country outlines for the inset and borders. Any Natural Earth
# admin-0 countries shapefile works; None or a missing file just skips it.
COUNTRIES_SHP = None

OUT_DIR = ROOT / "output" / "amazon_maps"

# Map window: (west, south, east, north), degrees
AMAZON_BBOX = (-80.0, -20.0, -44.0, 6.5)

# Width classes (m). Musher et al. 2022 call >1000 m "large" rivers and
# 100-1000 m "smaller" rivers, so 100 and 1000 m are class edges here.
WIDTH_EDGES  = [0, 100, 500, 1000, 3000, np.inf]
WIDTH_LABELS = ["< 100 m", "100–500 m", "500 m – 1 km", "1–3 km", "> 3 km"]
WIDTH_COLORS = ["#c6dbef", "#6baed6", "#2171b5", "#08519c", "#08306b"]
WIDTH_LW     = [0.25, 0.5, 0.9, 1.4, 2.0]           # line widths (overview)

# Regional close-ups. Rough boxes around major interfluves / river systems;
# edit freely. (W, S, E, N)
REGIONS = {
    "Negro_Branco":          (-68.5, -4.0, -58.0,  5.0),
    "Napo_Maranon_UpperAmazon": (-79.0, -7.0, -69.0,  1.5),
    "Solimoes_Japura_Jurua": (-73.0, -9.0, -62.0,  0.5),
    "Madeira_Purus":         (-69.0, -14.0, -58.0, -3.0),
    "Tapajos_TelesPires":    (-60.5, -14.5, -53.0, -2.0),
    "Xingu_Tocantins":       (-56.0, -14.0, -46.0, -1.0),
    "Lower_Amazon_Belem":    (-58.0, -4.5, -47.0,  1.5),
}

# Toggles
SHOW_DEM              = True
SHOW_GRWL_MASK        = True    # regional maps only (30 m tiles)
SHOW_GRWL_SUMMARY     = False   # GRWL segments under SWORD (mostly duplicates it)
SKIP_LAKES            = True    # drop SWORD lake-on-river / ghost reaches from maps
MAKE_OVERVIEW         = True
MAKE_CHANNEL_MAP      = True
MAKE_REGIONAL_MAPS    = True
EXPORT_GPKG           = True
EXPORT_TABLES         = True
LABEL_TOP_N_RIVERS    = 30      # named rivers labelled on the overview
MIN_LABEL_WIDTH_M     = 150     # only label rivers this wide somewhere

DEM_DECIMATE          = 10      # 3" x 10 = 30" (~0.9 km) cached DEM
DEM_MAX_ELEV          = 1500    # colour stretch top (m); Andes saturate
REGION_RASTER_PX      = 1600    # GRWL mask resolution across a regional map
DPI                   = 250

# =============================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.colors import LightSource, LinearSegmentedColormap

import geopandas as gpd
import shapely
from shapely.geometry import box


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def in_bbox(x, y, bb):
    w, s, e, n = bb
    return (x >= w) & (x <= e) & (y >= s) & (y <= n)


def as_str_array(var):
    """netCDF string variable (vlen str or char array) -> numpy array of str."""
    import netCDF4
    a = var[:]
    if getattr(a, "dtype", None) is not None and a.dtype.kind == "S" and a.ndim == 2:
        a = netCDF4.chartostring(a)
    a = np.asarray(a, dtype=object)
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in a.ravel()])


def clean_num(a, fill_below=0):
    a = np.ma.filled(np.asarray(a, dtype="float64") if not np.ma.isMaskedArray(a)
                     else a.astype("float64"), np.nan)
    a[a <= fill_below] = np.nan          # SWORD uses -9999 / 1 as fill for width
    return a


def width_class(w):
    idx = np.digitize(np.nan_to_num(w, nan=-1), WIDTH_EDGES[1:-1])
    idx[np.isnan(w)] = 0
    return idx


# -----------------------------------------------------------------------------
# SWORD
# -----------------------------------------------------------------------------
def load_sword(bb):
    import netCDF4
    log(f"Reading SWORD reaches and nodes: {SWORD_NC.name}")
    with netCDF4.Dataset(SWORD_NC) as ds:
        R, N = ds["reaches"], ds["nodes"]
        log(f"  reach variables: {', '.join(R.variables)}")
        nr = R.dimensions[list(R.dimensions)[0]].size

        def num(name, fill_below=None, as_int=False, default=np.nan):
            """Read a reach variable; missing -> filled with default (warns once)."""
            if name not in R.variables:
                log(f"  note: '{name}' not in this SWORD file; using {default}")
                return np.full(nr, default, dtype="int64" if as_int else "float64")
            if as_int:
                return np.ma.filled(R[name][:], default).astype("int64")
            return clean_num(R[name][:], fill_below=fill_below if fill_below is not None else -1e30)

        reach_id = np.asarray(R["reach_id"][:]).astype("int64")
        r = pd.DataFrame({
            "reach_id":   reach_id,
            "x":          np.asarray(R["x"][:], dtype="float64"),
            "y":          np.asarray(R["y"][:], dtype="float64"),
            "width_m":    num("width", fill_below=1),
            "max_width_m": num("max_width", fill_below=1),
            "n_chan_max": num("n_chan_max", as_int=True, default=1),
            "facc_km2":   num("facc", fill_below=0),
            "length_m":   num("reach_length", fill_below=0),
            "dist_out_m": num("dist_out", fill_below=-1),
            # SWORD reach type = last digit of reach_id
            # (1 river, 3 lake on river, 4 dam/waterfall, 5 unreliable, 6 ghost)
            "type":       (np.asarray(R["type"][:]).astype("int64") if "type" in R.variables
                           else reach_id % 10),
            "lakeflag":   num("lakeflag", as_int=True, default=0),
            "river_name": (as_str_array(R["river_name"]) if "river_name" in R.variables
                           else np.full(nr, "", dtype=object)),
        })
        r["n_chan_max"] = r.n_chan_max.clip(lower=1)
        if r.length_m.isna().all():          # fall back to SWORD's nominal ~10 km reaches
            r["length_m"] = 10_000.0
        nx = np.asarray(N["x"][:], dtype="float64")
        ny = np.asarray(N["y"][:], dtype="float64")
        keep = in_bbox(nx, ny, bb)
        nodes = pd.DataFrame({
            "x": nx[keep], "y": ny[keep],
            "reach_id": np.asarray(N["reach_id"][:])[keep].astype("int64"),
            "node_id":  np.asarray(N["node_id"][:])[keep].astype("int64"),
        })
    r = r[in_bbox(r.x.values, r.y.values, bb)].copy()
    log(f"  {len(r):,} reaches and {len(nodes):,} nodes inside the Amazon window")

    # Reach polylines from nodes, in node order
    nodes = nodes[nodes.reach_id.isin(r.reach_id)].sort_values(["reach_id", "node_id"])
    counts = nodes.groupby("reach_id").size()
    nodes = nodes[nodes.reach_id.isin(counts.index[counts >= 2])]
    ids, start = np.unique(nodes.reach_id.values, return_index=True)
    idx = np.repeat(np.arange(len(ids)), np.diff(np.append(start, len(nodes))))
    lines = shapely.linestrings(nodes[["x", "y"]].values, indices=idx)
    geom = pd.Series(lines, index=ids)

    r = r[r.reach_id.isin(ids)].copy()
    r["geometry"] = geom.loc[r.reach_id].values
    r["river_name"] = r.river_name.str.split(";").str[0].str.strip()
    r.loc[r.river_name.str.upper().isin(["NODATA", "", "NAN", "NONE"]), "river_name"] = ""
    r["width_class"] = width_class(r.width_m.values)
    r["width_label"] = np.array(WIDTH_LABELS)[r.width_class.values]
    r["is_lake_or_ghost"] = r.type.isin([3, 6]) | (r.lakeflag == 1)
    gdf = gpd.GeoDataFrame(r.drop(columns=["x", "y"]), geometry="geometry", crs="EPSG:4326")
    gdf["region"] = ""
    for name, rb in REGIONS.items():
        hit = gdf.intersects(box(*rb))
        gdf.loc[hit, "region"] = np.where(gdf.loc[hit, "region"] == "", name,
                                          gdf.loc[hit, "region"] + ";" + name)
    return gdf


# -----------------------------------------------------------------------------
# GRWL
# -----------------------------------------------------------------------------
def load_grwl_summary(bb):
    log("Reading GRWL summary segments")
    g = gpd.read_file(GRWL_SUMMARY, bbox=bb)
    g = g.rename(columns={"width_mean": "width_mean_m", "width_med_": "width_med_m",
                          "width_max_": "width_max_m", "width_min_": "width_min_m",
                          "width_sd_m": "width_sd_m"})
    if g.crs is None:
        g = g.set_crs("EPSG:4326")
    g = g.to_crs("EPSG:4326")
    log(f"  {len(g):,} GRWL segments")
    return g


_GRWL_INDEX = None
def grwl_tile_index():
    """[(path, (w,s,e,n))] for every mask tile, bounds in lon/lat."""
    global _GRWL_INDEX
    if _GRWL_INDEX is None:
        import rasterio
        from rasterio.warp import transform_bounds
        _GRWL_INDEX = []
        for p in sorted(glob.glob(str(GRWL_MASK_DIR / "*.tif"))):
            with rasterio.open(p) as src:
                if src.crs is None:
                    continue
                _GRWL_INDEX.append((p, transform_bounds(src.crs, "EPSG:4326", *src.bounds)))
    return _GRWL_INDEX


def grwl_mask_mosaic(rb, npx):
    """GRWL water mask resampled onto a lon/lat grid covering rb.
    Returns (array: 0 none, 1 river/tidal/canal, 2 lake, extent) or None."""
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_bounds
    w, s, e, n = rb
    nx = npx
    ny = max(1, int(round(npx * (n - s) / (e - w))))
    dst_t = from_bounds(w, s, e, n, nx, ny)
    river = np.zeros((ny, nx), "uint8")
    lake = np.zeros((ny, nx), "uint8")
    tiles = [p for p, (tw, ts, te, tn) in grwl_tile_index()
             if tw < e and te > w and ts < n and tn > s]
    if not tiles:
        return None
    dst_res_m = (e - w) / nx * 111_000
    for p in tiles:
        with rasterio.open(p) as src:
            f = max(1, int(dst_res_m / abs(src.res[0]) / 2))     # use .ovr levels
            h, wd = max(1, src.height // f), max(1, src.width // f)
            a = src.read(1, out_shape=(h, wd), resampling=Resampling.nearest)
            t = src.transform * src.transform.scale(src.width / wd, src.height / h)
            for val_set, target in (((255, 126, 86), river), ((180,), lake)):
                m = np.isin(a, val_set).astype("uint8")
                if not m.any():
                    continue
                tmp = np.zeros_like(target)
                reproject(m, tmp, src_transform=t, src_crs=src.crs,
                          dst_transform=dst_t, dst_crs="EPSG:4326",
                          resampling=Resampling.max)
                target |= tmp
    out = np.where(river > 0, 1, np.where(lake > 0, 2, 0)).astype("uint8")
    return out, (w, e, s, n)


# -----------------------------------------------------------------------------
# DEM
# -----------------------------------------------------------------------------
def cached_dem(bb):
    import rasterio
    from rasterio.windows import from_bounds as win_from_bounds
    from rasterio.enums import Resampling
    OUT_DERIVED = OUT_DIR / "derived"
    OUT_DERIVED.mkdir(parents=True, exist_ok=True)
    tag = "_".join(f"{v:g}" for v in bb)
    cache = OUT_DERIVED / f"dem_amazon_{tag}_x{DEM_DECIMATE}.tif"
    if cache.exists():
        with rasterio.open(cache) as c:
            return c.read(1, masked=True).filled(np.nan).astype("float32"), \
                   (c.bounds.left, c.bounds.right, c.bounds.bottom, c.bounds.top)

    log(f"Building reduced DEM (one-time; reads the full-resolution window, "
        f"expect a few minutes): {cache.name}")
    with rasterio.open(DEM_TIF) as src:
        win = win_from_bounds(*bb, transform=src.transform).round_offsets().round_lengths()
        win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        f = DEM_DECIMATE
        H, W = int(win.height) // f, int(win.width) // f
        out = np.full((H, W), np.nan, "float32")
        strip = 200                                     # output rows per strip
        nodata = src.nodata
        for r0 in range(0, H, strip):
            rows = min(strip, H - r0)
            sw = rasterio.windows.Window(win.col_off, win.row_off + r0 * f, W * f, rows * f)
            a = src.read(1, window=sw, out_shape=(rows, W), resampling=Resampling.average,
                         masked=True)
            a = a.astype("float32").filled(np.nan)
            if nodata is not None:
                a[a == nodata] = np.nan
            out[r0:r0 + rows] = a
            log(f"  DEM rows {r0 + rows}/{H}")
        t = src.window_transform(rasterio.windows.Window(win.col_off, win.row_off, W * f, H * f))
        t = t * t.scale(f, f)
        prof = dict(driver="GTiff", height=H, width=W, count=1, dtype="float32",
                    crs=src.crs or "EPSG:4326", transform=t, nodata=np.nan,
                    compress="deflate", tiled=True)
    with rasterio.open(cache, "w", **prof) as dst:
        dst.write(out, 1)
    left, top = t.c, t.f
    return out, (left, left + W * t.a, top + H * t.e, top)


def dem_rgb(dem, extent):
    """Subdued shaded relief: terrain tint x hillshade."""
    lat_mid = np.deg2rad(0.5 * (extent[2] + extent[3]))
    dx = (extent[1] - extent[0]) / dem.shape[1] * 111_320 * np.cos(lat_mid)
    dy = (extent[3] - extent[2]) / dem.shape[0] * 110_540
    z = np.nan_to_num(dem, nan=0.0)
    ls = LightSource(azdeg=315, altdeg=40)
    hs = ls.hillshade(z, vert_exag=6, dx=dx, dy=dy)
    cmap = LinearSegmentedColormap.from_list(
        "lowland", ["#eef3e6", "#dfe8cf", "#e8e0c8", "#d9c9a8", "#c9b69a", "#f4f1ea"])
    tint = cmap(np.clip(np.sqrt(np.clip(z, 0, None) / DEM_MAX_ELEV), 0, 1))[..., :3]
    rgb = tint * (0.55 + 0.45 * hs[..., None])
    sea = np.isnan(dem)
    rgb[sea] = (0.93, 0.95, 0.97)
    return rgb


def crop(rgb, extent, rb):
    """Crop a lon/lat RGB grid (extent = l, r, b, t) to bbox rb."""
    l, r, b, t = extent
    H, W = rgb.shape[:2]
    c0 = int(np.clip((rb[0] - l) / (r - l) * W, 0, W)); c1 = int(np.clip((rb[2] - l) / (r - l) * W, 0, W))
    r0 = int(np.clip((t - rb[3]) / (t - b) * H, 0, H)); r1 = int(np.clip((t - rb[1]) / (t - b) * H, 0, H))
    if c1 <= c0 or r1 <= r0:
        return None, None
    ext = (l + c0 / W * (r - l), l + c1 / W * (r - l), t - r1 / H * (t - b), t - r0 / H * (t - b))
    return rgb[r0:r1, c0:c1], ext


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------
def segments(geoms):
    return [np.asarray(g.coords) for g in geoms if g is not None and not g.is_empty]


def draw_rivers_by_class(ax, gdf, lw_scale=1.0):
    for k in range(len(WIDTH_LABELS)):
        sub = gdf[gdf.width_class == k]
        if len(sub):
            ax.add_collection(LineCollection(segments(sub.geometry), colors=WIDTH_COLORS[k],
                                             linewidths=WIDTH_LW[k] * lw_scale,
                                             capstyle="round", zorder=3 + k))


CHAN_COLORS = {1: "#2171b5", 2: "#41ab5d", 3: "#fd8d3c", 4: "#d7301f"}
CHAN_LABELS = {1: "1 channel", 2: "2", 3: "3", 4: "4+ (braided / anabranching)"}

def draw_rivers_by_channels(ax, gdf, lw_scale=1.0):
    nc = gdf.n_chan_max.clip(lower=1, upper=4)
    for k in (1, 2, 3, 4):
        sub = gdf[nc == k]
        if len(sub):
            lw = np.array(WIDTH_LW)[sub.width_class.values] * lw_scale
            ax.add_collection(LineCollection(segments(sub.geometry), colors=CHAN_COLORS[k],
                                             linewidths=lw, capstyle="round", zorder=3 + k))


def label_rivers(ax, gdf, rb, top_n, min_sep_frac=0.08, fontsize=6.5):
    named = gdf[(gdf.river_name != "") & gdf.width_m.notna()]
    if named.empty:
        return
    stats = named.groupby("river_name").agg(length=("length_m", "sum"),
                                             maxw=("width_m", "max"))
    stats = stats[stats.maxw >= MIN_LABEL_WIDTH_M]
    stats["score"] = stats.length * np.sqrt(stats.maxw)
    min_sep = min_sep_frac * max(rb[2] - rb[0], rb[3] - rb[1])
    placed = []
    for name in stats.sort_values("score", ascending=False).index:
        if len(placed) >= top_n:
            break
        sub = named[named.river_name == name]
        sub = sub[sub.intersects(box(*rb))]
        if sub.empty:
            continue
        mid = sub.iloc[np.argsort(sub.dist_out_m.values)[len(sub) // 2]].geometry
        p = mid.interpolate(0.5, normalized=True)
        if any(np.hypot(p.x - q[0], p.y - q[1]) < min_sep for q in placed):
            continue
        placed.append((p.x, p.y))
        ax.text(p.x, p.y, name, fontsize=fontsize, color="#08306b", style="italic",
                ha="center", va="center", zorder=20,
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white", alpha=0.9)])


def setup_axes(ax, rb):
    ax.set_xlim(rb[0], rb[2]); ax.set_ylim(rb[1], rb[3])
    ax.set_aspect(1 / np.cos(np.deg2rad(0.5 * (rb[1] + rb[3]))))
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{abs(v):g}°{'W' if v < 0 else 'E'}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{abs(v):g}°{'S' if v < 0 else 'N'}"))
    ax.grid(color="white", lw=0.4, alpha=0.6, zorder=1)


def draw_background(ax, dem_pack, rb):
    if dem_pack is None:
        ax.set_facecolor("#eef3e6")
        return
    rgb, ext = crop(*dem_pack, rb)
    if rgb is not None:
        ax.imshow(rgb, extent=ext, origin="upper", interpolation="bilinear", zorder=0)


def draw_countries(ax, countries):
    if countries is not None:
        countries.boundary.plot(ax=ax, color="#7a7a7a", lw=0.5, ls=(0, (4, 2)), zorder=2)


def width_legend(ax, title="SWORD channel width"):
    h = [Line2D([], [], color=c, lw=max(1.2, lw * 2)) for c, lw in zip(WIDTH_COLORS, WIDTH_LW)]
    ax.legend(h, WIDTH_LABELS, title=title, fontsize=6.5, title_fontsize=7,
              loc="lower left", framealpha=0.9).set_zorder(30)


def channel_legend(ax):
    h = [Line2D([], [], color=CHAN_COLORS[k], lw=2) for k in (1, 2, 3, 4)]
    ax.legend(h, [CHAN_LABELS[k] for k in (1, 2, 3, 4)], title="SWORD max. channels",
              fontsize=6.5, title_fontsize=7, loc="lower left", framealpha=0.9).set_zorder(30)


def draw_region_boxes(ax):
    for name, (w, s, e, n) in REGIONS.items():
        ax.plot([w, e, e, w, w], [s, s, n, n, s], color="#b2182b", lw=0.7, ls="--", zorder=15)
        ax.text(w + 0.15, n - 0.15, name.replace("_", " "), fontsize=5.5, color="#b2182b",
                va="top", zorder=16,
                path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])


def add_inset(fig, ax, countries):
    if countries is None:          # nothing to draw without outlines
        return
    ins = fig.add_axes([ax.get_position().x1 - 0.17, ax.get_position().y1 - 0.2, 0.16, 0.19])
    if countries is not None:
        sa = countries.cx[-92:-30, -57:14]
        sa.plot(ax=ins, color="#d9d9d9", edgecolor="white", lw=0.3)
    w, s, e, n = AMAZON_BBOX
    ins.plot([w, e, e, w, w], [s, s, n, n, s], color="#b2182b", lw=1)
    ins.set_xlim(-92, -32); ins.set_ylim(-57, 14); ins.set_aspect("equal")
    ins.set_xticks([]); ins.set_yticks([])
    ins.set_facecolor("#eef3f7")


def save(fig, path, pdf=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=DPI, bbox_inches="tight")
    if pdf:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {path.with_suffix('.png').relative_to(OUT_DIR)}")


# -----------------------------------------------------------------------------
# Maps
# -----------------------------------------------------------------------------
def overview(gdf, dem_pack, countries, grwl, mode):
    fig, ax = plt.subplots(figsize=(11, 8.6))
    rb = AMAZON_BBOX
    draw_background(ax, dem_pack, rb)
    draw_countries(ax, countries)
    if grwl is not None and SHOW_GRWL_SUMMARY:
        grwl.plot(ax=ax, color="#9e9ac8", lw=0.3, zorder=2)
    if mode == "width":
        draw_rivers_by_class(ax, gdf)
        width_legend(ax)
        title = f"Amazon rivers by channel width (SWORD {SWORD_VERSION})"
    else:
        draw_rivers_by_channels(ax, gdf)
        channel_legend(ax)
        title = f"Amazon rivers by number of channels (SWORD {SWORD_VERSION})"
    label_rivers(ax, gdf, rb, LABEL_TOP_N_RIVERS)
    draw_region_boxes(ax)
    setup_axes(ax, rb)
    ax.set_title(title, fontsize=11, loc="left")
    ax.text(0.0, -0.06, f"Rivers: SWORD {SWORD_VERSION} (Altenau et al. 2021). Relief: HydroSHEDS 3\" "
            "conditioned DEM. Dashed boxes: regional maps.", transform=ax.transAxes,
            fontsize=6.5, color="#555")
    add_inset(fig, ax, countries)
    save(fig, OUT_DIR / "maps" / f"amazon_overview_{'width' if mode == 'width' else 'channels'}",
         pdf=True)


def regional(name, rb, gdf, dem_pack, countries):
    sub = gdf[gdf.intersects(box(*rb))]
    fig, ax = plt.subplots(figsize=(9, 9 * (rb[3] - rb[1]) / (rb[2] - rb[0]) + 0.6))
    draw_background(ax, dem_pack, rb)
    draw_countries(ax, countries)
    if SHOW_GRWL_MASK and GRWL_MASK_DIR.exists():
        m = grwl_mask_mosaic(rb, REGION_RASTER_PX)
        if m is not None:
            arr, ext = m
            rgba = np.zeros(arr.shape + (4,))
            rgba[arr == 1] = (0.42, 0.68, 0.84, 0.9)
            rgba[arr == 2] = (0.55, 0.62, 0.75, 0.8)
            ax.imshow(rgba, extent=ext, origin="upper", interpolation="nearest", zorder=2)
    draw_rivers_by_class(ax, sub, lw_scale=1.6)
    label_rivers(ax, sub, rb, top_n=25, min_sep_frac=0.1, fontsize=7.5)
    setup_axes(ax, rb)
    width_legend(ax)
    km = sub.length_m.sum() / 1000
    ax.set_title(f"{name.replace('_', ' ')}: {len(sub):,} SWORD reaches, {km:,.0f} km of river",
                 fontsize=10, loc="left")
    if SHOW_GRWL_MASK:
        ax.text(0.0, -0.07 * 9 / fig.get_size_inches()[1] * 1.3,
                "Light blue fill: GRWL 30 m water mask (river, lake). Lines: SWORD reaches.",
                transform=ax.transAxes, fontsize=6.5, color="#555")
    save(fig, OUT_DIR / "maps" / "regions" / name)


# -----------------------------------------------------------------------------
# Tables / GeoPackage
# -----------------------------------------------------------------------------
def tables(gdf):
    T = OUT_DIR / "tables"; T.mkdir(parents=True, exist_ok=True)
    named = gdf[gdf.river_name != ""]
    rs = named.groupby("river_name").agg(
        n_reaches=("reach_id", "size"),
        length_km=("length_m", lambda v: v.sum() / 1000),
        median_width_m=("width_m", "median"),
        max_width_m=("width_m", "max"),
        max_channels=("n_chan_max", "max"),
        max_facc_km2=("facc_km2", "max"),
        regions=("region", lambda v: ";".join(sorted({x for s in v for x in s.split(";") if x}))),
    ).sort_values("length_km", ascending=False).round(1)
    rs.to_csv(T / "river_summary.csv")
    rows = []
    for name, rb in {"Amazon_window": AMAZON_BBOX, **REGIONS}.items():
        sub = gdf if name == "Amazon_window" else gdf[gdf.intersects(box(*rb))]
        row = {"region": name, "west": rb[0], "south": rb[1], "east": rb[2], "north": rb[3],
               "n_reaches": len(sub), "total_km": round(sub.length_m.sum() / 1000, 1)}
        for k, lab in enumerate(WIDTH_LABELS):
            row[f"km_{lab}"] = round(sub.loc[sub.width_class == k, "length_m"].sum() / 1000, 1)
        rows.append(row)
    pd.DataFrame(rows).to_csv(T / "region_width_summary.csv", index=False)
    log(f"  wrote tables/river_summary.csv ({len(rs)} named rivers), tables/region_width_summary.csv")


def export_gpkg(gdf, grwl):
    p = OUT_DIR / "amazon_rivers.gpkg"
    if p.exists():
        p.unlink()
    gdf.to_file(p, layer="sword_reaches", driver="GPKG")
    if grwl is not None:
        grwl.to_file(p, layer="grwl_segments", driver="GPKG")
    regs = gpd.GeoDataFrame({"region": list(REGIONS)},
                            geometry=[box(*b) for b in REGIONS.values()], crs="EPSG:4326")
    regs.to_file(p, layer="regions", driver="GPKG")
    log(f"  wrote amazon_rivers.gpkg (layers: sword_reaches, "
        f"{'grwl_segments, ' if grwl is not None else ''}regions)")


# -----------------------------------------------------------------------------
def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    gdf = load_sword(AMAZON_BBOX)
    gdf_map = gdf[~gdf.is_lake_or_ghost] if SKIP_LAKES else gdf

    grwl = None
    if (SHOW_GRWL_SUMMARY or EXPORT_GPKG) and GRWL_SUMMARY.exists():
        grwl = load_grwl_summary(AMAZON_BBOX)

    countries = None
    if COUNTRIES_SHP and Path(COUNTRIES_SHP).exists():
        countries = gpd.read_file(COUNTRIES_SHP).to_crs("EPSG:4326")

    dem_pack = None
    if SHOW_DEM and DEM_TIF.exists():
        dem, ext = cached_dem(AMAZON_BBOX)
        dem_pack = (dem_rgb(dem, ext), ext)

    log("Making maps")
    if MAKE_OVERVIEW:
        overview(gdf_map, dem_pack, countries, grwl, "width")
    if MAKE_CHANNEL_MAP:
        overview(gdf_map, dem_pack, countries, grwl, "channels")
    if MAKE_REGIONAL_MAPS:
        for name, rb in REGIONS.items():
            regional(name, rb, gdf_map, dem_pack, countries)

    if EXPORT_TABLES:
        tables(gdf)
    if EXPORT_GPKG:
        export_gpkg(gdf, grwl)
    log(f"Done in {time.time() - t0:.0f} s. Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
