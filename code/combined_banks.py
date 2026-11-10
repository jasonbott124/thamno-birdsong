"""
combined_banks.py  --  Thamno-2026

Steps 2 and 3 after the GBIF pull:
  2) Puts Xeno-canto, Macaulay and FNJV recordings through the SAME focal
     rivers and bank rules as xc_banks_map.py, and counts per species x river
     x bank:  ready   = Xeno-canto songs you can download now
              request = Macaulay / FNJV songs (or untagged recordings) whose
                        audio has to be requested
  3) Finds the species x river pairs that become testable only with requested
     recordings, and writes one request list per archive with just the
     recordings on the weaker bank(s) of those pairs.

Inputs:
  data/xc_metadata/thamnophilidae_amazon.csv      (xc_pull_metadata.py)
  data/other_metadata/other_archives_unified.csv  (gbif_pull_metadata.py)
  + the inputs of xc_banks_map.py (focal rivers are taken from its settings)

Outputs: output/combined_banks/
  bank_counts_by_source.csv     species x river x bank, counts by archive
  testable_now_vs_requested.csv species x river: testable now / with requests
  request_macaulay.csv          ML catalog numbers to request (+ why)
  request_fnjv.csv              FNJV catalog numbers to request (+ why)
  testable_by_river.png         testable species per river, now vs with requests
  possible_duplicates.csv       same species, day and site in two archives
  renamed_species.csv           older GBIF names matched to current names

Run from Thamno-2026/code:  python combined_banks.py
"""
from pathlib import Path
import time

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

import xc_banks_map as xb          # focal rivers, bank assignment, usable rules

# =============================================================================
# SETTINGS
# =============================================================================
ROOT = xb.ROOT
XC_CSV = xb.XC_CSV
OTHER_CSV = ROOT / "data" / "other_metadata" / "other_archives_unified.csv"
OUT = ROOT / "output" / "combined_banks"

MIN_SONGS, MIN_SITES = xb.MIN_SONGS, xb.MIN_SITES       # same bar as xc_banks_map.py
COUNT_UNTAGGED = True      # if xc_banks_map.STRICT_SONG is False: count untagged recordings too
ML_MIN_RATING = 3          # Macaulay community rating (1-5) needed when a rating exists
REQUEST_LIMIT_ML = 40_000  # Macaulay generally declines larger requests...
REQUEST_SPECIES_ML = 100   # ...or more species than this

# =============================================================================


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_xc():
    x = pd.read_csv(XC_CSV, low_memory=False)
    t = x["type"].fillna("").str.lower()
    return pd.DataFrame(dict(
        source="xeno-canto", catalog="XC" + x["id"].astype(str),
        species=(x.gen.fillna("") + " " + x.sp.fillna("")).str.strip(),
        lat=x.lat, lon=x.lon, date=x["date"].astype(str), recordist=x["rec"],
        sound_type=t, is_song=t.str.contains("song"), quality=x["q"],
        playback=x["playback-used"].fillna("").str.lower().eq("yes"),
        url="https://xeno-canto.org/" + x["id"].astype(str),
        ready=xb.usable_mask(x).values))


def load_other():
    if not OTHER_CSV.exists():
        raise SystemExit(f"Missing {OTHER_CSV}. Run gbif_pull_metadata.py first.")
    o = pd.read_csv(OTHER_CSV, low_memory=False)
    song = o.is_song.astype(str).str.lower()
    tagged_song = song.eq("true")
    untagged = ~song.isin(["true", "false"])
    pb = o.playback.astype(str).str.lower().eq("true")
    q = pd.to_numeric(o.quality, errors="coerce")
    ok_q = q.isna() | (q >= ML_MIN_RATING)
    if xb.STRICT_SONG:      # song only: no "song, call", no call, no untagged
        songish = xb.is_pure_song(o.sound_type)
    else:
        songish = tagged_song | (untagged & COUNT_UNTAGGED)
    o["candidate"] = songish & ~(pb & xb.EXCLUDE_PLAYBACK) & ok_q
    o["ready"] = False
    return o


def _stem(epithet):
    """Strip Latin gender endings so hyperythra / hyperythrus match."""
    e = str(epithet).lower()
    for end in ("us", "um", "a", "is", "e"):
        if e.endswith(end) and len(e) > len(end) + 3:
            return e[: -len(end)]
    return e


# Old genus -> genera it was split into (Isler et al. 2013 and later revisions).
# An older GBIF name is renamed only into one of these genera, and only when the
# species epithet (ignoring Latin gender endings) matches exactly one
# Xeno-canto species there. Add pairs here if renamed_species.csv misses any.
GENUS_SPLITS = {
    "Myrmeciza": {"Myrmeciza", "Poliocrania", "Myrmoderus", "Myrmophylax", "Sciaphylax",
                  "Percnostola", "Sipia", "Ampelornis", "Aprositornis", "Ammonastes",
                  "Myrmelastes", "Hafferia", "Akletos"},
    "Cercomacra": {"Cercomacra", "Cercomacroides"},
    "Schistocichla": {"Myrmelastes"},
    "Terenura": {"Euchrepomis", "Terenura"},
    "Gymnopithys": {"Gymnopithys", "Oneillornis"},
    "Percnostola": {"Percnostola", "Myrmoborus", "Akletos", "Myrmelastes"},
    "Myrmotherula": {"Myrmotherula", "Isleria", "Epinecrophylla"},
    "Hylophylax": {"Hylophylax", "Willisornis"},
    "Thamnophilus": {"Thamnophilus", "Sakesphorus"},
}


def harmonize_names(other, xc_species):
    """Rename older GBIF species names to the current (Xeno-canto) name, within
    GENUS_SPLITS only; genus-only or family-only names are dropped."""
    by_key = {}
    for sp in xc_species:
        parts = sp.split()
        if len(parts) == 2:
            by_key.setdefault(_stem(parts[1]), set()).add(sp)
    known = set(xc_species)
    out, renamed = [], {}
    for sp in other.species.astype(str):
        parts = sp.split()
        if len(parts) != 2 or not parts[1].islower():
            out.append(""); continue                     # not a species-level name
        if sp in known or parts[0] not in GENUS_SPLITS:
            out.append(sp); continue
        hits = {h for h in by_key.get(_stem(parts[1]), set())
                if h.split()[0] in GENUS_SPLITS[parts[0]]}
        if len(hits) == 1:
            new = next(iter(hits)); renamed[sp] = new; out.append(new)
        else:
            out.append(sp)
    other = other.assign(species=out)
    return other[other.species != ""], renamed


def flag_duplicates(df):
    d = df.copy()
    d["day"] = d.date.astype(str).str[:10]
    d["key"] = (d.species + "|" + d.day + "|" + d.lat.round(2).astype(str) + "," + d.lon.round(2).astype(str))
    multi = d[d.duplicated("key", keep=False) & d.day.str.match(r"\d{4}-\d{2}-\d{2}")]
    multi = multi[multi.groupby("key").source.transform("nunique") > 1]
    multi.sort_values("key").to_csv(OUT / "possible_duplicates.csv", index=False)
    # keep the Xeno-canto copy (ready now) when one exists
    pri = d.source.map({"xeno-canto": 0, "macaulay": 1, "fnjv": 2}).fillna(3)
    drop = d.assign(p=pri).sort_values("p").duplicated("key", keep="first") & d.index.isin(multi.index)
    log(f"  {int(drop.sum()):,} likely cross-archive duplicates dropped (same species, day, site)")
    return df[~drop]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    xc, other = load_xc(), load_other()
    xc["candidate"] = False
    other, renamed = harmonize_names(other, set(xc.species))
    pd.Series(renamed, name="current_name").rename_axis("gbif_name").to_csv(OUT / "renamed_species.csv")
    log(f"  {len(renamed)} older GBIF species names matched to current Xeno-canto names "
        f"(see renamed_species.csv)")
    rec = pd.concat([xc, other], ignore_index=True)
    w, s, e, n = xb.AMAZON_BBOX
    rec = rec[rec.lat.between(s, n) & rec.lon.between(w, e)].copy()
    rec = flag_duplicates(rec).reset_index(drop=True)
    rec["site"] = rec.lat.round(2).astype(str) + "," + rec.lon.round(2).astype(str)
    log("Records in the Amazon window: " + ", ".join(
        f"{k} {v:,}" for k, v in rec.source.value_counts().items()))

    rv, g = xb.trace_rivers()
    gdf = gpd.GeoDataFrame(rec, geometry=gpd.points_from_xy(rec.lon, rec.lat), crs=4326)
    use = gdf[gdf.ready | gdf.candidate]
    log(f"Assigning banks to {len(use):,} ready + candidate songs")
    banks = xb.assign_banks(use, rv, g).merge(
        rec[["source", "catalog", "species", "site", "ready", "candidate", "url", "lat", "lon"]],
        left_on="rec_index", right_index=True)

    # ---- counts per species x river x bank ---------------------------------
    def agg(df):
        r = df[df.ready]; c = df[df.candidate]
        return pd.Series(dict(
            ready_songs=len(r), ready_sites=r.site.nunique(),
            request_ml=int((c.source == "macaulay").sum()), request_fnjv=int((c.source == "fnjv").sum()),
            all_songs=len(df), all_sites=df.site.nunique()))
    per = banks.groupby(["species", "river", "side", "bank"]).apply(agg, include_groups=False).reset_index()
    per.to_csv(OUT / "bank_counts_by_source.csv", index=False)

    ok_now = (per.ready_songs >= MIN_SONGS) & (per.ready_sites >= MIN_SITES)
    ok_all = (per.all_songs >= MIN_SONGS) & (per.all_sites >= MIN_SITES)
    per["ok_now"], per["ok_all"] = ok_now, ok_all
    pair = per.groupby(["species", "river"]).agg(
        sides=("side", "nunique"), now=("ok_now", "sum"), all=("ok_all", "sum")).reset_index()
    pair["testable_now"] = (pair.sides == 2) & (pair.now == 2)
    pair["testable_with_requests"] = (pair.sides == 2) & (pair["all"] == 2)
    pair["gained"] = pair.testable_with_requests & ~pair.testable_now
    pair = pair.merge(rv[["name", "median_width_m"]], left_on="river", right_on="name").drop(columns="name")
    pair.drop(columns=["sides", "now", "all"]).sort_values(
        ["testable_with_requests", "testable_now", "species"], ascending=[False, False, True]
    ).to_csv(OUT / "testable_now_vs_requested.csv", index=False)
    n_now, n_all = int(pair.testable_now.sum()), int(pair.testable_with_requests.sum())
    log(f"Testable species x river pairs: {n_now} now (Xeno-canto only) -> {n_all} with requested "
        f"recordings ({pair[pair.testable_with_requests].species.nunique()} species)")

    # ---- step 3: request lists ---------------------------------------------
    gained = pair[pair.gained][["species", "river"]]
    weak = per.merge(gained, on=["species", "river"])
    weak = weak[~weak.ok_now][["species", "river", "side", "bank"]]      # sides that need help
    req = banks[banks.candidate].merge(weak, on=["species", "river", "side", "bank"])
    req = req.assign(reason="fills the " + req.bank + " bank of the " + req.river + " for this species")
    cols = ["catalog", "species", "river", "bank", "lat", "lon", "url", "reason"]
    for src, fname in (("macaulay", "request_macaulay.csv"), ("fnjv", "request_fnjv.csv")):
        r = req[req.source == src].drop_duplicates("catalog")[cols].sort_values(["species", "river"])
        r.to_csv(OUT / fname, index=False)
        log(f"  {fname}: {len(r):,} recordings, {r.species.nunique()} species")
        if src == "macaulay" and (len(r) > REQUEST_LIMIT_ML or r.species.nunique() > REQUEST_SPECIES_ML):
            log("  note: above Macaulay's usual request limits; split it or trim to the "
                "pairs you care about most")

    # ---- figure: testable species per river, now vs with requests ----------
    byr = (pair.groupby("river")[["testable_now", "testable_with_requests"]].sum()
             .query("testable_with_requests > 0")
             .sort_values("testable_with_requests"))
    if len(byr):
        fig, ax = plt.subplots(figsize=(8, 0.42 * len(byr) + 1.6))
        y = np.arange(len(byr))
        ax.barh(y, byr.testable_with_requests, height=0.7, color="#c6dbef",
                label="With requested Macaulay / FNJV audio")
        ax.barh(y, byr.testable_now, height=0.36, color="#08519c", label="Now (Xeno-canto only)")
        for i, (a, b) in enumerate(zip(byr.testable_now, byr.testable_with_requests)):
            ax.text(b + 0.15, i, f"{int(a)} → {int(b)}", va="center", fontsize=8, color="#333")
        ax.set_yticks(y, byr.index)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.set_xlabel("Species testable across the river (both banks with data)")
        ax.set_title(f"Testable species per river: both banks with >= {MIN_SONGS} songs from "
                     f">= {MIN_SITES} sites", loc="left", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#e6e6e6"); ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        fig.savefig(OUT / "testable_by_river.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    log(f"Done. Outputs in {OUT}")


if __name__ == "__main__":
    main()
