"""
xc_pull_metadata.py  --  Thamno-2026

Pulls Xeno-canto METADATA ONLY (no audio) for every antbird (Thamnophilidae)
recording, through API v3, page by page, genus by genus.

Outputs (Thamno-2026/data/xc_metadata/):
  thamnophilidae_all.csv         every recording returned, every API field
  thamnophilidae_amazon.csv      inside the Amazon window, with helper columns
  species_coverage_amazon.csv    per species: recordings, songs, A/B songs, sites
  raw_pages/<genus>_pNNN.json    raw API responses (re-runs reuse them; delete to refresh)

API key: set XENO_CANTO_API_KEY in your environment, or put a line
  XENO_CANTO_API_KEY=yourkey
in Thamno-2026/.env, or paste it into API_KEY below.

Run:  python xc_pull_metadata.py
Needs: requests, pandas
"""

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

# =============================================================================
# SETTINGS
# =============================================================================
_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent if (_HERE.parent / "data").exists() \
    else Path.home() / "Desktop" / "Thamno-2026"
OUT = ROOT / "data" / "xc_metadata"

API_KEY = ""                     # leave empty to use env var / .env
API_URL = "https://xeno-canto.org/api/3/recordings"
PER_PAGE = 500                   # v3 page size (the API caps it; 500 is the max)
PAUSE_S = 1.0                    # politeness delay between requests
EXTRA_QUERY = ""                 # e.g. 'cnt:"Brazil"' or 'q:">C"' to narrow every query

# Same window as amazon_river_maps.py: (west, south, east, north)
AMAZON_BBOX = (-80.0, -20.0, -44.0, 6.5)

# Thamnophilidae genera (current names plus a few older ones; a genus with no
# recordings just returns zero, so extras are harmless). Add any that are missing.
GENERA = [
    "Akletos", "Ammonastes", "Ampelornis", "Aprositornis", "Batara", "Biatas",
    "Cercomacra", "Cercomacroides", "Clytoctantes", "Cymbilaimus", "Dichrozona",
    "Drymophila", "Dysithamnus", "Epinecrophylla", "Euchrepomis", "Formicivora",
    "Frederickena", "Gymnocichla", "Gymnopithys", "Hafferia", "Herpsilochmus",
    "Hylophylax", "Hypocnemis", "Hypocnemoides", "Hypoedaleus", "Isleria",
    "Mackenziaena", "Megastictus", "Microrhopias", "Myrmeciza", "Myrmelastes",
    "Myrmoborus", "Myrmochanes", "Myrmoderus", "Myrmophylax", "Myrmorchilus",
    "Myrmornis", "Myrmotherula", "Neoctantes", "Oneillornis", "Percnostola",
    "Phaenostictus", "Phlegopsis", "Pithys", "Poliocrania", "Pygiptila",
    "Pyriglena", "Rhegmatorhina", "Rhopias", "Rhopornis", "Sakesphorus",
    "Schistocichla", "Sciaphylax", "Sclateria", "Sipia", "Stymphalornis",
    "Taraba", "Terenura", "Thamnistes", "Thamnomanes", "Thamnophilus",
    "Willisornis", "Xenornis",
]

# =============================================================================


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_key():
    if API_KEY:
        return API_KEY
    k = os.environ.get("XENO_CANTO_API_KEY", "")
    if not k:
        for envf in (ROOT / ".env", _HERE / ".env"):
            if envf.exists():
                for line in envf.read_text().splitlines():
                    if line.strip().startswith("XENO_CANTO_API_KEY"):
                        k = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not k:
        sys.exit("No API key. Set XENO_CANTO_API_KEY, add it to Thamno-2026/.env, "
                 "or paste it into API_KEY at the top of this script.")
    return k


def fetch_page(session, key, query, page, cache):
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    params = {"query": query, "key": key, "page": page, "per_page": PER_PAGE}
    for attempt in range(5):
        try:
            r = session.get(API_URL, params=params, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            if r.status_code != 200:
                # 4xx other than 429: bad query or key -> show the message and stop
                sys.exit(f"API error {r.status_code} for query {query!r}: {r.text[:500]}")
            data = r.json()
            cache.write_text(json.dumps(data), encoding="utf-8")
            time.sleep(PAUSE_S)
            return data
        except (requests.RequestException, ValueError) as e:
            wait = 5 * 2 ** attempt
            log(f"    {e}; retrying in {wait} s")
            time.sleep(wait)
    sys.exit(f"Giving up on {query!r} page {page}")


def pull_genus(session, key, genus, raw_dir):
    query = f"gen:{genus}" + (f" {EXTRA_QUERY}" if EXTRA_QUERY else "")
    recs, page, npages = [], 1, 1
    while page <= npages:
        data = fetch_page(session, key, query, page, raw_dir / f"{genus}_p{page:03d}.json")
        if "error" in data:
            sys.exit(f"API error for {query!r}: {data}")
        npages = int(data.get("numPages", 1) or 1)
        recs.extend(data.get("recordings", []))
        if page == 1:
            log(f"  {genus:<16} {int(data.get('numRecordings', 0) or 0):>6,} recordings, "
                f"{npages} page(s)")
        page += 1
    return recs


def tidy(df):
    # v3 uses 'lon'; v2 used 'lng'
    if "lon" not in df and "lng" in df:
        df = df.rename(columns={"lng": "lon"})
    df["lat"] = pd.to_numeric(df.get("lat"), errors="coerce")
    df["lon"] = pd.to_numeric(df.get("lon"), errors="coerce")
    df["species"] = (df["gen"].fillna("") + " " + df["sp"].fillna("")).str.strip()
    t = df.get("type", pd.Series("", index=df.index)).fillna("").str.lower()
    df["has_song"] = t.str.contains("song")
    df["song_only"] = df.has_song & ~t.str.contains("call")
    df["playback_used"] = df.get("playback-used", pd.Series("", index=df.index)) \
        .fillna("").str.lower().eq("yes")
    w, s, e, n = AMAZON_BBOX
    df["in_amazon_box"] = df.lat.between(s, n) & df.lon.between(w, e)
    df["site"] = df.lat.round(2).astype(str) + "," + df.lon.round(2).astype(str)
    return df


def main():
    key = get_key()
    raw_dir = OUT / "raw_pages"
    raw_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "Thamno-2026 research metadata pull"

    log(f"Pulling metadata for {len(GENERA)} genera -> {OUT}")
    all_recs = []
    for g in GENERA:
        all_recs.extend(pull_genus(session, key, g, raw_dir))
    if not all_recs:
        sys.exit("No recordings returned. Check the key and the query syntax.")

    df = tidy(pd.DataFrame(all_recs).drop_duplicates(subset="id"))
    df.to_csv(OUT / "thamnophilidae_all.csv", index=False)

    am = df[df.in_amazon_box].copy()
    am.to_csv(OUT / "thamnophilidae_amazon.csv", index=False)

    q = am.get("q", pd.Series("", index=am.index)).fillna("")
    good_song = am.has_song & q.isin(["A", "B"]) & ~am.playback_used
    cov = (am.assign(good_song=good_song)
             .groupby("species")
             .agg(recordings=("id", "size"),
                  songs=("has_song", "sum"),
                  songs_AB_no_playback=("good_song", "sum"),
                  sites=("site", "nunique"),
                  sites_with_good_song=("site", lambda s: s[good_song.loc[s.index]].nunique()))
             .sort_values("songs_AB_no_playback", ascending=False))
    cov.to_csv(OUT / "species_coverage_amazon.csv")

    log(f"Done. {len(df):,} recordings of {df.species.nunique()} species in total; "
        f"{len(am):,} recordings of {am.species.nunique()} species inside the Amazon window.")
    log("Top species by usable songs (A/B quality, no playback):")
    print(cov.head(15).to_string())


if __name__ == "__main__":
    main()
