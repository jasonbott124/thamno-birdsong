"""
gbif_pull_metadata.py  --  Thamno-2026

Pulls antbird (Thamnophilidae) recording METADATA (no audio) for the
Macaulay Library and FNJV from the GBIF occurrence API, and optionally merges
a Macaulay Library website export, into one table with the same columns as
the Xeno-canto pull.

Outputs (Thamno-2026/data/other_metadata/):
  gbif_raw/<source>_pNNNN.json   raw GBIF pages (re-runs reuse them; delete to refresh)
  macaulay_gbif.csv, fnjv_gbif.csv   flattened GBIF records
  other_archives_unified.csv     Macaulay + FNJV in the unified schema

Unified schema (also used for Xeno-canto in combined_banks.py):
  source, catalog, species, lat, lon, date, recordist, sound_type, is_song,
  quality, playback, url

Optional: Macaulay website export. In Macaulay Library Search, filter to
Thamnophilidae / audio / the Amazon countries, press Export (up to 10,000 rows
per export when signed in) and save the CSV(s) in data/other_metadata/
macaulay_exports/. They are merged, and preferred over GBIF rows with the
same catalog number because they carry ratings, behaviours and playback.

Run:  python gbif_pull_metadata.py
Needs: requests, pandas
"""
import glob
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

# =============================================================================
# SETTINGS
# =============================================================================
_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent if (_HERE.parent / "data").exists() else Path.home() / "Desktop" / "Thamno-2026"
OUT = ROOT / "data" / "other_metadata"
ML_EXPORT_DIR = OUT / "macaulay_exports"

API = "https://api.gbif.org/v1"
DATASETS = {   # GBIF dataset keys
    "macaulay": "7f6dd0f7-9ed4-49c0-bb71-b2a9c7fed9f1",   # Macaulay Library Audio and Video Collection
    "fnjv":     "ca2e422f-be48-42a5-bca5-a37cada435c8",   # Fonoteca Neotropical Jacques Vielliard
}
FAMILY = "Thamnophilidae"
PAGE = 300            # GBIF maximum page size
PAUSE_S = 0.5

# =============================================================================


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get(session, url, params):
    for attempt in range(5):
        try:
            r = session.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}")
            sys.exit(f"GBIF error {r.status_code} for {url} {params}: {r.text[:300]}")
        except (requests.RequestException, ValueError) as e:
            wait = 5 * 2 ** attempt
            log(f"    {e}; retrying in {wait} s")
            time.sleep(wait)
    sys.exit(f"Giving up on {url}")


def family_key(session):
    m = get(session, f"{API}/species/match", {"name": FAMILY, "rank": "FAMILY", "strict": "true"})
    k = m.get("familyKey") or m.get("usageKey")
    if not k:
        sys.exit(f"GBIF could not match {FAMILY}: {m}")
    log(f"{FAMILY}: GBIF familyKey {k}")
    return k


def pull(session, source, dkey, fkey, raw_dir):
    recs, offset, page = [], 0, 0
    while True:
        cache = raw_dir / f"{source}_p{page:04d}.json"
        if cache.exists():
            data = json.loads(cache.read_text(encoding="utf-8"))
        else:
            data = get(session, f"{API}/occurrence/search",
                       {"datasetKey": dkey, "familyKey": fkey, "limit": PAGE, "offset": offset})
            cache.write_text(json.dumps(data), encoding="utf-8")
            time.sleep(PAUSE_S)
        if page == 0:
            n = data.get("count", 0)
            log(f"  {source}: {n:,} antbird records in GBIF")
            if n > 100_000:
                log("  more than 100,000 records: GBIF search paging stops there; "
                    "use a GBIF download instead for the rest")
        recs.extend(data.get("results", []))
        if data.get("endOfRecords", True) or offset + PAGE >= 100_000:
            break
        offset += PAGE; page += 1
    return recs


SONG_WORDS = re.compile(r"\b(song|canto|singing|dawn song)\b", re.I)
CALL_WORDS = re.compile(r"\b(call|chamado|alarm|chip)\b", re.I)


def sound_info(text):
    t = text or ""
    s, c = bool(SONG_WORDS.search(t)), bool(CALL_WORDS.search(t))
    if s and not c:
        return "song", True
    if c and not s:
        return "call", False
    if s and c:
        return "song, call", True
    return "", pd.NA           # unknown


def flatten_gbif(recs, source):
    rows = []
    for r in recs:
        text = " ".join(str(r.get(k, "")) for k in
                        ("occurrenceRemarks", "behavior", "dynamicProperties", "fieldNotes"))
        stype, is_song = sound_info(text)
        media = [m.get("identifier") or m.get("references") for m in r.get("media", [])
                 if str(m.get("type", "")).lower() == "sound"]
        rows.append(dict(
            source=source, catalog=str(r.get("catalogNumber") or r.get("occurrenceID") or r.get("key")),
            species=r.get("species") or r.get("scientificName", ""),
            lat=r.get("decimalLatitude"), lon=r.get("decimalLongitude"),
            date=r.get("eventDate", ""), recordist=r.get("recordedBy", ""),
            sound_type=stype, is_song=is_song, quality=pd.NA, playback=pd.NA,
            url=r.get("references") or (media[0] if media else ""),
            gbif_key=r.get("key"), has_sound_media=bool(media),
            basis=r.get("basisOfRecord", ""), country=r.get("country", ""),
            locality=r.get("locality", ""), remarks=text.strip()))
    return pd.DataFrame(rows)


def col(df, *names):
    """First matching column (case-insensitive), else None."""
    low = {c.lower().strip(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return df[low[n.lower()]]
    return None


def macaulay_exports():
    files = sorted(glob.glob(str(ML_EXPORT_DIR / "*.csv")))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    cat = col(df, "ML Catalog Number", "Catalog Number", "catalogNumber")
    if cat is None:
        log("  Macaulay export(s) found but no 'ML Catalog Number' column; skipped")
        return None
    beh = col(df, "Behaviors", "Behavior")
    beh = beh.fillna("").astype(str) if beh is not None else pd.Series("", index=df.index)
    info = beh.map(sound_info)
    pb = col(df, "Playback")
    rating = col(df, "Average Community Rating", "Rating")
    fmt = col(df, "Format", "Media Type")
    out = pd.DataFrame(dict(
        source="macaulay", catalog=("ML" + cat.astype(str).str.replace(r"\.0$", "", regex=True)),
        species=col(df, "Scientific Name", "scientificName"),
        lat=pd.to_numeric(col(df, "Latitude"), errors="coerce"),
        lon=pd.to_numeric(col(df, "Longitude"), errors="coerce"),
        date=col(df, "Date"), recordist=col(df, "Recordist"),
        sound_type=[a for a, _ in info], is_song=[b for _, b in info],
        quality=pd.to_numeric(rating, errors="coerce") if rating is not None else pd.NA,
        playback=(pb.astype(str).str.lower().isin(["yes", "true", "1"]) if pb is not None else pd.NA),
        url="https://macaulaylibrary.org/asset/" + cat.astype(str).str.replace(r"\.0$", "", regex=True)))
    if fmt is not None:
        out = out[fmt.astype(str).str.lower().str.contains("audio")]
    log(f"  Macaulay website export(s): {len(out):,} audio records from {len(files)} file(s)")
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = OUT / "gbif_raw"; raw.mkdir(exist_ok=True)
    ML_EXPORT_DIR.mkdir(exist_ok=True)
    s = requests.Session()
    s.headers["User-Agent"] = "Thamno-2026 research metadata pull"
    fkey = family_key(s)

    frames = []
    for source, dkey in DATASETS.items():
        df = flatten_gbif(pull(s, source, dkey, fkey, raw), source)
        df.to_csv(OUT / f"{source}_gbif.csv", index=False)
        log(f"  {source}: {len(df):,} rows, {df.lat.notna().sum():,} with coordinates, "
            f"{df.species.nunique()} species; song tagged {int((df.is_song == True).sum()):,}, "
            f"call {int((df.is_song == False).sum()):,}, unknown {int(df.is_song.isna().sum()):,}")
        frames.append(df)

    # Macaulay catalog numbers as 'ML123456' (the form their request page uses)
    g = frames[0]
    digits = g.catalog.astype(str).str.extract(r"(\d+)")[0]
    g["catalog"] = ("ML" + digits).where(digits.notna(), g.catalog)
    ml = macaulay_exports()
    if ml is not None:          # exports carry ratings/playback, so they replace GBIF rows
        frames[0] = g[~g.catalog.isin(ml.catalog)]
        frames.append(ml)

    uni = pd.concat(frames, ignore_index=True)
    keep = ["source", "catalog", "species", "lat", "lon", "date", "recordist",
            "sound_type", "is_song", "quality", "playback", "url"]
    uni[keep].to_csv(OUT / "other_archives_unified.csv", index=False)
    log(f"Done. {len(uni):,} records -> {OUT / 'other_archives_unified.csv'}")
    print(uni.groupby("source").agg(records=("catalog", "size"),
                                    with_coords=("lat", lambda v: v.notna().sum()),
                                    species=("species", "nunique")).to_string())


if __name__ == "__main__":
    main()
