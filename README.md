# thamno-birdsong

Antbird (Thamnophilidae) song variation across Amazonian rivers. The project tests whether song differs more across rivers than along them, and whether that difference scales with river width. The aim is to estimate how permeable each river is as a barrier.
Advisor: Dr. Timothy Keitt, UT Austin.

## Folder layout

```
Thamno-2026/
  code/      this repo's scripts
  data/      geodata/ (SWORD v16, GRWL, HydroSHEDS), xc_metadata/, other_metadata/, audio/   (not in git)
  output/    figures and tables written by the scripts                                (not in git)
```

Each script finds `Thamno-2026/` by itself. It uses the folder above `code/` if `data/` is there, and `~/Desktop/Thamno-2026` otherwise. Settings are capital-letter constants at the top of each file.

## Setup

```
conda env create -f environment.yml
conda activate antbird
```

You need a Xeno-canto API key. Set `XENO_CANTO_API_KEY` in your environment, or put the line `XENO_CANTO_API_KEY=yourkey` in `Thamno-2026/.env`. Don't commit the key.

## Scripts (run in this order)

| # | Script | What it does | Writes |
|---|---|---|---|
| 1 | `amazon_river_maps.py` | Maps SWORD rivers over HydroSHEDS relief (by width class and by number of channels), with regional close-ups using the GRWL water mask | `output/amazon_maps/` (maps, `amazon_rivers.gpkg`, cached DEM, tables) |
| 2 | `xc_pull_metadata.py` | Pulls Xeno-canto metadata (API v3) for 63 antbird genera | `data/xc_metadata/` |
| 3 | `plot_xc_coverage.py` | Recordings, songs and sites for the top 20 species | `output/xc_coverage/` |
| 4 | `xc_banks_map.py` | Core module: traces 19 focal rivers, assigns recordings to the left or right bank, counts usable songs per species per bank | `output/xc_banks/` |
| 5 | `gbif_pull_metadata.py` | Pulls Macaulay Library and FNJV metadata through GBIF; can merge Macaulay website exports | `data/other_metadata/` |
| 6 | `combined_banks.py` | Runs all archives through the bank rules; finds species × river pairs that become testable with requested audio; writes request lists | `output/combined_banks/` |
| 7 | `plot_archives_map.py` | One map of every recording, by archive | `output/combined_banks/recordings_by_archive_map.png` |
| 8 | `genus_song_clusters.py` | Downloads songs for one genus, measures them, runs k-means with no labels, and scores the clusters against species (silhouette, ARI, NMI) | `output/genus_clusters/<GENUS>/` |
| 9 | `song_vs_geo_distance.py` | Per species: Euclidean song distance (site means, z-scored features) vs great-circle distance; Mantel test | `output/song_vs_geo/<GENUS>/` |

Scripts 6–8 import `xc_banks_map.py`, so its settings (`STRICT_SONG`, `FOCAL_RIVERS`, thresholds) carry through to them.

## Key definitions

- **Site**: lat/lon rounded to 0.01°.
- **Pure song**: every comma-separated type tag contains "song". Recordings tagged "call, song", and untagged recordings, are left out.
- **Usable (Xeno-canto)**: pure song, quality A or B, no playback.
- **Bank has data**: at least 5 usable songs from at least 2 sites, within 100 km of the river.
- **Testable pair**: both banks of one river have data for one species.
- **Left/right bank**: looking downstream.

## Data sources

- Xeno-canto: https://xeno-canto.org (API v3)
- Macaulay Library (Cornell): https://www.macaulaylibrary.org (metadata via GBIF; audio by research request)
- FNJV, Unicamp: https://www2.ib.unicamp.br/fnjv/ (metadata via GBIF)
- SWORD v17c (Altenau et al. 2021); GRWL (Allen & Pavelsky 2018); HydroSHEDS (Lehner et al. 2008)
