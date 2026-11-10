"""
genus_song_clusters.py  --  Thamno-2026

Can unsupervised clustering separate species from song alone?

For one GENUS, takes the usable Xeno-canto songs in the Amazon window (same
rules as xc_banks_map.py: pure song, quality A/B, no playback), downloads the
audio, measures each song, clusters the songs with k-means WITHOUT species
labels, then compares the clusters with the species names and maps them.

Per-recording song features (medians over the song bouts found in the file):
  bout duration, notes per bout, note rate, rate change (2nd vs 1st half),
  inter-note interval variability, peak frequency, 10th/90th percentile
  frequency, bandwidth, frequency slope, bouts per minute, + 13 MFCC means.

Outputs: output/genus_clusters/<GENUS>/
  features.csv         one row per recording (+ species, lat, lon, site)
  clusters.csv         cluster labels for every k tried
  summary.png          silhouette by k | PCA by species | PCA by cluster |
                       species x cluster table (k = number of species)
  cluster_map.png      clusters at k = number of species on the river map
  scores.csv           silhouette, ARI and NMI against species for each k

Audio is cached in data/audio/<GENUS>/ (re-runs skip downloaded files).

Run from Thamno-2026/code:   python genus_song_clusters.py
Needs: librosa, soundfile, scikit-learn, requests (+ the base-map inputs)
"""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# =============================================================================
# SETTINGS
# =============================================================================
GENUS = "Thamnophilus"       # try "Hypocnemis" next (river-split cantator complex)
MIN_PER_SPECIES = 20         # species with fewer usable songs are left out
MAX_PER_SPECIES = 40         # cap per species so common species don't dominate
MAX_SECONDS = 60             # analyse at most this much of each file
FMIN, FMAX = 600, 10_000     # song band (Hz)
K_RANGE = range(2, 16)       # k values to try
SEED = 0
PAUSE_S = 0.5                # between downloads

import xc_banks_map as xb    # paths, usable-song rules, base map
ROOT = xb.ROOT
OUT = ROOT / "output" / "genus_clusters" / GENUS
AUDIO = ROOT / "data" / "audio" / GENUS

# =============================================================================


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def api_key():
    k = os.environ.get("XENO_CANTO_API_KEY", "")
    for envf in (ROOT / ".env", Path(__file__).resolve().parent / ".env"):
        if not k and envf.exists():
            for line in envf.read_text().splitlines():
                if line.strip().startswith("XENO_CANTO_API_KEY"):
                    k = line.split("=", 1)[1].strip().strip('"').strip("'")
    return k


def pick_recordings():
    x = pd.read_csv(xb.XC_CSV, low_memory=False)
    x = x[(x.gen == GENUS) & x.lat.notna()]
    x = x[xb.usable_mask(x).values].copy()
    x["species"] = x.gen + " " + x.sp
    keep = x.species.value_counts()
    keep = keep[keep >= MIN_PER_SPECIES].index
    x = x[x.species.isin(keep)]
    # cap per species, spreading picks across sites (one per site first)
    x["site"] = x.lat.round(2).astype(str) + "," + x.lon.round(2).astype(str)
    x = (x.sample(frac=1, random_state=SEED)
           .assign(rank_in_site=lambda d: d.groupby(["species", "site"]).cumcount())
           .sort_values(["species", "rank_in_site"])
           .groupby("species").head(MAX_PER_SPECIES))
    log(f"{GENUS}: {x.species.nunique()} species with >= {MIN_PER_SPECIES} usable songs; "
        f"{len(x)} recordings selected (max {MAX_PER_SPECIES} per species)")
    return x.reset_index(drop=True)


def download(rec):
    AUDIO.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers["User-Agent"] = "Thamno-2026 research"
    key = api_key()
    paths, fails = [], 0
    for _, r in rec.iterrows():
        p = AUDIO / f"XC{r['id']}.mp3"
        if not p.exists():
            url = r["file"] if str(r.get("file", "")).startswith("http") else \
                f"https://xeno-canto.org/{r['id']}/download"
            try:
                resp = s.get(url, params={"key": key} if key else None, timeout=120)
                if resp.status_code != 200 or len(resp.content) < 1000:
                    raise IOError(f"HTTP {resp.status_code}")
                p.write_bytes(resp.content)
                time.sleep(PAUSE_S)
            except Exception as e:
                fails += 1
                log(f"  download failed XC{r['id']}: {e}")
                paths.append(None); continue
        paths.append(p)
    log(f"Audio: {sum(p is not None for p in paths)} files ready, {fails} failed")
    return paths


def song_features(path):
    import librosa
    y, sr = librosa.load(path, sr=22050, mono=True, duration=MAX_SECONDS)
    if len(y) < sr:                         # under 1 s
        return None
    hop, nfft = 256, 1024
    S = np.abs(librosa.stft(y, n_fft=nfft, hop_length=hop))
    f = librosa.fft_frequencies(sr=sr, n_fft=nfft)
    band = (f >= FMIN) & (f <= FMAX)
    Sb, fb = S[band], f[band]
    e = 10 * np.log10((Sb ** 2).sum(axis=0) + 1e-12)
    active = e > np.percentile(e, 20) + 10                       # 10 dB over the noise floor
    t = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop)
    if active.sum() < 10:
        return None
    dom = fb[np.argmax(Sb, axis=0)]
    onset_env = librosa.onset.onset_strength(S=librosa.amplitude_to_db(Sb), sr=sr)
    on = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop, units="frames")
    on = on[active[np.clip(on, 0, len(active) - 1)]]
    on_t = t[on] if len(on) else np.array([])

    # bouts: active runs, merging gaps < 0.6 s, keeping runs >= 0.5 s
    gap = int(0.6 * sr / hop)
    act = active.copy()
    idx = np.where(act)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= gap:
            act[a:b] = True
    edges = np.diff(np.r_[0, act.astype(int), 0])
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    bouts = [(s0, e0) for s0, e0 in zip(starts, ends) if (e0 - s0) * hop / sr >= 0.5]
    if not bouts:
        return None
    rows = []
    for s0, e0 in bouts:
        tb0, tb1 = t[s0], t[min(e0, len(t) - 1)]
        dur = tb1 - tb0
        notes = on_t[(on_t >= tb0) & (on_t <= tb1)]
        fr = active[s0:e0]
        fd = dom[s0:e0][fr]
        tt = t[s0:e0][fr]
        if len(fd) < 5:
            continue
        ioi = np.diff(notes)
        half = tb0 + dur / 2
        n1, n2 = (notes < half).sum(), (notes >= half).sum()
        rows.append(dict(
            bout_s=dur, notes=len(notes), rate_hz=len(notes) / dur if dur else np.nan,
            rate_change=(n2 - n1) / max(n1 + n2, 1),
            ioi_cv=(ioi.std() / ioi.mean()) if len(ioi) > 2 and ioi.mean() > 0 else np.nan,
            peak_hz=np.median(fd), f10_hz=np.percentile(fd, 10), f90_hz=np.percentile(fd, 90),
            bw_hz=np.percentile(fd, 90) - np.percentile(fd, 10),
            slope_hz_s=np.polyfit(tt - tt[0], fd, 1)[0] if tt[-1] > tt[0] else 0.0))
    if not rows:
        return None
    feat = pd.DataFrame(rows).median().to_dict()
    feat["bouts_per_min"] = len(rows) / (len(y) / sr / 60)
    mf = librosa.feature.mfcc(S=librosa.power_to_db(librosa.feature.melspectrogram(
        S=S ** 2, sr=sr, fmin=FMIN, fmax=FMAX)), n_mfcc=13)[:, active]
    for i, v in enumerate(mf.mean(axis=1)):
        feat[f"mfcc{i:02d}"] = v
    return feat


def cluster(feat):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
    cols = [c for c in feat.columns if c not in ("id", "species", "lat", "lon", "site")]
    X = feat[cols].astype(float)
    X = X.fillna(X.median())
    Z = StandardScaler().fit_transform(X)
    pca = PCA(n_components=0.95, random_state=SEED).fit(Z)
    P = pca.transform(Z)
    n_sp = feat.species.nunique()
    ks = sorted(set(k for k in K_RANGE if k < len(feat)) | {n_sp})
    labels, scores = {}, []
    for k in ks:
        lab = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit_predict(P)
        labels[k] = lab
        scores.append(dict(k=k, silhouette=silhouette_score(P, lab),
                           ARI_vs_species=adjusted_rand_score(feat.species, lab),
                           NMI_vs_species=normalized_mutual_info_score(feat.species, lab)))
    scores = pd.DataFrame(scores)
    return P, pca, labels, scores, n_sp


def figures(feat, P, labels, scores, n_sp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    species = sorted(feat.species.unique())
    cmap = plt.get_cmap("tab20")
    col_sp = {s: cmap(i % 20) for i, s in enumerate(species)}
    lab = labels[n_sp]
    best_k = int(scores.loc[scores.silhouette.idxmax(), "k"])

    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.42)
    a = fig.add_subplot(gs[0, 0])
    a.plot(scores.k, scores.silhouette, "o-", color="#08519c", label="Silhouette (cluster separation)")
    a.plot(scores.k, scores.ARI_vs_species, "s--", color="#e07b39", label="ARI vs species names")
    a.plot(scores.k, scores.NMI_vs_species, "^:", color="#555", label="NMI vs species names")
    a.axvline(n_sp, color="#999", lw=0.8); a.text(n_sp, a.get_ylim()[1], f" k = {n_sp} species",
                                                  fontsize=8, va="top", color="#666")
    a.set_xlabel("k (number of clusters)"); a.set_ylabel("score")
    a.set_title("a  How well clusters separate, and how well they match species", loc="left", fontsize=10)
    a.legend(fontsize=8, frameon=False); a.grid(color="#eee")

    b = fig.add_subplot(gs[0, 1])
    for s in species:
        m = feat.species == s
        b.scatter(P[m, 0], P[m, 1], s=16, color=col_sp[s], label=s.split()[1], alpha=0.85, edgecolor="none")
    b.set_xlabel("PC1"); b.set_ylabel("PC2")
    b.set_title("b  Songs in feature space, coloured by species name", loc="left", fontsize=10)
    b.legend(fontsize=7, frameon=False, markerscale=1.2, loc="upper left", bbox_to_anchor=(1.01, 1),
             title="species", title_fontsize=7.5)

    c = fig.add_subplot(gs[1, 0])
    for k in range(n_sp):
        m = lab == k
        c.scatter(P[m, 0], P[m, 1], s=16, color=cmap(k % 20), alpha=0.85, edgecolor="none")
    c.set_xlabel("PC1"); c.set_ylabel("PC2")
    c.set_title(f"c  Same songs, coloured by k-means cluster (k = {n_sp}, no labels used)",
                loc="left", fontsize=10)

    d = fig.add_subplot(gs[1, 1])
    ct = pd.crosstab(feat.species, lab, normalize="index")
    order = ct.idxmax(axis=1).sort_values().index
    ct = ct.loc[order]
    ct = ct[sorted(ct.columns, key=lambda c_: ct[c_].values.argmax())]   # diagonal-ish order
    im = d.imshow(ct.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    d.set_yticks(range(len(ct)), [s.split()[1] for s in ct.index], fontsize=8, style="italic")
    d.set_xticks(range(ct.shape[1]), [str(c_) for c_ in ct.columns], fontsize=8)
    d.set_xlabel("k-means cluster")
    d.set_title("d  Share of each species' songs in each cluster", loc="left", fontsize=10)
    fig.colorbar(im, ax=d, fraction=0.04, pad=0.02).set_label("share of species' songs", fontsize=8)
    ari = scores.set_index("k").loc[n_sp, "ARI_vs_species"]
    fig.suptitle(f"{GENUS}: {len(feat)} songs, {n_sp} species. At k = {n_sp}, agreement with species "
                 f"names ARI = {ari:.2f} (0 = chance, 1 = perfect); best-separated k = {best_k}",
                 x=0.01, ha="left", fontsize=12)
    fig.savefig(OUT / "summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # map of clusters at k = number of species
    import geopandas as gpd
    from matplotlib.lines import Line2D
    sword = gpd.read_file(xb.GPKG, layer="sword_reaches")
    fig, ax = plt.subplots(figsize=(12, 9.4))
    xb.base_map(ax, sword)
    markers = "os^Dv<>Pph*X"
    for k in range(n_sp):
        m = lab == k
        ax.scatter(feat.lon[m], feat.lat[m], s=34, color=cmap(k % 20), marker=markers[k % len(markers)],
                   edgecolor="white", lw=0.5, zorder=10, alpha=0.9)
    dom = pd.crosstab(lab, feat.species).idxmax(axis=1)
    hs = [Line2D([], [], ls="", marker=markers[k % len(markers)], ms=7, mfc=cmap(k % 20), mec="white",
                 label=f"cluster {k}: mostly {dom[k].split()[1]}") for k in range(n_sp)]
    leg = ax.legend(handles=hs, loc="lower left", fontsize=7.5, framealpha=0.92, title="k-means cluster",
                    title_fontsize=8)
    leg.set_zorder(30)
    ax.set_title(f"{GENUS} songs by k-means cluster (k = {n_sp}); labels name each cluster's "
                 "most common species", loc="left", fontsize=10)
    fig.savefig(OUT / "cluster_map.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rec = pick_recordings()
    rec["path"] = download(rec)
    rows = []
    for i, r in rec.iterrows():
        if r.path is None:
            continue
        try:
            f = song_features(r.path)
        except Exception as e:
            log(f"  could not read XC{r['id']}: {e}"); f = None
        if f:
            rows.append(dict(id=r["id"], species=r.species, lat=r.lat, lon=r.lon, site=r.site, **f))
        if (i + 1) % 50 == 0:
            log(f"  features: {i + 1}/{len(rec)}")
    feat = pd.DataFrame(rows)
    feat = feat[feat.species.map(feat.species.value_counts()) >= 5]       # need a few per species
    feat.to_csv(OUT / "features.csv", index=False)
    log(f"Features for {len(feat)} recordings of {feat.species.nunique()} species")

    P, pca, labels, scores, n_sp = cluster(feat)
    scores.round(3).to_csv(OUT / "scores.csv", index=False)
    cl = feat[["id", "species", "lat", "lon"]].copy()
    for k, lab in labels.items():
        cl[f"k{k}"] = lab
    cl.to_csv(OUT / "clusters.csv", index=False)
    figures(feat, P, labels, scores, n_sp)
    print(scores.round(3).to_string(index=False))
    log(f"Done. Outputs in {OUT}")


if __name__ == "__main__":
    main()
