#!/usr/bin/env python3
import os
import time
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.io.streamline import save_tractogram
from dipy.tracking.distances import bundles_distances_mam
from dipy.segment.bundles import bundles_distances_mdf
from dipy.tracking.streamline import set_number_of_points

from .tracklib import loadTractogram, trk2tck
from .dissimilarity_common import compute_dissimilarity
from .io_utils import ensure_dir, write_json

import os
try:
    import psutil
    
    def log_mem(prefix=""):
        proc = psutil.Process(os.getpid())
        rss_gb = proc.memory_info().rss / (1024 ** 3)
        vm = psutil.virtual_memory()
        avail_gb = vm.available / (1024 ** 3)
        total_gb = vm.total / (1024 ** 3)
        print(f"[mem] {prefix} rss={rss_gb:.2f} GB | avail={avail_gb:.2f} GB / total={total_gb:.2f} GB", flush=True)
except:
    import resource    
    def log_mem(prefix=""):
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_gb = rss_kb / (1024 ** 2)
        print(f"[mem] {prefix} maxrss={rss_gb:.2f} GB", flush=True)
    
def _padnum(num, n=4):
    return str(num).zfill(n)


def _resolve_distance(distance_name: str):
    distance_name = (distance_name or "mdf").lower()
    if distance_name == "mdf":
        return bundles_distances_mdf
    if distance_name == "mam":
        return bundles_distances_mam
    raise ValueError(f"Unsupported distance: {distance_name}")

from multiprocessing import Pool

def _resample_chunk(args):
    chunk, nb_points = args

    seq = nib.streamlines.ArraySequence()
    for sl in chunk:
        seq.append(np.asarray(sl, dtype=np.float32))

    if nb_points is None:
        return np.array([np.asarray(sl, dtype=np.float32) for sl in seq], dtype=object)

    resampled = set_number_of_points(seq, nb_points)
    return np.array([np.asarray(sl, dtype=np.float32) for sl in resampled], dtype=object)


def resample_streamlines_parallel(streamlines, nb_points, n_jobs=1, chunk_size=50000):
    chunks = [
        (streamlines[i:i+chunk_size], nb_points)
        for i in range(0, len(streamlines), chunk_size)
    ]

    if n_jobs == 1:
        results = [_resample_chunk(c) for c in chunks]
    else:
        with Pool(n_jobs) as p:
            results = p.map(_resample_chunk, chunks)

    return np.concatenate(results, axis=0)

def _resample_streamlines(streamlines, nb_points: int):
    seq = nib.streamlines.ArraySequence()
    for sl in streamlines:
        seq.append(np.asarray(sl, dtype=np.float32))

    if nb_points is None:
        return np.array([np.asarray(sl, dtype=np.float32) for sl in seq], dtype=object)

    resampled = set_number_of_points(seq, nb_points)
    return np.array([np.asarray(sl, dtype=np.float32) for sl in resampled], dtype=object)

def compute_embedding(streamlines, distance_name="mdf", num_prototypes=64,
                      prototype_policy="sff", size_limit=5_000_000,
                      n_jobs=None, verbose=False):
    distance_fn = _resolve_distance(distance_name)
    data = np.asarray(streamlines, dtype=object)
    return compute_dissimilarity(
        data,
        distance=distance_fn,
        prototype_policy=prototype_policy,
        num_prototypes=num_prototypes,
        verbose=verbose,
        size_limit=size_limit,
        n_jobs=n_jobs,
    )


def fit_mbkm(embedding, n_clusters, batch_size=1000, n_init=10,
             max_no_improvement=5, random_state=0, verbose=False):
    km = MiniBatchKMeans(
        init="random",
        n_clusters=n_clusters,
        batch_size=batch_size,
        n_init=n_init,
        max_no_improvement=max_no_improvement,
        random_state=random_state,
    )
    t0 = time.time()
    km.fit(np.nan_to_num(embedding))
    if verbose:
        print(f"[clustering] MBKM fit in {time.time()-t0:.2f}s")
    return km


def select_medoids_from_embedding(embedding, labels, centers):
    n_clusters = centers.shape[0]
    medoids = np.zeros(n_clusters, dtype=int)
    clusters = []
    for i in range(n_clusters):
        idx = np.where(labels == i)[0]
        if len(idx) == 0:
            medoids[i] = 0
            clusters.append(np.array([0], dtype=int))
            continue
        diff = embedding[idx] - centers[i]
        medoids[i] = idx[np.argmin((diff * diff).sum(axis=1))]
        clusters.append(idx)
    return clusters, medoids


def save_merged_medoids(tract, medoids, reference, out_path):
    ensure_dir(Path(out_path).parent)
    sls = [tract[i] for i in medoids.tolist()]
    sft = StatefulTractogram(sls, reference, Space.RASMM)
    sft.remove_invalid_streamlines()
    save_tractogram(sft, out_path, bbox_valid_check=False)
    trk2tck([out_path])


def save_individual_medoids(tract, medoids, reference, out_dir):
    ensure_dir(out_dir)
    pad_width = len(str(max(1, len(medoids)-1)))
    for i, medoid_idx in enumerate(medoids.tolist()):
        sft = StatefulTractogram([tract[medoid_idx]], reference, Space.RASMM)
        sft.remove_invalid_streamlines()
        save_tractogram(sft, os.path.join(out_dir, f"medoid_{_padnum(i, pad_width)}.trk"), bbox_valid_check=False)


def save_individual_clusters(tract, clusters, reference, out_dir):
    ensure_dir(out_dir)
    pad_width = len(str(max(1, len(clusters)-1)))
    for i, cluster_idx in enumerate(clusters):
        sft = StatefulTractogram(tract[cluster_idx.tolist()], reference, Space.RASMM)
        sft.remove_invalid_streamlines()
        save_tractogram(sft, os.path.join(out_dir, f"cluster_{_padnum(i, pad_width)}.trk"), bbox_valid_check=False)


def cluster_size_stats(labels, n_clusters):
    counts = np.bincount(labels, minlength=n_clusters)
    return {
        "min": int(counts.min()) if len(counts) else 0,
        "median": float(np.median(counts)) if len(counts) else 0.0,
        "max": int(counts.max()) if len(counts) else 0,
        "singleton_clusters": int((counts == 1).sum()),
        "empty_clusters": int((counts == 0).sum()),
    }


from multiprocessing import cpu_count

def run_from_config(cfg):
    t_global = time.time()
    track_path = cfg["track"]
    reference_path = cfg.get("reference")

    # ---- FIX n_jobs HERE ----
    n_jobs = int(cfg.get("n_jobs", -1))

    if n_jobs == -1:
        n_jobs = cpu_count()

    if n_jobs < 1:
        n_jobs = 1

    # --------------------------------

    out_medoids = cfg.get("out_medoids", "out_medoids")
    out_labels = cfg.get("out_labels", "out_labels")
    out_qc = cfg.get("out_qc", "out_qc")
    ensure_dir(out_medoids)
    ensure_dir(out_labels)
    ensure_dir(out_qc)
    
    print(f"loading tractogram: {track_path}")
    print(f"using n_jobs={n_jobs}")
    log_mem("before tractogram loading")
    verbose = bool(cfg.get("verbose", True))

    streamlines, affine, header = loadTractogram(
        track_path,
        n_jobs=n_jobs,
        reference_img=reference_path,
        verbose=verbose, 
        max_num=cfg.get("max_num_streamlines"),
    )
    
    log_mem("after tractogram loading")
    n_streamlines = len(streamlines)

    print(f"loaded {n_streamlines:,} streamlines")
    
    n_clusters_requested = int(cfg.get("n_clusters", 1))
    n_clusters_final = max(1, min(n_clusters_requested, n_streamlines))
    if n_clusters_final != n_clusters_requested:
        print(f"[clustering] capping n_clusters from {n_clusters_requested} to {n_clusters_final}")

    reference = nib.load(reference_path) if reference_path else header

    nb_points = cfg.get("nb_points", 20)
    distance = cfg.get("distance", "mdf")
    
    print(f"resampling streamlines to {nb_points} points")
    t0 = time.time()
    if n_jobs == 1:
        resampled = _resample_streamlines(streamlines, nb_points)
    else:
        chunk_size = 20000
        resampled = resample_streamlines_parallel(streamlines, nb_points, n_jobs=n_jobs)
    resample_sec = time.time() - t0
    log_mem("after resample")
    print(f"resampling done in {resample_sec:.2f}s")
    log_mem("after deleting original streamlines")
    log_mem("before embedding")
    t0 = time.time()

    n_prototypes = int(cfg.get("n_prototypes", 64))
    if n_prototypes > n_streamlines:
        n_prototypes = n_streamlines
    print(f"computing embedding ({distance}, {n_prototypes} prototypes) with {n_jobs} workers...")
    t0 = time.time()
    embedding = compute_embedding(
        resampled,
        distance_name=distance,
        num_prototypes=n_prototypes,
        prototype_policy=cfg.get("prototype_policy", "sff"),
        size_limit=int(cfg.get("size_limit", 5_000_000)),
        n_jobs=n_jobs,
        verbose=verbose,
    )
    embed_sec = time.time() - t0
    log_mem("after embedding")
    print(f"embedding done in {embed_sec:.2f}s")
    
    log_mem("before clustering")
    print(f"embedding shape: {embedding.shape}, dtype={embedding.dtype}")
    t0 = time.time()
    km = fit_mbkm(
        embedding,
        n_clusters=n_clusters_final,
        batch_size=int(cfg.get("batch_size", 1000)),
        n_init=int(cfg.get("n_init", 10)),
        max_no_improvement=int(cfg.get("max_no_improvement", 5)),
        random_state=int(cfg.get("random_state", 0)),
        verbose=verbose,
    )

    cluster_sec = time.time() - t0
    
    log_mem("after clustering")
    print(f"clustering done in {cluster_sec:.2f}s")
    
    clusters, medoids = select_medoids_from_embedding(embedding, km.labels_, km.cluster_centers_)
    
    print("selecting medoids...")

    merged_medoid_path = os.path.join(out_medoids, "track.trk")

    print("saving merged medoids...")
    
    save_merged_medoids(streamlines, medoids, reference, merged_medoid_path)

    if cfg.get("save_individual_medoids", False):
        save_individual_medoids(streamlines, medoids, reference, os.path.join(out_medoids, "medoids"))
    if cfg.get("save_individual_clusters", False):
        save_individual_clusters(streamlines, clusters, reference, os.path.join(out_medoids, "clusters"))

    if cfg.get("save_labels", True):
        np.savez_compressed(
            os.path.join(out_labels, "labels.npz"),
            labels=km.labels_.astype(np.int32),
            medoids=medoids.astype(np.int32),
            cluster_centers=km.cluster_centers_.astype(np.float32),
        )

    summary = {
        "fold_id": cfg.get("fold_id", "unspecified"),
        "track": track_path,
        "reference": reference_path,
        "distance": distance,
        "nb_points": nb_points,
        "n_prototypes": n_prototypes,
        "n_clusters_requested": n_clusters_requested,
        "n_clusters_final": n_clusters_final,
        "input_streamlines": n_streamlines,
        "embedding_shape": list(embedding.shape),
        "embedding_runtime_sec": embed_sec,
        "medoids_path": merged_medoid_path,
        "cluster_stats": cluster_size_stats(km.labels_, n_clusters_final),
        "random_state": int(cfg.get("random_state", 0)),
    }

    if cfg.get("save_summary", True):
        write_json(os.path.join(out_qc, "summary.json"), summary)
    total_sec = time.time() - t_global
    print(f"[app] total runtime: {total_sec:.2f}s")
    return summary
