#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 17:05:44 2021

#########################################################################################################################
#########################################################################################################################
###################                                                                                   ###################
################### title:                      tractography library                                  ###################
###################                                                                                   ###################
###################  description:                                                                     ###################
###################  version:        0.6.8                                                             ###################
###################  notes:          need to Install: py modules - nibabel, dipy,                      ###################
###################                                                                                   ###################             
###################  bash version:   tested on GNU bash, version 5.2.21                               ###################
###################                                                                                   ###################
################### autor: gamorosino                                                                 ###################
###################                                                                                   ###################
#########################################################################################################################
#########################################################################################################################
###################                                                                                   ###################
################### update:                                                                            ###################
###################                                                                                   ###################
###################   Add streaming-capable tractogram editing utilities                              ###################
###################                                                                                   ###################
###################   - Implemented new track_edit_stream() pipeline in tracklib.py:                  ###################
###################       • Fully streaming TCK loader (mmap-based, chunked processing)               ###################
###################       • Parallel streamline loading and ROI filtering                             ###################
###################       • Memory-safe design for tractograms >400GB                                 ###################
###################       • Support for both .tck and .trk outputs                                    ###################
###################       • Automatic header creation & incremental append modes                      ###################
###################       • Fallbacks for systems without numba                                       ###################
###################                                                                                   ###################
###################   - Updated classic track_edit() API:                                             ###################
###################       • Unified job control via --jobs                                             ###################
###################       • Parallel filtering & conditional parallel loading                          ###################
###################       • Improved ROI mask union logic & endpoint checks                            ###################
###################                                                                                   ###################
###################   - Improved robustness in TCK parsing:                                            ###################
###################       • Safe END-of-header detection                                               ###################
###################       • Correct handling of "file:" offsets                                        ###################
###################       • Streamline boundary detection via NaN delimiters                           ###################
###################                                                                                   ###################
###################   - Introduced low-memory TRK/TCK writers:                                         ###################
###################       • _append_streamlines_to_tck()                                               ###################
###################       • _append_streamlines_to_trk()                                               ###################
###################                                                                                   ###################
###################   These additions enable efficient filtering of extremely large                    ###################
###################   tractograms (hundreds of millions of streamlines) while keeping                 ###################
###################   memory usage bounded and supporting multi-core execution.                        ###################
###################                                                                                   ###################
#########################################################################################################################
#########################################################################################################################


@author: gamorosino
"""



import sys
import nibabel as nib
import numpy as np
from dipy.tracking.vox2track import streamline_mapping
from dipy.io.streamline import  save_tractogram, load_tractogram
from dipy.io.stateful_tractogram import Space, StatefulTractogram
import dipy.io.streamline
import argparse
from nibabel.streamlines import Field
from nibabel.orientations import aff2axcodes
from dipy.tracking.streamline import set_number_of_points
from dipy.tracking.metrics import length as streamline_length
import scipy.interpolate as si
from dipy.segment.bundles import bundles_distances_mam,bundles_distances_mdf 
from dipy.segment.clustering import QuickBundles
from dipy.align.streamlinear import StreamlineLinearRegistration
from dipy.tracking.streamline import Streamlines
from nibabel import load as load_nifti
import os
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree

#try:
#	from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map
#except:
#	pass



def load_mask(mask_path):
    img = nib.load(mask_path)
    data = np.asanyarray(img.get_fdata()) > 0
    return data, img.affine

def point_in_mask(point, mask, affine):
    """Return True if a point (in world space) lies inside a binary mask."""
    ijk = np.round(nib.affines.apply_affine(np.linalg.inv(affine), point)).astype(int)
    if np.any(ijk < 0) or np.any(ijk >= mask.shape):
        return False
    return mask[tuple(ijk)]

def streamline_hits_roi(streamline, mask, affine):
    """Check if streamline intersects mask (in voxel coordinates)."""
    for p in streamline:
        if point_in_mask(p, mask, affine):
            return True
    return False

def streamline_ends_in_roi(streamline, mask, affine):
    """Check if both endpoints are inside ROI."""
    return point_in_mask(streamline[0], mask, affine) and point_in_mask(streamline[-1], mask, affine)


# ===============================================================
# 1. LOADING 
# ===============================================================

def load_tracks(in_file, roi_includes, roi_excludes, reference=None):
    # ---- reference detection for .tck ----
    if in_file.endswith(".tck") and reference is None:
        if roi_includes:
            reference = roi_includes[0]
            print(f"Using ROI {reference} as reference space for .tck")
        elif roi_excludes:
            reference = roi_excludes[0]
            print(f"Using ROI {reference} as reference space for .tck")
        else:
            raise ValueError(
                f"No reference or ROI provided for {os.path.basename(in_file)}. "
                "Cannot infer space for .tck tractogram."
            )

    # ---- tractogram loading ----
    if in_file.endswith(".trk"):
        sft = load_tractogram(in_file, 'same', to_space=Space.RASMM)
    else:
        sft = load_tractogram(in_file, reference, to_space=Space.RASMM)

    streamlines = Streamlines(sft.streamlines)

    if hasattr(sft, "affine_to_rasmm"):
        tract_affine = sft.affine_to_rasmm
    else:
        tract_affine = sft.affine

    return sft, streamlines, tract_affine


# ===============================================================
# 2. ROI MASKS + FILTERING HELPERS
# ===============================================================

def _xyz3(p):
    p = np.asarray(p)
    return p[..., :3] if p.shape[-1] >= 3 else p


def _voxel_ijk(point_xyz, affine):
    xyz = _xyz3(point_xyz)
    return np.round(
        nib.affines.apply_affine(np.linalg.inv(affine), xyz)
    ).astype(np.int64)


def point_in_mask(point, mask, affine):
    ijk = _voxel_ijk(point, affine)
    if ijk.ndim == 1:
        # single point
        if np.any(ijk < 0) or np.any(ijk >= np.array(mask.shape[:3])):
            return False
        return bool(mask[tuple(ijk)])
    else:
        valid = np.all((ijk >= 0) & (ijk < np.array(mask.shape[:3])), axis=-1)
        return bool(np.any(mask[tuple(ijk[valid].T)])) if np.any(valid) else False


def streamline_hits_roi(sl, mask, affine):
    return any(point_in_mask(p, mask, affine) for p in sl)


def streamline_ends_in_roi(sl, mask, affine):
    return (point_in_mask(sl[0], mask, affine) and
            point_in_mask(sl[-1], mask, affine))


def streamline_one_end_in_roi(sl, mask, affine):
    return (point_in_mask(sl[0], mask, affine) or
            point_in_mask(sl[-1], mask, affine))


# ===============================================================
# 3. FILTERING (PARALLELIZABLE)
# ===============================================================

def filter_streamlines_chunk(
    chunk,
    masks_inc,
    aff_inc,
    masks_exc,
    aff_exc,
    union_inc_mask,
    union_inc_aff,
    ends_only,
    one_end,
):
    kept = []

    for sl in chunk:
        include_flag = True
        exclude_flag = False

        # ---- inclusion ----
        if masks_inc:
            if ends_only:
                if union_inc_mask is not None:
                    include_flag = streamline_ends_in_roi(sl, union_inc_mask, union_inc_aff)
                else:
                    include_flag = any(streamline_ends_in_roi(sl, m, a)
                                       for m, a in zip(masks_inc, aff_inc))

            elif one_end:
                if union_inc_mask is not None:
                    include_flag = streamline_one_end_in_roi(sl, union_inc_mask, union_inc_aff)
                else:
                    include_flag = any(streamline_one_end_in_roi(sl, m, a)
                                       for m, a in zip(masks_inc, aff_inc))

            else:
                include_flag = any(streamline_hits_roi(sl, m, a)
                                   for m, a in zip(masks_inc, aff_inc))

        # ---- exclusion ----
        if masks_exc:
            exclude_flag = any(streamline_hits_roi(sl, m, a)
                               for m, a in zip(masks_exc, aff_exc))

        # ---- decision ----
        if include_flag and not exclude_flag:
            kept.append(sl)

    return kept


# ===============================================================
# 4. SAVE
# ===============================================================

def write_tracks(sft_loader_obj, streamlines, out_file, reference=None):
    """
    Save streamlines using DIPY StatefulTractogram.
    If `reference` is provided (path to NIfTI), use its affine + shape.
    Otherwise fall back to 1×1×1 identity reference.
    """
    import nibabel as nib
    from dipy.io.stateful_tractogram import StatefulTractogram, Space
    from dipy.io.streamline import save_tractogram
    import numpy as np

    # -----------------------------------------------
    # Determine the affine to use
    # -----------------------------------------------
    if reference is not None:
        ref_img = nib.load(reference)
        affine = ref_img.affine
        reference_img = ref_img
    else:
        # fallback if sft_loader has affine
        if hasattr(sft_loader_obj, "affine_to_rasmm"):
            affine = sft_loader_obj.affine_to_rasmm
        else:
            affine = np.eye(4)

        # identity 1×1×1 reference
        reference_img = nib.Nifti1Image(
            np.zeros((1, 1, 1), dtype=np.uint8),
            affine
        )

    # -----------------------------------------------
    # Build StatefulTractogram in RASMM
    # -----------------------------------------------
    sft = StatefulTractogram(
        Streamlines(streamlines),
        reference_img,
        space=Space.RASMM
    )

    save_tractogram(sft, out_file, bbox_valid_check=False)



# ===============================================================
# 5. TOP-LEVEL track_edit() — now supports n_jobs
# ===============================================================

def _filter_wrapper(args):
    chunk, masks_inc, aff_inc, masks_exc, aff_exc, union_inc_mask, union_inc_aff, ends_only, one_end = args
    from .tracklib import filter_streamlines_chunk  # or correct path if needed
    return filter_streamlines_chunk(
        chunk,
        masks_inc, aff_inc,
        masks_exc, aff_exc,
        union_inc_mask, union_inc_aff,
        ends_only, one_end
    )

def _wrap_filter(args):
    (
        chunk,
        masks_inc, aff_inc,
        masks_exc, aff_exc,
        union_inc_mask, union_inc_aff,
        ends_only, one_end
    ) = args

    # call your existing function
    return filter_streamlines_chunk(
        chunk,
        masks_inc, aff_inc,
        masks_exc, aff_exc,
        union_inc_mask, union_inc_aff,
        ends_only, one_end
    )

# ===============================================================
# 6. LOAD TRACKS CONDITIONAL
# ===============================================================

# ===============================================================
# 6. LOAD TRACKS CONDITIONAL  (UPDATED VERSION)
# ===============================================================

def load_tracks_conditional(
    in_file,
    roi_includes=None,
    roi_excludes=None,
    reference=None,
    n_jobs=1,
    verbose=False
):
    """
    If n_jobs == 1 → use original serial load_tracks()
    If n_jobs > 1  → use fast mmap-safe parallel loader (TRK/TCK)
    """

    ext = os.path.splitext(in_file)[1].lower()

    # -------------------------------
    # SERIAL PATH — existing DIPY loader
    # -------------------------------
    if n_jobs == 1:
        if verbose:
            print("[LOAD] Serial DIPY loader")
        return load_tracks(in_file, roi_includes, roi_excludes, reference)

    # -------------------------------
    # PARALLEL PATH — TRK/TCK loaders
    # -------------------------------
    if verbose:
        print(f"[LOAD] Parallel loader activated ({n_jobs} workers)")

    # ----------- TRK -----------
    if ext == ".trk":
        streamlines, header, lengths, idxs = load_streamlines_parallel(
            in_file,
            idxs=None,
            apply_affine=True,
            container="list",
            n_jobs=n_jobs,
            verbose=verbose
        )

        affine = nib.streamlines.trk.get_affine_trackvis_to_rasmm(header)

        class FakeSFT:
            def __init__(self, affine, header):
                self.affine_to_rasmm = affine
                self.header = header

        sft = FakeSFT(affine, header)
        return sft, streamlines, affine

    # ----------- TCK -----------
    elif ext == ".tck":
        streamlines, affine = load_tck_parallel(
            in_file,
            n_jobs=n_jobs,
            apply_affine=True,
            verbose=verbose
        )

        class FakeSFT:
            def __init__(self, affine):
                self.affine_to_rasmm = affine

        sft = FakeSFT(affine)
        return sft, streamlines, affine

    else:
        raise ValueError(f"Unsupported tract format: {ext}")

# ===============================================================
# 6. Track EDIT
# ===============================================================


from nibabel.streamlines import TrkFile

def _endpoint_region(point, masks, affines):
    """
    Return the index of the FIRST ROI that contains the endpoint.
    If no ROI matches → return None.
    """
    import numpy as np
    import nibabel as nib

    for idx, (mask, A) in enumerate(zip(masks, affines)):
        # Transform RASMM → voxel
        ijk = nib.affines.apply_affine(np.linalg.inv(A), point)
        ijk = np.round(ijk).astype(int)

        # Bounds check
        if (0 <= ijk[0] < mask.shape[0] and
            0 <= ijk[1] < mask.shape[1] and
            0 <= ijk[2] < mask.shape[2]):
            if mask[ijk[0], ijk[1], ijk[2]]:
                return idx

    return None


def _create_empty_tck(path, affine):
    """
    Creates a valid TCK header with correct byte offset.
    """
    # Build transform lines (4 lines required by MRtrix)
    T = affine
    transform_lines = [
        "transform: " + " ".join(map(str, T[0])),
        "transform: " + " ".join(map(str, T[1])),
        "transform: " + " ".join(map(str, T[2])),
        "transform: 0 0 0 1"
    ]

    # Build header text (we will compute OFFSET next)
    header_lines = [
        "mrtrix tracks",
        "datatype: Float32LE",
        "count: -1",
    ] + transform_lines

    # Join header + END
    header = "\n".join(header_lines) + "\nEND\n"

    # Compute offset: number of bytes in header
    offset = len(header.encode("utf-8"))

    # Now rewrite header with correct offset
    header = header.replace("count: -1", "count: -1") \
                   .replace("END",
                            f"file: . {offset}\nEND")

    # Write to file
    with open(path, "wb") as f:
        f.write(header.encode("utf-8"))

def _append_streamlines_to_tck(path, streamlines):
    """
    Appends streamlines to an existing TCK by writing binary data at end.
    """
    with open(path, "ab") as f:
        for sl in streamlines:
            sl = np.asarray(sl, dtype=np.float32)
            f.write(sl.tobytes())
            # delimiter NaN row
            f.write(np.array([np.nan, np.nan, np.nan], dtype=np.float32).tobytes())




def _append_streamlines_to_trk(out_file, new_streamlines, affine):
    """
    Append streamlines to an existing TRK file safely.
    If file doesn't exist or is empty: create clean TRK header.
    """

    # Case 1 — file does not exist yet → create it
    if not os.path.exists(out_file) or os.path.getsize(out_file) < 1000:
        hdr = TrkFile.create_empty_header()
        hdr["nb_streamlines"] = 0
        hdr["voxel_to_rasmm"] = affine
        trk = TrkFile(Streamlines(new_streamlines), header=hdr)
        trk.save(out_file)
        return

    # Case 2 — append mode
    trk_old = TrkFile.load(out_file)
    old_sl = list(trk_old.streamlines)

    merged = Streamlines(old_sl + list(new_streamlines))

    hdr = trk_old.header.copy()
    hdr["nb_streamlines"] = len(merged)

    new_trk = TrkFile(merged, header=hdr)
    new_trk.save(out_file)


def _filter_chunk_with_order(args):
    (streamlines,
     masks_inc, aff_inc,
     masks_exc, aff_exc,
     ends_only, one_end,
     include_order) = args

    out = []
    n_roi = len(masks_inc)

    for sl in streamlines:

        p0 = sl[0]
        p1 = sl[-1]

        # EXCLUDE masks first
        excluded = False
        for m, A in zip(masks_exc, aff_exc):
            ijk = nib.affines.apply_affine(np.linalg.inv(A), p0)
            ijk = np.round(ijk).astype(int)
            if (0 <= ijk[0] < m.shape[0] and
                0 <= ijk[1] < m.shape[1] and
                0 <= ijk[2] < m.shape[2] and
                m[ijk[0], ijk[1], ijk[2]]):
                excluded = True
                break

            ijk = nib.affines.apply_affine(np.linalg.inv(A), p1)
            ijk = np.round(ijk).astype(int)
            if (0 <= ijk[0] < m.shape[0] and
                0 <= ijk[1] < m.shape[1] and
                0 <= ijk[2] < m.shape[2] and
                m[ijk[0], ijk[1], ijk[2]]):
                excluded = True
                break

        if excluded:
            continue

        # ORDERED ROI ENDPOINTS CHECK
        if include_order:
            r0 = _endpoint_region(p0, masks_inc, aff_inc)
            r1 = _endpoint_region(p1, masks_inc, aff_inc)

            # must match exactly ROI0 → ROI1 → ROI2…
            if r0 is None or r1 is None:
                continue
            if r0 >= r1:      # wrong direction OR same ROI
                continue
            out.append(sl)
            continue

        # CLASSIC ends_only / one_end logic
        in0 = _endpoint_region(p0, masks_inc, aff_inc) is not None
        in1 = _endpoint_region(p1, masks_inc, aff_inc) is not None

        if ends_only:
            if in0 and in1:
                out.append(sl)
        elif one_end:
            if in0 or in1:
                out.append(sl)
        else:
            # non-endpoint filtering → accept all
            out.append(sl)

    return out


def track_edit_stream(
    in_file,
    roi_includes=None,
    roi_excludes=None,
    out_file="filtered.tck",
    ends_only=False,
    one_end=False,
    reference=None,
    n_jobs=1,
    chunk_size=500000   # number of streamlines per streaming-chunk
):
    import numpy as np
    from multiprocessing import Pool, cpu_count

    # ------------------------------
    # ROI masks
    # ------------------------------
    roi_includes = roi_includes or []
    roi_excludes = roi_excludes or []

    def load_mask(path):
        img = nib.load(path)
        return (img.get_fdata() > 0), img.affine

    masks_inc, aff_inc = zip(*(load_mask(r) for r in roi_includes)) if roi_includes else ([], [])
    masks_exc, aff_exc = zip(*(load_mask(r) for r in roi_excludes)) if roi_excludes else ([], [])

    # union mask
    union_inc_mask = None
    union_inc_aff = None
    if (ends_only or one_end) and masks_inc:
        if all(np.allclose(aff_inc[0], a) for a in aff_inc):
            union_inc_aff = aff_inc[0]
            union_inc_mask = np.zeros_like(masks_inc[0], dtype=bool)
            for m in masks_inc:
                union_inc_mask |= m

    # ------------------------------
    # multiprocessing
    # ------------------------------
    if n_jobs == -1:
        n_jobs = cpu_count()
    n_jobs = max(1, min(n_jobs, 32))  # safety cap

    print(f"[STREAM] Using {n_jobs} workers")
    print(f"[STREAM] Chunk size = {chunk_size}")

    # ------------------------------
    # Scan TCK boundaries (first pass)
    # ------------------------------
    header, header_end = _parse_tck_header(in_file)

    f = open(in_file, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    mm.seek(header_end)
    raw = np.ndarray(
        shape=((len(mm) - header_end)//4,),
        dtype=np.float32,
        buffer=mm,
        offset=header_end
    )
    points = raw.reshape(-1, 3)

    print("[STREAM] Finding streamline boundaries…")
    nan_idx = _find_nan_blocks(points)

    starts = []
    ends = []
    prev = 0
    for idx in nan_idx:
        if idx > prev:
            starts.append(prev)
            ends.append(idx)
        prev = idx + 1

    starts = np.array(starts)
    ends   = np.array(ends)
    num_sl = len(starts)

    print(f"[STREAM] Total streamlines: {num_sl:,}")

    # affine
    if "transform" in header:
        affine = np.fromstring(header["transform"], sep=" ").reshape(4,4).astype(np.float32)
    else:
        affine = np.eye(4, dtype=np.float32)

    total_kept = 0


    # ------------------------------
    # Prepare output file once (BEFORE LOOP)
    # ------------------------------
    ext_out = os.path.splitext(out_file)[1].lower()

    if ext_out == ".tck":
        _create_empty_tck(out_file, affine)
        # nothing to accumulate
        filtered_all = None

    elif ext_out == ".trk":
        # Start with empty TRK
        if os.path.exists(out_file):
            os.remove(out_file)
        filtered_all = []   # <<< IMPORTANT

    else:
        raise ValueError("Unsupported output format")



    # ------------------------------
    # MAIN STREAMING LOOP
    # ------------------------------
    for i in range(0, num_sl, chunk_size):

        sl_st = starts[i:i+chunk_size]
        sl_en = ends[i:i+chunk_size]

        # -------- load chunk in parallel --------
        args_load = [
            (in_file, header_end, sc, ec, affine, True)
            for sc, ec in zip(np.array_split(sl_st, n_jobs),
                              np.array_split(sl_en, n_jobs))
        ]

        if n_jobs == 1:
            parts = [_load_chunk_tck(args_load[0])]
        else:
            with Pool(n_jobs) as p:
                parts = p.map(_load_chunk_tck, args_load)

        chunk_sl = [sl for sub in parts for sl in sub]

        # -------- filter in parallel --------
        args_filt = [
            (subset,
             masks_inc, aff_inc,
             masks_exc, aff_exc,
             union_inc_mask, union_inc_aff,
             ends_only, one_end)
            for subset in np.array_split(chunk_sl, n_jobs)
        ]

        if n_jobs == 1:
            filt_parts = [_wrap_filter(args_filt[0])]
        else:
            with Pool(n_jobs) as p:
                filt_parts = p.map(_wrap_filter, args_filt)

        filtered_chunk = [sl for sub in filt_parts for sl in sub]
        total_kept += len(filtered_chunk)

        # -------- append or accumulate --------
        if ext_out == ".tck":
            if filtered_chunk:
                _append_streamlines_to_tck(out_file, filtered_chunk)

        elif ext_out == ".trk":
            if filtered_chunk:
                filtered_all.extend(filtered_chunk)

        print(f"[STREAM] {min(i+chunk_size, num_sl):,}/{num_sl:,}  → kept {total_kept:,}")

    # ------------------------------
    # close mmap/file
    # ------------------------------
    mm.close()
    f.close()

    # ------------------------------
    # Final TRK write (TCK already written)
    # ------------------------------
    if ext_out == ".trk":
        write_tracks(
            sft_loader_obj=type("Fake", (), {"affine_to_rasmm": affine}),
            streamlines=filtered_all,
            out_file=out_file,
            reference=reference
        )

    print(f"\n[STREAM] COMPLETED — total kept: {total_kept:,}")
    print(f"Output saved to {out_file}")


def track_edit(
    in_file,
    roi_includes=None,
    roi_excludes=None,
    out_file="filtered.trk",
    ends_only=False,
    one_end=False,
    include_order=False,   # <<< NEW ARGUMENT
    reference=None,
    n_jobs=1
):

    import numpy as np
    from multiprocessing import Pool, cpu_count

    roi_includes = roi_includes or []
    roi_excludes = roi_excludes or []

    # Load input
    sft, streamlines, tract_affine = load_tracks_conditional(
        in_file,
        roi_includes=roi_includes,
        reference=reference,
        n_jobs=n_jobs,
        verbose=True
    )

    # Load masks
    def load_mask(path):
        img = nib.load(path)
        return (img.get_fdata() > 0), img.affine

    masks_inc, aff_inc = zip(*(load_mask(r) for r in roi_includes)) if roi_includes else ([], [])
    masks_exc, aff_exc = zip(*(load_mask(r) for r in roi_excludes)) if roi_excludes else ([], [])

    # Prepare jobs
    if n_jobs == -1:
        n_jobs = cpu_count()
    n_jobs = max(1, n_jobs)

    chunks = np.array_split(streamlines, n_jobs)

    args = [
        (chunk,
         masks_inc, aff_inc,
         masks_exc, aff_exc,
         ends_only, one_end,
         include_order)
        for chunk in chunks
    ]

    if n_jobs == 1:
        results = [_filter_chunk_with_order(args[0])]
    else:
        with Pool(n_jobs) as p:
            results = p.map(_filter_chunk_with_order, args)

    filtered = [sl for lst in results for sl in lst]
    print(f"→ {len(filtered)} kept / {len(streamlines)} total")

    write_tracks(sft, filtered, out_file, reference=reference)
    print(f"Saved: {out_file}")



def similarity_distance(distance,lambda_=0.005):
    return np.exp(-1*lambda_*distance**2)

def loadTrk(filename):
    data = nib.streamlines.load(filename)
    s = data.streamlines
    aff = data.affine
    header = data.header
    return s,aff,header

def compress_with_terminations(streamlines, type='array', add_midpoint=False):
    compressed_streamlines = []
    for sl in streamlines:
        if type=='array':
            if add_midpoint:
                mid = len(sl)/2
                compressed_sl = np.concatenate(
                                    (sl[0],sl[mid],sl[-1]), axis=0)
                assert len(compressed_sl) == 9
            else:
                compressed_sl = np.concatenate((sl[0],sl[-1]), axis=0)
                assert len(compressed_sl) == 6
        elif type=='tuple':
            if add_midpoint:
                mid = len(sl)/2
                compressed_sl = (sl[0][0], sl[0][1], sl[0][2],
                                sl[mid][0], sl[mid][1], sl[mid][2],
                                sl[-1][0], sl[-1][1], sl[-1][2])
            else:
                compressed_sl = (sl[0][0], sl[0][1], sl[0][2],
                                sl[-1][0], sl[-1][1], sl[-1][2])
        compressed_streamlines.append(compressed_sl)
    return compressed_streamlines

def extractTerminations(track_filename,structural_filename=None,affine=None,plot=False):
    
    
    if structural_filename is not None:
    
        func_nib=nib.load(structural_filename)
        affine=func_nib.affine

    track,affine_,_ = loadTrk(track_filename)
    
    #track, aff = loadTrk(track_filename)
    
    terminations=np.array(compress_with_terminations(track))
    startpoints=terminations[:,0:3]
    endpoints=terminations[:,3:]
    
    terminations_=[]
    for i in range(terminations.shape[0]):
        terminations_.append(np.array([startpoints[i,:],endpoints[i,:]]))
    
    if affine is None:
        affine = affine_
    
    stream_map = streamline_mapping(terminations_,affine=affine) #,mapping_as_streamlines=True)
    
    
    stream = dict()
    
    Points=np.zeros([len(stream_map.keys()),3])
    for idx,i in enumerate(stream_map.keys()):
        Points[idx,:]=np.array(i)
        for j in stream_map[i]:
            try:
                list_vox=stream[j]
                list_vox.append(i)
            except KeyError:
                list_vox=[i]
            stream[j] = list_vox
    
    X1=[]
    X2=[]
    Y1=[]
    Y2=[]
    Z1=[]
    Z2=[]
    
    
    vox_0=np.min(Points,axis=0)
    #vox_0=[0,0,0]
    vox=[]

    for idx,j in enumerate(stream.keys()):
        
        vox_t=stream[idx]
        if len(vox_t) < 2:
            sys.stderr.write(f"Warning: Streamline {idx} starts and ends in the same voxel: {vox[0]}\n")
            vox=[list(vox_t[0]),list(vox_t[0])]        
        else:
            vox=[list(vox_t[0]),list(vox_t[1])]
        #print('vox 1:'+str(vox))
        #vox.append(list(vox_t[1]))
        #print('vox 2:'+str(vox))
        
        dist_1 = np.linalg.norm(vox_0-np.array(vox[0][:]))
        dist_2 = np.linalg.norm(vox_0-np.array(vox[1][:]))
        
        if dist_1 > dist_2:
            temp = vox[0][:]
            vox[0][:] = vox[1][:]
            vox[1][:] = temp
        

        X1.append(vox[0][0])
        Y1.append(vox[0][1])
        Z1.append(vox[0][2])
        
        X2.append(vox[1][0])
        Y2.append(vox[1][1])
        Z2.append(vox[1][2])        
    
    
        if plot:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.scatter3D(X1,Y1,Z1, color='green')
            ax.scatter3D(X2,Y2,Z2, color='red')

    return X1,Y1,Z1,X2,Y2,Z2 


def extractDensityTerminations(track_filename,structural_filename=None,output_filename=None,affine=None,plot=False):
    
    
    if structural_filename is not None:
    
        func_nib=nib.load(structural_filename)
        affine=func_nib.affine

    track,affine_,_ = loadTrk(track_filename)
    
    #track, aff = loadTrk(track_filename)
    
    terminations=np.array(compress_with_terminations(track))
    startpoints=terminations[:,0:3]
    endpoints=terminations[:,3:]
    
    terminations_=[]
    for i in range(terminations.shape[0]):
        terminations_.append(np.array([startpoints[i,:],endpoints[i,:]]))
    
    if affine is None:
        affine = affine_
    
    stream_map = streamline_mapping(terminations_,affine=affine) 
    if output_filename is not None:
        nii_output = nib.Nifti1Image(Volume, affine=affine, header=header)
        nii_output.to_filename(output_filename)
    return stream_map

def saveTrackTerminations(track_filename,output_filename,structural_filename=None):

    affine=None
    header=None
    
    if structural_filename is not None:
        
        func_nib=nib.load(structural_filename)
        affine = func_nib.affine
        header = func_nib.header
        dims=header.get_data_shape()[:3]
        
    else:
        data = nib.streamlines.load(track_filename)
        dims = list(data.header['dimensions'])
        
    Volume = np.zeros(dims)    
    
    X1,Y1,Z1,X2,Y2,Z2 =  extractTerminations(track_filename,structural_filename=None,affine=affine,plot=False)

    Volume[X1,Y1,Z1] = 1
    Volume[X2,Y2,Z2] = 2  
    
    nii_output = nib.Nifti1Image(Volume, affine=affine, header=header)
    nii_output.to_filename(output_filename)


def saveTerminationsTrackfile(track_filename,output_filename,structural_filename=None):
    
    track,affine_,header_ = loadTrk(track_filename)
    if structural_filename is not None:
    
        func_nib=nib.load(structural_filename)
        affine=func_nib.affine
        header_ = None

    
    
    #track, aff = loadTrk(track_filename)
    
    terminations=np.array(compress_with_terminations(track))

    return saveTrackDipy(terminations, output_filename,
                    structural_filename=structural_filename,
                    remove_invalid_streamlines=True,
                    header=header_,
                    bbox_valid_check=True)

def saveTrackDipy(track, output_file,
                  structural_filename=None,
                  remove_invalid_streamlines=True,
                  header=None,
                  bbox_valid_check=True):

    # --------------------------------------------------
    # Resolve reference image
    # --------------------------------------------------
    if structural_filename is not None:
        struct_nib = nib.load(structural_filename)

    elif header is not None:
        struct_nib = header

    else:
        raise ValueError(
            "saveTrackDipy(): no reference provided.\n"
            "You must supply either:\n"
            "  - structural_filename (path to NIfTI), or\n"
            "  - header/reference object (Nifti1Image or compatible).\n"
            f"Output file: {output_file}"
        )

    # --------------------------------------------------
    # Build StatefulTractogram
    # --------------------------------------------------
    track_sft = StatefulTractogram(track, struct_nib, Space.RASMM)

    if remove_invalid_streamlines:
        track_sft.remove_invalid_streamlines()

    print(f"bbox_valid_check: {bbox_valid_check}")

    save_tractogram(
        track_sft,
        output_file,
        bbox_valid_check=bbox_valid_check
    )

     
def vtk2trk(input_file, output_file, structural_filename=None):

	saveTrackDipy(dipy.io.streamline.load_vtk_streamlines(input_file), output_file, structural_filename)
	

def build_argparser_tck2trk():
    DESCRIPTION = "Convert tractograms (TCK -> TRK)."
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument('anatomy', help='reference anatomy (.nii|.nii.gz.')
    p.add_argument('tractograms', metavar='tractogram', nargs="+", help='list of tractograms (.tck).')
    p.add_argument('-f', '--force', action="store_true", help='overwrite existing output files.')
    return p

def tck2trk(args=None):	
    parser = build_argparser_tck2trk()
    if args is not None:
        args = parser.parse_args(args)
    else:
        args = parser.parse_args()

    try:
        nii = nib.load(args.anatomy)
    except:
        parser.error("Expecting anatomy image as first agument.")

    for tractogram in args.tractograms:
        if nib.streamlines.detect_format(tractogram) is not nib.streamlines.TckFile:
            print("Skipping non TCK file: '{}'".format(tractogram))
            continue

        output_filename = tractogram[:-4] + '.trk'
        if os.path.isfile(output_filename) and not args.force:
            print("Skipping existing file: '{}'. Use -f to overwrite.".format(output_filename))
            continue

        header = {}
        header[Field.VOXEL_TO_RASMM] = nii.affine.copy()
        header[Field.VOXEL_SIZES] = nii.header.get_zooms()[:3]
        header[Field.DIMENSIONS] = nii.shape[:3]
        header[Field.VOXEL_ORDER] = "".join(aff2axcodes(nii.affine))

        tck = nib.streamlines.load(tractogram)
        nib.streamlines.save(tck.tractogram, output_filename, header=header)
	



def build_argparser_trk2tck():
    DESCRIPTION = "Convert tractograms (TRK -> TCK)."
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument('tractograms', metavar='bundle', nargs="+", help='list of tractograms.')
    p.add_argument('-f', '--force', action="store_true", help='overwrite existing output files.')
    return p


def trk2tck(args=None):
    parser = build_argparser_trk2tck()
    if args is not None:
        args = parser.parse_args(args)
    else:
        args = parser.parse_args()
    for tractogram in args.tractograms:
        if nib.streamlines.detect_format(tractogram) is not nib.streamlines.TrkFile:
            print("Skipping non TRK file: '{}'".format(tractogram))
            continue

        output_filename = tractogram[:-4] + '.tck'
        if os.path.isfile(output_filename) and not args.force:
            print("Skipping existing file: '{}'. Use -f to overwrite.".format(output_filename))
            continue

        trk = nib.streamlines.load(tractogram)
        nib.streamlines.save(trk.tractogram, output_filename)




def get_average_streamline(track_filename, output_file, structural_filename, N_points=32):
    
    # %%load data

   
    track,track_aff,track_header = loadTrk(track_filename)
 

    #%% Resample Streamlines
    print("setting the same number of points...")
    track=set_number_of_points( track , N_points )



    #%% Mean
    num_streamlines=len(track)
    mean_streamline=[]
    print("number of streamlines: "+str(num_streamlines))



    #%flip streamlines
    streamline_0=track[0]
    length = np.zeros(len(track))
    for i, streamline in enumerate(track):
        dist=np.linalg.norm((streamline_0 - streamline).reshape(-1, N_points*3), axis=1)
        dist_flip=np.linalg.norm((streamline_0 - streamline[::-1,:]).reshape(-1, N_points*3), axis=1)
        idx_flip=dist_flip<dist

        if idx_flip:

            streamline = streamline[::-1,:]
        track[i] = streamline
        length[i]=streamline_length(streamline)

    length_max=np.array(length)>(0.9*np.max(np.array(length)))
    track_ = track[length_max]
    track = track_
    
    #%compute the mean
    
    for point in range(len(track[0])):
        #print(point)
        trip=[]
        for i, streamline in enumerate(track):
            trip.append(streamline[point])
            #print(streamline[point])
        mean_point=np.mean(trip,axis=0)
        #print(mean_point)
        mean_streamline.append(mean_point)
    mean_streamline = [mean_streamline]    
    mean_streamline=np.array(mean_streamline)


    saveTrackDipy(mean_streamline, output_file, structural_filename=structural_filename, remove_invalid_streamlines=False)
    return mean_streamline

def get_streamlines_count(track_filename,output=None,smooth_density=True):
    track,track_aff,track_header = loadTrk(track_filename)
    sft = load_tractogram(track_filename, 'same',
                              bbox_valid_check=False)
    sft.to_vox()
    sft.to_corner()
    transformation, dimensions, _, _ = sft.space_attributes
    streamlines_count_array = streamlines_count(track,track_aff,dimensions,stream_map=None,smooth_density=smooth_density)
    if output is not None:
        nib.save(nib.Nifti1Image(streamlines_count_array, transformation),
             output)		
    return streamlines_count_array

def streamlines_count(track,track_aff,dimensions,stream_map=None,smooth_density=True):
    if stream_map is None:
        stream_map = streamline_mapping(track,affine=track_aff)
    streamline_count = np.zeros(dimensions)
    for idx in list(stream_map.keys()):
         streamline_count[idx] = len(stream_map[idx])
    if smooth_density:
        print('smooth density')		
        from scipy import signal
        kernel=gkernel(l=4, sig=2)
        streamline_count = signal.fftconvolve(streamline_count, kernel, mode='same')

    return streamline_count

def gkernel(l=3, sig=2):
    """\
    Gaussian Kernel Creator via given length and sigma
    """

    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy, zz = np.meshgrid(ax, ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)+ np.square(zz)) / np.square(sig))

    return kernel / np.sum(kernel)

def get_core_streamlines(track_filename, perc=0.75, output_file=None, structural_filename=None,smooth_density=True):
    
    # %%load data	
    track,track_aff,track_header = loadTrk(track_filename)
    # %%calculate density map
    print("calculate density map...")
    # --- Safe tractogram loading ---
    if track_filename.endswith(".trk"):
        # .trk files contain their own reference info
        sft = load_tractogram(track_filename, "same",
                            to_space=Space.RASMM,
                            bbox_valid_check=True)
        #track_aff=sft.affine
        print("Affine transformation will be taken directly from the TRK file header.")
    else:
        # for .tck files, use the provided reference (if any)
        if structural_filename is not None:
            ref = nib.load(structural_filename)
        else:
            # fall back to first ROI or raise error
            raise ValueError(
                "Reference image (structural_filename) must be provided for .tck files."
            )
        sft = load_tractogram(track_filename, ref,
                            to_space=Space.RASMM,
                            bbox_valid_check=True)
        track_aff=ref.affine
        print("Affine transformation will be taken from the reference NIfTI image for spatial alignment.")
    print(track_aff)
    #track = sft.streamlines
    num_streamlines=len(track)
    print("number of streamlines: "+str(num_streamlines))
    sft.to_vox()
    sft.to_corner()
    transformation, dimensions, _, _ = sft.space_attributes
    stream_map = streamline_mapping(track,affine=track_aff)
    streamline_count_array = streamlines_count(track,track_aff,dimensions,stream_map,smooth_density=smooth_density)
    #perc = np.percentile(np.unique(streamline_count_array), 75)
    #percent = np.percentile(streamline_count_array[streamline_count_array>0.5*np.max(streamline_count_array)],perc)
    #indices = np.where(streamline_count_array > percent )
    
     
    if perc != 0:
        indices = np.where(streamline_count_array >= perc*np.max(streamline_count_array) )
    else:
        indices = np.where(streamline_count_array > perc*np.max(streamline_count_array) )
    streamlines_to_keep = []
    streamlines_to_keep_dic = dict()
    for i in range(len(indices[0])):
        idx = (indices[0][i],indices[1][i],indices[2][i])
        for element in stream_map[idx]:
                streamlines_to_keep.append(element)
                try:
                      streamlines_to_keep_dic[element] = streamlines_to_keep_dic[element] + 1
                except:
                      streamlines_to_keep_dic[element] =  1
    streamlines_to_keep=np.unique(streamlines_to_keep)
    track = track[streamlines_to_keep]
    num_streamlines=len(track)
    print("number of cores streamlines: "+str(num_streamlines))
    if  output_file is not None:
        saveTrackDipy(track, output_file, structural_filename=structural_filename, remove_invalid_streamlines=False)
 
    return track


def get_bundle_backbone(track_filename, output_file, structural_filename, 
                        N_points=32, perc=0.75, smooth_density=True, 
                        length_thr=0.9,
                        keep_endpoints=False,
                        average_type="mean",
                        endpoint_mode="median",       # "mean", "median", "median_project"
                        representative=False,         # choose closest streamline instead of backbone
                        spline_smooth=None):          # apply spline smoothing at the end
    """
    Compute backbone of a streamline bundle, optionally extract representative streamline,
    and apply optional spline smoothing at the end.

    endpoint_mode:
        "mean"            → endpoints = mean(coords)
        "median"          → endpoints = median(coords)
        "median_project"  → project median point to closest actual streamline endpoint
    """

    # ----------------------------------------------------------
    # 1. Load & preprocess bundle
    # ----------------------------------------------------------
    if perc != 0:
        print("get core streamlines...")
        track = get_core_streamlines(
            track_filename,
            structural_filename=structural_filename,
            perc=perc,
            smooth_density=smooth_density
        )
    else:
        track, track_aff, track_header = loadTrk(track_filename)

    print("Resampling streamlines to N_points =", N_points)
    track = set_number_of_points(track, N_points)

    # ----------------------------------------------------------
    # 2. Orient streamlines consistently
    # ----------------------------------------------------------
    streamline_0 = track[0]
    length = np.zeros(len(track))

    for i, sl in enumerate(track):
        dist      = np.linalg.norm((streamline_0 - sl).reshape(-1, N_points*3), axis=1)
        dist_flip = np.linalg.norm((streamline_0 - sl[::-1]).reshape(-1, N_points*3), axis=1)

        if dist_flip < dist:
            sl = sl[::-1]

        track[i] = sl
        length[i] = streamline_length(sl)

    # keep only long-enough streamlines
    length_mask = length > (length_thr * np.max(length))
    track = track[length_mask]

    # ----------------------------------------------------------
    # 3. Compute backbone (mean/median)
    # ----------------------------------------------------------
    backbone = []
    num_points = len(track[0])

    for p in range(num_points):
        coords = np.array([sl[p] for sl in track])

        # ---- Endpoint logic ----
        if keep_endpoints and (p == 0 or p == num_points - 1):

            if endpoint_mode == "mean":
                agg = np.mean(coords, axis=0)

            elif endpoint_mode == "median":
                agg = np.median(coords, axis=0)

            elif endpoint_mode == "median_project":
                med = np.median(coords, axis=0)
                endpoints = np.array([sl[p] for sl in track])
                d = np.linalg.norm(endpoints - med, axis=1)
                agg = endpoints[d.argmin()]

            else:
                raise ValueError(f"Unknown endpoint_mode: {endpoint_mode}")

        # ---- Interior points ----
        else:
            agg = np.median(coords, axis=0) if average_type == "median" else np.mean(coords, axis=0)

        backbone.append(agg)

    backbone = np.array(backbone)  # shape: (N_points, 3)

    # ----------------------------------------------------------
    # 4. If representative mode → choose closest streamline
    # ----------------------------------------------------------
    if representative:
        print("Selecting representative streamline (closest to backbone).")

        tree = cKDTree(np.array(track).reshape(len(track), -1))
        dist, idx = tree.query(backbone.reshape(-1), k=1)
        rep = track[idx]

        final = rep.copy()

    else:
        final = backbone.copy()

    # ----------------------------------------------------------
    # 5. Apply spline smoothing LAST
    # ----------------------------------------------------------
    if spline_smooth is not None:
        print(f"Applying spline smoothing with s={spline_smooth}")

        tck, u = splprep(final.T, s=float(spline_smooth))
        u_new = np.linspace(0, 1, num_points)
        x, y, z = splev(u_new, tck)
        smoothed = np.vstack([x, y, z]).T

        # preserve endpoints if requested (projected/median already computed)
        if keep_endpoints:
            smoothed[0]  = final[0]
            smoothed[-1] = final[-1]

        final = smoothed

    # reshape & save
    final = np.array([final])
    saveTrackDipy(final, output_file, structural_filename=structural_filename)

    return final





def bspline(cv, n=100, degree=3, periodic=False):
    """ Calculate n samples on a bspline
        cv :      Array ov control vertices
        n  :      Number of samples to return
        degree:   Curve degree
        periodic: True - Curve is closed
                  False - Curve is open
    """
    # If periodic, extend the point array by count+degree+1
    cv = np.asarray(cv)
    count = len(cv)
    if periodic:
        factor, fraction = divmod(count + degree + 1, count)
        cv = np.concatenate((cv,) * factor + (cv[:fraction],))
        count = len(cv)
        degree = np.clip(degree, 1, degree)
    # If opened, prevent degree from exceeding count-1
    else:
        degree = np.clip(degree, 1, count - 1)
    # Calculate knot vector
    kv = None
    if periodic:
        kv = np.arange(0 - degree, count + degree + degree - 1)
    else:
        kv = np.clip(np.arange(count + degree + 1) - degree, 0, count - degree)
    # Calculate query range
    u = np.linspace(periodic, (count - degree), n)
    # Calculate result
    return np.array(si.splev(u, (kv, cv.T, degree))).T

def resample_streamlines(streamlines, type='bspline', n_pts=20):
    if type is None or type == 'None':
            np.array(streamlines)
            resampled = set_number_of_points(streamlines, n_pts)
    else:
        resampled = nib.streamlines.ArraySequence()
        for sl in streamlines:
            if type == 'bspline':
                resampled.append(bspline(sl, n=n_pts))

    return resampled


def get_resampled_streamlines(track_filename, output_file, structural_filename, N_points=32, type_='bspline'):
    
    track,track_aff,track_header = loadTrk(track_filename)
    resampled_streamlines=resample_streamlines(track, type=type_, n_pts=N_points)
    saveTrackDipy(resampled_streamlines, output_file, structural_filename=structural_filename, remove_invalid_streamlines=False)
    return resampled_streamlines
    
    
def track2mask(track_filename,output_filename=None,structural_filename=None):
	 


    track,track_aff,_ = loadTrk(track_filename)
    
    if structural_filename is None or len(structural_filename)==0:

        sft = load_tractogram(track_filename, 'same',
                              bbox_valid_check=False)
        sft.to_vox()
        sft.to_corner()
        transformation, dimensions, _, _ = sft.space_attributes
        stream_map = streamline_mapping(track,affine=track_aff)
        Points=np.zeros(dimensions)
        for i,idx in enumerate(stream_map.keys()):
            Points[idx[0],idx[1],idx[2]] = 1
        if output_filename is not None:
            nib.save(nib.Nifti1Image(Points, transformation),
                output_filename)
    else:	

        struct_nib=nib.load(structural_filename)
        affine=struct_nib.affine
        header=struct_nib.header
        struct_data=struct_nib.get_data()
        stream_map = streamline_mapping(track,affine=affine)
        Points=np.zeros_like(struct_data)
        for i,idx in enumerate(stream_map.keys()):
            Points[idx[0],idx[1],idx[2]] = 1
        if output_filename is not None:
            nii_output = nib.Nifti1Image(Points, affine=affine, header=header)
            nii_output.to_filename(output_filename)

    return  Points     

def streamline_distance(track1,track2,distance='euclidean', N_points=None, check_flip=False):
	 
    if N_points is None:
		 
        N_points = np.array(track2).shape[1]
    else:
        print('resampling streamlines to same number of points..')
        track1=set_number_of_points( track1 , N_points )
        track2=set_number_of_points( track2 , N_points )
	
        if check_flip:
                if len(track1) == 1:
                         print('check streamline flip')
                         track1=np.array(track1)
                         track2=np.array(track2)
                         dist=np.linalg.norm((track1 - track2).reshape(-1, N_points*3), axis=1)
                         dist_flip=np.linalg.norm((track1 - track2[:,::-1,:]).reshape(-1, N_points*3), axis=1)
                         idx_flip=dist_flip<dist
                         if idx_flip:
                             print('flip streamline')
                             track1 = track1[:,::-1,:]        
    
    if distance == 'euclidean':
        track1=np.array(track1)
        track2=np.array(track2)	
        distance=np.linalg.norm((track1 - track2).reshape(-1, N_points*3), axis=1)[0]
	 
    if distance == 'mam':
        distance=bundles_distances_mam(track1,track2)[0][0]
	 
    if distance == 'mdf':
        distance=bundles_distances_mdf(track1,track2)[0][0]
		 		 
    return distance
    
    
def get_streamline_distance(track_filename1,track_filename2,distance='euclidean', N_points=None, check_flip=False):
    track1,track_aff1,track_header1 = loadTrk(track_filename1)
    track2,track_aff2,track_header2 = loadTrk(track_filename2)
    distance=streamline_distance(track1,track2,distance=distance, N_points=N_points, check_flip=check_flip)
    return distance

def clustering_track(track, threshold_length=40.0, qb_threshold=16.0, nb_res_points=20):
    """Clustering for SLR.
    """
    # Parameters as in [Garyfallidis et al. 2015]."):
    # threshold_length = 40.0 # 50mm / 1.25
    # qb_threshold = 16.0  # 20mm / 1.25
    # nb_res_points = 20
    #print("Loading: %s" % filename)
    #
    if track is str:
        tractogram = nib.streamlines.load(track).streamlines
    else:
        tractogram = track
    print("Performing QuickBundles")
    tractogram = np.array([s for s in tractogram if len(s) > threshold_length], dtype=np.object)
    qb = QuickBundles(threshold=qb_threshold)
    clusters = [cluster.centroid for cluster in qb.cluster(tractogram)]
    if nb_res_points is not None:
        print("Resampling to %s points" % nb_res_points)
        clusters = set_number_of_points(clusters, nb_res_points)
    return clusters
       	
def SLR(track_moving,track_fixed,N_points=32,transf='rigid', apply_to=None, clustering=True, num_threads = None, verbose=False ):
    
    
    #%% Resample Streamlines
    #
    #track_moving=set_number_of_points( track_moving , N_points )
    #track_fixed=set_number_of_points( track_fixed ,  N_points )
    #print("Streamline clusrering...")

    if track_moving is str:
        print("load streamlines (moving)")
        track_moving, _, _ = loadTrk(track_moving)
        
    if track_fixed is str:
        print("load streamlines (fixed)")
        track_fixed, _, _ = loadTrk(track_fixed)
        
    # print("setting the same number of points for both the tracts...")
    
    
    # track_moving=set_number_of_points( track_moving , N_points )
    # track_fixed=set_number_of_points( track_fixed , N_points)
    if clustering:
        track_moving = clustering_track(track_moving,threshold_length=40.0, qb_threshold=16.0, nb_res_points=N_points )
        track_fixed = clustering_track(track_fixed,threshold_length=40.0, qb_threshold=16.0, nb_res_points=N_points )
    
    
    print("Linear Registration ("+str(transf)+") of the tracts using SLR...")
    if transf == 'transl':
        bounds=[(-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (0, 0), (0, 0), (0, 0)]
        srr = StreamlineLinearRegistration(bounds=bounds,num_threads=num_threads,verbose=verbose)
    else:
        srr = StreamlineLinearRegistration(x0=transf,num_threads=num_threads,verbose=verbose)
    srm = srr.optimize(static=track_fixed, moving=track_moving)
    
    if apply_to is None:
        track_warped = None
    else:
        track_warped = srm.transform(apply_to)
        
    return srm,track_warped

def apply_SLR(track,srm,file_out=None,header=None, reference=None, remove_invalid_streamlines=True,bbox_valid_check=True):
    #if os.path.isfile(track):
    #    print('load track...')
    if type(track) is str:
        track, aff, head = loadTrk(track)
        
    print('apply SLR transformation...')
    track_warped = srm.transform(track)

    if reference is None:
        if header is None:
            header=head
    else:
        header=None
    
    if file_out is not None:
        print('bbox_valid_check: '+str(bbox_valid_check))
        saveTrackDipy(track_warped, file_out,  structural_filename=reference, remove_invalid_streamlines=remove_invalid_streamlines,header=header,bbox_valid_check=bbox_valid_check)
    
    return track_warped



def orient_streamlines(streamlines,reference=None, distance='euclidean', n_pts=None,verbose=False):
    if n_pts is not None:
        if n_pts != 'None':
            n_pts=int(n_pts)
            np.array(streamlines)
            streamlines = set_number_of_points(streamlines, n_pts)
    if reference is None:
        reference = streamlines[0]
    else:
        if len(reference) > 1:
            reference = np.array(reference)[0]
    if n_pts is not None:
        if n_pts != 'None':
            reference = set_number_of_points(reference, n_pts)

    reoriented = nib.streamlines.ArraySequence()
    for sl in streamlines:
                        track1=np.array(sl)
                        track2=np.array(reference)
                        track1_flipped=track1[::-1,:]
                        
                        track1=np.expand_dims(track1,axis=1)
                        track2=np.expand_dims(track2,axis=1)
                        track1_flipped=np.expand_dims(track1_flipped,axis=1)
                        
                        dist=streamline_distance(track1,track2, distance=distance)

                        dist_flip=streamline_distance(track1_flipped ,track2  , distance=distance)
                        
                        idx_flip=dist_flip<dist
                        if verbose:
                            print(idx_flip)
                        if idx_flip:
                             if verbose:
                                print('flip streamline')
                             track1 = np.squeeze(track1)
                             track1 = track1[::-1,:]
                        
                        reoriented.append(np.squeeze(track1))               

    return reoriented


def get_oriented_streamlines(track_filename, output_file, reference=None, distance='mam', n_pts=None , structural_filename=None, verbose=True):
    
    track,track_aff,track_header = loadTrk(track_filename)
    if structural_filename is None or structural_filename=='None':
        header=track_header
        structural_filename=None
    else:
        header = None
    if reference is not None and reference != 'None':
        reference_track,reference_aff,reference_header = loadTrk(reference)
        reference = reference_track
    else:
        reference = None
    reoriented=orient_streamlines(track,reference=reference, distance=distance, n_pts=n_pts,verbose=verbose)
    saveTrackDipy(reoriented, output_file, structural_filename=structural_filename,header=header, remove_invalid_streamlines=False)
    return reoriented



from dipy.io.streamline import load_trk, load_tck
from dipy.tracking.streamline import length

def track_info(track_file):
    """
    Return basic information about a tractogram (.trk or .tck).

    Parameters
    ----------
    track_file : str
        Path to a .trk or .tck streamline file.

    Returns
    -------
    info : dict
        Dictionary containing:
            - number of streamlines
            - mean length (mm)
            - min length (mm)
            - max length (mm)
            - bounding box (mm)
    """
    if not os.path.isfile(track_file):
        raise FileNotFoundError(f"Track file not found: {track_file}")

    # --- Load streamlines ---
    if track_file.endswith(".trk"):
        trk_obj = load_trk(track_file, reference="same")
        streamlines = trk_obj.streamlines

    elif track_file.endswith(".tck"):
        # Identity reference image (required by DIPY)
        import nibabel as nib
        identity_ref = nib.Nifti1Image(
            np.zeros((1, 1, 1), dtype=np.uint8),
            np.eye(4)
        )
        tck_obj = load_tck(track_file, reference=identity_ref)
        streamlines = tck_obj.streamlines

    else:
        raise ValueError(f"Unsupported file format: {track_file}")

    # --- Empty ---
    n_streamlines = len(streamlines)
    if n_streamlines == 0:
        return {
            "number of streamlines": 0,
            "mean length (mm)": 0.0,
            "min length (mm)": 0.0,
            "max length (mm)": 0.0,
            "bounding box (mm)": [0, 0, 0, 0, 0, 0],
        }

    # --- Lengths ---
    lengths = list(length(streamlines))
    all_pts = np.vstack(streamlines)

    return {
        "number of streamlines": int(n_streamlines),
        "mean length (mm)": float(np.mean(lengths)),
        "min length (mm)": float(np.min(lengths)),
        "max length (mm)": float(np.max(lengths)),
        "bounding box (mm)": [float(v) for v in np.concatenate([all_pts.min(0), all_pts.max(0)])]
    }

    return info

##############################
# Parallel streamlines loading (TRK + TCK)
##############################

from multiprocessing import Pool, cpu_count
import numpy as np
import mmap
import nibabel as nib
import os

# ------------------------------------------------------------
# Optional Numba import (safe)
# ------------------------------------------------------------
try:
    from numba import njit
    numba_available = True
except Exception:
    numba_available = False


# ============================================================
# PART 1 — TRK LOADING
# ============================================================

# ------------------------------------------------------------
# TRK parsing functions — Numba or fallback
# ------------------------------------------------------------

if numba_available:

    @njit
    def parse_lengths(buffer, lengths, point_size, n_properties):
        pointer = 0
        for idx in range(lengths.size):
            l = buffer[pointer]
            lengths[idx] = l
            pointer += 1 + l * point_size + n_properties
        return lengths

    @njit
    def parse_streamlines(buffer, idxs, split_points, n_floats, affine, apply_affine=True):
        streamlines = []
        R = affine[:3, :3].copy()
        t = affine[:3, 3].copy()
        for idx in idxs:
            s = buffer[
                split_points[idx] : split_points[idx] + n_floats[idx]
            ].reshape(-1, 3)
            if apply_affine:
                s = (s @ R.T) + t
            streamlines.append(s)
        return streamlines

else:
    # ---------- Fallback: pure NumPy versions ----------
    def parse_lengths(buffer, lengths, point_size, n_properties):
        pointer = 0
        for idx in range(lengths.size):
            l = buffer[pointer]
            lengths[idx] = l
            pointer += 1 + l * point_size + n_properties
        return lengths

    def parse_streamlines(buffer, idxs, split_points, n_floats, affine, apply_affine=True):
        streamlines = []
        R = affine[:3, :3]
        t = affine[:3, 3]
        for idx in idxs:
            s = buffer[
                split_points[idx] : split_points[idx] + n_floats[idx]
            ].reshape(-1, 3)
            if apply_affine:
                s = s @ R.T + t
            streamlines.append(s)
        return streamlines


# ------------------------------------------------------------
# Worker wrapper for multiprocessing
# ------------------------------------------------------------
def _parse_chunk(args):
    buffer, idxs_chunk, split_points, n_floats, aff, apply_affine = args
    return parse_streamlines(buffer, idxs_chunk, split_points, n_floats, aff, apply_affine)


# ------------------------------------------------------------
# MAIN TRK PARALLEL LOADER
# ------------------------------------------------------------
def load_streamlines_parallel(trk_fn, idxs=None, apply_affine=True,
                              container='list', n_jobs=-1, verbose=False):

    # ---------- parse header ----------
    lazy_trk = nib.streamlines.load(trk_fn, lazy_load=True)
    header = lazy_trk.header
    nb_streamlines = header["nb_streamlines"]
    n_scalars = header["nb_scalars_per_point"]
    n_properties = header["nb_properties_per_streamline"]

    point_size = 3 + n_scalars  # floats per point
    header_size = header["hdr_size"]

    # ---------- load entire buffer ----------
    buffer = np.empty(
        os.path.getsize(trk_fn) // 4,
        dtype=np.float32
    )

    with open(trk_fn, "rb") as f:
        f.seek(header_size)
        f.readinto(buffer)

    buffer_int32 = buffer.view(np.int32)

    lengths = np.empty(nb_streamlines, dtype=np.int32)
    lengths = parse_lengths(buffer_int32, lengths, point_size, n_properties)

    if idxs is None:
        idxs = np.arange(nb_streamlines, dtype=np.int64)

    n_floats = lengths * point_size
    split_points = (n_floats + 1 + n_properties).cumsum() - n_floats - n_properties

    # ---------- determine workers ----------
    if n_jobs == -1:
        n_jobs = cpu_count()
    n_jobs = max(1, min(n_jobs, len(idxs)))

    idx_chunks = np.array_split(idxs, n_jobs)

    # ---------- parallel parsing ----------
    if n_jobs == 1:
        streamlines = parse_streamlines(buffer, idxs, split_points, n_floats,
                                        nib.streamlines.trk.get_affine_trackvis_to_rasmm(header),
                                        apply_affine)
    else:
        args_list = [
            (
                buffer,
                chunk.astype(np.int64),
                split_points,
                n_floats,
                nib.streamlines.trk.get_affine_trackvis_to_rasmm(header),
                apply_affine
            )
            for chunk in idx_chunks
        ]

        with Pool(n_jobs) as p:
            parts = p.map(_parse_chunk, args_list)

        streamlines = [sl for sub in parts for sl in sub]

    # ---------- container handling ----------
    if container == "array":
        streamlines = np.array(streamlines, dtype=object)
    elif container == "ArraySequence":
        streamlines = nib.streamlines.ArraySequence(streamlines)
    elif container == "array_flat":
        streamlines = np.concatenate(streamlines)
    # else keep as list

    return streamlines, header, lengths[idxs], idxs



# ============================================================
# PART 2 — TCK LOADING (FIXED FOR HUGE FILES)
# ============================================================

# ---------- step 1: parse ASCII header ----------
def _parse_tck_header(path):
    """
    Parse MRtrix .tck header and return (header_dict, data_offset_bytes).
    """
    header = {}
    file_offset = None

    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if line == b"":
                break
            text = line.decode("utf-8", errors="ignore").strip()

            if text == "END":
                break

            if ":" in text:
                key, value = text.split(":", 1)
                key = key.strip()
                value = value.strip()
                header[key] = value

                if key == "file":
                    parts = value.split()
                    if len(parts) >= 2:
                        try:
                            file_offset = int(parts[-1])
                        except ValueError:
                            file_offset = None

        if file_offset is None:
            file_offset = f.tell()

    return header, file_offset



# ---------- step 2: find NaN delimiters ----------
if numba_available:
    @njit
    def _find_nan_blocks(data):
        out = []
        for i in range(data.shape[0]):
            if np.isnan(data[i, 0]):
                out.append(i)
        return np.array(out, dtype=np.int64)
else:
    def _find_nan_blocks(data):
        return np.where(np.isnan(data[:, 0]))[0].astype(np.int64)


# ---------- step 3: worker ----------
def _load_chunk(args):
    data, starts, ends, affine, apply_affine = args
    out = []
    R = affine[:3, :3]
    t = affine[:3, 3]

    for s, e in zip(starts, ends):
        sl = data[s:e].copy()
        if apply_affine:
            sl = sl @ R.T + t
        out.append(sl)
    return out

def _load_chunk_tck(args):
    """
    Worker that opens the file locally and reads its own streamline slices.
    This avoids sending huge arrays through multiprocessing.
    """
    path, header_end, starts, ends, affine, apply_affine = args

    # reconstruct mmap inside worker
    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    mm.seek(header_end)
    raw = np.ndarray(
        shape=((len(mm) - header_end) // 4,),
        dtype=np.float32,
        buffer=mm,
        offset=header_end,
    )

    n = raw.shape[0] - (raw.shape[0] % 3)
    points = raw[:n].reshape(-1, 3)



    R = affine[:3, :3]
    t = affine[:3, 3]

    out = []
    for s, e in zip(starts, ends):
        sl = points[s:e].copy()
        if apply_affine:
            sl = sl @ R.T + t
        out.append(sl)

    mm.close()
    f.close()
    return out


# ---------- step 4: MAIN TCK PARALLEL LOADER ----------
def load_tck_parallel(path, n_jobs=-1, apply_affine=True, verbose=False):

    header, header_end = _parse_tck_header(path)

    # affine
    if "transform" in header:
        mat = np.fromstring(header["transform"], sep=" ").reshape(4, 4)
        affine = mat.astype(np.float32)
    else:
        affine = np.eye(4, dtype=np.float32)

    # main process mmap to detect NaNs only
    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    mm.seek(header_end)
    raw = np.ndarray(
        shape=((len(mm) - header_end) // 4,),
        dtype=np.float32,
        buffer=mm,
        offset=header_end,
    )
    points = raw.reshape(-1, 3)

    # NaN delimiters
    nan_idx = _find_nan_blocks(points)

    # compute streamline ranges
    starts = []
    ends = []
    prev = 0
    for idx in nan_idx:
        if idx > prev:
            starts.append(prev)
            ends.append(idx)
        prev = idx + 1

    starts = np.array(starts, dtype=np.int64)
    ends   = np.array(ends, dtype=np.int64)

    n_sl = len(starts)

    # workers
    if n_jobs == -1:
        n_jobs = cpu_count()
    n_jobs = max(1, min(n_jobs, n_sl))

    start_chunks = np.array_split(starts, n_jobs)
    end_chunks   = np.array_split(ends, n_jobs)

    args_list = [
        (path, header_end, sc, ec, affine, apply_affine)
        for sc, ec in zip(start_chunks, end_chunks)
    ]

    if n_jobs == 1:
        parts = [_load_chunk_tck(args_list[0])]
    else:
        with Pool(n_jobs) as p:
            parts = p.map(_load_chunk_tck, args_list)

    mm.close()
    f.close()

    # flatten
    streamlines = [sl for sub in parts for sl in sub]

    return streamlines, affine




# ============================================================
# PART 3 — Unified entry point
# ============================================================

def load_tck_serial(path, apply_affine=True, verbose=False):
    """
    Serial TCK loader using the same logic as load_tck_parallel(),
    but without multiprocessing.

    Returns
    -------
    streamlines : list of (N_i, 3) float32 arrays
    affine : (4, 4) float32 array
    """
    header, header_end = _parse_tck_header(path)

    # Recover affine from TCK header if present
    if "transform" in header:
        mat = np.fromstring(header["transform"], sep=" ")
        if mat.size == 16:
            affine = mat.reshape(4, 4).astype(np.float32)
        else:
            affine = np.eye(4, dtype=np.float32)
    else:
        affine = np.eye(4, dtype=np.float32)

    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    try:
        raw = np.ndarray(
            shape=((len(mm) - header_end) // 4,),
            dtype=np.float32,
            buffer=mm,
            offset=header_end,
        )

        # Safe trim to multiple of 3
        n = raw.shape[0] - (raw.shape[0] % 3)
        points = raw[:n].reshape(-1, 3)

        nan_idx = _find_nan_blocks(points)

        starts = []
        ends = []
        prev = 0
        for idx in nan_idx:
            if idx > prev:
                starts.append(prev)
                ends.append(idx)
            prev = idx + 1

        starts = np.array(starts, dtype=np.int64)
        ends = np.array(ends, dtype=np.int64)

        if verbose:
            print(f"[load_tck_serial] streamlines found: {len(starts):,}")

        R = affine[:3, :3]
        t = affine[:3, 3]

        streamlines = []
        for s, e in zip(starts, ends):
            sl = points[s:e].copy()
            if apply_affine:
                sl = sl @ R.T + t
            streamlines.append(sl)

    finally:
        mm.close()
        f.close()

    return streamlines, affine

def load_any_tractogram(path, n_jobs=1, reference_img=None):
    ext = os.path.splitext(path)[1].lower()

    if n_jobs is None:
        n_jobs = 1

    if n_jobs == 1:
        if ext == ".trk":
            return load_tractogram(path, reference="same", to_space=Space.RASMM)
        elif ext == ".tck":
            return load_tck_serial(path, apply_affine=True)
        else:
            raise ValueError(f"Unsupported extension: {ext}")

    if n_jobs > 1:
        if ext == ".tck":
            return load_tck_parallel(path, n_jobs=n_jobs, apply_affine=True)
        elif ext == ".trk":
            return load_streamlines_parallel(path, n_jobs=n_jobs)
        else:
            raise ValueError(f"Unsupported extension: {ext}")

    raise ValueError(f"Invalid n_jobs value: {n_jobs}")



def load_tracks_parallel(in_file, roi_includes=None, reference=None, n_jobs_io=-1, verbose=False):
    ext = os.path.splitext(in_file)[1].lower()

    if ext == ".trk":
        streamlines, header, lengths, idxs = load_streamlines_parallel(
            in_file, idxs=None, apply_affine=True,
            container="list", n_jobs=n_jobs_io, verbose=verbose
        )

        affine = nib.streamlines.trk.get_affine_trackvis_to_rasmm(header)

        class FakeSFT:
            def __init__(self, affine, header):
                self.affine_to_rasmm = affine
                self.header = header

        return FakeSFT(affine, header), streamlines, affine

    elif ext == ".tck":
        streamlines, affine = load_tck_parallel(
            in_file, n_jobs=n_jobs_io, apply_affine=True, verbose=verbose
        )

        class FakeSFT:
            def __init__(self, affine):
                self.affine_to_rasmm = affine

        return FakeSFT(affine), streamlines, affine

    else:
        raise ValueError(f"Unsupported tract format: {ext}")

##################################################
# Streaming + Parallel TCK→TRK (HYBRID)
##################################################

def _load_chunk_tck_raw(args):
    """
    Raw worker: load streamline segments from a TCK file WITHOUT applying any affine.
    This should reproduce nibabel's tck loader behaviour at the coordinate level
    (assuming the TCK is already in world / scanner space).
    """
    path, header_end, starts, ends = args
    import numpy as np
    import mmap

    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    mm.seek(header_end)
    raw = np.ndarray(
        shape=((len(mm) - header_end) // 4,),
        dtype=np.float32,
        buffer=mm,
        offset=header_end,
    )
    points = raw.reshape(-1, 3)

    out = []
    for s, e in zip(starts, ends):
        sl = points[s:e].copy()
        out.append(sl)

    mm.close()
    f.close()
    return out



def tck_to_trk_stream_hybrid(
    in_tck,
    out_trk,
    reference_nii,
    n_jobs=8,
    chunk_size=500000,
    verbose=True,
):
    """
    Hybrid TCK→TRK:
      - Streaming + parallel TCK reading
      - Then classic nib.streamlines.save() with TRK header
    Good for:
      * Comparing with existing tck2trk()
      * Moderate tractograms (not fully 400GB-safe)
    """

    import numpy as np
    from multiprocessing import Pool, cpu_count

    if verbose:
        print(f"[TCK→TRK hybrid] Converting: {in_tck}")
        print(f"[TCK→TRK hybrid] Output:     {out_trk}")
        print(f"[TCK→TRK hybrid] Workers:    {n_jobs}")
        print(f"[TCK→TRK hybrid] Chunk size: {chunk_size}")

    # --------------------------------------------------
    # Load reference anatomy – defines TRK header
    # --------------------------------------------------
    nii = nib.load(reference_nii)

    # Build TRK header exactly like classic tck2trk
    header = {}
    header[Field.VOXEL_TO_RASMM] = nii.affine.copy()
    header[Field.VOXEL_SIZES]    = nii.header.get_zooms()[:3]
    header[Field.DIMENSIONS]     = nii.shape[:3]
    header[Field.VOXEL_ORDER]    = "".join(aff2axcodes(nii.affine))

    # --------------------------------------------------
    # Parse TCK header and boundaries
    # --------------------------------------------------
    header_tck, header_end = _parse_tck_header(in_tck)

    f = open(in_tck, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    mm.seek(header_end)
    raw = np.ndarray(
        shape=((len(mm) - header_end) // 4,),
        dtype=np.float32,
        buffer=mm,
        offset=header_end,
    )
    points = raw.reshape(-1, 3)

    if verbose:
        print("[TCK→TRK hybrid] Detecting streamline boundaries…")

    nan_idx = _find_nan_blocks(points)

    starts, ends = [], []
    prev = 0
    for idx in nan_idx:
        if idx > prev:
            starts.append(prev)
            ends.append(idx)
        prev = idx + 1

    starts = np.array(starts, dtype=np.int64)
    ends   = np.array(ends,   dtype=np.int64)
    num_sl = len(starts)

    if verbose:
        print(f"[TCK→TRK hybrid] Total streamlines: {num_sl:,}")

    # --------------------------------------------------
    # STREAM-INTO-MEMORY (still accumulates, but fast)
    # --------------------------------------------------
    all_streamlines = []

    if n_jobs == -1:
        n_jobs = cpu_count()
    n_jobs = max(1, min(n_jobs, num_sl))

    for i in range(0, num_sl, chunk_size):

        sl_st = starts[i:i+chunk_size]
        sl_en = ends[i:i+chunk_size]

        args_load = [
            (in_tck, header_end, sc, ec)
            for sc, ec in zip(np.array_split(sl_st, n_jobs),
                              np.array_split(sl_en, n_jobs))
        ]

        if n_jobs == 1:
            parts = [_load_chunk_tck_raw(args_load[0])]
        else:
            with Pool(n_jobs) as p:
                parts = p.map(_load_chunk_tck_raw, args_load)

        chunk_sl = [sl for sub in parts for sl in sub]
        all_streamlines.extend(chunk_sl)

        if verbose:
            print(f"[TCK→TRK hybrid] {min(i+chunk_size, num_sl):,}/{num_sl:,} processed")

    mm.close()
    f.close()

    # --------------------------------------------------
    # Final TRK save using nibabel (like classic tck2trk)
    # --------------------------------------------------
    if verbose:
        print(f"[TCK→TRK hybrid] Writing TRK: {out_trk}")

    from nibabel.streamlines.array_sequence import ArraySequence
    from nibabel.streamlines.tractogram import Tractogram

    tractogram = Tractogram(ArraySequence(all_streamlines), affine_to_rasmm=None)
    nib.streamlines.save(tractogram, out_trk, header=header)

    if verbose:
        print(f"[TCK→TRK hybrid] DONE → saved {len(all_streamlines):,} streamlines.")


##################################################
# Streaming + Parallel TCK→TRK (PURE STREAMING)
##################################################

import mmap
import struct
import numpy as np
import nibabel as nib
from multiprocessing import Pool, cpu_count
from nibabel.streamlines.trk import TrkFile, Field
from nibabel.orientations import aff2axcodes
from nibabel.streamlines import Tractogram

import tempfile
import os
import struct
import numpy as np
import nibabel as nib
from nibabel.streamlines import Tractogram
from nibabel.streamlines.trk import TrkFile, Field
from nibabel.orientations import aff2axcodes
import mmap
from multiprocessing import Pool, cpu_count


def _build_trk_header_from_nii(nii):
    """
    Build a valid TrackVis TRK header as a NumPy structured array.
    Works with ALL nibabel versions.

    Ensures:
    - return type is numpy.void structured header
    - never returns dict
    """

    # 1) Try safest mechanism: extract dtype from a dummy TrkFile
    try:
        dummy = TrkFile(Tractogram([]))
        hdr_dtype = dummy.header.dtype
    except Exception:
        # Extremely old nibabel versions fallback
        # Load a fake small TRK to discover dtype
        from nibabel.streamlines.trk import trk_header
        hdr_dtype = trk_header

    # 2) Allocate proper structured header
    hdr = np.zeros((), dtype=hdr_dtype)

    # 3) Fill required fields
    hdr[Field.VOXEL_TO_RASMM] = nii.affine.astype(np.float32)
    hdr[Field.VOXEL_SIZES]    = np.asarray(nii.header.get_zooms()[:3], dtype=np.float32)
    hdr[Field.DIMENSIONS]     = np.asarray(nii.shape[:3], dtype=np.int16)

    # TrackVis requires bytes, padded to 4 chars
    voxel_order = "".join(aff2axcodes(nii.affine))
    hdr[Field.VOXEL_ORDER] = voxel_order.ljust(4).encode("ascii")

    hdr[Field.NB_STREAMLINES] = 0

    return hdr

import numpy as np
import struct

def build_trk_header_manual(affine, dims, vox, voxel_order, n_streamlines):
    """
    Construct a fully valid 1000-byte TrackVis TRK header.
    Compliant with TrackVis 2.0 layout AND nibabel expectations.
    """

    dims = np.asarray(dims, dtype=np.int16)
    vox  = np.asarray(vox, dtype=np.float32)

    # ---------------------------------------------
    # Prepare voxel_order (4 bytes)
    # ---------------------------------------------
    # Use exactly 3 letters + NULL terminator
    voxel_order = voxel_order[:3] + "\x00"
    voxel_order = voxel_order.encode("ascii")

    # Preallocate 1000 bytes
    header = bytearray(1000)

    # ---------------------------------------------
    # FIX 1: dims (offset 0)
    # ---------------------------------------------
    struct.pack_into("<3h", header, 0, *dims)
    struct.pack_into("<h",  header, 6, 0)

    # ---------------------------------------------
    # voxel sizes (offset 8)
    # ---------------------------------------------
    struct.pack_into("<3f", header, 8, *vox)

    # ---------------------------------------------
    # origin (offset 20)
    # ---------------------------------------------
    struct.pack_into("<3f", header, 20, 0.0, 0.0, 0.0)

    # ---------------------------------------------
    # n_scalars + scalars block (offset 32)
    # ---------------------------------------------
    struct.pack_into("<i", header, 32, 0)

    # ---------------------------------------------
    # n_properties + properties block (offset 236)
    # ---------------------------------------------
    struct.pack_into("<i", header, 236, 0)

    # ---------------------------------------------
    # FIX 2: Affine matrix (offset 440)
    # TrackVis expects COLUMN-MAJOR ordering!
    # ---------------------------------------------
    aff_cm = affine.astype(np.float32).T.ravel()   # transpose!!
    struct.pack_into("<16f", header, 440, *aff_cm)

    # ---------------------------------------------
    # FIX 3: voxel_order (offset 504)
    # ---------------------------------------------
    struct.pack_into("4s", header, 504, voxel_order)

    # ---------------------------------------------
    # track_threshold (offset 512)
    # ---------------------------------------------
    struct.pack_into("<f", header, 512, 0.0)

    # ---------------------------------------------
    # FIX 4: counts + version + hdr_size
    # ---------------------------------------------
    struct.pack_into("<i", header, 988, n_streamlines)  # n_count
    struct.pack_into("<i", header, 992, 2)              # version = 2
    struct.pack_into("<i", header, 996, 1000)           # hdr_size = 1000

    return bytes(header)



def _make_trk_header_bytes_from_nii(nii):
    """
    Ask nibabel to create a *real* TRK file with the proper header for this
    reference NIfTI, then read back the first 1000 bytes of that file.

    This way we never have to guess the TrackVis header layout: nibabel
    does it for us, and we just reuse the raw header bytes.
    """

    # Build the header dict exactly like your classic tck2trk()
    header = {}
    header[Field.VOXEL_TO_RASMM] = nii.affine.copy()
    header[Field.VOXEL_SIZES]    = nii.header.get_zooms()[:3]
    header[Field.DIMENSIONS]     = nii.shape[:3]
    header[Field.VOXEL_ORDER]    = "".join(aff2axcodes(nii.affine))

    # Empty tractogram: we only care about header
    empty_tg = Tractogram([])

    # Create a temporary TRK, let nibabel write a correct header
    tmp = tempfile.NamedTemporaryFile(suffix=".trk", delete=False)
    tmp_name = tmp.name
    tmp.close()

    trk = TrkFile(empty_tg, header=header)
    trk.save(tmp_name)

    # Read back the first 1000 bytes (TrackVis header size)
    with open(tmp_name, "rb") as f:
        hdr_bytes = f.read(1000)

    os.remove(tmp_name)

    # Sanity check
    if len(hdr_bytes) != 1000:
        raise RuntimeError(f"TRK header must be 1000 bytes, got {len(hdr_bytes)}")

    return hdr_bytes



def _load_chunk_tck_raw(args):
    path, header_end, starts, ends = args

    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    mm.seek(header_end)
    raw = np.ndarray(
        shape=((len(mm) - header_end)//4,),
        dtype=np.float32,
        buffer=mm,
        offset=header_end
    )
    points = raw.reshape(-1, 3)

    out = [points[s:e].copy() for s, e in zip(starts, ends)]

    mm.close()
    f.close()
    return out


def parse_affine_from_tck_header(header):
    if "transform" not in header:
        return np.eye(4, dtype=np.float32)

    values = header["transform"]

    a = np.fromstring(values, sep=" ")
    if a.size == 16:
        return a.reshape(4,4).astype(np.float32)

    rows = []
    for k, v in header.items():
        if k == "transform":
            r = np.fromstring(v, sep=" ")
            if r.size == 4:
                rows.append(r)

    if len(rows) == 4:
        return np.vstack(rows).astype(np.float32)

    return np.eye(4, dtype=np.float32)

def tck_to_trk_stream(
    in_tck,
    out_trk,
    reference_nii,
    n_jobs=8,
    chunk_size=500000,
    verbose=True,
):
    """
    PURE streaming TCK → TRK (TrackVis .trk):
      - Parallel TCK reading
      - Writes TRK incrementally in TrackVis format
      - Never builds a full Tractogram in memory

    Assumes:
      - TCK coordinates can be mapped into reference_nii space via the
        MRtrix 'transform' header (or identity if missing).
      - Header semantics match the classic tck2trk() (VOXEL_TO_RASMM, etc.).
    """

    if verbose:
        print(f"[TCK→TRK pure] Converting: {in_tck}")
        print(f"[TCK→TRK pure] Output:     {out_trk}")
        print(f"[TCK→TRK pure] Workers:    {n_jobs}")
        print(f"[TCK→TRK pure] Chunk size: {chunk_size}")

    # --------------------------------------------------
    # Load reference NIfTI and get a VALID TRK header
    # --------------------------------------------------
    nii = nib.load(reference_nii)
    aff = nii.affine
    dims = nii.shape[:3]
    vox  = nii.header.get_zooms()[:3]
    order = "".join(aff2axcodes(aff))

    # Let nibabel write a correct TrackVis header for us
    hdr = _make_trk_header_bytes_from_nii(nii)

    # --------------------------------------------------
    # Parse TCK header
    # --------------------------------------------------
    header_tck, header_end = _parse_tck_header(in_tck)

    # --------------------------------------------------
    # Compute TCK→Ref transform (world/RASMM)
    # --------------------------------------------------
    aff_tck = parse_affine_from_tck_header(header_tck)
    M = aff @ np.linalg.inv(aff_tck)
    R_tck_to_ref = M[:3, :3].astype(np.float32)
    t_tck_to_ref = M[:3, 3].astype(np.float32)

    # --------------------------------------------------
    # mmap and find boundaries
    # --------------------------------------------------
    f = open(in_tck, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    mm.seek(header_end)
    raw = np.ndarray(
        shape=((len(mm) - header_end)//4,),
        dtype=np.float32,
        buffer=mm,
        offset=header_end
    )
    points = raw.reshape(-1, 3)

    nan_idx = _find_nan_blocks(points)

    starts, ends = [], []
    prev = 0
    for idx in nan_idx:
        if idx > prev:
            starts.append(prev)
            ends.append(idx)
        prev = idx + 1

    starts = np.array(starts, dtype=np.int64)
    ends   = np.array(ends,   dtype=np.int64)
    num_sl = len(starts)

    if verbose:
        print(f"[TCK→TRK pure] Total streamlines: {num_sl:,}")

    if n_jobs == -1:
        n_jobs = cpu_count()
    n_jobs = max(1, min(n_jobs, num_sl))

    # --------------------------------------------------
    # Open TRK for write+update, write header once
    # --------------------------------------------------
    with open(out_trk, "wb+") as fout:
        # write the 1000-byte header nibabel generated
        fout.write(hdr)

        total_written = 0
        RESERVED = b"\x00" * 12  # per TrackVis: 3 x float32 scalars we keep at 0

        for i in range(0, num_sl, chunk_size):

            sl_st = starts[i:i+chunk_size]
            sl_en = ends[i:i+chunk_size]

            args = [
                (in_tck, header_end, sc, ec)
                for sc, ec in zip(np.array_split(sl_st, n_jobs),
                                  np.array_split(sl_en, n_jobs))
            ]

            if n_jobs == 1:
                parts = [_load_chunk_tck_raw(args[0])]
            else:
                with Pool(n_jobs) as p:
                    parts = p.map(_load_chunk_tck_raw, args)

            chunk = [sl for sub in parts for sl in sub]

            # ----- Transform + write each streamline -----
            for sl in chunk:
                # TCK world → reference world (RASMM)
                sl = (sl @ R_tck_to_ref.T + t_tck_to_ref).astype("<f4")

                # TrackVis: [int32 n_points] + [float32 x,y,z]*n + 12 reserved bytes
                fout.write(struct.pack("<i", len(sl)))
                fout.write(sl.tobytes())
                fout.write(RESERVED)
                total_written += 1

            if verbose:
                print(f"[TCK→TRK pure] {min(i+chunk_size, num_sl):,}/{num_sl:,} processed")

        # NOTE: we do NOT rewrite the header.
        # n_count in the header will remain at 0, which TrackVis/nibabel
        # interpret as “unknown, read until EOF”.

    mm.close()
    f.close()

    if verbose:
        print(f"[TCK→TRK pure] DONE → saved {total_written:,} streamlines.")

##################################################
# Streaming + Parallel TCK and TRK
##################################################
def log(msg):
    if verbose:
        print(f"[tracklib] {msg}", flush=True)
		
def loadTractogram(filename, max_num=None, n_jobs=None, reference_img=None, verbose=False):
    """
    Load a tractogram and return:
        - streamlines : np.ndarray (dtype=object)
        - affine      : np.ndarray (4x4)
        - header      : dict or None
    """

    def log(msg):
        if verbose:
            print(f"[tracklib] {msg}", flush=True)

    log(f"loading: {filename}")

    # ---------------------------------------------------------
    # Lazy TRK metadata recovery (zero-cost)
    # ---------------------------------------------------------
    trk_header = None
    trk_affine = None

    if filename.lower().endswith(".trk"):
        log("detected TRK file (lazy header read)")
        lazy_trk = nib.streamlines.load(filename, lazy_load=True)
        trk_header = lazy_trk.header
        trk_affine = nib.streamlines.trk.get_affine_trackvis_to_rasmm(trk_header)

    def _header_from_reference(nii):
        return {
            Field.VOXEL_TO_RASMM: nii.affine.copy(),
            Field.VOXEL_SIZES:    nii.header.get_zooms()[:3],
            Field.DIMENSIONS:    nii.shape[:3],
            Field.VOXEL_ORDER:   "".join(aff2axcodes(nii.affine)),
        }

    # ---------------------------------------------------------
    # Decide loader behavior
    # ---------------------------------------------------------
    if n_jobs is None:
        n_jobs = 1

    log(f"using n_jobs={n_jobs}")

    t0 = time.time()
    tg = load_any_tractogram(filename, n_jobs=n_jobs)
    log(f"loader returned type: {type(tg)} in {time.time()-t0:.2f}s")

    # ---------------------------------------------------------
    # CASE A — DIPY loader
    # ---------------------------------------------------------
    if hasattr(tg, "streamlines"):
        log("detected DIPY StatefulTractogram")
        streamlines = np.array(tg.streamlines, dtype=object)

        if trk_affine is not None:
            affine = trk_affine
            header = trk_header
            log("using TRK header + affine")

        else:
            # TCK via DIPY
            if reference_img is not None:
                log("using reference image for TCK")
                nii = nib.load(reference_img)
                affine = nii.affine
                header = _header_from_reference(nii)
            else:
                log("WARNING: no reference for TCK, using tg.affine")
                affine = tg.affine
                header = None

    # ---------------------------------------------------------
    # CASE B — parallel TRK loader
    # ---------------------------------------------------------
    elif isinstance(tg, tuple) and len(tg) == 4:
        log("detected parallel TRK loader output")
        streamlines, header, _, _ = tg
        streamlines = np.array(streamlines, dtype=object)
        affine = nib.streamlines.trk.get_affine_trackvis_to_rasmm(header)

    # ---------------------------------------------------------
    # CASE C — parallel TCK loader
    # ---------------------------------------------------------
    elif isinstance(tg, tuple) and len(tg) == 2:
        log("detected TCK loader output (serial/parallel)")
        streamlines, affine = tg
        streamlines = np.array(streamlines, dtype=object)
        header = None

        if reference_img is not None:
            log("injecting reference image into TCK header")
            nii = nib.load(reference_img)
            affine = nii.affine
            header = _header_from_reference(nii)

    else:
        raise TypeError(f"Unknown tractogram loader output: {type(tg)}")

    log(f"loaded {len(streamlines):,} streamlines")

    # ---------------------------------------------------------
    # Optional subsampling
    # ---------------------------------------------------------
    if max_num is not None and len(streamlines) > max_num:
        log(f"subsampling from {len(streamlines):,} to {max_num:,}")
        idxs = np.random.choice(len(streamlines), size=max_num, replace=False)
        streamlines = streamlines[idxs]

    log("done loading tractogram")

    return streamlines, affine, header
##################################################
# get backbone new version
##################################################
def get_core_streamlines_from_streamlines(
    track,
    track_aff,
    dimensions,
    perc=0.75,
    smooth_density=True,
    verbose=True
):
    """
    In-memory version of get_core_streamlines.

    Parameters
    ----------
    track : np.ndarray (N, P, 3)
        Streamlines in RASMM
    track_aff : (4,4) ndarray
        Affine mapping RASMM → voxel space
    dimensions : tuple (X, Y, Z)
        Voxel grid dimensions
    """
    if verbose:
        print("calculate density map...")

    # ----------------------------------------------------------
    # 1. Streamline → voxel mapping
    # ----------------------------------------------------------
    stream_map = streamline_mapping(track, affine=track_aff)

    # ----------------------------------------------------------
    # 2. Density volume
    # ----------------------------------------------------------
    streamline_count_array = streamlines_count(
        track,
        track_aff,
        dimensions,
        stream_map=stream_map,
        smooth_density=smooth_density
    )

    # ----------------------------------------------------------
    # 3. Thresholding
    # ----------------------------------------------------------
    if perc != 0:
        indices = np.where(
            streamline_count_array >= perc * np.max(streamline_count_array)
        )
    else:
        indices = np.where(streamline_count_array > 0)

    # ----------------------------------------------------------
    # 4. Collect streamlines touching dense voxels
    # ----------------------------------------------------------
    streamlines_to_keep = []

    for i in range(len(indices[0])):
        idx = (indices[0][i], indices[1][i], indices[2][i])
        for sl_idx in stream_map.get(idx, []):
            streamlines_to_keep.append(sl_idx)

    streamlines_to_keep = np.unique(streamlines_to_keep)
    track = np.asarray(track, dtype=object)
    core_track = track[streamlines_to_keep]
    if verbose:
            print(f"number of core streamlines: {len(core_track)}")

    return core_track

def _ensure_streamlines(track):
    if isinstance(track, Streamlines):
        return track

    track = np.asarray(track, dtype=object)

    if track.ndim == 2:      # single streamline (N,3)
        track = [track]

    return Streamlines(track)

def _as_float32_streamline(sl):
    sl = np.asarray(sl, dtype=np.float32)
    if sl.ndim != 2 or sl.shape[1] != 3:
        return None
    return sl


def get_bundle_backbone_from_streamlines(
    track,
    track_aff,
    dimensions,
    N_points=32,
    perc=0,
    smooth_density=True,
    length_thr=0.9,
    keep_endpoints=False,
    average_type="mean",
    endpoint_mode="median",
    representative=False,
    spline_smooth=None,
    verbose=True
):
    """
    In-memory backbone with optional core-streamline extraction.
    Returns: np.ndarray of shape (1, N_points, 3)
    """

    # ----------------------------------------------------------
    # 1. Core streamline extraction
    # ----------------------------------------------------------
    track = _ensure_streamlines(track)

    if perc != 0:
        if verbose:
            print("get core streamlines...")

        track = get_core_streamlines_from_streamlines(
            track,
            track_aff,
            dimensions,
            perc=perc,
            smooth_density=smooth_density,
            verbose=verbose
        )
        track = _ensure_streamlines(track)

    if len(track) == 0:
        return None

    # ----------------------------------------------------------
    # 2. Resample 
    # ----------------------------------------------------------
    track = Streamlines([_as_float32_streamline(sl) for sl in track if _as_float32_streamline(sl) is not None])
    track = set_number_of_points(track, N_points)

    # after resample, all should be same length -> stack into real float array
    track = np.asarray(track, dtype=np.float32)      # (M, N_points, 3) float32

    # ----------------------------------------------------------
    # 3. Orientation + length filtering
    # ----------------------------------------------------------
    ref = track[0]
    lengths = np.zeros(len(track))

    for i, sl in enumerate(track):
        d = np.linalg.norm(ref - sl)
        d_flip = np.linalg.norm(ref - sl[::-1])
        if d_flip < d:
            track[i] = sl[::-1]
        diff = np.diff(track[i], axis=0)          # (N-1,3) float32
        lengths[i] = np.linalg.norm(diff, axis=1).sum()


    keep = lengths > (length_thr * np.max(lengths))
    track = track[keep]

    if len(track) == 0:
        return None

    # ----------------------------------------------------------
    # 4. Backbone computation
    # ----------------------------------------------------------
    num_points = track.shape[1]
    backbone = np.zeros((num_points, 3))

    for p in range(num_points):
        coords = np.array([sl[p] for sl in track])

        if keep_endpoints and (p == 0 or p == num_points - 1):
            if endpoint_mode == "mean":
                backbone[p] = np.mean(coords, axis=0)
            elif endpoint_mode == "median":
                backbone[p] = np.median(coords, axis=0)
            elif endpoint_mode == "median_project":
                med = np.median(coords, axis=0)
                backbone[p] = coords[np.linalg.norm(coords - med, axis=1).argmin()]
            else:
                raise ValueError(endpoint_mode)
        else:
            backbone[p] = (
                np.median(coords, axis=0)
                if average_type == "median"
                else np.mean(coords, axis=0)
            )

    # ----------------------------------------------------------
    # 5. Representative (SAFE)
    # ----------------------------------------------------------
    if representative:
        from scipy.spatial import cKDTree
        tree = cKDTree(track.reshape(len(track), -1))
        _, idx = tree.query(backbone.reshape(-1), k=1)
        final = track[idx]              # (N,3)
    else:
        final = backbone                # (N,3)

    # ----------------------------------------------------------
    # 6. Spline smoothing
    # ----------------------------------------------------------
    if spline_smooth is not None:
        tck, _ = splprep(final.T, s=float(spline_smooth))
        u = np.linspace(0, 1, num_points)
        x, y, z = splev(u, tck)
        smoothed = np.vstack([x, y, z]).T

        if keep_endpoints:
            smoothed[0] = final[0]
            smoothed[-1] = final[-1]

        final = smoothed

    return final[np.newaxis, :, :]
