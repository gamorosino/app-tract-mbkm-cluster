# Tractogram Clasturing using dissimilarity embedding and MiniBatch K-Means (app-tract-mbkm-cluster)

Orientation-invariant streamline dissimilarity embedding, MiniBatch K-Means clustering, and medoid extraction

## Overview

This app performs large-scale clustering of tractography streamlines using a dissimilarity representation and MiniBatch K-Means (MBKM), followed by extraction of representative streamlines (medoids).

Each streamline is embedded into a vector space defined by its distances to a set of prototype fibers, using an orientation-invariant metric. Clustering is then performed in this embedding space, and a representative streamline is selected per cluster.

This approach follows the dissimilarity projection framework introduced by Emanuele Olivetti and colleagues, enabling scalable clustering of very large tractograms while preserving anatomical validity (Olivetti et al., 2012, 2013).

---

## Key Features

* Orientation-invariant streamline distance (MDF)
* Dissimilarity embedding using prototype streamlines
* Scalable clustering via MiniBatch K-Means
* Medoid extraction (valid streamlines, not synthetic centroids)
* Streaming-friendly design for large datasets
* Supports `.trk` and `.tck` tractograms
* Parallel processing support
* Built on tools from DIPY (Garyfallidis et al., 2014)

---

## Method Summary

### 1. Streamline Embedding

Each streamline ( s_i ) is embedded as:

[
\phi(s_i) = [d(s_i, p_1), \dots, d(s_i, p_m)]
]

where:

* ( p_j ) are prototype streamlines
* ( d(\cdot,\cdot) ) is an orientation-invariant distance (MDF)

This formulation follows the dissimilarity projection framework described in:

* Olivetti et al., 2012
* Olivetti et al., 2013

Prototypes are selected using subset farthest-first traversal.

---

### 2. Clustering

MiniBatch K-Means is applied to the embedded representation:

* Handles large-scale datasets efficiently
* Avoids full pairwise distance computation
* Controlled via batch updates

This follows the web-scale clustering paradigm introduced by David Sculley (2010).

---

### 3. Medoid Extraction

For each cluster:

* Default: select streamline closest to centroid in embedding space
* Optional refinement: approximate true medoid using subset evaluation

Unlike centroids, medoids correspond to real streamlines and preserve anatomical plausibility.

---

## Inputs

| Name                 | Type                            | Description                             |
| -------------------- | ------------------------------- | --------------------------------------- |
| tractogram           | `track/tck` or `track/trk`      | Input streamlines                       |
| reference (optional) | `neuro/dwi` or `neuro/anat/t1w` | Required for `.tck` → `.trk` conversion |

---

## Outputs

| Name         | Type                        | Description                                |
| ------------ | --------------------------- | ------------------------------------------ |
| medoids      | `track/trk` or `track/tck`  | Representative streamlines (1 per cluster) |
| labels       | `raw`                       | Streamline → cluster assignments           |
| summary      | `raw`                       | JSON with clustering statistics            |
| qc           | `generic/images` (optional) | QC plots (cluster sizes, etc.)             |
| product.json | metadata                    | Brainlife visualization + logs             |

---

## Configuration Parameters

Example `config.json`:

```json
{
  "track": "input.tck",
  "reference": "dwi.nii.gz",
  "n_clusters": 80000,
  "nb_points": 20,
  "n_prototypes": 64,
  "distance": "mdf",
  "random_state": 42,
  "batch_size": 256,
  "medoid_mode": "centroid_proxy"
}
```

---

## Usage

### Local execution

```bash
singularity exec -e docker://gamorosino/tract_align:latest \
    micromamba run -n tract_align \
    python3 main.py
```

---

## References

* Olivetti, E., Nguyen, T. B., & Garyfallidis, E. (2012).
  *The approximation of the dissimilarity projection.*
  2012 2nd International Workshop on Pattern Recognition in NeuroImaging (PRNI).
  [https://doi.org/10.1109/prni.2012.13](https://doi.org/10.1109/prni.2012.13)

* Olivetti, E., Nguyen, T. B., Garyfallidis, E., Agarwal, N., & Avesani, P. (2013).
  *Fast clustering for interactive tractography segmentation.*
  2013 International Workshop on Pattern Recognition in Neuroimaging (PRNI).
  [https://doi.org/10.1109/prni.2013.20](https://doi.org/10.1109/prni.2013.20)

* Garyfallidis, E., Brett, M., Amirbekian, B., Rokem, A., van der Walt, S., Descoteaux, M., Nimmo-Smith, I., & DIPY Contributors. (2014).
  *DIPY, a library for the analysis of diffusion MRI data.*
  Frontiers in Neuroinformatics, 8, 8.
  [https://doi.org/10.3389/fninf.2014.00008](https://doi.org/10.3389/fninf.2014.00008)

* Sculley, D. (2010).
  *Web-scale k-means clustering.*

---

## Author

Gamorosino

---

## License

Specify your license (e.g., MIT, BSD-3-Clause)