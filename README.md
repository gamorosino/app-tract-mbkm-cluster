<h1 align="center">Tractogram clustering</h1>

<p align="center">
  Clusters a large tractogram by embedding streamlines with an orientation-invariant dissimilarity to prototype fibres and running MiniBatch K-Means, outputting one medoid streamline per cluster.
</p>

<p align="center">
  <a href="https://github.com/brainlife/abcd-spec"><img src="https://img.shields.io/badge/ABCD_Spec-v1.1-green.svg" alt="ABCD Spec v1.1"></a>
  <a href="https://doi.org/10.25663/brainlife.app.916"><img src="https://img.shields.io/badge/Brainlife-brainlife.app.916-blue.svg" alt="Run on brainlife.io"></a>
</p>

## At a glance

<!-- brainlife-app:begin (generated from the App manifest; edit the manifest, not this block) -->

> **Clusters a large tractogram by embedding streamlines with an orientation-invariant dissimilarity to prototype fibres and running MiniBatch K-Means, outputting one medoid streamline per cluster.**

Resamples the streamlines, embeds each one as its MDF distances to a set of prototype streamlines chosen by farthest-first traversal (Olivetti dissimilarity projection), clusters the embedding with MiniBatch K-Means and selects a real streamline (medoid) per cluster. Use it to compress a very large tractogram into a representative set of streamlines. A reference image is needed to write the output with the right spatial header.

| | |
|---|---|
| **Inputs** | `neuro/track/tck` · `neuro/anat/t1w` |
| **Outputs** | `neuro/track/tck` |
| **Container** | `gamorosino/tract_align:v1.0` |
| **Resources** | 1 CPUs · 62 GB memory · 5 h walltime |
| **Run it** | [doi.org/10.25663/brainlife.app.916](https://doi.org/10.25663/brainlife.app.916) |

#### Inputs

| Input | Datatype | Required | Description |
|---|---|---|---|
| `track` | `neuro/track/tck` | yes | Whole-brain tractogram in .tck format. |
| `t1` | `neuro/anat/t1w` | yes | Reference anatomical image defining the tractogram's space, used for format conversion and headers. |

#### Outputs

| Output | Datatype | Description |
|---|---|---|
| `out_medoids` | `neuro/track/tck` | Tractogram with one medoid (representative real streamline) per cluster. |

<details>
<summary><b>Parameters</b> (3)</summary>

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_clusters` | number | `1000` | Number of MiniBatch K-Means clusters, i.e. the number of medoid streamlines in the output. |
| `n_jobs` | number | `-1` | Number of parallel jobs; -1 uses all available cores. |
| `max_num_streamlines` | number |  | Randomly subsample the input tractogram to a fixed number of streamlines. This is useful for debugging and reducing runtime on large datasets. Note that subsampling may affect clustering quality and reproducibility. |

</details>

<!-- brainlife-app:end -->

## Running the App

### On brainlife.io

Open the App at [doi.org/10.25663/brainlife.app.916](https://doi.org/10.25663/brainlife.app.916), choose `neuro/track/tck`, `neuro/anat/t1w` from your project, set the parameters and press **Execute**. The results are saved to your project as `neuro/track/tck`.

### Locally

You need [Singularity/Apptainer](https://apptainer.org/) (the App runs in `gamorosino/tract_align:v1.0`), [`jq`](https://jqlang.github.io/jq/).

1. Clone this repository.
2. Create `config.json` with the paths to your inputs and the parameters:

   ```json
   {
       "track": "/path/to/track/track",
       "reference": "/path/to/t1/reference",
       "n_clusters": 1000,
       "n_jobs": -1,
       "max_num_streamlines": "<number>"
   }
   ```

3. Run `./main`.

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

## 1. Streamline Embedding

Each streamline ( $s_i$ ) is embedded as:
```math
\phi(s_i) = [d(s_i, p_1), \dots, d(s_i, p_m)]
```
where:

* ( $p_j$ ) are prototype streamlines
* ( $d(\cdot,\cdot)$ ) is an orientation-invariant distance (MDF)

This formulation follows the dissimilarity projection framework described in:

* Olivetti et al., 2012
* Olivetti et al., 2013

Prototypes are selected using subset farthest-first traversal.

---

## 2. Clustering

MiniBatch K-Means is applied to the embedded representation:

* Handles large-scale datasets efficiently
* Avoids full pairwise distance computation
* Controlled via batch updates

This follows the web-scale clustering paradigm introduced by David Sculley (2010).

---

## 3. Medoid Extraction

For each cluster:

* Default: select streamline closest to centroid in embedding space
* Optional refinement: approximate true medoid using subset evaluation

Unlike centroids, medoids correspond to real streamlines and preserve anatomical plausibility.

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

## Local execution

```bash
singularity exec -e docker://gamorosino/tract_align:latest \
    micromamba run -n tract_align \
    python3 main.py
```

---

## Citing

If you use this App, please cite it, the software it builds on and brainlife.io:

- Tractogram clustering. brainlife.io App. [doi:10.25663/brainlife.app.916](https://doi.org/10.25663/brainlife.app.916)
- Olivetti E, Nguyen TB, Garyfallidis E. The approximation of the dissimilarity projection. PRNI (2012). [doi:10.1109/prni.2012.13](https://doi.org/10.1109/prni.2012.13)
- Olivetti E, Nguyen TB, Garyfallidis E, Agarwal N, Avesani P. Fast clustering for interactive tractography segmentation. PRNI (2013). [doi:10.1109/prni.2013.20](https://doi.org/10.1109/prni.2013.20)
- Garyfallidis E, Brett M, Amirbekian B, et al. DIPY, a library for the analysis of diffusion MRI data. Front Neuroinform 8:8 (2014). [doi:10.3389/fninf.2014.00008](https://doi.org/10.3389/fninf.2014.00008)
- Sculley D. Web-scale k-means clustering. WWW (2010).
- Sculley, D. (2010). *Web-scale k-means clustering.* ---
- Hayashi S, Caron BA, Heinsfeld AS, et al. brainlife.io: a decentralized and open-source cloud platform to support neuroscience research. *Nature Methods* 21:809–813 (2024). [doi:10.1038/s41592-024-02237-2](https://doi.org/10.1038/s41592-024-02237-2)
- Avesani P, McPherson B, Hayashi S, et al. The open diffusion data derivatives, brain data upcycling via integrated publishing of derivatives and reproducible open cloud services. *Scientific Data* 6:69 (2019). [doi:10.1038/s41597-019-0073-y](https://doi.org/10.1038/s41597-019-0073-y)

## Authors

Gabriele amorosino

---
