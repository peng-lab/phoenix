"""
Plotting helpers for comparing true and predicted spatial gene expression.

© Peng Lab / Helmholtz Munich
"""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import issparse


# -------------------------------------------------------------------------------
def to_dense(X):
    """
    Return `X` as a dense `numpy.ndarray`, converting it if it is sparse.

    Parameters
    ----------
    X
        A dense array-like or a `scipy.sparse` matrix.

    Returns
    -------
    Dense array.
    """
    return X.toarray() if issparse(X) else np.asarray(X)


def norm01(x, q=0.99):
    """
    Clip and rescale `x` to the ``[0, 1]`` range using its min and a given quantile.

    Parameters
    ----------
    x
        Array-like of values to normalize.
    q
        Quantile used as the upper bound of the rescaling range.

    Returns
    -------
    Array of the same shape as `x`, clipped to ``[0, 1]``.
    """
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), np.quantile(x, q)
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((x - lo) / (hi - lo), 0, 1)


def spatial_plot(adata, gex_true, gex_pred):
    """
    Plot true vs. predicted spatial expression for a fixed marker gene panel.

    Renders a two-column, one-row-per-gene grid of scatter plots over the spatial
    coordinates in `adata.obsm['spatial']`, and saves it to ``spatial-heatmap.png``.

    Parameters
    ----------
    adata
        Anndata object providing `var_names` and spatial coordinates.
    gex_true
        Ground-truth gene expression matrix, cells x genes.
    gex_pred
        Predicted gene expression matrix, cells x genes.
    """
    genes = ["PECAM1", "MMRN2", "MYH11", "SFRP2", "COL5A2", "PLIN4"]
    var_names = adata.var_names.tolist()
    coords = adata.obsm["spatial"]

    top_idx = [var_names.index(g) for g in genes]
    n_genes = len(genes)

    # apply spatial smoothing
    # adj = adata.obsp["spatial_connectivities"].copy()
    # adj.setdiag(1)
    # n_neighbors = np.array(adj.sum(1)).flatten()
    # n_neighbors[n_neighbors == 0] = 1
    # deg_inv = sp.diags(1.0 / n_neighbors)
    # gex_true = (deg_inv @ adj) @ gex_true
    # gex_pred = (deg_inv @ adj) @ gex_pred

    fig, axes = plt.subplots(
        n_genes,
        2,
        figsize=(9, 3.6 * n_genes),
        squeeze=False,
    )

    for row, (index, gene) in enumerate(zip(top_idx, genes, strict=True)):
        cols = [
            (norm01(gex_true[:, index]), "TRUE"),
            (norm01(gex_pred[:, index]), "PRED (Demo, tol=1e-0)"),
        ]
        for col, (values, label) in enumerate(cols):
            ax = axes[row, col]
            sc = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=values,
                s=30,
                cmap="viridis",
                vmin=0,
                vmax=1,
                edgecolors="k",
                linewidths=0.15,
            )
            ax.set_title(f"{gene} — {label}")
            ax.set_aspect("equal")
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    plt.savefig("spatial-heatmap.png", dpi=600, bbox_inches="tight")
    plt.show()


# -------------------------------------------------------------------------------
# SpatialData zarr plotting helpers
# -------------------------------------------------------------------------------

NORM = mcolors.Normalize(vmin=0, vmax=1)  # passed to render_shapes; set to a Normalize instance if needed
SHOW_KWARGS: dict = {}


def set_active_layer(table, layer_name):
    """
    Set `table.X` from a named layer.

    Parameters
    ----------
    table
        Anndata table whose active matrix (`X`) is set in place.
    layer_name
        Key into `table.layers` to make active.
    """
    table.X = table.layers[layer_name]


def plot_gene(sdata_obj, shape_key, gene, save_path, layer_name):
    """
    Render gene expression for a shape element and save it to disk.

    Parameters
    ----------
    sdata_obj
        SpatialData object (or a `spatialdata_plot`-compatible view of one) to render.
    shape_key
        Key of the shapes element to color by `gene`.
    gene
        Gene (column of `table.X`) to render.
    save_path
        Path the rendered figure is written to.
    layer_name
        Label used in the figure title, identifying which layer was rendered.
    """
    sdata_obj.pl.render_images("he_image").pl.render_shapes(
        shape_key, color=gene, fill_alpha=0.7, norm=NORM, method="matplotlib"
    ).pl.show(save=save_path, title=f"{gene}_{layer_name}", colorbar=True, **SHOW_KWARGS)


def spatial_zarr_plot(sdata, gex_pred, shape_key="cell_boundaries", table_key="table", genes=None, save_dir="."):
    """
    Plot true vs. predicted gene expression side-by-side using a SpatialData object.

    For each gene in `genes`, renders one figure for the ground-truth layer and one
    for the predicted layer, saving both under `save_dir`. The table's original `X`
    is restored before returning.

    Parameters
    ----------
    sdata
        SpatialData object containing the table and shapes elements to render.
    gex_pred
        Predicted gene expression matrix, cells x genes, aligned to `sdata.tables[table_key]`.
    shape_key
        Key of the shapes element to render expression on.
    table_key
        Key of the anndata table in `sdata.tables` to read/write expression layers on.
    genes
        Genes to plot; defaults to a fixed six-gene marker panel.
    save_dir
        Directory the rendered figures are written to; created if missing.
    """
    import os

    if genes is None:
        genes = ["PECAM1", "MMRN2", "MYH11", "SFRP2", "COL5A2", "PLIN4"]

    table = sdata.tables[table_key]
    var_names = table.var_names.tolist()
    table.layers["gt_raw"] = table.X

    # Use norm01 for normalization
    gt_dense = table.X.toarray() if issparse(table.X) else np.asarray(table.X, dtype=float)
    table.layers["gt"] = norm01(gt_dense)

    pred_dense = gex_pred.toarray() if issparse(gex_pred) else np.asarray(gex_pred, dtype=float)
    table.layers["pred"] = norm01(pred_dense)

    os.makedirs(save_dir, exist_ok=True)

    for gene in genes:
        if gene not in var_names:
            print(f"[spatial_zarr_plot] Warning: '{gene}' not found in table.var_names – skipping.")
            continue

        for layer_name in ["gt", "pred"]:
            set_active_layer(table, layer_name)
            save_path = os.path.join(save_dir, f"{gene}_{layer_name}.png")
            plot_gene(sdata, shape_key, gene, save_path, layer_name)

    if "gt_raw" in table.layers:
        table.X = table.layers["gt_raw"]
