import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse import issparse

#-------------------------------------------------------------------------------


def to_dense(X):
    return X.toarray() if issparse(X) else np.asarray(X)


def norm01(x, q=0.99):
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), np.quantile(x, q)
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((x - lo) / (hi - lo), 0, 1)


def spatial_plot(adata, gex_true, gex_pred):
    """
    Spatial plot of experimental Xenium and Demo model outputs.
    """
    genes = ["PECAM1", "MMRN2", "MYH11", "SFRP2", "COL5A2", "PLIN4"]
    var_names = adata.var_names.tolist()
    coords = adata.obsm['spatial']

    top_idx = [var_names.index(g) for g in genes]
    n_genes = len(genes)

    # apply spatial smoothing
    #adj = adata.obsp["spatial_connectivities"].copy()
    #adj.setdiag(1)
    #n_neighbors = np.array(adj.sum(1)).flatten()
    #n_neighbors[n_neighbors == 0] = 1
    #deg_inv = sp.diags(1.0 / n_neighbors)
    #gex_true = (deg_inv @ adj) @ gex_true
    #gex_pred = (deg_inv @ adj) @ gex_pred

    fig, axes = plt.subplots(
        n_genes,
        2,
        figsize=(9, 3.6 * n_genes),
        squeeze=False,
    )

    for row, (index, gene) in enumerate(zip(top_idx, genes)):
        cols = [
            (norm01(gex_true[:, index]), 'TRUE'),
            (norm01(gex_pred[:, index]), 'PRED (Demo, tol=1e-0)'),
        ]
        for col, (values, label) in enumerate(cols):
            ax = axes[row, col]
            sc = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=values,
                s=30,
                cmap='viridis',
                vmin=0,
                vmax=1,
                edgecolors='k',
                linewidths=0.15,
            )
            ax.set_title(f"{gene} — {label}")
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    plt.savefig("spatial-heatmap.png", dpi=600, bbox_inches='tight')
    plt.show()
