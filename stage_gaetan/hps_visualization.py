import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def to_numpy(t):
    return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)


def viewing_directions(rot_mats):
    return rot_mats[..., :, 2]


def plot_search_round(rot_mats, 
                      errors, 
                      kept_idx, 
                      iteration,
                      save_fold,
                      index):
    """
    Génère un graphique 3D des directions de vue candidates et des erreurs associées pour une itération donnée de HPS
    - rot_mats: matrices de rotation candidates
    - errors: erreurs associées à chaque matrice de rotation candidate
    - kept_idx: indices des meilleures matrices de rotation conservées
    - iteration: numéro de l'itération actuelle
    - save_fold: dossier où sauvegarder le graphique
    - index: indice de la vue traitée
    """
    rot_mats = to_numpy(rot_mats)
    errors = to_numpy(errors)
    kept_idx = to_numpy(kept_idx)

    if rot_mats.ndim == 4: 
        rot_mats = rot_mats[0]
    if kept_idx.ndim == 2:
        kept_idx = kept_idx[0]

    dirs = viewing_directions(rot_mats)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    sx, sy, sz = np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v)
    ax.plot_wireframe(sx, sy, sz, color="0.85", linewidth=0.3, zorder=0)

    poses = ax.scatter(dirs[:, 0], 
                     dirs[:, 1], 
                     dirs[:, 2],
                     c=errors, 
                     cmap="YlOrBr", 
                     s=35, 
                     edgecolors="none",
                     depthshade=True, 
                     zorder=2)

    kept_dirs = dirs[kept_idx]
    kept = ax.scatter(kept_dirs[:, 0], 
                kept_dirs[:, 1], 
                kept_dirs[:, 2],
                facecolors="none", 
                edgecolors="teal", 
                linewidths=2,
                s=90, 
                zorder=3, 
                label="kept")

    cbar = fig.colorbar(poses, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("error")

    n_kept = len(kept_idx)
    ax.set_title(f"Iter {iteration}: {dirs.shape[0]} candidates, {n_kept} kept")
    ax.set_box_aspect([1, 1, 1])
    ax.legend(loc="upper right", fontsize=8)

    os.makedirs(save_fold, exist_ok=True)
    out_path = os.path.join(save_fold, f"view_{index}_round_{iteration}.png")
    fig.savefig(out_path)
    plt.close(fig)