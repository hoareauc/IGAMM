"""
Module post_traitement.py
Fonctions de post-traitement, évaluation sur grille régulière, visualisation des modes
et validation analytique pour l'acoustique 2D en IGA.
Approche 100% procédurale (fonctions pures, sans POO).
"""

import numpy as np
import matplotlib.pyplot as plt
from .bspline import trouver_intervalle, fonctions_base_1d


def evaluer_matrice_base_1d(n_ctrl, p, U, coords_1d):
    """
    Calcule la matrice des fonctions de base 1D B(k, i) = N_{i, p}(coords[k])
    pour un ensemble de coordonnées données.
    
    Paramètres :
        n_ctrl (int) : Nombre de points de contrôle
        p (int) : Degré polynomial
        U (ndarray) : Vecteur de nœuds
        coords_1d (ndarray de taille N) : Points d'évaluation
        
    Retourne :
        B (ndarray de forme (N, n_ctrl)) : Matrice des fonctions de base
    """
    N_pts = len(coords_1d)
    B = np.zeros((N_pts, n_ctrl), dtype=float)
    
    for k, u in enumerate(coords_1d):
        # Clip pour sécurité numérique
        u_val = np.clip(u, U[0], U[-1])
        i_span = trouver_intervalle(n_ctrl, p, u_val, U)
        vals = fonctions_base_1d(i_span, u_val, p, U)
        for a in range(p + 1):
            ctrl_idx = i_span - p + a
            B[k, ctrl_idx] = vals[a]
            
    return B


def evaluer_champ_pression_grille(geo, vecteur_mode, n_pts_x=100, n_pts_y=100):
    """
    Évalue le champ de pression acoustique p(x, y) sur une grille cartésienne régulière
    en utilisant l'accélération tensorielle B_xi * Phi * B_eta^T.
    
    Paramètres :
        geo (dict) : Description de la géométrie IGA
        vecteur_mode (ndarray de taille n_dofs) : Degrés de liberté du mode propre
        n_pts_x (int) : Nombre de points de discrétisation selon x
        n_pts_y (int) : Nombre de points de discrétisation selon y
        
    Retourne :
        X_grille (ndarray 2D) : Coordonnées x (m)
        Y_grille (ndarray 2D) : Coordonnées y (m)
        P_grille (ndarray 2D de forme (n_pts_y, n_pts_x)) : Valeurs du champ de pression acoustique
    """
    Lx = geo['Lx']
    Ly = geo['Ly']
    p_xi = geo['p_xi']
    p_eta = geo['p_eta']
    U_xi = geo['U_xi']
    U_eta = geo['U_eta']
    n_ctrl_xi = geo['n_ctrl_xi']
    n_ctrl_eta = geo['n_ctrl_eta']
    
    # Coordonnées physiques régulières
    x_lin = np.linspace(0.0, Lx, n_pts_x)
    y_lin = np.linspace(0.0, Ly, n_pts_y)
    X_grille, Y_grille = np.meshgrid(x_lin, y_lin)
    
    # Coordonnées paramétriques normalisées [0, 1]
    xi_lin = np.linspace(0.0, 1.0, n_pts_x)
    eta_lin = np.linspace(0.0, 1.0, n_pts_y)
    
    # Évaluation des matrices de base 1D
    B_xi = evaluer_matrice_base_1d(n_ctrl_xi, p_xi, U_xi, xi_lin)     # (n_pts_x, n_ctrl_xi)
    B_eta = evaluer_matrice_base_1d(n_ctrl_eta, p_eta, U_eta, eta_lin) # (n_pts_y, n_ctrl_eta)
    
    # Reformatage du vecteur modal en matrice 2D (ordre [i_xi, j_eta])
    # Dans l'assemblage, dof = j * n_ctrl_xi + i -> ordre 'C' pour reshape((n_ctrl_eta, n_ctrl_xi))
    Phi_2d = vecteur_mode.reshape((n_ctrl_eta, n_ctrl_xi))
    
    # Évaluation par produit tensoriel :
    # P(y, x) = B_eta * Phi_2d * B_xi^T
    P_grille = np.dot(B_eta, np.dot(Phi_2d, B_xi.T))
    
    return X_grille, Y_grille, P_grille


def afficher_comparaison_modes(frequences_iga, frequences_exactes, ordres_modaux, num_modes=8):
    """
    Affiche un tableau comparatif formaté entre les fréquences propres IGA et les fréquences analytiques.
    
    Paramètres :
        frequences_iga (ndarray) : Fréquences calculées par IGA (Hz)
        frequences_exactes (ndarray) : Fréquences exactes analytiques (Hz)
        ordres_modaux (list of tuple) : Couples d'indices (m, n)
        num_modes (int) : Nombre de modes à afficher
    """
    n_affiche = min(num_modes, len(frequences_iga), len(frequences_exactes))
    
    print("\n" + "=" * 80)
    print(f"{'Mode':<6} | {'(m, n)':<8} | {'f_analytique (Hz)':<18} | {'f_IGA (Hz)':<14} | {'Erreur abs (Hz)':<16} | {'Erreur rel (%)'}")
    print("-" * 80)
    
    for i in range(n_affiche):
        m, n = ordres_modaux[i]
        f_exact = frequences_exactes[i]
        f_iga = frequences_iga[i]
        err_abs = abs(f_iga - f_exact)
        if f_exact > 1e-12:
            err_rel = (err_abs / f_exact) * 100.0
            print(f"{i:<6} | ({m}, {n}){' '*(4 - len(str(m)) - len(str(n)))} | {f_exact:<18.4f} | {f_iga:<14.4f} | {err_abs:<16.4e} | {err_rel:<.4f}%")
        else:
            print(f"{i:<6} | ({m}, {n}){' '*(4 - len(str(m)) - len(str(n)))} | {f_exact:<18.4f} | {f_iga:<14.4f} | {err_abs:<16.4e} | 0.0000% (Mode constant)")
            
    print("=" * 80 + "\n")


def visualiser_modes_acoustiques(geo, frequences_iga, modes_propres, indices_modes=None,
                                 ordres_modaux=None, chemin_sauvegarde=None, afficher=False):
    """
    Trace les cartographies de pression acoustique des modes propres sélectionnés
    avec les lignes nodales (pression nulle).
    
    Paramètres :
        geo (dict) : Géométrie IGA
        frequences_iga (ndarray) : Fréquences calculées (Hz)
        modes_propres (ndarray) : Matrice des vecteurs propres
        indices_modes (list of int, optionnel) : Liste des indices des modes à tracer (défaut [1, 2, 3, 4, 5, 6])
        ordres_modaux (list of tuple, optionnel) : Indices analytiques (m, n) pour les titres
        chemin_sauvegarde (str, optionnel) : Chemin du fichier image à sauvegarder (png, pdf, ...)
        afficher (bool) : Appeler plt.show() si True
    """
    if indices_modes is None:
        # Par défaut, on affiche les 6 premiers modes acoustiques dynamiques (hors mode rigide 0)
        n_dispo = modes_propres.shape[1]
        indices_modes = [i for i in range(1, min(7, n_dispo))]
        
    n_plots = len(indices_modes)
    if n_plots == 0:
        return
        
    # Disposition des sous-figures en grille (ex: 2 colonnes)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.2 * n_rows), squeeze=False)
    
    Lx = geo['Lx']
    Ly = geo['Ly']
    
    for idx_plot, mode_idx in enumerate(indices_modes):
        row = idx_plot // n_cols
        col = idx_plot % n_cols
        ax = axes[row, col]
        
        # Vecteur propre correspondant
        phi_mode = modes_propres[:, mode_idx]
        
        # Évaluation sur grille fine
        X_g, Y_g, P_g = evaluer_champ_pression_grille(geo, phi_mode, n_pts_x=120, n_pts_y=120)
        
        # Normalisation pour la visualisation [-1, 1]
        p_max = np.max(np.abs(P_g))
        if p_max > 1e-14:
            P_norm = P_g / p_max
        else:
            P_norm = P_g
            
        # Tracé des isovaleurs avec carte de couleur divergente
        cf = ax.contourf(X_g, Y_g, P_norm, levels=40, cmap='RdBu_r', vmin=-1.0, vmax=1.0)
        
        # Tracé de la ligne nodale (p = 0)
        ax.contour(X_g, Y_g, P_norm, levels=[0.0], colors='black', linewidths=1.5, linestyles='--')
        
        cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Pression acoustique normalisée", fontsize=9)
        
        # Titre avec fréquence et éventuellement indice (m, n)
        titre = f"Mode #{mode_idx} : f = {frequences_iga[mode_idx]:.2f} Hz"
        if ordres_modaux is not None and mode_idx < len(ordres_modaux):
            m, n = ordres_modaux[mode_idx]
            titre += f" (m={m}, n={n})"
            
        ax.set_title(titre, fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel("x (m)", fontsize=10)
        ax.set_ylabel("y (m)", fontsize=10)
        ax.set_xlim([0, Lx])
        ax.set_ylim([0, Ly])
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.5)
        
    # Cacher les axes inutilisés
    for idx_plot in range(n_plots, n_rows * n_cols):
        row = idx_plot // n_cols
        col = idx_plot % n_cols
        axes[row, col].axis('off')
        
    plt.tight_layout()
    
    if chemin_sauvegarde:
        plt.savefig(chemin_sauvegarde, dpi=300, bbox_inches='tight')
        print(f"[Visualisation] Figure sauvegardée avec succès sous : {chemin_sauvegarde}")
        
    if afficher:
        plt.show()
        
    plt.close(fig)
