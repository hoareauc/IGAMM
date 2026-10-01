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


# =============================================================================
# Visualisation 3D avancée avec PyVista
# =============================================================================

def evaluer_mode_sur_grille_nurbs(geo, phi_mode, n_grid=90):
    """
    Évalue les coordonnées cartésiennes physiques (X, Y) et le champ modal P
    sur une grille paramétrique uniforme (u, v) in [0, 1]^2 pour un patch NURBS 2D.
    
    Paramètres :
        geo (dict) : Géométrie NURBS 2D (disque, ellipse ou patch général)
        phi_mode (ndarray) : Vecteur propre modal de taille n_dofs
        n_grid (int) : Résolution de la grille paramétrique (n_grid x n_grid)
        
    Retourne :
        X (ndarray 2D) : Coordonnées x physiques
        Y (ndarray 2D) : Coordonnées y physiques
        P (ndarray 2D) : Amplitude de pression acoustique modale
    """
    from .bspline import fonctions_base_et_derivees_nurbs_2d
    
    n_ctrl_xi = geo['n_ctrl_xi']
    n_ctrl_eta = geo['n_ctrl_eta']
    U_xi = geo['U_xi']
    U_eta = geo['U_eta']
    P_ctrl = geo['points_ctrl']
    W_ctrl = geo['poids']
    
    u_lin = np.linspace(0.0, 1.0, n_grid)
    v_lin = np.linspace(0.0, 1.0, n_grid)
    
    X = np.zeros((n_grid, n_grid), dtype=float)
    Y = np.zeros((n_grid, n_grid), dtype=float)
    P = np.zeros((n_grid, n_grid), dtype=float)
    
    for j, v in enumerate(v_lin):
        j_span = trouver_intervalle(n_ctrl_eta, 2, v, U_eta)
        for i, u in enumerate(u_lin):
            i_span = trouver_intervalle(n_ctrl_xi, 2, u, U_xi)
            W_loc = W_ctrl[i_span - 2:i_span + 1, j_span - 2:j_span + 1]
            P_loc = P_ctrl[i_span - 2:i_span + 1, j_span - 2:j_span + 1]
            R_loc, _, _ = fonctions_base_et_derivees_nurbs_2d(
                i_span, j_span, u, v, 2, 2, U_xi, U_eta, W_loc
            )
            
            x_pt = 0.0
            y_pt = 0.0
            p_pt = 0.0
            loc = 0
            for b_idx in range(3):
                for a_idx in range(3):
                    dof = (j_span - 2 + b_idx) * n_ctrl_xi + (i_span - 2 + a_idx)
                    R_val = R_loc[loc]
                    x_pt += R_val * P_loc[a_idx, b_idx, 0]
                    y_pt += R_val * P_loc[a_idx, b_idx, 1]
                    p_pt += R_val * phi_mode[dof]
                    loc += 1
                    
            X[j, i] = x_pt
            Y[j, i] = y_pt
            P[j, i] = p_pt
            
    return X, Y, P


def creer_grille_pyvista_mode(geo, phi_mode, n_grid=90, elevation_3d=True, amplitude_z=0.35):
    """
    Construit un maillage structuré PyVista (StructuredGrid) représentant le mode acoustique.
    Peut être représenté à plat (z=0) ou en nappe vibrante 3D (z proportionnel à la pression).
    
    Paramètres :
        geo (dict) : Géométrie IGA
        phi_mode (ndarray) : Vecteur propre modal
        n_grid (int) : Résolution de la grille (n_grid x n_grid)
        elevation_3d (bool) : Activer le relief / gauchissement 3D (z = amplitude * p_norm)
        amplitude_z (float) : Facteur d'échelle de hauteur pour la nappe vibrante 3D
        
    Retourne :
        grid (pyvista.StructuredGrid) : Maillage 3D PyVista avec champs scalaires
    """
    import pyvista as pv
    
    X, Y, P = evaluer_mode_sur_grille_nurbs(geo, phi_mode, n_grid=n_grid)
    
    p_max = np.max(np.abs(P))
    P_norm = P / p_max if p_max > 1e-14 else P
    
    if elevation_3d:
        Z = amplitude_z * P_norm
    else:
        Z = np.zeros_like(P_norm)
        
    grid = pv.StructuredGrid(X.T, Y.T, Z.T)
    grid.point_data['Pression'] = P.ravel(order='C')
    grid.point_data['Pression_Normalisee'] = P_norm.ravel(order='C')
    grid.point_data['Elevation'] = Z.ravel(order='C')
    
    return grid


def creer_contour_bord_pyvista(geo, n_pts=250):
    """
    Génère une courbe 3D fermée (PolyData) matérialisant la frontière rigide du domaine au plan z=0.
    """
    import pyvista as pv
    
    g_type = geo.get('type', 'rectangle')
    theta = np.linspace(0.0, 2.0 * np.pi, n_pts)
    
    if g_type == 'disque':
        R = geo['R']
        pts = np.column_stack([R * np.cos(theta), R * np.sin(theta), np.zeros(n_pts)])
    elif g_type == 'ellipse':
        a = geo['a']
        b = geo['b']
        pts = np.column_stack([a * np.cos(theta), b * np.sin(theta), np.zeros(n_pts)])
    else:
        Lx, Ly = geo['Lx'], geo['Ly']
        pts = np.array([
            [0, 0, 0], [Lx, 0, 0], [Lx, Ly, 0], [0, Ly, 0], [0, 0, 0]
        ], dtype=float)
        
    poly = pv.lines_from_points(pts, close=True)
    return poly


def visualiser_mode_pyvista_3d(
    geo,
    phi_mode,
    freq=None,
    mode_id=1,
    amplitude_z=0.35,
    chemin_sauvegarde=None,
    afficher=False
):
    """
    Génère un rendu 3D haute qualité d'un mode acoustique avec PyVista :
    - Surface 3D gauchie en élévation (onde stationnaire / nappe vibrante)
    - Carte de couleur divergente symétrique ('RdBu_r')
    - Ligne nodale (p = 0) en noir
    - Bordure rigide circulaire / elliptique au plan moyen z = 0
    - Éclairage réaliste et barre d'échelle
    
    Paramètres :
        geo (dict) : Géométrie IGA
        phi_mode (ndarray) : Vecteur modal
        freq (float, optionnel) : Fréquence du mode propre en Hz
        mode_id (int) : Numéro d'indice du mode
        amplitude_z (float) : Amplitude de l'élévation 3D
        chemin_sauvegarde (str, optionnel) : Chemin de sortie de l'image PNG
        afficher (bool) : Ouvrir la fenêtre interactive si True
    """
    import pyvista as pv
    
    grid = creer_grille_pyvista_mode(geo, phi_mode, n_grid=110, elevation_3d=True, amplitude_z=amplitude_z)
    poly_bord = creer_contour_bord_pyvista(geo)
    
    # Extraction de la ligne nodale 3D (p = 0)
    contour_nodal = grid.contour(isosurfaces=[0.0], scalars='Pression_Normalisee')
    
    pl = pv.Plotter(off_screen=not afficher, window_size=(1400, 1000))
    pl.set_background('white')
    
    # 1. Surface vibrante gauchie
    sbar_kwargs = dict(
        title="Pression acoustique normalisée",
        vertical=False,
        position_x=0.25,
        position_y=0.06,
        width=0.50,
        height=0.08,
        title_font_size=13,
        label_font_size=11,
        color='black'
    )
    pl.add_mesh(
        grid,
        scalars='Pression_Normalisee',
        cmap='RdBu_r',
        clim=[-1.0, 1.0],
        smooth_shading=True,
        specular=0.25,
        ambient=0.20,
        diffuse=0.85,
        scalar_bar_args=sbar_kwargs
    )
    
    # 2. Ligne nodale (pression nulle)
    if contour_nodal.n_points > 0:
        tube_nodal = contour_nodal.tube(radius=0.012)
        pl.add_mesh(tube_nodal, color='black', label="Ligne nodale (p = 0)")
        
    # 3. Contour rigide de référence à z = 0
    tube_bord = poly_bord.tube(radius=0.008)
    pl.add_mesh(tube_bord, color='#444444', label="Bord du domaine (z = 0)")
    
    # Titre
    titre_txt = f"Cavité Circulaire IGA NURBS - Mode #{mode_id}"
    if freq is not None:
        titre_txt += f" : f = {freq:.2f} Hz"
    pl.add_text(titre_txt, position='upper_left', font_size=13, color='black')
    pl.add_text("Visualisation 3D PyVista - Surface en onde stationnaire z = p(x, y)", position='upper_right', font_size=10, color='gray')
    
    # Position de la caméra isométrique optimisée
    pl.camera_position = [(2.2, -2.1, 1.7), (0.0, 0.0, 0.0), (-0.3, 0.3, 0.9)]
    
    if afficher:
        if chemin_sauvegarde:
            pl.show(screenshot=chemin_sauvegarde)
            print(f"[PyVista 3D] Rendu sauvegardé sous : {chemin_sauvegarde}")
        else:
            pl.show()
    else:
        if chemin_sauvegarde:
            pl.show(screenshot=chemin_sauvegarde)
            print(f"[PyVista 3D] Rendu sauvegardé sous : {chemin_sauvegarde}")
            
    try:
        pl.close()
    except Exception:
        pass
        
    return chemin_sauvegarde


def visualiser_planche_modes_pyvista_3d(
    geo,
    freqs_iga,
    modes_propres,
    indices_modes=(1, 2, 3, 4, 5, 6),
    ordres_modaux=None,
    amplitude_z=0.30,
    chemin_sauvegarde=None,
    afficher=False
):
    """
    Génère une planche maîtresse 2x3 avec PyVista montrant les 6 premiers modes dynamiques
    en surfaces ondulatoires 3D (relief de pression).
    """
    import pyvista as pv
    
    n_modes = len(indices_modes)
    n_rows = 2
    n_cols = 3
    
    pl = pv.Plotter(shape=(n_rows, n_cols), off_screen=not afficher, window_size=(1920, 1150))
    poly_bord = creer_contour_bord_pyvista(geo)
    tube_bord = poly_bord.tube(radius=0.009)
    
    for idx_plot, mode_idx in enumerate(indices_modes[:6]):
        r = idx_plot // n_cols
        c = idx_plot % n_cols
        pl.subplot(r, c)
        pl.set_background('white')
        
        phi = modes_propres[:, mode_idx]
        grid = creer_grille_pyvista_mode(geo, phi, n_grid=85, elevation_3d=True, amplitude_z=amplitude_z)
        contour_nodal = grid.contour(isosurfaces=[0.0], scalars='Pression_Normalisee')
        
        show_sbar = (r == 1 and c == 1)
        sbar_kwargs = dict(
            title="Pression normalisée",
            vertical=False,
            position_x=0.20,
            position_y=0.04,
            width=0.60,
            height=0.10,
            title_font_size=11,
            label_font_size=9,
            color='black'
        ) if show_sbar else None
        
        pl.add_mesh(
            grid,
            scalars='Pression_Normalisee',
            cmap='RdBu_r',
            clim=[-1.0, 1.0],
            smooth_shading=True,
            specular=0.2,
            ambient=0.25,
            show_scalar_bar=show_sbar,
            scalar_bar_args=sbar_kwargs
        )
        
        if contour_nodal.n_points > 0:
            pl.add_mesh(contour_nodal.tube(radius=0.012), color='black')
            
        pl.add_mesh(tube_bord, color='#555555')
        
        f_val = freqs_iga[mode_idx]
        titre = f"Mode #{mode_idx} : {f_val:.2f} Hz"
        if ordres_modaux is not None and mode_idx < len(ordres_modaux):
            m, n = ordres_modaux[mode_idx]
            titre += f" (m={m}, n={n})"
            
        pl.add_text(titre, position='upper_left', font_size=11, color='black')
        pl.camera_position = [(2.1, -2.1, 1.6), (0.0, 0.0, 0.0), (-0.3, 0.3, 0.9)]
        
    if afficher:
        if chemin_sauvegarde:
            pl.show(screenshot=chemin_sauvegarde)
            print(f"[PyVista Planche 3D] Rendu sauvegardé sous : {chemin_sauvegarde}")
        else:
            pl.show()
    else:
        if chemin_sauvegarde:
            pl.show(screenshot=chemin_sauvegarde)
            print(f"[PyVista Planche 3D] Rendu sauvegardé sous : {chemin_sauvegarde}")
            
    try:
        pl.close()
    except Exception:
        pass
        
    return chemin_sauvegarde

