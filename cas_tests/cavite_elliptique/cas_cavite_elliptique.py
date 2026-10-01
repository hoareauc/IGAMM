"""
Cas test : Modes d'une cavité acoustique 2D elliptique par IGA NURBS (Patch unique)
Géométrie : Ellipse de demi-grand axe a et demi-petit axe b (« carré gonflé » anisotrope sans rotation).
Approche 100% procédurale (fonctions pures, sans POO).

Principe physique et mathématique :
    - Équation de Helmholtz : -Delta(p) = (omega / c)^2 * p dans l'ellipse x^2/a^2 + y^2/b^2 <= 1
    - Conditions aux limites : Parois rigides (Neumann homogène dp/dn = 0 sur le bord elliptique)
    - Levée de dégénérescence :
        Contrairement au cercle où les modes azimutaux (m >= 1) sont doublement dégénérés (fréquences identiques),
        l'excentricité de l'ellipse brise la symétrie de révolution SO(2).
        Les modes se séparent en modes pairs (symétriques / cosinus-elliptiques) et modes impairs (antisymétriques / sinus-elliptiques).
    - Représentation exacte par un seul patch NURBS :
        L'ellipse s'obtient par transformation affine du disque NURBS exact.
        Chaque point du bord satisfait (x/a)^2 + (y/b)^2 = 1 strictement à la précision machine.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

chemin_courant = os.path.dirname(os.path.abspath(__file__))
chemin_racine = os.path.abspath(os.path.join(chemin_courant, "..", ".."))
if chemin_racine not in sys.path:
    sys.path.insert(0, chemin_racine)

from iga import (
    creer_geometrie_ellipse_nurbs,
    assembler_systeme_acoustique_2d,
    resoudre_modes_acoustiques_2d,
)
from iga.bspline import trouver_intervalle, fonctions_base_et_derivees_nurbs_2d


def tracer_geometrie_et_control_net_ellipse(
    geo, nom_fichier="geometrie_control_net_cavite_elliptique.png", sauvegarder=True, afficher=False
):
    """
    Génère une figure explicative de la géométrie elliptique NURBS à patch unique :
    - Panneau 1 : Le patch de base grossier (3x3 points de contrôle avec poids) et ellipse exacte
    - Panneau 2 : Le patch raffiné avec les éléments IGA physiques, les points de contrôle et le « control net ».
    """
    a = geo['a']
    b = geo['b']
    P_ctrl = geo['points_ctrl']
    W_ctrl = geo['poids']
    P_base = geo['points_ctrl_base']
    W_base = geo['poids_base']
    n_xi, n_eta = P_ctrl.shape[0], P_ctrl.shape[1]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    
    # Ellipse analytique exacte
    theta_e = np.linspace(0, 2 * np.pi, 250)
    x_e = a * np.cos(theta_e)
    y_e = b * np.sin(theta_e)
    
    # -------------------------------------------------------------------------
    # PANNEAU 1 : Patch de base (Carré gonflé elliptique 3x3)
    # -------------------------------------------------------------------------
    ax1 = axes[0]
    ax1.plot(x_e, y_e, 'k-', linewidth=2.2, label=f"Contour elliptique exact ($a={a}, b={b}$ m)")
    
    # Lignes du réseau de contrôle de base
    for i in range(3):
        ax1.plot(P_base[i, :, 0], P_base[i, :, 1], 'r--', linewidth=1.5, alpha=0.8)
    for j in range(3):
        ax1.plot(P_base[:, j, 0], P_base[:, j, 1], 'r--', linewidth=1.5, alpha=0.8)
    ax1.plot([], [], 'r--', linewidth=1.5, label="Réseau de contrôle (Control Net)")
    
    # Points de contrôle avec annotations
    for i in range(3):
        for j in range(3):
            px, py = P_base[i, j]
            w = W_base[i, j]
            if i in (0, 2) and j in (0, 2):
                col = '#1f77b4'  # coin (w=1)
                lbl = f"$P_{{{i},{j}}}$\n$w=1$"
            elif i == 1 and j == 1:
                col = '#2ca02c'  # centre (w=1)
                lbl = f"$P_{{{i},{j}}}$ (centre)\n$w=1$"
            else:
                col = '#d62728'  # milieu bord (w=1/sqrt(2))
                lbl = f"$P_{{{i},{j}}}$\n$w=1/\\sqrt{{2}}$"
                
            ax1.plot(px, py, 'o', color=col, markersize=8, markeredgecolor='black', zorder=5)
            
            # Positionnement robuste des étiquettes
            if abs(py) > 1.2 * b:
                offset_x = 0.0
                offset_y = 0.16 * a if py > 0 else -0.18 * a
            elif abs(px) > 1.2 * a:
                offset_x = 0.18 * a if px > 0 else -0.18 * a
                offset_y = 0.0
            elif abs(px) < 0.05 and abs(py) < 0.05:
                offset_x = 0.0
                offset_y = -0.16 * a
            else:
                offset_x = 0.11 * a * (1 if px >= 0 else -1)
                offset_y = 0.11 * a * (1 if py >= 0 else -1)
                
            ax1.text(
                px + offset_x, py + offset_y, lbl, fontsize=9, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.25", facecolor='white', alpha=0.85, edgecolor='#cccccc')
            )
            
    ax1.set_title(
        f"Patch de base $3 \\times 3$ (Quadratique $p=2$)\n« Carré gonflé » elliptique sans rotation",
        fontsize=11, fontweight='bold', pad=10
    )
    ax1.set_xlabel("x (m)", fontsize=10)
    ax1.set_ylabel("y (m)", fontsize=10)
    ax1.set_xlim([-1.75 * a, 1.75 * a])
    ax1.set_ylim([-1.65 * a, 1.65 * a])
    ax1.set_aspect('equal')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)
    
    # -------------------------------------------------------------------------
    # PANNEAU 2 : Patch raffiné (Éléments IGA physiques & Réseau de contrôle)
    # -------------------------------------------------------------------------
    ax2 = axes[1]
    ax2.plot(x_e, y_e, 'k-', linewidth=2.2, label="Bord du domaine")
    
    U_xi = geo['U_xi']
    U_eta = geo['U_eta']
    knots_u = np.unique(U_xi)
    knots_v = np.unique(U_eta)
    lin_samp = np.linspace(0, 1, 60)
    
    def eval_pt(u_val, v_val):
        i_sp = trouver_intervalle(n_xi, 2, u_val, U_xi)
        j_sp = trouver_intervalle(n_eta, 2, v_val, U_eta)
        W_loc = W_ctrl[i_sp-2:i_sp+1, j_sp-2:j_sp+1]
        P_loc = P_ctrl[i_sp-2:i_sp+1, j_sp-2:j_sp+1]
        R_loc, _, _ = fonctions_base_et_derivees_nurbs_2d(i_sp, j_sp, u_val, v_val, 2, 2, U_xi, U_eta, W_loc)
        pt = np.zeros(2)
        loc = 0
        for j_idx in range(3):
            for i_idx in range(3):
                pt += R_loc[loc] * P_loc[i_idx, j_idx]
                loc += 1
        return pt

    # Lignes d'éléments iso-u
    for u in knots_u:
        pts = np.array([eval_pt(u, v) for v in lin_samp])
        ax2.plot(pts[:, 0], pts[:, 1], color='#1f77b4', linewidth=0.9, alpha=0.7)
    # Lignes d'éléments iso-v
    for v in knots_v:
        pts = np.array([eval_pt(u, v) for u in lin_samp])
        ax2.plot(pts[:, 0], pts[:, 1], color='#1f77b4', linewidth=0.9, alpha=0.7)
    ax2.plot([], [], color='#1f77b4', linewidth=1.2, label=f"Éléments IGA ({geo['n_el_xi']} x {geo['n_el_eta']})")
    
    # Réseau de contrôle raffiné (Control Net)
    for i in range(n_xi):
        ax2.plot(P_ctrl[i, :, 0], P_ctrl[i, :, 1], 'r--', linewidth=0.8, alpha=0.6)
    for j in range(n_eta):
        ax2.plot(P_ctrl[:, j, 0], P_ctrl[:, j, 1], 'r--', linewidth=0.8, alpha=0.6)
    ax2.plot([], [], 'r--', linewidth=1.2, label=f"Réseau de contrôle ({n_xi}x{n_eta})")
    
    # Points de contrôle raffinés
    P_flat = P_ctrl.reshape((-1, 2))
    ax2.plot(P_flat[:, 0], P_flat[:, 1], 'ro', markersize=4.5, markeredgecolor='black', alpha=0.85, label=f"Points de contrôle ({geo['n_dofs']} DDL)")
    
    ax2.set_title(
        f"Maillage IGA raffiné par insertion de nœuds\nÉléments physiques et Réseau de contrôle",
        fontsize=11, fontweight='bold', pad=10
    )
    ax2.set_xlabel("x (m)", fontsize=10)
    ax2.set_ylabel("y (m)", fontsize=10)
    ax2.set_xlim([-1.35 * a, 1.35 * a])
    ax2.set_ylim([-1.35 * a, 1.35 * a])
    ax2.set_aspect('equal')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)
    
    plt.suptitle(
        f"Paramétrisation Isogéométrique NURBS de la Cavité Elliptique ($a = {a}$ m, $b = {b}$ m)\n"
        f"Patch unique « Carré gonflé » sans rotation - Représentation stricte du contour elliptique",
        fontsize=13, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    
    chemin_sauvegarde = None
    if sauvegarder:
        dossier_sortie = os.path.dirname(os.path.abspath(__file__))
        chemin_sauvegarde = os.path.join(dossier_sortie, nom_fichier)
        plt.savefig(chemin_sauvegarde, dpi=300, bbox_inches='tight')
        print(f"[Géométrie] Figure du réseau de contrôle sauvegardée sous :\n  -> {chemin_sauvegarde}")
        
    if afficher:
        plt.show()
    plt.close(fig)
    return chemin_sauvegarde


def evaluer_modes_ellipse_grille(geo, modes, mode_indices, n_grid=100):
    """
    Évalue les champs de pression des modes acoustiques sur une grille cartésienne paramétrique pour l'ellipse.
    """
    n_ctrl_xi = geo['n_ctrl_xi']
    n_ctrl_eta = geo['n_ctrl_eta']
    U_xi = geo['U_xi']
    U_eta = geo['U_eta']
    P_ctrl = geo['points_ctrl']
    W_ctrl = geo['poids']
    
    u_lin = np.linspace(0.0, 1.0, n_grid)
    v_lin = np.linspace(0.0, 1.0, n_grid)
    
    X = np.zeros((n_grid, n_grid))
    Y = np.zeros((n_grid, n_grid))
    P_modes = {m_idx: np.zeros((n_grid, n_grid)) for m_idx in mode_indices}
    
    for j, v in enumerate(v_lin):
        j_span = trouver_intervalle(n_ctrl_eta, 2, v, U_eta)
        for i, u in enumerate(u_lin):
            i_span = trouver_intervalle(n_ctrl_xi, 2, u, U_xi)
            W_loc = W_ctrl[i_span-2:i_span+1, j_span-2:j_span+1]
            P_loc = P_ctrl[i_span-2:i_span+1, j_span-2:j_span+1]
            R_loc, _, _ = fonctions_base_et_derivees_nurbs_2d(i_span, j_span, u, v, 2, 2, U_xi, U_eta, W_loc)
            
            x_pt = 0.0
            y_pt = 0.0
            p_vals = {m_idx: 0.0 for m_idx in mode_indices}
            
            loc = 0
            for b_idx in range(3):
                for a_idx in range(3):
                    dof = (j_span - 2 + b_idx) * n_ctrl_xi + (i_span - 2 + a_idx)
                    R_val = R_loc[loc]
                    x_pt += R_val * P_loc[a_idx, b_idx, 0]
                    y_pt += R_val * P_loc[a_idx, b_idx, 1]
                    for m_idx in mode_indices:
                        p_vals[m_idx] += R_val * modes[dof, m_idx]
                    loc += 1
                    
            X[j, i] = x_pt
            Y[j, i] = y_pt
            for m_idx in mode_indices:
                P_modes[m_idx][j, i] = p_vals[m_idx]
                
    return X, Y, P_modes


def tracer_modes_cavite_elliptique(
    geo, freqs_iga, modes_propres,
    nom_fichier="modes_cavite_elliptique.png", sauvegarder=True, afficher=False
):
    """
    Trace les cartographies 2D de pression acoustique avec lignes nodales pour la cavité elliptique.
    Affiche les 6 premiers modes dynamiques (indices 1 à 6).
    """
    a = geo['a']
    b = geo['b']
    indices = [1, 2, 3, 4, 5, 6]
    X, Y, P_modes = evaluer_modes_ellipse_grille(geo, modes_propres, indices, n_grid=100)
    
    # Description physique des modes pour cavité elliptique (fonctions de Mathieu)
    descriptions = {
        1: "Dipôle longitudinal (axe x)\nMode pair $ce_1$",
        2: "Dipôle transversal (axe y)\nMode impair $se_1$",
        3: "Quadripôle mixte\nMode pair $ce_2$",
        4: "Quadripôle diagonal\nMode impair $se_2$",
        5: "Mode radial elliptique\nMode pair $ce_0$ (anneau déformé)",
        6: "Hexapôle longitudinal\nMode pair $ce_3$"
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.5))
    
    theta_e = np.linspace(0, 2 * np.pi, 250)
    x_e = a * np.cos(theta_e)
    y_e = b * np.sin(theta_e)
    
    for idx_plot, mode_idx in enumerate(indices):
        row = idx_plot // 3
        col = idx_plot % 3
        ax = axes[row, col]
        
        P_g = P_modes[mode_idx]
        p_max = np.max(np.abs(P_g))
        P_norm = P_g / p_max if p_max > 1e-14 else P_g
        
        levels_p = np.linspace(-1.0, 1.0, 51)
        cf = ax.contourf(X, Y, P_norm, levels=levels_p, cmap='RdBu_r', vmin=-1.0, vmax=1.0)
        # Lignes nodales (p = 0)
        ax.contour(X, Y, P_norm, levels=[0.0], colors='black', linewidths=1.8, linestyles='--')
        # Contour elliptique exact
        ax.plot(x_e, y_e, 'k-', linewidth=2.0)
        
        cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Pression normalisée", fontsize=9)
        
        f_calc = freqs_iga[mode_idx]
        desc = descriptions.get(mode_idx, "")
        ax.set_title(f"Mode #{mode_idx} : $f = {f_calc:.2f}$ Hz\n{desc}", fontsize=10.5, fontweight='bold', pad=8)
        ax.set_xlabel("x (m)", fontsize=9.5)
        ax.set_ylabel("y (m)", fontsize=9.5)
        ax.set_xlim([-1.18 * a, 1.18 * a])
        ax.set_ylim([-1.25 * b, 1.25 * b])
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.5)
        
    plt.suptitle(
        f"Modes propres acoustiques d'une cavité elliptique ($a = {a}$ m, $b = {b}$ m, $c = 343$ m/s)\n"
        f"Analyse Isogéométrique NURBS 2D (Patch unique « Carré gonflé » sans rotation)",
        fontsize=13, fontweight='bold', y=0.99
    )
    plt.tight_layout()
    
    chemin_sauvegarde = None
    if sauvegarder:
        dossier_sortie = os.path.dirname(os.path.abspath(__file__))
        chemin_sauvegarde = os.path.join(dossier_sortie, nom_fichier)
        plt.savefig(chemin_sauvegarde, dpi=300, bbox_inches='tight')
        print(f"[Visualisation] Modes acoustiques enregistrés sous :\n  -> {chemin_sauvegarde}")
        
    if afficher:
        plt.show()
    plt.close(fig)
    return chemin_sauvegarde


def executer_cas_cavite_elliptique(
    a=1.0,
    b=0.6,
    c=343.0,
    n_el_xi=14,
    n_el_eta=14,
    num_modes=8,
    sauvegarder_figures=True
):
    """
    Exécute le cas test de la cavité acoustique 2D elliptique par IGA NURBS.
    """
    print("=" * 88)
    print("  SIMULATION IGA NURBS 2D : CAVITÉ ACOUSTIQUE ELLIPTIQUE (PATCH UNIQUE)")
    print("=" * 88)
    print(f"  Demi-grand axe : a = {a:.3f} m")
    print(f"  Demi-petit axe : b = {b:.3f} m")
    print(f"  Rapport d'aspect : a/b = {a/b:.3f} | Excentricité : e = {np.sqrt(1.0 - (b/a)**2):.4f}")
    print(f"  Célérité du son : c = {c:.1f} m/s")
    print(f"  Discrétisation : {n_el_xi} x {n_el_eta} éléments NURBS quadratiques")
    
    # 1. Création de la géométrie NURBS carré gonflé elliptique
    t0 = time.time()
    geo = creer_geometrie_ellipse_nurbs(a=a, b=b, n_el_xi=n_el_xi, n_el_eta=n_el_eta)
    t_geo = time.time() - t0
    print(f"  Nombre de degrés de liberté (DDL) : {geo['n_dofs']}")
    print(f"  [1/4] Géométrie NURBS elliptique créée en {t_geo:.4f} s")
    
    # 2. Assemblage des matrices globales K et M
    t0 = time.time()
    K, M = assembler_systeme_acoustique_2d(geo)
    t_ass = time.time() - t0
    print(f"  [2/4] Matrices K et M assemblées en {t_ass:.4f} s (K non-nuls : {K.nnz})")
    
    # 3. Résolution du problème aux valeurs propres
    t0 = time.time()
    freqs_iga, modes_propres, _ = resoudre_modes_acoustiques_2d(K, M, num_modes=num_modes, c=c)
    t_solv = time.time() - t0
    print(f"  [3/4] {len(freqs_iga)} modes propres calculés en {t_solv:.4f} s")
    
    # 4. Affichage du tableau des fréquences obtenues
    print("\n" + "=" * 80)
    print(f"{'Mode':<6} | {'Fréquence IGA (Hz)':<20} | {'Description physique'}")
    print("-" * 80)
    descriptions = [
        "Mode rigide / pression constante (f = 0 Hz)",
        "Dipôle longitudinal selon le grand axe x (mode pair)",
        "Dipôle transversal selon le petit axe y (mode impair)",
        "Quadripôle mixte (mode pair)",
        "Quadripôle diagonal (mode impair)",
        "Mode radial elliptique (anneau déformé)",
        "Hexapôle longitudinal",
        "Hexapôle transversal"
    ]
    for m_idx in range(min(num_modes, len(freqs_iga))):
        desc = descriptions[m_idx] if m_idx < len(descriptions) else f"Mode #{m_idx}"
        f_val = freqs_iga[m_idx]
        print(f"{m_idx:<6} | {f_val:<20.4f} | {desc}")
    print("=" * 80 + "\n")
    
    # 5. Graphiques
    fig_net = None
    fig_modes = None
    if sauvegarder_figures:
        fig_net = tracer_geometrie_et_control_net_ellipse(geo, nom_fichier="geometrie_control_net_cavite_elliptique.png")
        fig_modes = tracer_modes_cavite_elliptique(geo, freqs_iga, modes_propres, nom_fichier="modes_cavite_elliptique.png")
        print(f"  [4/4] Figures géométrie et modes propres enregistrées avec succès.")
        
    print("=" * 88)
    print("  SIMULATION DE LA CAVITÉ ELLIPTIQUE TERMINÉE AVEC SUCCÈS")
    print("=" * 88)
    
    return {
        'geo': geo,
        'freqs_iga': freqs_iga,
        'modes_propres': modes_propres,
        'figure_control_net': fig_net,
        'figure_modes': fig_modes
    }


if __name__ == "__main__":
    executer_cas_cavite_elliptique(
        a=1.0,
        b=0.6,
        c=343.0,
        n_el_xi=14,
        n_el_eta=14,
        num_modes=8,
        sauvegarder_figures=True
    )
