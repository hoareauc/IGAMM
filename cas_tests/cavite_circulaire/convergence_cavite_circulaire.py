"""
Cas test : Analyse de convergence et de performance (temps de calcul) de la cavité circulaire 2D par IGA NURBS
Géométrie : Patch unique « Carré gonflé » sans rotation.
Discrétisation : Raffinement h par insertion de nœuds (préservation exacte de la géométrie circulaire)
Modes acoustiques cibles :
    - Mode 1 : (m=1, n=1) Dipolaire (f ≈ 100.51 Hz)
    - Mode 3 : (m=2, n=1) Quadripolaire (f ≈ 166.73 Hz)
    - Mode 5 : (m=0, n=1) Radial / Monopolaire (f ≈ 209.17 Hz)
    - Mode 6 : (m=3, n=1) Hexapolaire (f ≈ 229.34 Hz)
    - Mode 10 : (m=1, n=2) Harmonique radiale dipolaire (f ≈ 291.04 Hz)

Génère 3 visualisations graphiques détaillées :
1. convergence_cavite_circulaire_modes.png : Erreur relative en fréquence vs DDL
2. temps_calcul_cavite_circulaire.png : Temps d'assemblage, temps de résolution et total vs DDL
3. analyse_complete_cavite_circulaire.png : Planche maîtresse complète

Approche 100% procédurale (fonctions pures, sans POO).
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
    creer_geometrie_disque_nurbs,
    assembler_systeme_acoustique_2d,
    resoudre_modes_acoustiques_2d,
    frequences_analytiques_cavite_circulaire,
)


def executer_analyse_convergence_circulaire(
    R=1.0,
    c=343.0,
    liste_n_el=(4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 40),
    modes_cibles=(1, 3, 5, 6, 10),
    nb_repetitions=2,
    sauvegarder_figures=True,
    afficher=False
):
    """
    Exécute l'analyse paramétrique de convergence h et de temps de calcul pour la cavité circulaire NURBS.
    """
    print("=" * 96)
    print("  ANALYSE DE CONVERGENCE & TEMPS DE CALCUL : CAVITÉ CIRCULAIRE NURBS (IGA 2D)")
    print("=" * 96)
    print(f"  Rayon de la cavité : R = {R} m | Célérité du son : c = {c} m/s")
    print(f"  Patch NURBS : Patch unique « Carré gonflé » (degré p=2)")
    print(f"  Discrétisations testées (N_el x N_el) : {list(liste_n_el)}")
    print(f"  Modes acoustiques cibles : {list(modes_cibles)}")
    print("-" * 96)
    
    # 1. Calcul des fréquences analytiques exactes (zéros de Bessel J'_m)
    max_mode_idx = max(modes_cibles)
    freqs_ana, ordres_modaux, zeros_bessel = frequences_analytiques_cavite_circulaire(
        R=R, c=c, max_m=8, max_n=6
    )
    
    infos_modes = {}
    for m_idx in modes_cibles:
        f_exact = freqs_ana[m_idx]
        ordre = ordres_modaux[m_idx]
        z_val = zeros_bessel[m_idx]
        infos_modes[m_idx] = {'f_exact': f_exact, 'ordre': ordre, 'zero': z_val}
        print(f"  -> Mode #{m_idx:2d} : f_analytique = {f_exact:.4f} Hz | (m={ordre[0]}, n={ordre[1]}) | alpha' = {z_val:.4f}")
        
    print("-" * 96)
    
    # Structure de résultats
    resultats = {
        'n_el': [],
        'dofs': [],
        'h': [],
        't_ass': [],
        't_solv': [],
        't_total': [],
        'erreurs': {m_idx: [] for m_idx in modes_cibles},
        'freqs_iga': {m_idx: [] for m_idx in modes_cibles}
    }
    
    t_global_0 = time.time()
    
    for n_el in liste_n_el:
        geo = creer_geometrie_disque_nurbs(R=R, n_el_xi=n_el, n_el_eta=n_el)
        n_dofs = geo['n_dofs']
        h_approx = 2.0 * R / n_el
        
        t_ass_passes = []
        t_solv_passes = []
        freqs_iga = None
        
        for _ in range(nb_repetitions):
            # Temps d'assemblage
            t1 = time.perf_counter()
            K, M = assembler_systeme_acoustique_2d(geo, methode='auto')
            t_ass_passes.append(time.perf_counter() - t1)
            
            # Temps de résolution
            t2 = time.perf_counter()
            freqs_iga, _, _ = resoudre_modes_acoustiques_2d(K, M, num_modes=max_mode_idx + 1, c=c)
            t_solv_passes.append(time.perf_counter() - t2)
            
        t_ass_opt = min(t_ass_passes)
        t_solv_opt = min(t_solv_passes)
        t_tot_opt = t_ass_opt + t_solv_opt
        
        resultats['n_el'].append(n_el)
        resultats['dofs'].append(n_dofs)
        resultats['h'].append(h_approx)
        resultats['t_ass'].append(t_ass_opt)
        resultats['t_solv'].append(t_solv_opt)
        resultats['t_total'].append(t_tot_opt)
        
        for m_idx in modes_cibles:
            f_exact = infos_modes[m_idx]['f_exact']
            f_calc = freqs_iga[m_idx]
            err_rel = abs(f_calc - f_exact) / f_exact
            err_rel_clip = max(err_rel, 1e-16)
            resultats['erreurs'][m_idx].append(err_rel_clip)
            resultats['freqs_iga'][m_idx].append(f_calc)
            
        err_m1 = resultats['erreurs'][modes_cibles[0]][-1] * 100.0
        print(f"  [N_el = {n_el:2d} x {n_el:2d}] DDL = {n_dofs:4d} | "
              f"t_ass = {t_ass_opt*1000:6.1f} ms | t_solv = {t_solv_opt*1000:5.1f} ms | "
              f"Err Mode #{modes_cibles[0]} = {err_m1:.5e} %")
        
    t_total_calc = time.time() - t_global_0
    print("-" * 96)
    print(f"  Calculs terminés avec succès en {t_total_calc:.2f} s")
    print("=" * 96)
    
    # -------------------------------------------------------------------------
    # Visualisation graphique
    # -------------------------------------------------------------------------
    couleurs_modes = {
        1: '#1f77b4',  # bleu
        3: '#2ca02c',  # vert
        5: '#d62728',  # rouge
        6: '#ff7f0e',  # orange
        10: '#9467bd', # violet
    }
    symboles_modes = {
        1: 'o',
        3: 's',
        5: '^',
        6: 'D',
        10: 'v',
    }
    
    dossier_sortie = os.path.dirname(os.path.abspath(__file__))
    fichiers_generes = []
    
    # -------------------------------------------------------------------------
    # FIGURE 1 : Erreur relative en fréquence vs Degrés de liberté (log-log)
    # -------------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(9.5, 6.8))
    dofs = np.array(resultats['dofs'])
    
    for m_idx in modes_cibles:
        errs = np.array(resultats['erreurs'][m_idx])
        m, n = infos_modes[m_idx]['ordre']
        f_ex = infos_modes[m_idx]['f_exact']
        lbl = f"Mode #{m_idx} : $(m={m}, n={n})$ | $f = {f_ex:.1f}$ Hz"
        
        ax1.loglog(
            dofs, errs,
            marker=symboles_modes.get(m_idx, 'o'),
            color=couleurs_modes.get(m_idx, 'black'),
            linewidth=2.2,
            markersize=7,
            label=lbl,
            alpha=0.92
        )
        
    # Pente théorique de convergence O(h^4) = O(N_ddl^-2) pour p=2
    # log(err) = -2 * log(dof) + C
    dof_ref = dofs[3:8]
    pente_ref = 1.2 * resultats['erreurs'][modes_cibles[0]][3] * (dof_ref[0] / dof_ref)**2
    ax1.loglog(
        dof_ref, pente_ref, 'k--', linewidth=1.8,
        label=r"Pente théorique $O(h^{2p}) = O(N_{\mathrm{ddl}}^{-2})$ ($p=2$)"
    )
    
    ax1.set_xlabel("Nombre de degrés de liberté (DDL)", fontsize=11, fontweight='bold')
    ax1.set_ylabel(r"Erreur relative $\frac{|f_{\mathrm{IGA}} - f_{\mathrm{exact}}|}{f_{\mathrm{exact}}}$", fontsize=11, fontweight='bold')
    ax1.set_title(
        f"Convergence Isogéométrique NURBS (Patch unique « Carré gonflé »)\n"
        f"Cavité Acoustique Circulaire 2D ($R = {R}$ m, $c = {c}$ m/s)",
        fontsize=12, fontweight='bold', pad=12
    )
    ax1.grid(True, which='both', linestyle=':', alpha=0.6)
    ax1.legend(loc='lower left', fontsize=9.5, framealpha=0.95)
    plt.tight_layout()
    
    nom_fig1 = "convergence_cavite_circulaire_modes.png"
    chemin_fig1 = os.path.join(dossier_sortie, nom_fig1)
    if sauvegarder_figures:
        fig1.savefig(chemin_fig1, dpi=300, bbox_inches='tight')
        fichiers_generes.append(chemin_fig1)
        print(f"[Graphique 1/3] Enregistré : {chemin_fig1}")
    if afficher:
        plt.show()
    plt.close(fig1)
    
    # -------------------------------------------------------------------------
    # FIGURE 2 : Temps d'assemblage et de résolution vs Degrés de liberté
    # -------------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(9.5, 6.8))
    t_ass_ms = np.array(resultats['t_ass']) * 1000.0
    t_solv_ms = np.array(resultats['t_solv']) * 1000.0
    t_tot_ms = np.array(resultats['t_total']) * 1000.0
    
    ax2.loglog(dofs, t_ass_ms, 'o-', color='#1f77b4', linewidth=2.2, markersize=7, label="Temps d'assemblage (K & M)")
    ax2.loglog(dofs, t_solv_ms, 's-', color='#2ca02c', linewidth=2.2, markersize=7, label="Temps de résolution (Eigensolver)")
    ax2.loglog(dofs, t_tot_ms, '^-', color='#d62728', linewidth=2.4, markersize=8, label="Temps total (Assemblage + Solveur)")
    
    # Pentes de complexité
    dof_sub = dofs[3:9]
    c_lin = t_ass_ms[3] * (dof_sub / dof_sub[0])**1.0
    ax2.loglog(dof_sub, c_lin, 'k:', linewidth=1.5, label=r"Complexité linéaire théorique $O(N_{\mathrm{ddl}})$")
    
    ax2.set_xlabel("Nombre de degrés de liberté (DDL)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Temps de calcul CPU (ms)", fontsize=11, fontweight='bold')
    ax2.set_title(
        f"Performance de Calcul IGA NURBS 2D vs DDL\n"
        f"Cavité Acoustique Circulaire ($R = {R}$ m, $c = {c}$ m/s)",
        fontsize=12, fontweight='bold', pad=12
    )
    ax2.grid(True, which='both', linestyle=':', alpha=0.6)
    ax2.legend(loc='upper left', fontsize=10, framealpha=0.95)
    plt.tight_layout()
    
    nom_fig2 = "temps_calcul_cavite_circulaire.png"
    chemin_fig2 = os.path.join(dossier_sortie, nom_fig2)
    if sauvegarder_figures:
        fig2.savefig(chemin_fig2, dpi=300, bbox_inches='tight')
        fichiers_generes.append(chemin_fig2)
        print(f"[Graphique 2/3] Enregistré : {chemin_fig2}")
    if afficher:
        plt.show()
    plt.close(fig2)
    
    # -------------------------------------------------------------------------
    # FIGURE 3 : Planche maîtresse complète (Convergence & Performance)
    # -------------------------------------------------------------------------
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(17, 6.8))
    
    # Volet gauche : Convergence
    for m_idx in modes_cibles:
        errs = np.array(resultats['erreurs'][m_idx])
        m, n = infos_modes[m_idx]['ordre']
        f_ex = infos_modes[m_idx]['f_exact']
        lbl = f"Mode #{m_idx} : $(m={m}, n={n})$ | $f = {f_ex:.1f}$ Hz"
        ax3a.loglog(
            dofs, errs,
            marker=symboles_modes.get(m_idx, 'o'),
            color=couleurs_modes.get(m_idx, 'black'),
            linewidth=2.0,
            markersize=6.5,
            label=lbl,
            alpha=0.92
        )
    ax3a.loglog(dof_ref, pente_ref, 'k--', linewidth=1.6, label=r"Pente $O(N_{\mathrm{ddl}}^{-2})$ ($p=2$)")
    ax3a.set_xlabel("Nombre de degrés de liberté (DDL)", fontsize=11, fontweight='bold')
    ax3a.set_ylabel(r"Erreur relative $\frac{|f_{\mathrm{IGA}} - f_{\mathrm{exact}}|}{f_{\mathrm{exact}}}$", fontsize=11, fontweight='bold')
    ax3a.set_title("Précision Modale & Taux de Convergence $h$", fontsize=12, fontweight='bold', pad=10)
    ax3a.grid(True, which='both', linestyle=':', alpha=0.6)
    ax3a.legend(loc='lower left', fontsize=9, framealpha=0.92)
    
    # Volet droit : Temps de calcul
    ax3b.loglog(dofs, t_ass_ms, 'o-', color='#1f77b4', linewidth=2.0, markersize=6.5, label="Temps d'assemblage (K & M)")
    ax3b.loglog(dofs, t_solv_ms, 's-', color='#2ca02c', linewidth=2.0, markersize=6.5, label="Temps de résolution (Eigensolver)")
    ax3b.loglog(dofs, t_tot_ms, '^-', color='#d62728', linewidth=2.2, markersize=7.5, label="Temps total")
    ax3b.loglog(dof_sub, c_lin, 'k:', linewidth=1.5, label=r"Référence $O(N_{\mathrm{ddl}})$")
    ax3b.set_xlabel("Nombre de degrés de liberté (DDL)", fontsize=11, fontweight='bold')
    ax3b.set_ylabel("Temps de calcul CPU (ms)", fontsize=11, fontweight='bold')
    ax3b.set_title("Efficacité Computationnelle & Scalabilité CPU", fontsize=12, fontweight='bold', pad=10)
    ax3b.grid(True, which='both', linestyle=':', alpha=0.6)
    ax3b.legend(loc='upper left', fontsize=9.5, framealpha=0.92)
    
    plt.suptitle(
        f"Étude de Convergence et de Performance IGA NURBS 2D : Cavité Acoustique Circulaire\n"
        f"Patch unique « Carré gonflé » sans rotation ($R = {R}$ m, $c = {c}$ m/s)",
        fontsize=13, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    
    nom_fig3 = "analyse_complete_cavite_circulaire.png"
    chemin_fig3 = os.path.join(dossier_sortie, nom_fig3)
    if sauvegarder_figures:
        fig3.savefig(chemin_fig3, dpi=300, bbox_inches='tight')
        fichiers_generes.append(chemin_fig3)
        print(f"[Graphique 3/3] Enregistré : {chemin_fig3}")
    if afficher:
        plt.show()
    plt.close(fig3)
    
    return {
        'resultats': resultats,
        'infos_modes': infos_modes,
        'fichiers_figures': fichiers_generes
    }


if __name__ == "__main__":
    executer_analyse_convergence_circulaire(
        R=1.0,
        c=343.0,
        liste_n_el=(4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 40),
        modes_cibles=(1, 3, 5, 6, 10),
        nb_repetitions=2,
        sauvegarder_figures=True,
        afficher=False
    )
