"""
Cas test : Analyse de convergence et de performance (temps de calcul) du k-rafinement (IGA 2D)
Modes acoustiques cibles : Mode 1 (1, 0), Mode 5 (2, 1), Mode 10 (2, 2)
Degrés polynomiaux : p = 2, 3, 4, 5, 6, 7
Discrétisation : n_el_xi = n_el_eta = N_el croissant

Génère deux visualisations graphiques détaillées :
1. convergence_k_rafinement_modes_1_5_10.png : Erreur relative en fréquence vs DDL
2. temps_calcul_k_rafinement.png : Temps d'assemblage, temps de résolution et temps total vs DDL
3. analyse_complete_k_rafinement.png : Planche maîtresse combinée (Précision & Temps de calcul)

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
    creer_geometrie_rectangle,
    assembler_systeme_acoustique_2d,
    resoudre_modes_acoustiques_2d,
    frequences_analytiques_cavite_rectangulaire,
)


def executer_analyse_convergence(
    Lx=1.0,
    Ly=0.6,
    c=343.0,
    liste_p=(2, 3, 4, 5, 6, 7),
    liste_n_el=(4, 6, 8, 10, 12, 16, 20, 24, 30),
    modes_cibles=(1, 5, 10),
    nb_repetitions=2,
    sauvegarder_figures=True,
    afficher=False
):
    """
    Exécute l'analyse paramétrique de convergence et de temps de calcul du k-rafinement.
    
    Paramètres :
        Lx, Ly (float) : Dimensions physiques de la cavité acoustique (m)
        c (float) : Célérité acoustique (m/s)
        liste_p (tuple/list) : Liste des degrés polynomiaux à tester (ex: 2 à 7)
        liste_n_el (tuple/list) : Nombres d'éléments selon xi et eta (n_el_xi = n_el_eta)
        modes_cibles (tuple/list) : Indices des modes propres cibles (1, 5, 10)
        nb_repetitions (int) : Nombre de passes pour stabiliser la mesure des temps CPU
        sauvegarder_figures (bool) : Sauvegarder les figures sur disque si True
        afficher (bool) : Afficher les figures interactivement si True
        
    Retourne :
        dict : Résultats complets structurés (erreurs, temps d'assemblage, temps de résolution)
    """
    print("=" * 96)
    print("  ANALYSE DE CONVERGENCE & TEMPS DE CALCUL DU k-RAFFINEMENT (IGA 2D ACOUSTIQUE)")
    print("=" * 96)
    print(f"  Domaine : {Lx} m x {Ly} m | Célérité : {c} m/s")
    print(f"  Degrés polynomiaux : {list(liste_p)}")
    print(f"  Tailles de maillage (N_el = n_el_xi = n_el_eta) : {list(liste_n_el)}")
    print(f"  Modes acoustiques cibles : {list(modes_cibles)}")
    print("-" * 96)
    
    # 1. Calcul des fréquences analytiques exactes
    max_mode_idx = max(modes_cibles)
    freqs_ana, ordres_modaux = frequences_analytiques_cavite_rectangulaire(
        Lx=Lx, Ly=Ly, c=c, max_m=10, max_n=10
    )
    
    infos_modes = {}
    for m_idx in modes_cibles:
        f_exact = freqs_ana[m_idx]
        ordre = ordres_modaux[m_idx]
        infos_modes[m_idx] = {'f_exact': f_exact, 'ordre': ordre}
        print(f"  -> Mode #{m_idx:2d} : f_analytique = {f_exact:.4f} Hz (m={ordre[0]}, n={ordre[1]})")
        
    print("-" * 96)
    
    # Structure de données pour stocker les résultats
    # resultats_p[p] = {'dofs': [], 'n_el': [], 't_ass': [], 't_solv': [], 't_total': [], 'erreurs': {m_idx: []}}
    resultats_p = {
        p: {
            'dofs': [],
            'n_el': [],
            't_ass': [],
            't_solv': [],
            't_total': [],
            'erreurs': {m_idx: [] for m_idx in modes_cibles}
        }
        for p in liste_p
    }
    
    t_global_0 = time.time()
    
    for p in liste_p:
        t_p_0 = time.time()
        for n_el in liste_n_el:
            geo = creer_geometrie_rectangle(Lx=Lx, Ly=Ly, p_xi=p, p_eta=p, n_el_xi=n_el, n_el_eta=n_el)
            n_dofs = geo['n_dofs']
            
            # Mesure précise des temps d'assemblage et de résolution (minimum sur nb_repetitions passes)
            t_ass_passes = []
            t_solv_passes = []
            freqs_iga = None
            
            for _ in range(nb_repetitions):
                # Temps d'assemblage
                t1 = time.perf_counter()
                K, M = assembler_systeme_acoustique_2d(geo, methode='auto')
                t_ass_passes.append(time.perf_counter() - t1)
                
                # Temps de résolution du problème aux valeurs propres
                t2 = time.perf_counter()
                freqs_iga, _, _ = resoudre_modes_acoustiques_2d(K, M, num_modes=max_mode_idx + 1, c=c)
                t_solv_passes.append(time.perf_counter() - t2)
                
            t_ass_opt = min(t_ass_passes)
            t_solv_opt = min(t_solv_passes)
            t_tot_opt = t_ass_opt + t_solv_opt
            
            resultats_p[p]['dofs'].append(n_dofs)
            resultats_p[p]['n_el'].append(n_el)
            resultats_p[p]['t_ass'].append(t_ass_opt)
            resultats_p[p]['t_solv'].append(t_solv_opt)
            resultats_p[p]['t_total'].append(t_tot_opt)
            
            # Erreurs relatives
            for m_idx in modes_cibles:
                f_exact = infos_modes[m_idx]['f_exact']
                f_calc = freqs_iga[m_idx]
                err_rel = abs(f_calc - f_exact) / f_exact
                err_rel_clip = max(err_rel, 1e-16)
                resultats_p[p]['erreurs'][m_idx].append(err_rel_clip)
                
        duree_p = time.time() - t_p_0
        t_tot_max = max(resultats_p[p]['t_total']) * 1000.0
        print(f"  [p = {p}] 9 maillages calculés en {duree_p:.3f} s (Temps max par calcul = {t_tot_max:.1f} ms)")
        
    t_total = time.time() - t_global_0
    print(f"  -> Total : {len(liste_p) * len(liste_n_el)} configurations IGA résolues en {t_total:.2f} s")
    print("=" * 96)
    
    # Style visuel cohérent
    couleurs = {
        2: '#1f77b4',  # bleu
        3: '#2ca02c',  # vert
        4: '#ff7f0e',  # orange
        5: '#9467bd',  # violet
        6: '#d62728',  # rouge
        7: '#8c564b',  # marron
    }
    symboles = {
        2: 'o',
        3: 's',
        4: '^',
        5: 'D',
        6: 'v',
        7: 'P',
    }
    
    dossier_sortie = os.path.dirname(os.path.abspath(__file__))
    fichiers_generes = []
    
    # -------------------------------------------------------------------------
    # FIGURE 1 : Erreur en fréquence vs DDL (Modes 1, 5, 10)
    # -------------------------------------------------------------------------
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5.8), sharey=True)
    
    for idx_ax, m_idx in enumerate(modes_cibles):
        ax = axes1[idx_ax]
        f_exact = infos_modes[m_idx]['f_exact']
        m, n = infos_modes[m_idx]['ordre']
        
        for p in liste_p:
            dofs = np.array(resultats_p[p]['dofs'])
            errs = np.array(resultats_p[p]['erreurs'][m_idx])
            ax.loglog(
                dofs, errs,
                marker=symboles.get(p, 'o'),
                color=couleurs.get(p, 'black'),
                linewidth=2.0,
                markersize=6,
                label=f"p = {p} ($C^{{{p-1}}}$)",
                alpha=0.9
            )
            
        ax.axhline(y=2.2e-14, color='gray', linestyle=':', linewidth=1.2, alpha=0.7)
        if idx_ax == 0:
            ax.text(
                dofs[0] * 1.05, 3e-14, "Précision machine IEEE 754 ($~10^{-14}$)",
                color='gray', fontsize=8.5, fontstyle='italic'
            )
            
        ax.set_title(f"Mode #{m_idx} : $(m={m}, n={n})$ | $f = {f_exact:.2f}$ Hz", fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("Nombre de degrés de liberté (DDL)", fontsize=11)
        if idx_ax == 0:
            ax.set_ylabel(r"Erreur relative $\frac{|f_{\mathrm{IGA}} - f_{\mathrm{exact}}|}{f_{\mathrm{exact}}}$", fontsize=12)
        ax.grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.6)
        ax.set_ylim([1e-15, 1e-1])
        ax.legend(loc='upper right', fontsize=9.5, framealpha=0.9)
        
    fig1.suptitle(
        f"Analyse de convergence du k-rafinement IGA 2D [{Lx} m x {Ly} m]\n"
        f"Erreur en fréquence propre vs DDL pour $p = 2$ à $p = 7$ (Continuité $C^{{p-1}}$)",
        fontsize=13, fontweight='bold', y=1.02
    )
    fig1.tight_layout()
    
    chemin_fig1 = os.path.join(dossier_sortie, "convergence_k_rafinement_modes_1_5_10.png")
    if sauvegarder_figures:
        fig1.savefig(chemin_fig1, dpi=300, bbox_inches='tight')
        fichiers_generes.append(chemin_fig1)
        print(f"[Graphique 1/3] Courbes de convergence sauvegardées dans :\n  -> {chemin_fig1}")
    plt.close(fig1)
    
    # -------------------------------------------------------------------------
    # FIGURE 2 : Temps de calculs associés (Assemblage & Résolution vs DDL)
    # -------------------------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5.8))
    
    # Sous-figure (a) : Temps d'assemblage
    ax_ass = axes2[0]
    for p in liste_p:
        dofs = np.array(resultats_p[p]['dofs'])
        t_ass_ms = np.array(resultats_p[p]['t_ass']) * 1000.0  # en millisecondes
        ax_ass.loglog(
            dofs, t_ass_ms,
            marker=symboles.get(p, 'o'),
            color=couleurs.get(p, 'black'),
            linewidth=2.0,
            markersize=6,
            label=f"p = {p} ($C^{{{p-1}}}$)"
        )
    ax_ass.set_title("Temps d'assemblage des matrices K et M", fontsize=12, fontweight='bold', pad=10)
    ax_ass.set_xlabel("Nombre de degrés de liberté (DDL)", fontsize=11)
    ax_ass.set_ylabel("Temps d'assemblage (ms)", fontsize=11)
    ax_ass.grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.6)
    ax_ass.legend(loc='upper left', fontsize=9.5, framealpha=0.9)
    
    # Sous-figure (b) : Temps de résolution aux valeurs propres
    ax_solv = axes2[1]
    for p in liste_p:
        dofs = np.array(resultats_p[p]['dofs'])
        t_solv_ms = np.array(resultats_p[p]['t_solv']) * 1000.0  # en millisecondes
        ax_solv.loglog(
            dofs, t_solv_ms,
            marker=symboles.get(p, 'o'),
            color=couleurs.get(p, 'black'),
            linewidth=2.0,
            markersize=6,
            label=f"p = {p} ($C^{{{p-1}}}$)"
        )
    ax_solv.set_title("Temps de résolution aux valeurs propres", fontsize=12, fontweight='bold', pad=10)
    ax_solv.set_xlabel("Nombre de degrés de liberté (DDL)", fontsize=11)
    ax_solv.set_ylabel("Temps de résolution (ms)", fontsize=11)
    ax_solv.grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.6)
    ax_solv.legend(loc='upper left', fontsize=9.5, framealpha=0.9)
    
    # Sous-figure (c) : Temps de calcul total
    ax_tot = axes2[2]
    for p in liste_p:
        dofs = np.array(resultats_p[p]['dofs'])
        t_tot_ms = np.array(resultats_p[p]['t_total']) * 1000.0  # en millisecondes
        ax_tot.loglog(
            dofs, t_tot_ms,
            marker=symboles.get(p, 'o'),
            color=couleurs.get(p, 'black'),
            linewidth=2.0,
            markersize=6,
            label=f"p = {p} ($C^{{{p-1}}}$)"
        )
    ax_tot.set_title("Temps total de calcul (Assemblage + Résolution)", fontsize=12, fontweight='bold', pad=10)
    ax_tot.set_xlabel("Nombre de degrés de liberté (DDL)", fontsize=11)
    ax_tot.set_ylabel("Temps total (ms)", fontsize=11)
    ax_tot.grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.6)
    ax_tot.legend(loc='upper left', fontsize=9.5, framealpha=0.9)
    
    fig2.suptitle(
        f"Temps de calculs associés au k-rafinement IGA 2D vs Nombre de DDL\n"
        f"Décomposition en temps d'assemblage (K, M) et temps de solveur spectral (p = 2 à 7)",
        fontsize=13, fontweight='bold', y=1.02
    )
    fig2.tight_layout()
    
    chemin_fig2 = os.path.join(dossier_sortie, "temps_calcul_k_rafinement.png")
    if sauvegarder_figures:
        fig2.savefig(chemin_fig2, dpi=300, bbox_inches='tight')
        fichiers_generes.append(chemin_fig2)
        print(f"[Graphique 2/3] Courbes des temps de calcul sauvegardées dans :\n  -> {chemin_fig2}")
    plt.close(fig2)
    
    # -------------------------------------------------------------------------
    # FIGURE 3 : Planche Maîtresse Combinée (2 lignes x 3 colonnes)
    # -------------------------------------------------------------------------
    fig3, axes3 = plt.subplots(2, 3, figsize=(18, 10.5))
    
    # Ligne 1 : Erreurs Modes 1, 5, 10
    for idx_m, m_idx in enumerate(modes_cibles):
        ax = axes3[0, idx_m]
        f_exact = infos_modes[m_idx]['f_exact']
        m, n = infos_modes[m_idx]['ordre']
        for p in liste_p:
            dofs = np.array(resultats_p[p]['dofs'])
            errs = np.array(resultats_p[p]['erreurs'][m_idx])
            ax.loglog(
                dofs, errs,
                marker=symboles.get(p, 'o'),
                color=couleurs.get(p, 'black'),
                linewidth=1.8,
                markersize=5,
                label=f"p = {p} ($C^{{{p-1}}}$)"
            )
        ax.axhline(y=2.2e-14, color='gray', linestyle=':', linewidth=1.1, alpha=0.7)
        ax.set_title(f"Erreur Mode #{m_idx} : $(m={m}, n={n})$ | $f = {f_exact:.2f}$ Hz", fontsize=11, fontweight='bold')
        ax.set_xlabel("Nombre de DDL", fontsize=10)
        if idx_m == 0:
            ax.set_ylabel(r"Erreur relative $\frac{|f_{\mathrm{IGA}} - f_{\mathrm{exact}}|}{f_{\mathrm{exact}}}$", fontsize=11)
        ax.grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.6)
        ax.set_ylim([1e-15, 1e-1])
        ax.legend(loc='upper right', fontsize=8.5, framealpha=0.9)
        
    # Ligne 2 : Temps d'assemblage, Temps de résolution, Temps total
    titres_temps = [
        ("Temps d'assemblage (K, M)", 't_ass', "Temps assemblage (ms)"),
        ("Temps de résolution spectral", 't_solv', "Temps résolution (ms)"),
        ("Temps total de calcul", 't_total', "Temps total (ms)")
    ]
    for idx_t, (titre, cle_t, y_lbl) in enumerate(titres_temps):
        ax = axes3[1, idx_t]
        for p in liste_p:
            dofs = np.array(resultats_p[p]['dofs'])
            t_ms = np.array(resultats_p[p][cle_t]) * 1000.0
            ax.loglog(
                dofs, t_ms,
                marker=symboles.get(p, 'o'),
                color=couleurs.get(p, 'black'),
                linewidth=1.8,
                markersize=5,
                label=f"p = {p} ($C^{{{p-1}}}$)"
            )
        ax.set_title(titre, fontsize=11, fontweight='bold')
        ax.set_xlabel("Nombre de DDL", fontsize=10)
        ax.set_ylabel(y_lbl, fontsize=10)
        ax.grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.6)
        ax.legend(loc='upper left', fontsize=8.5, framealpha=0.9)
        
    fig3.suptitle(
        f"Analyse Complète du k-rafinement IGA 2D pour Cavité Acoustique Rectangulaire\n"
        f"Précision spectrale (Ligne supérieure) et Temps de calcul CPU (Ligne inférieure) pour p = 2 à 7",
        fontsize=13, fontweight='bold', y=0.99
    )
    fig3.tight_layout()
    
    chemin_fig3 = os.path.join(dossier_sortie, "analyse_complete_k_rafinement.png")
    if sauvegarder_figures:
        fig3.savefig(chemin_fig3, dpi=300, bbox_inches='tight')
        fichiers_generes.append(chemin_fig3)
        print(f"[Graphique 3/3] Planche complète sauvegardée dans :\n  -> {chemin_fig3}")
    plt.close(fig3)
    
    # -------------------------------------------------------------------------
    # 4. Tableau récapitulatif dans la console
    # -------------------------------------------------------------------------
    print("\n" + "=" * 96)
    print(f"{'Degré p':<8} | {'Plage DDL':<14} | {'T_assemblage (min-max)':<24} | {'T_solveur (min-max)':<24} | {'T_total (min-max)'}")
    print("-" * 96)
    for p in liste_p:
        dofs = resultats_p[p]['dofs']
        dofs_str = f"{dofs[0]} - {dofs[-1]}"
        t_ass_min = min(resultats_p[p]['t_ass']) * 1000.0
        t_ass_max = max(resultats_p[p]['t_ass']) * 1000.0
        t_solv_min = min(resultats_p[p]['t_solv']) * 1000.0
        t_solv_max = max(resultats_p[p]['t_solv']) * 1000.0
        t_tot_min = min(resultats_p[p]['t_total']) * 1000.0
        t_tot_max = max(resultats_p[p]['t_total']) * 1000.0
        
        ass_str = f"{t_ass_min:5.2f} ms - {t_ass_max:5.2f} ms"
        solv_str = f"{t_solv_min:5.2f} ms - {t_solv_max:5.2f} ms"
        tot_str = f"{t_tot_min:5.2f} ms - {t_tot_max:5.2f} ms"
        print(f"p = {p:<4} | {dofs_str:<14} | {ass_str:<24} | {solv_str:<24} | {tot_str}")
    print("=" * 96 + "\n")
    
    if afficher:
        plt.show()
        
    return {
        'resultats_p': resultats_p,
        'infos_modes': infos_modes,
        'fichiers_figures': fichiers_generes
    }


if __name__ == "__main__":
    executer_analyse_convergence(
        Lx=1.0,
        Ly=0.6,
        c=343.0,
        liste_p=(2, 3, 4, 5, 6, 7),
        liste_n_el=(4, 6, 8, 10, 12, 16, 20, 24, 30),
        modes_cibles=(1, 5, 10),
        nb_repetitions=2,
        sauvegarder_figures=True,
        afficher=False
    )
