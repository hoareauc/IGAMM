"""
Cas test : Analyse de convergence du k-rafinement (IGA 2D)
Modes acoustiques cibles : Mode 1 (1, 0), Mode 5 (2, 1), Mode 10 (2, 2)
Degrés polynomiaux : p = 2, 3, 4, 5, 6, 7
Discrétisation : n_el_xi = n_el_eta = N_el croissant

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
    nom_fichier_figure="convergence_k_rafinement_modes_1_5_10.png",
    sauvegarder_figure=True,
    afficher=False
):
    """
    Exécute l'analyse paramétrique de convergence du k-rafinement pour différents degrés polynomiaux.
    
    Paramètres :
        Lx, Ly (float) : Dimensions physiques de la cavité acoustique (m)
        c (float) : Célérité acoustique (m/s)
        liste_p (tuple/list) : Liste des degrés polynomiaux à tester (ex: 2 à 7)
        liste_n_el (tuple/list) : Nombres d'éléments selon xi et eta (n_el_xi = n_el_eta)
        modes_cibles (tuple/list) : Indices des modes propres cibles (1, 5, 10)
        nom_fichier_figure (str) : Nom du fichier image de sortie
        sauvegarder_figure (bool) : Sauvegarder la figure sur disque si True
        afficher (bool) : Afficher la figure interactivement si True
        
    Retourne :
        dict : Résultats complets structurés de la convergence
    """
    print("=" * 90)
    print("  ANALYSE DE CONVERGENCE DU k-RAFFINEMENT (IGA 2D ACOUSTIQUE)")
    print("=" * 90)
    print(f"  Domaine : {Lx} m x {Ly} m | Célérité : {c} m/s")
    print(f"  Degrés polynomiaux : {list(liste_p)}")
    print(f"  Tailles de maillage (N_el = n_el_xi = n_el_eta) : {list(liste_n_el)}")
    print(f"  Modes acoustiques cibles : {list(modes_cibles)}")
    print("-" * 90)
    
    # 1. Calcul des fréquences analytiques de référence
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
        
    print("-" * 90)
    
    # Structure pour stocker les résultats : donnees[m_idx][p] = {'dofs': [], 'erreurs': [], 'n_el': []}
    donnees = {m_idx: {p: {'dofs': [], 'erreurs': [], 'n_el': []} for p in liste_p} for m_idx in modes_cibles}
    
    t_global_0 = time.time()
    total_simulations = len(liste_p) * len(liste_n_el)
    compteur = 0
    
    for p in liste_p:
        t_p_0 = time.time()
        for n_el in liste_n_el:
            compteur += 1
            # Création de la géométrie k-raffinée (continuité C^{p-1})
            geo = creer_geometrie_rectangle(Lx=Lx, Ly=Ly, p_xi=p, p_eta=p, n_el_xi=n_el, n_el_eta=n_el)
            n_dofs = geo['n_dofs']
            
            # Assemblage (utilise l'assemblage tensoriel rapide)
            K, M = assembler_systeme_acoustique_2d(geo, methode='auto')
            
            # Résolution des modes propres jusqu'au mode cible maximal
            freqs_iga, _, _ = resoudre_modes_acoustiques_2d(K, M, num_modes=max_mode_idx + 1, c=c)
            
            # Enregistrement des erreurs relatives pour chaque mode cible
            for m_idx in modes_cibles:
                f_exact = infos_modes[m_idx]['f_exact']
                f_calc = freqs_iga[m_idx]
                err_rel = abs(f_calc - f_exact) / f_exact
                # Seuil de sécurité pour le log (bruit numérique de précision double ~ 1e-16)
                err_rel_propre = max(err_rel, 1e-16)
                
                donnees[m_idx][p]['dofs'].append(n_dofs)
                donnees[m_idx][p]['erreurs'].append(err_rel_propre)
                donnees[m_idx][p]['n_el'].append(n_el)
                
        duree_p = time.time() - t_p_0
        err_min_m1 = min(donnees[modes_cibles[0]][p]['erreurs'])
        print(f"  [p = {p}] Calculé en {duree_p:.3f} s (Erreur min Mode 1 = {err_min_m1:.2e})")
        
    t_total = time.time() - t_global_0
    print(f"  -> Total : {total_simulations} résolutions IGA effectuées en {t_total:.2f} s")
    print("=" * 90)
    
    # 2. Tracé des courbes de convergence
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
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), sharey=True)
    
    for idx_ax, m_idx in enumerate(modes_cibles):
        ax = axes[idx_ax]
        f_exact = infos_modes[m_idx]['f_exact']
        m, n = infos_modes[m_idx]['ordre']
        
        for p in liste_p:
            dofs = np.array(donnees[m_idx][p]['dofs'])
            errs = np.array(donnees[m_idx][p]['erreurs'])
            
            label_p = f"p = {p} ($C^{{{p-1}}}$)"
            ax.loglog(
                dofs, errs,
                marker=symboles.get(p, 'o'),
                color=couleurs.get(p, 'black'),
                linewidth=2.0,
                markersize=6,
                label=label_p,
                alpha=0.9
            )
            
        # Ligne de référence : précision machine standard double (~ 10^-14)
        ax.axhline(y=2.2e-14, color='gray', linestyle=':', linewidth=1.2, alpha=0.7)
        if idx_ax == 0:
            ax.text(
                dofs[0] * 1.05, 3e-14, "Précision machine IEEE 754 ($~10^{-14}$)",
                color='gray', fontsize=8.5, fontstyle='italic'
            )
            
        ax.set_title(
            f"Mode #{m_idx} : $(m={m}, n={n})$ | $f = {f_exact:.2f}$ Hz",
            fontsize=12, fontweight='bold', pad=10
        )
        ax.set_xlabel("Nombre de degrés de liberté (DDL)", fontsize=11, fontweight='medium')
        if idx_ax == 0:
            ax.set_ylabel(r"Erreur relative $\frac{|f_{\mathrm{IGA}} - f_{\mathrm{exact}}|}{f_{\mathrm{exact}}}$", fontsize=12)
            
        ax.grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.6)
        ax.set_ylim([1e-15, 1e-1])
        ax.legend(loc='upper right', fontsize=9.5, framealpha=0.9)
        
    plt.suptitle(
        f"Analyse de convergence du k-rafinement IGA 2D pour la cavité acoustique [{Lx} m x {Ly} m]\n"
        f"Erreur en fréquence propre vs DDL pour $p = 2$ à $p = 7$ (Continuité $C^{{p-1}}$ maximale)",
        fontsize=13, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    
    chemin_fig = None
    if sauvegarder_figure:
        dossier_sortie = os.path.dirname(os.path.abspath(__file__))
        chemin_fig = os.path.join(dossier_sortie, nom_fichier_figure)
        plt.savefig(chemin_fig, dpi=300, bbox_inches='tight')
        print(f"[Visualisation] Figure de convergence enregistrée sous :\n  -> {chemin_fig}")
        
    if afficher:
        plt.show()
        
    plt.close(fig)
    
    # 3. Affichage d'un tableau récapitulatif synthétique
    print("\n" + "=" * 90)
    print(f"{'Degré p':<8} | {'Pente théorique':<16} | {'Erreur Mode 1 (min)':<22} | {'Erreur Mode 5 (min)':<22} | {'Erreur Mode 10 (min)'}")
    print("-" * 90)
    for p in liste_p:
        e1_min = min(donnees[1][p]['erreurs'])
        e5_min = min(donnees[5][p]['erreurs'])
        e10_min = min(donnees[10][p]['erreurs'])
        pente = f"O(h^{2*p}) = O(N^-{p})"
        print(f"p = {p:<4} | {pente:<16} | {e1_min:<22.2e} | {e5_min:<22.2e} | {e10_min:.2e}")
    print("=" * 90 + "\n")
    
    return {
        'donnees': donnees,
        'infos_modes': infos_modes,
        'liste_p': liste_p,
        'liste_n_el': liste_n_el,
        'modes_cibles': modes_cibles,
        'chemin_figure': chemin_fig
    }


if __name__ == "__main__":
    executer_analyse_convergence(
        Lx=1.0,
        Ly=0.6,
        c=343.0,
        liste_p=(2, 3, 4, 5, 6, 7),
        liste_n_el=(4, 6, 8, 10, 12, 16, 20, 24, 30),
        modes_cibles=(1, 5, 10),
        sauvegarder_figure=True,
        afficher=False
    )
