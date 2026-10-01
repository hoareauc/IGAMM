"""
Cas test : Modes d'une cavité acoustique 2D rectangulaire (Lx x Ly) par IGA
Approche 100% procédurale / non orientée objet (fonctions pures).

Problème physique :
    - Équation de Helmholtz : -Delta(p) = (omega / c)^2 * p dans Omega = [0, Lx] x [0, Ly]
    - Conditions aux limites : Parois rigides (Neumann homogène dp/dn = 0 sur dOmega)
    - Solution analytique :
        f_{m, n} = (c / 2) * sqrt( (m / Lx)^2 + (n / Ly)^2 )
        p_{m, n}(x, y) = cos(m * pi * x / Lx) * cos(n * pi * y / Ly)
"""

import os
import sys
import time

# Ajout du répertoire racine IGAMM au sys.path pour permettre l'import du module `iga`
chemin_courant = os.path.dirname(os.path.abspath(__file__))
chemin_racine = os.path.abspath(os.path.join(chemin_courant, "..", ".."))
if chemin_racine not in sys.path:
    sys.path.insert(0, chemin_racine)

from iga import (
    creer_geometrie_rectangle,
    assembler_systeme_acoustique_2d,
    resoudre_modes_acoustiques_2d,
    frequences_analytiques_cavite_rectangulaire,
    afficher_comparaison_modes,
    visualiser_modes_acoustiques,
)


def executer_cas_cavite_rectangulaire(
    Lx=1.0,
    Ly=0.6,
    c=343.0,
    p_xi=2,
    p_eta=2,
    n_el_xi=16,
    n_el_eta=10,
    num_modes=8,
    sauvegarder_figure=True,
    nom_fichier_figure="modes_cavite_rectangulaire.png"
):
    """
    Fonction principale procédurale pour exécuter la simulation modale de la cavité rectangulaire.
    
    Paramètres :
        Lx (float) : Longueur selon x (m)
        Ly (float) : Hauteur selon y (m)
        c (float) : Célérité acoustique (m/s)
        p_xi, p_eta (int) : Degrés polynomiaux IGA
        n_el_xi, n_el_eta (int) : Nombres d'éléments B-splines
        num_modes (int) : Nombre de modes propres à calculer et comparer
        sauvegarder_figure (bool) : Si True, sauvegarde l'image des modes propres
        nom_fichier_figure (str) : Nom du fichier image de sortie
        
    Retourne :
        dict : Résultats complets du calcul (géométrie, fréquences IGA, fréquences analytiques, etc.)
    """
    print("=" * 80)
    print("  SIMULATION IGA 2D : MODES PROPRES D'UNE CAVITÉ ACOUSTIQUE RECTANGULAIRE")
    print("=" * 80)
    print(f"  Dimensions de la cavité : Lx = {Lx:.3f} m, Ly = {Ly:.3f} m")
    print(f"  Célérité du son : c = {c:.1f} m/s")
    print(f"  Degrés polynomiaux : p_xi = {p_xi}, p_eta = {p_eta}")
    print(f"  Discrétisation : {n_el_xi} x {n_el_eta} éléments ({n_el_xi * n_el_eta} éléments au total)")
    
    # 1. Création de la géométrie IGA
    t0 = time.time()
    geo = creer_geometrie_rectangle(Lx=Lx, Ly=Ly, p_xi=p_xi, p_eta=p_eta, n_el_xi=n_el_xi, n_el_eta=n_el_eta)
    t_geo = time.time() - t0
    print(f"  Nombre de degrés de liberté (DDL) : {geo['n_dofs']}")
    print(f"  [1/4] Géométrie créée en {t_geo:.4f} s")
    
    # 2. Assemblage des matrices K et M
    t0 = time.time()
    K, M = assembler_systeme_acoustique_2d(geo)
    t_ass = time.time() - t0
    print(f"  [2/4] Matrices K et M assemblées en {t_ass:.4f} s (K non-nuls : {K.nnz})")
    
    # 3. Résolution du problème aux valeurs propres
    t0 = time.time()
    freqs_iga, modes_propres, val_propres = resoudre_modes_acoustiques_2d(K, M, num_modes=num_modes, c=c)
    t_solv = time.time() - t0
    print(f"  [3/4] {len(freqs_iga)} modes propres calculés en {t_solv:.4f} s")
    
    # 4. Solution analytique exacte
    freqs_ana, ordres_modaux = frequences_analytiques_cavite_rectangulaire(Lx=Lx, Ly=Ly, c=c, max_m=6, max_n=6)
    
    # 5. Affichage du tableau de comparaison
    afficher_comparaison_modes(freqs_iga, freqs_ana, ordres_modaux, num_modes=num_modes)
    
    # 6. Visualisation graphique
    chemin_fig = None
    if sauvegarder_figure:
        dossier_sortie = os.path.dirname(os.path.abspath(__file__))
        chemin_fig = os.path.join(dossier_sortie, nom_fichier_figure)
        visualiser_modes_acoustiques(
            geo=geo,
            frequences_iga=freqs_iga,
            modes_propres=modes_propres,
            indices_modes=[i for i in range(1, min(7, len(freqs_iga)))],
            ordres_modaux=ordres_modaux,
            chemin_sauvegarde=chemin_fig,
            afficher=False
        )
        print(f"  [4/4] Graphiques des modes acoustiques enregistrés dans :\n        {chemin_fig}")
        
    print("=" * 80)
    print("  SIMULATION TERMINÉE AVEC SUCCÈS")
    print("=" * 80)
    
    resultats = {
        'geo': geo,
        'K': K,
        'M': M,
        'frequences_iga': freqs_iga,
        'frequences_analytiques': freqs_ana[:num_modes],
        'ordres_modaux': ordres_modaux[:num_modes],
        'modes_propres': modes_propres,
        'chemin_figure': chemin_fig
    }
    
    return resultats


if __name__ == "__main__":
    # Paramètres d'exécution par défaut
    executer_cas_cavite_rectangulaire(
        Lx=1.0,
        Ly=0.6,
        c=343.0,
        p_xi=2,
        p_eta=2,
        n_el_xi=16,
        n_el_eta=10,
        num_modes=8,
        sauvegarder_figure=True
    )
