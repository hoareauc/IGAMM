"""
Visualisation 3D avancée des modes acoustiques de la cavité circulaire avec PyVista.
Permet d'observer les ondes stationnaires en relief 3D (nappes vibrantes z = p(x, y)),
les lignes nodales 3D (p = 0) et le contour circulaire du domaine physique.

Génère 3 rendus 3D haute définition :
1. mode_1_3d_pyvista_circulaire.png : Mode dipolaire #1 en élévation 3D
2. mode_5_radial_3d_pyvista_circulaire.png : Mode radial #5 (anneau nodal) en relief 3D
3. planche_6modes_3d_pyvista_circulaire.png : Planche maîtresse 2x3 des 6 premiers modes en 3D

Usage :
    python visualisation_pyvista_circulaire.py
    python visualisation_pyvista_circulaire.py --interactif   (pour ouvrir la fenêtre 3D interactive)
"""

import os
import sys
import argparse
import numpy as np

chemin_courant = os.path.dirname(os.path.abspath(__file__))
chemin_racine = os.path.abspath(os.path.join(chemin_courant, "..", ".."))
if chemin_racine not in sys.path:
    sys.path.insert(0, chemin_racine)

from iga import (
    creer_geometrie_disque_nurbs,
    assembler_systeme_acoustique_2d,
    resoudre_modes_acoustiques_2d,
    frequences_analytiques_cavite_circulaire,
    afficher_comparaison_modes,
    visualiser_mode_pyvista_3d,
    visualiser_planche_modes_pyvista_3d,
)


def executer_visualisation_pyvista_circulaire(
    R=1.0,
    c=343.0,
    n_el_xi=14,
    n_el_eta=14,
    num_modes=8,
    sauvegarder=True,
    afficher=False
):
    """
    Calcule les modes propres et produit les visualisations 3D PyVista de la cavité circulaire.
    """
    print("=" * 88)
    print("  VISUALISATION 3D PYVISTA : MODES ACOUSTIQUES DE LA CAVITÉ CIRCULAIRE (IGA NURBS)")
    print("=" * 88)
    print(f"  Rayon R = {R:.2f} m | Célérité c = {c:.1f} m/s")
    print(f"  Maillage : {n_el_xi} x {n_el_eta} éléments NURBS quadratiques")
    
    # 1. Calcul IGA
    geo = creer_geometrie_disque_nurbs(R=R, n_el_xi=n_el_xi, n_el_eta=n_el_eta)
    print(f"  Degrés de liberté (DDL) : {geo['n_dofs']}")
    
    K, M = assembler_systeme_acoustique_2d(geo)
    freqs_iga, modes_propres, _ = resoudre_modes_acoustiques_2d(K, M, num_modes=num_modes, c=c)
    freqs_ana, ordres_modaux, _ = frequences_analytiques_cavite_circulaire(R=R, c=c, max_m=6, max_n=4)
    
    # Affichage du tableau comparatif
    afficher_comparaison_modes(freqs_iga, freqs_ana, ordres_modaux, num_modes=num_modes)
    
    dossier_sortie = os.path.dirname(os.path.abspath(__file__))
    fichiers_rendus = []
    
    # 2. Rendu 3D du Mode #1 (Dipolaire - 100.51 Hz)
    f_m1 = freqs_iga[1]
    nom_m1 = "mode_1_3d_pyvista_circulaire.png"
    p_m1 = os.path.join(dossier_sortie, nom_m1)
    visualiser_mode_pyvista_3d(
        geo,
        modes_propres[:, 1],
        freq=f_m1,
        mode_id=1,
        amplitude_z=0.35 * R,
        chemin_sauvegarde=p_m1 if sauvegarder else None,
        afficher=afficher
    )
    if sauvegarder:
        fichiers_rendus.append(p_m1)
        
    # 3. Rendu 3D du Mode #5 (Radial Monopolaire - 209.18 Hz)
    f_m5 = freqs_iga[5]
    nom_m5 = "mode_5_radial_3d_pyvista_circulaire.png"
    p_m5 = os.path.join(dossier_sortie, nom_m5)
    visualiser_mode_pyvista_3d(
        geo,
        modes_propres[:, 5],
        freq=f_m5,
        mode_id=5,
        amplitude_z=0.35 * R,
        chemin_sauvegarde=p_m5 if sauvegarder else None,
        afficher=afficher
    )
    if sauvegarder:
        fichiers_rendus.append(p_m5)
        
    # 4. Planche maîtresse 3D des 6 premiers modes (indices 1 à 6)
    nom_planche = "planche_6modes_3d_pyvista_circulaire.png"
    p_planche = os.path.join(dossier_sortie, nom_planche)
    visualiser_planche_modes_pyvista_3d(
        geo,
        freqs_iga,
        modes_propres,
        indices_modes=(1, 2, 3, 4, 5, 6),
        ordres_modaux=ordres_modaux,
        amplitude_z=0.28 * R,
        chemin_sauvegarde=p_planche if sauvegarder else None,
        afficher=afficher
    )
    if sauvegarder:
        fichiers_rendus.append(p_planche)
        
    print("=" * 88)
    print("  VISUALISATION 3D PYVISTA TERMINÉE AVEC SUCCÈS")
    for f in fichiers_rendus:
        print(f"  -> {f}")
    print("=" * 88)
    
    return fichiers_rendus


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualisation 3D PyVista des modes acoustiques de la cavité circulaire IGA")
    parser.add_argument("--interactif", action="store_true", help="Ouvrir les fenêtres 3D interactives")
    args = parser.parse_args()
    
    executer_visualisation_pyvista_circulaire(
        R=1.0,
        c=343.0,
        n_el_xi=14,
        n_el_eta=14,
        num_modes=8,
        sauvegarder=True,
        afficher=args.interactif
    )
