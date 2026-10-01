"""
Script de validation numérique et test de non-régression pour la cavité elliptique 2D IGA NURBS.
Peut être exécuté directement ou via un lanceur de tests (pytest / unittest).
"""

import os
import sys
import unittest
import numpy as np

chemin_courant = os.path.dirname(os.path.abspath(__file__))
chemin_racine = os.path.abspath(os.path.join(chemin_courant, "..", ".."))
if chemin_racine not in sys.path:
    sys.path.insert(0, chemin_racine)

from iga import (
    creer_geometrie_ellipse_nurbs,
    assembler_systeme_acoustique_2d,
    resoudre_modes_acoustiques_2d,
    frequences_analytiques_cavite_circulaire,
)
from iga.bspline import trouver_intervalle, fonctions_base_et_derivees_nurbs_2d


class TestCaviteAcoustiqueElliptique(unittest.TestCase):
    """Tests unitaires et de validation analytique pour l'ellipse NURBS (patch unique)."""

    def test_precision_contour_elliptique(self):
        """Vérifie que le contour physique de l'ellipse satisfait (x/a)^2 + (y/b)^2 = 1 à la précision machine."""
        a, b = 1.25, 0.75
        geo = creer_geometrie_ellipse_nurbs(a=a, b=b, n_el_xi=8, n_el_eta=8)
        
        U_xi = geo['U_xi']
        U_eta = geo['U_eta']
        n_xi = geo['n_ctrl_xi']
        n_eta = geo['n_ctrl_eta']
        P_ctrl = geo['points_ctrl']
        W_ctrl = geo['poids']
        
        def eval_pt(u, v):
            i_sp = trouver_intervalle(n_xi, 2, u, U_xi)
            j_sp = trouver_intervalle(n_eta, 2, v, U_eta)
            W_loc = W_ctrl[i_sp-2:i_sp+1, j_sp-2:j_sp+1]
            P_loc = P_ctrl[i_sp-2:i_sp+1, j_sp-2:j_sp+1]
            R_loc, _, _ = fonctions_base_et_derivees_nurbs_2d(i_sp, j_sp, u, v, 2, 2, U_xi, U_eta, W_loc)
            pt = np.zeros(2)
            loc = 0
            for j in range(3):
                for i in range(3):
                    pt += R_loc[loc] * P_loc[i, j]
                    loc += 1
            return pt

        # Échantillonnage fin le long des 4 bords paramétriques
        for xi in np.linspace(0.0, 1.0, 50):
            for eta in (0.0, 1.0):
                pt = eval_pt(xi, eta)
                val = (pt[0] / a)**2 + (pt[1] / b)**2
                self.assertAlmostEqual(
                    val, 1.0, places=12,
                    msg=f"Erreur sur le contour elliptique à (xi={xi}, eta={eta}) : |val - 1| = {abs(val - 1.0)}"
                )
                
        for eta in np.linspace(0.0, 1.0, 50):
            for xi in (0.0, 1.0):
                pt = eval_pt(xi, eta)
                val = (pt[0] / a)**2 + (pt[1] / b)**2
                self.assertAlmostEqual(
                    val, 1.0, places=12,
                    msg=f"Erreur sur le contour elliptique à (xi={xi}, eta={eta}) : |val - 1| = {abs(val - 1.0)}"
                )

    def test_modes_cavite_elliptique(self):
        """Vérifie le mode rigide et la levée de dégénérescence des modes dipolaires (f1 != f2)."""
        a, b = 1.0, 0.6
        c = 343.0
        geo = creer_geometrie_ellipse_nurbs(a=a, b=b, n_el_xi=10, n_el_eta=10)
        K, M = assembler_systeme_acoustique_2d(geo)
        freqs_iga, _, _ = resoudre_modes_acoustiques_2d(K, M, num_modes=6, c=c)
        
        # Mode rigide ~ 0 Hz
        self.assertAlmostEqual(freqs_iga[0], 0.0, delta=1e-3, msg="Le mode 0 doit être nul")
        
        # Levée de dégénérescence due à l'excentricité
        f1 = freqs_iga[1]
        f2 = freqs_iga[2]
        self.assertGreater(f1, 50.0, "Fréquence dipolaire trop basse")
        self.assertGreater(f2, f1 + 20.0, "La dégénérescence azimutale doit être nettement levée par l'excentricité")

    def test_limite_circulaire(self):
        """Vérifie que lorsque a = b = R, les fréquences tendent vers la solution circulaire de Bessel."""
        R = 1.0
        c = 343.0
        geo = creer_geometrie_ellipse_nurbs(a=R, b=R, n_el_xi=12, n_el_eta=12)
        K, M = assembler_systeme_acoustique_2d(geo)
        freqs_iga, _, _ = resoudre_modes_acoustiques_2d(K, M, num_modes=4, c=c)
        freqs_ana, _, _ = frequences_analytiques_cavite_circulaire(R=R, c=c, max_m=4, max_n=3)
        
        # Mode 1 et Mode 2 doivent être quasi identiques et proches de Bessel (100.51 Hz)
        self.assertAlmostEqual(freqs_iga[1], freqs_ana[1], delta=0.01)
        self.assertAlmostEqual(freqs_iga[2], freqs_ana[2], delta=0.01)


if __name__ == "__main__":
    unittest.main()
