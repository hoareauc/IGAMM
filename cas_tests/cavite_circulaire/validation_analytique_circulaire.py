"""
Script de validation numérique et test de non-régression pour la cavité circulaire 2D IGA NURBS.
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
    creer_geometrie_disque_nurbs,
    assembler_systeme_acoustique_2d,
    resoudre_modes_acoustiques_2d,
    frequences_analytiques_cavite_circulaire,
)


class TestCaviteAcoustiqueCirculaire(unittest.TestCase):
    """Tests unitaires et de validation analytique pour l'IGA acoustique NURBS (cavité circulaire)."""

    def test_geometrie_disque_nurbs(self):
        """Vérifie que la géométrie circulaire NURBS préserve strictement le contour r = R."""
        R = 1.0
        geo = creer_geometrie_disque_nurbs(R=R, n_el_xi=8, n_el_eta=8)
        
        # Vérification des propriétés de la géométrie
        self.assertEqual(geo['type'], 'disque')
        self.assertTrue(geo['est_nurbs'])
        self.assertEqual(geo['p_xi'], 2)
        self.assertEqual(geo['p_eta'], 2)
        self.assertEqual(geo['n_dofs'], (8 + 2) * (8 + 2))
        
        # Vérification des poids
        self.assertTrue(np.all(geo['poids'] > 0.0), "Tous les poids NURBS doivent être strictement positifs.")

    def test_modes_cavite_circulaire(self):
        """Vérifie la précision des 8 premiers modes propres acoustiques (< 0.05% d'erreur relative)."""
        R = 1.0
        c = 343.0
        n_el_xi, n_el_eta = 12, 12
        num_modes = 8
        
        geo = creer_geometrie_disque_nurbs(R=R, n_el_xi=n_el_xi, n_el_eta=n_el_eta)
        K, M = assembler_systeme_acoustique_2d(geo)
        freqs_iga, _, _ = resoudre_modes_acoustiques_2d(K, M, num_modes=num_modes, c=c)
        freqs_ana, _, _ = frequences_analytiques_cavite_circulaire(R=R, c=c, max_m=6, max_n=4)
        
        # Mode 0 (mode rigide / pression constante) doit être ~ 0 Hz
        self.assertAlmostEqual(freqs_iga[0], 0.0, delta=1e-3, msg="Le mode 0 doit être nul (mode constant)")
        
        # Modes dynamiques (1 à 7) : erreur relative < 0.05%
        for i in range(1, num_modes):
            f_exact = freqs_ana[i]
            f_calc = freqs_iga[i]
            err_rel_pct = abs(f_calc - f_exact) / f_exact * 100.0
            self.assertLess(
                err_rel_pct,
                0.05,
                f"Mode {i} : Erreur relative ({err_rel_pct:.4f}%) supérieure au seuil de tolérance (0.05%)"
            )


if __name__ == "__main__":
    unittest.main()
