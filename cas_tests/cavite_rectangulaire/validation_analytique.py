"""
Script de validation numérique et test de non-régression pour la cavité rectangulaire 2D IGA.
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
    creer_geometrie_rectangle,
    assembler_systeme_acoustique_2d,
    resoudre_modes_acoustiques_2d,
    frequences_analytiques_cavite_rectangulaire,
)


class TestCaviteAcoustiqueRectangulaire(unittest.TestCase):
    """Tests unitaires et de validation physique pour l'IGA acoustique."""

    def test_modes_cavite_2d(self):
        """Vérifie la précision des 6 premiers modes propres acoustiques (< 0.05% d'erreur relative)."""
        Lx = 1.0
        Ly = 0.6
        c = 343.0
        p_xi, p_eta = 2, 2
        n_el_xi, n_el_eta = 16, 10
        num_modes = 6
        
        geo = creer_geometrie_rectangle(Lx=Lx, Ly=Ly, p_xi=p_xi, p_eta=p_eta, n_el_xi=n_el_xi, n_el_eta=n_el_eta)
        K, M = assembler_systeme_acoustique_2d(geo)
        freqs_iga, _, _ = resoudre_modes_acoustiques_2d(K, M, num_modes=num_modes, c=c)
        freqs_ana, _ = frequences_analytiques_cavite_rectangulaire(Lx=Lx, Ly=Ly, c=c, max_m=4, max_n=4)
        
        # Mode 0 (mode constant de pression) doit être ~ 0 Hz
        self.assertAlmostEqual(freqs_iga[0], 0.0, delta=1e-3, msg="Le mode 0 doit être nul (mode rigide acoustique)")
        
        # Modes dynamiques : erreur relative < 0.05%
        for i in range(1, num_modes):
            f_exact = freqs_ana[i]
            f_calc = freqs_iga[i]
            err_rel_pct = abs(f_calc - f_exact) / f_exact * 100.0
            self.assertLess(
                err_rel_pct,
                0.05,
                f"Mode {i} : Erreur relative trop élevée ({err_rel_pct:.4f}%) par rapport à la solution exacte ({f_exact:.2f} Hz)"
            )


if __name__ == "__main__":
    unittest.main()
