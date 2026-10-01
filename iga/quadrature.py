"""
Module quadrature.py
Calcul fonctionnel des points et poids de quadrature de Gauss-Legendre pour l'intégration numérique en IGA 2D.
Approche 100% procédurale (fonctions pures, sans classes / POO).
"""

import numpy as np


def quadrature_gauss_1d(n_pts):
    """
    Calcule les points et poids de quadrature de Gauss-Legendre sur le domaine canonique [-1, 1].
    
    Paramètres :
        n_pts (int) : Nombre de points de Gauss (n_pts >= 1)
        
    Retourne :
        pts (ndarray de taille n_pts) : Coordonnées des points de Gauss
        wts (ndarray de taille n_pts) : Poids associés
    """
    assert n_pts >= 1, "Le nombre de points de quadrature doit être >= 1"
    pts, wts = np.polynomial.legendre.leggauss(n_pts)
    return pts, wts


def quadrature_gauss_2d(n_pts_xi, n_pts_eta):
    """
    Génère la grille 2D de points et poids de Gauss-Legendre sur [-1, 1] x [-1, 1]
    par produit tensoriel 1D x 1D.
    
    Paramètres :
        n_pts_xi (int) : Nombre de points de Gauss selon xi
        n_pts_eta (int) : Nombre de points de Gauss selon eta
        
    Retourne :
        pts_2d (ndarray de forme (n_pts_xi * n_pts_eta, 2)) : Coordonnées (xi_canon, eta_canon)
        wts_2d (ndarray de taille n_pts_xi * n_pts_eta) : Poids 2D w_xi * w_eta
    """
    pts_xi, wts_xi = quadrature_gauss_1d(n_pts_xi)
    pts_eta, wts_eta = quadrature_gauss_1d(n_pts_eta)
    
    total_pts = n_pts_xi * n_pts_eta
    pts_2d = np.zeros((total_pts, 2), dtype=float)
    wts_2d = np.zeros(total_pts, dtype=float)
    
    idx = 0
    for j in range(n_pts_eta):
        for i in range(n_pts_xi):
            pts_2d[idx, 0] = pts_xi[i]
            pts_2d[idx, 1] = pts_eta[j]
            wts_2d[idx] = wts_xi[i] * wts_eta[j]
            idx += 1
            
    return pts_2d, wts_2d


def mapper_quadrature_vers_element(xi_min, xi_max, eta_min, eta_max, pts_canon, wts_canon):
    """
    Mappe les points de quadrature du domaine de référence [-1, 1] x [-1, 1]
    vers l'élément physique/paramétrique [xi_min, xi_max] x [eta_min, eta_max].
    
    Paramètres :
        xi_min, xi_max (float) : Bornes en xi de l'élément (knot span)
        eta_min, eta_max (float) : Bornes en eta de l'élément (knot span)
        pts_canon (ndarray de forme (N, 2)) : Points dans [-1, 1]^2
        wts_canon (ndarray de taille N) : Poids sur [-1, 1]^2
        
    Retourne :
        pts_elem (ndarray de forme (N, 2)) : Coordonnées physiques dans le span
        wts_elem (ndarray de taille N) : Poids mis à l'échelle par le déterminant jacobien de la transformation affine
    """
    d_xi = 0.5 * (xi_max - xi_min)
    c_xi = 0.5 * (xi_max + xi_min)
    
    d_eta = 0.5 * (eta_max - eta_min)
    c_eta = 0.5 * (eta_max + eta_min)
    
    pts_elem = np.zeros_like(pts_canon)
    pts_elem[:, 0] = c_xi + d_xi * pts_canon[:, 0]
    pts_elem[:, 1] = c_eta + d_eta * pts_canon[:, 1]
    
    # Déterminant du changement de variables de [-1, 1]^2 vers [xi_min, xi_max] x [eta_min, eta_max]
    facteur_jacobien = d_xi * d_eta
    wts_elem = wts_canon * facteur_jacobien
    
    return pts_elem, wts_elem
