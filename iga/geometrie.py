"""
Module geometrie.py
Représentation et calculs géométriques pour l'analyse isogéométrique 2D.
Génération du maillage IGA (points de contrôle, vecteurs de nœuds) et calcul de la métrique
(matrice jacobienne, déterminant jacobien, gradients physiques).
Approche 100% procédurale (dictionnaires et tableaux NumPy, sans POO).
"""

import numpy as np
from .bspline import generer_vecteur_noeuds_uniforme


def calculer_abscisses_greville_1d(n_ctrl, p, U):
    """
    Calcule les abscisses de Greville (moyennes de nœuds) pour un vecteur de nœuds ouvert.
    xi*_i = (1 / p) * sum_{k=1}^p U[i + k]
    Ces points garantissent la reproduction linéaire exacte de la géométrie.
    
    Paramètres :
        n_ctrl (int) : Nombre de points de contrôle
        p (int) : Degré polynomial
        U (ndarray) : Vecteur de nœuds
        
    Retourne :
        ndarray (taille n_ctrl) : Abscisses de Greville
    """
    greville = np.zeros(n_ctrl, dtype=float)
    for i in range(n_ctrl):
        if p == 0:
            greville[i] = 0.5 * (U[i] + U[i + 1])
        else:
            greville[i] = np.sum(U[i + 1:i + 1 + p]) / float(p)
    return greville


def creer_geometrie_rectangle(Lx=1.0, Ly=1.0, p_xi=2, p_eta=2, n_el_xi=8, n_el_eta=8):
    """
    Crée une géométrie IGA rectangulaire [0, Lx] x [0, Ly].
    Utilise une discrétisation B-spline tensorielle.
    
    Paramètres :
        Lx (float) : Longueur du domaine selon x (m)
        Ly (float) : Hauteur du domaine selon y (m)
        p_xi (int) : Degré polynomial selon la direction xi (x)
        p_eta (int) : Degré polynomial selon la direction eta (y)
        n_el_xi (int) : Nombre d'éléments (spans non nuls) selon xi
        n_el_eta (int) : Nombre d'éléments (spans non nuls) selon eta
        
    Retourne :
        dict : Dictionnaire contenant la description géométrique et la discrétisation IGA :
            - 'Lx', 'Ly' : Dimensions physiques
            - 'p_xi', 'p_eta' : Degrés polynomiaux
            - 'n_el_xi', 'n_el_eta' : Nombres d'éléments
            - 'n_ctrl_xi', 'n_ctrl_eta' : Nombres de points de contrôle
            - 'n_dofs' : Nombre total de degrés de liberté
            - 'U_xi', 'U_eta' : Vecteurs de nœuds
            - 'points_ctrl' : Coordonnées des points de contrôle (shape (n_ctrl_xi, n_ctrl_eta, 2))
            - 'elements' : Liste des éléments actifs (indices i_span, j_span, [u_min, u_max], [v_min, v_max])
    """
    assert Lx > 0.0 and Ly > 0.0, "Les dimensions Lx et Ly doivent être strictement positives."
    assert p_xi >= 1 and p_eta >= 1, "Les degrés p_xi et p_eta doivent être >= 1."
    assert n_el_xi >= 1 and n_el_eta >= 1, "Le nombre d'éléments doit être >= 1."
    
    n_ctrl_xi = n_el_xi + p_xi
    n_ctrl_eta = n_el_eta + p_eta
    
    # Vecteurs de nœuds ouverts uniformes sur [0, 1]
    U_xi = generer_vecteur_noeuds_uniforme(n_ctrl_xi, p_xi, 0.0, 1.0)
    U_eta = generer_vecteur_noeuds_uniforme(n_ctrl_eta, p_eta, 0.0, 1.0)
    
    # Abscisses de Greville pour placer les points de contrôle
    greville_xi = calculer_abscisses_greville_1d(n_ctrl_xi, p_xi, U_xi)
    greville_eta = calculer_abscisses_greville_1d(n_ctrl_eta, p_eta, U_eta)
    
    # Grille 2D des points de contrôle
    points_ctrl = np.zeros((n_ctrl_xi, n_ctrl_eta, 2), dtype=float)
    for j in range(n_ctrl_eta):
        for i in range(n_ctrl_xi):
            points_ctrl[i, j, 0] = greville_xi[i] * Lx
            points_ctrl[i, j, 1] = greville_eta[j] * Ly
            
    # Extraction de la liste des éléments actifs (intervalles de nœuds de mesure non nulle)
    elements = []
    # Les nœuds intérieurs non répétés sont situés entre l'indice p et m-(p+1)
    for j_span in range(p_eta, n_ctrl_eta):
        eta_min = U_eta[j_span]
        eta_max = U_eta[j_span + 1]
        if eta_max <= eta_min:
            continue  # span de mesure nulle
            
        for i_span in range(p_xi, n_ctrl_xi):
            xi_min = U_xi[i_span]
            xi_max = U_xi[i_span + 1]
            if xi_max <= xi_min:
                continue  # span de mesure nulle
                
            elements.append({
                'i_span': i_span,
                'j_span': j_span,
                'xi_lim': (xi_min, xi_max),
                'eta_lim': (eta_min, eta_max)
            })
            
    geo = {
        'Lx': float(Lx),
        'Ly': float(Ly),
        'p_xi': int(p_xi),
        'p_eta': int(p_eta),
        'n_el_xi': int(n_el_xi),
        'n_el_eta': int(n_el_eta),
        'n_ctrl_xi': int(n_ctrl_xi),
        'n_ctrl_eta': int(n_ctrl_eta),
        'n_dofs': int(n_ctrl_xi * n_ctrl_eta),
        'U_xi': U_xi,
        'U_eta': U_eta,
        'points_ctrl': points_ctrl,
        'elements': elements
    }
    
    return geo


def evaluer_jacobienne_et_gradients_physiques(geo, i_span, j_span, dR_dxi, dR_deta):
    """
    Calcule la matrice jacobienne de la transformation géométrique (xi, eta) -> (x, y),
    son déterminant et les gradients physiques des fonctions de base dR/dx et dR/dy.
    
    Paramètres :
        geo (dict) : Géométrie IGA
        i_span, j_span (int) : Indices d'intervalle actifs
        dR_dxi (ndarray de taille n_loc) : Dérivées partielles locales par rapport à xi
        dR_deta (ndarray de taille n_loc) : Dérivées partielles locales par rapport à eta
        
    Retourne :
        det_J (float) : Déterminant jacobien de la transformation géométrique
        dR_dx (ndarray de taille n_loc) : Dérivées spatiales par rapport à x
        dR_dy (ndarray de taille n_loc) : Dérivées spatiales par rapport à y
    """
    p_xi = geo['p_xi']
    p_eta = geo['p_eta']
    pts_ctrl = geo['points_ctrl']
    
    # Calcul de la matrice jacobienne J = [[dx/dxi, dy/dxi], [dx/deta, dy/deta]]
    dx_dxi = 0.0
    dy_dxi = 0.0
    dx_deta = 0.0
    dy_deta = 0.0
    
    loc_idx = 0
    for b in range(p_eta + 1):
        j_glob = j_span - p_eta + b
        for a in range(p_xi + 1):
            i_glob = i_span - p_xi + a
            
            x_ab = pts_ctrl[i_glob, j_glob, 0]
            y_ab = pts_ctrl[i_glob, j_glob, 1]
            
            dxi = dR_dxi[loc_idx]
            deta = dR_deta[loc_idx]
            
            dx_dxi += dxi * x_ab
            dy_dxi += dxi * y_ab
            dx_deta += deta * x_ab
            dy_deta += deta * y_ab
            
            loc_idx += 1
            
    # Déterminant de J
    det_J = dx_dxi * dy_deta - dy_dxi * dx_deta
    assert det_J > 1e-14, f"Déterminant jacobien géométrique non strictement positif : {det_J}"
    
    inv_det = 1.0 / det_J
    
    # Inversion de J :
    # [dR/dx]   1   [  dy/deta  -dy/dxi ] [dR/dxi ]
    # [dR/dy] = --- [ -dx/deta   dx/dxi ] [dR/deta]
    #           det
    dR_dx = inv_det * (dy_deta * dR_dxi - dy_dxi * dR_deta)
    dR_dy = inv_det * (-dx_deta * dR_dxi + dx_dxi * dR_deta)
    
    return det_J, dR_dx, dR_dy
