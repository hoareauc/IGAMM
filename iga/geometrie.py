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


def inserer_noeud_1d(U, P, u_bar, p):
    """
    Insère un nœud u_bar dans le vecteur de nœuds U et calcule les nouveaux points de contrôle.
    Algorithme A5.1 de Piegl & Tiller ("The NURBS Book").
    
    Paramètres :
        U (ndarray) : Vecteur de nœuds initial
        P (ndarray) : Points de contrôle (1D ou coordonnées homogènes multidimensionnelles)
        u_bar (float) : Nouvelle valeur paramétrique à insérer
        p (int) : Degré polynomial
        
    Retourne :
        U_new (ndarray) : Nouveau vecteur de nœuds avec u_bar inséré
        Q (ndarray) : Nouveaux points de contrôle
    """
    n = len(P) - 1
    k = -1
    for i in range(p, len(U) - p - 1):
        if U[i] <= u_bar < U[i + 1]:
            k = i
            break
    if k == -1:
        if u_bar >= U[len(U) - p - 1]:
            k = len(U) - p - 2
        else:
            return U, P
            
    dim = P.shape[1] if P.ndim > 1 else 1
    Q = np.zeros((n + 2, dim), dtype=float)
    P_mat = P.reshape((n + 1, dim))
    
    for i in range(0, k - p + 1):
        Q[i] = P_mat[i]
    for i in range(k + 1, n + 2):
        Q[i] = P_mat[i - 1]
    for i in range(k - p + 1, k + 1):
        denom = U[i + p] - U[i]
        alpha = (u_bar - U[i]) / denom if denom != 0.0 else 0.0
        Q[i] = (1.0 - alpha) * P_mat[i - 1] + alpha * P_mat[i]
        
    U_new = np.insert(U, k + 1, u_bar)
    if P.ndim == 1:
        Q = Q.ravel()
    return U_new, Q


def raffiner_surface_nurbs_2d(U_xi, U_eta, P_homog, p_xi, p_eta, noeuds_xi, noeuds_eta):
    """
    Raffine un patch NURBS 2D par insertion de nœuds en coordonnées projectives (homogènes).
    Garantit l'invariance exacte de la géométrie et du contour circulaire.
    
    Paramètres :
        U_xi, U_eta (ndarray) : Vecteurs de nœuds initiaux
        P_homog (ndarray de forme (n_xi, n_eta, 3)) : Coordonnées homogènes [w*x, w*y, w]
        p_xi, p_eta (int) : Degrés polynomiaux
        noeuds_xi (list/tuple) : Valeurs des nœuds à insérer en xi
        noeuds_eta (list/tuple) : Valeurs des nœuds à insérer en eta
        
    Retourne :
        U_xi_r, U_eta_r : Nouveaux vecteurs de nœuds
        ctrl_pts (ndarray) : Nouveaux points de contrôle physiques (x, y)
        weights (ndarray) : Nouveaux poids NURBS associés
    """
    U_xi_cur = U_xi.copy()
    P_cur = P_homog.copy()
    
    # 1. Insertion selon la direction xi
    for u in noeuds_xi:
        n_eta = P_cur.shape[1]
        P_list = []
        for j in range(n_eta):
            _, col_j = inserer_noeud_1d(U_xi_cur, P_cur[:, j, :], u, p_xi)
            P_list.append(col_j)
        U_xi_cur, _ = inserer_noeud_1d(U_xi_cur, P_cur[:, 0, :], u, p_xi)
        P_cur = np.stack(P_list, axis=1)
        
    # 2. Insertion selon la direction eta
    U_eta_cur = U_eta.copy()
    for v in noeuds_eta:
        n_xi = P_cur.shape[0]
        P_list = []
        for i in range(n_xi):
            _, row_i = inserer_noeud_1d(U_eta_cur, P_cur[i, :, :], v, p_eta)
            P_list.append(row_i)
        U_eta_cur, _ = inserer_noeud_1d(U_eta_cur, P_cur[0, :, :], v, p_eta)
        P_cur = np.stack(P_list, axis=0)
        
    weights = P_cur[:, :, 2]
    ctrl_pts = P_cur[:, :, :2] / weights[:, :, None]
    
    return U_xi_cur, U_eta_cur, ctrl_pts, weights


def creer_geometrie_disque_nurbs(R=1.0, n_el_xi=8, n_el_eta=8):
    """
    Crée une géométrie de cavité circulaire 2D (rayon R) avec un SEUL PATCH NURBS
    sous la forme d'un « carré gonflé » (inflated square), SANS ROTATION ni singularité centrale.
    
    Le patch initial quadratique (p=2, 3x3 = 9 points de contrôle) possède des bords
    qui forment 4 arcs circulaires exacts de 90° chacun grâce aux poids w = 1/sqrt(2).
    Le raffinement par insertion de nœuds (h-refinement) préserve strictement
    la géométrie circulaire exacte à la précision machine.
    
    Paramètres :
        R (float) : Rayon du disque (m)
        n_el_xi (int) : Nombre d'éléments selon la direction xi
        n_el_eta (int) : Nombre d'éléments selon la direction eta
        
    Retourne :
        dict : Description complète de la géométrie IGA NURBS du disque :
            - 'type' : 'disque'
            - 'est_nurbs' : True
            - 'R' : Rayon
            - 'p_xi', 'p_eta' : 2, 2 (quadratique)
            - 'n_el_xi', 'n_el_eta' : Nombres d'éléments
            - 'n_ctrl_xi', 'n_ctrl_eta' : Nombres de points de contrôle
            - 'n_dofs' : Nombre total de DDL
            - 'U_xi', 'U_eta' : Vecteurs de nœuds
            - 'points_ctrl' : Coordonnées physiques des points de contrôle (shape (n_xi, n_eta, 2))
            - 'poids' : Poids NURBS (shape (n_xi, n_eta))
            - 'elements' : Liste des éléments actifs
            - 'points_ctrl_base', 'poids_base' : Patch initial 3x3 non raffiné
    """
    assert R > 0.0, "Le rayon R doit être strictement positif."
    assert n_el_xi >= 1 and n_el_eta >= 1, "Le nombre d'éléments doit être >= 1."
    
    s2 = 1.0 / np.sqrt(2.0)
    
    # Patch de base quadratique (3 x 3 points de contrôle)
    P_base = np.zeros((3, 3, 2), dtype=float)
    W_base = np.ones((3, 3), dtype=float)
    
    # 4 coins (angles 225°, 315°, 45°, 135°)
    P_base[0, 0] = [-R * s2, -R * s2]
    P_base[2, 0] = [ R * s2, -R * s2]
    P_base[2, 2] = [ R * s2,  R * s2]
    P_base[0, 2] = [-R * s2,  R * s2]
    
    # 4 milieux d'arêtes (points de contrôle tangentiels avec poids w = 1/sqrt(2))
    P_base[1, 0] = [0.0, -np.sqrt(2.0) * R]
    P_base[2, 1] = [ np.sqrt(2.0) * R, 0.0]
    P_base[1, 2] = [0.0,  np.sqrt(2.0) * R]
    P_base[0, 1] = [-np.sqrt(2.0) * R, 0.0]
    
    W_base[1, 0] = s2
    W_base[2, 1] = s2
    W_base[1, 2] = s2
    W_base[0, 1] = s2
    
    # Centre (poids 1.0)
    P_base[1, 1] = [0.0, 0.0]
    W_base[1, 1] = 1.0
    
    # Forme projective homogène (w*x, w*y, w)
    P_homog = np.zeros((3, 3, 3), dtype=float)
    P_homog[:, :, :2] = P_base * W_base[:, :, None]
    P_homog[:, :, 2] = W_base
    
    U_xi_base = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=float)
    U_eta_base = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=float)
    
    # Nœuds uniformes à insérer pour obtenir n_el_xi et n_el_eta
    noeuds_xi = [i / float(n_el_xi) for i in range(1, n_el_xi)]
    noeuds_eta = [j / float(n_el_eta) for j in range(1, n_el_eta)]
    
    if len(noeuds_xi) > 0 or len(noeuds_eta) > 0:
        U_xi, U_eta, points_ctrl, poids = raffiner_surface_nurbs_2d(
            U_xi_base, U_eta_base, P_homog, 2, 2, noeuds_xi, noeuds_eta
        )
    else:
        U_xi = U_xi_base
        U_eta = U_eta_base
        points_ctrl = P_base
        poids = W_base
        
    n_ctrl_xi = points_ctrl.shape[0]
    n_ctrl_eta = points_ctrl.shape[1]
    
    elements = []
    for j_span in range(2, n_ctrl_eta):
        eta_min = U_eta[j_span]
        eta_max = U_eta[j_span + 1]
        if eta_max <= eta_min:
            continue
        for i_span in range(2, n_ctrl_xi):
            xi_min = U_xi[i_span]
            xi_max = U_xi[i_span + 1]
            if xi_max <= xi_min:
                continue
            elements.append({
                'i_span': i_span,
                'j_span': j_span,
                'xi_lim': (xi_min, xi_max),
                'eta_lim': (eta_min, eta_max)
            })
            
    geo = {
        'type': 'disque',
        'est_nurbs': True,
        'R': float(R),
        'Lx': 2.0 * float(R),
        'Ly': 2.0 * float(R),
        'p_xi': 2,
        'p_eta': 2,
        'n_el_xi': int(n_el_xi),
        'n_el_eta': int(n_el_eta),
        'n_ctrl_xi': int(n_ctrl_xi),
        'n_ctrl_eta': int(n_ctrl_eta),
        'n_dofs': int(n_ctrl_xi * n_ctrl_eta),
        'U_xi': U_xi,
        'U_eta': U_eta,
        'points_ctrl': points_ctrl,
        'poids': poids,
        'elements': elements,
        'points_ctrl_base': P_base,
        'poids_base': W_base
    }
    
    return geo

