"""
Module bspline.py
Calcul fonctionnel des fonctions de base B-splines et de leurs dérivées 1D et 2D.
Implémente les algorithmes classiques de Piegl & Tiller ("The NURBS Book").
Approche 100% procédurale (fonctions pures, sans classes / POO).
"""

import numpy as np


def generer_vecteur_noeuds_uniforme(n_ctrl, p, u_min=0.0, u_max=1.0):
    """
    Génère un vecteur de nœuds ouvert et uniforme pour n_ctrl points de contrôle
    et un degré polynomial p.
    
    Un vecteur ouvert possède une multiplicité (p + 1) aux extrémités u_min et u_max,
    ce qui permet l'interpolation exacte aux bords.
    
    Paramètres :
        n_ctrl (int) : Nombre de points de contrôle (n_ctrl >= p + 1)
        p (int) : Degré polynomial (p >= 1)
        u_min (float) : Valeur minimale du domaine paramétrique (défaut 0.0)
        u_max (float) : Valeur maximale du domaine paramétrique (défaut 1.0)
        
    Retourne :
        ndarray (taille n_ctrl + p + 1) : Vecteur de nœuds
    """
    assert n_ctrl >= p + 1, f"Le nombre de points de contrôle ({n_ctrl}) doit être >= p + 1 ({p + 1})"
    
    m = n_ctrl + p + 1
    vecteur = np.zeros(m, dtype=float)
    
    # Nœuds initiaux de multiplicité p + 1
    vecteur[:p + 1] = u_min
    
    # Nœuds finaux de multiplicité p + 1
    vecteur[m - (p + 1):] = u_max
    
    # Nœuds intérieurs uniformément espacés
    n_spans = n_ctrl - p  # nombre d'éléments (spans)
    if n_spans > 1:
        pas = (u_max - u_min) / float(n_spans)
        for i in range(1, n_spans):
            vecteur[p + i] = u_min + i * pas
            
    return vecteur


def trouver_intervalle(n_ctrl, p, u, U):
    """
    Détermine l'indice i de l'intervalle de nœuds tel que u in [U[i], U[i+1]).
    Algorithme A2.1 de Piegl & Tiller (recherche binaire).
    
    Paramètres :
        n_ctrl (int) : Nombre total de points de contrôle
        p (int) : Degré polynomial
        u (float) : Coordonnée paramétrique
        U (ndarray) : Vecteur de nœuds
        
    Retourne :
        int : Indice i de l'intervalle de nœuds actif
    """
    n = n_ctrl - 1  # dernier indice de point de contrôle
    
    # Cas limite de l'extrémité droite
    if u >= U[n + 1]:
        return n
    if u <= U[p]:
        return p
        
    low = p
    high = n + 1
    mid = (low + high) // 2
    
    while u < U[mid] or u >= U[mid + 1]:
        if u < U[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2
        
    return mid


def fonctions_base_1d(i, u, p, U):
    """
    Calcule les (p + 1) fonctions de base B-splines non nulles au point u.
    Algorithme A2.2 de Piegl & Tiller.
    
    Paramètres :
        i (int) : Indice d'intervalle tel que u in [U[i], U[i+1])
        u (float) : Coordonnée paramétrique
        p (int) : Degré polynomial
        U (ndarray) : Vecteur de nœuds
        
    Retourne :
        ndarray (taille p + 1) : Valeurs [N_{i-p, p}(u), ..., N_{i, p}(u)]
    """
    N = np.zeros(p + 1, dtype=float)
    N[0] = 1.0
    left = np.zeros(p + 1, dtype=float)
    right = np.zeros(p + 1, dtype=float)
    
    for j in range(1, p + 1):
        left[j] = u - U[i + 1 - j]
        right[j] = U[i + j] - u
        saved = 0.0
        for r in range(j):
            denom = right[r + 1] + left[j - r]
            if denom != 0.0:
                temp = N[r] / denom
                N[r] = saved + right[r + 1] * temp
                saved = left[j - r] * temp
            else:
                saved = 0.0
        N[j] = saved
        
    return N


def derivees_fonctions_base_1d(i, u, p, n_derivees, U):
    """
    Calcule les fonctions de base B-splines et leurs dérivées jusqu'à l'ordre n_derivees
    au point u pour l'intervalle i.
    Algorithme A2.3 de Piegl & Tiller.
    
    Paramètres :
        i (int) : Indice d'intervalle tel que u in [U[i], U[i+1])
        u (float) : Coordonnée paramétrique
        p (int) : Degré polynomial
        n_derivees (int) : Ordre maximal des dérivées souhaitées
        U (ndarray) : Vecteur de nœuds
        
    Retourne :
        ndarray (taille n_derivees + 1, p + 1) :
            ders[k, j] = k-ième dérivée de N_{i-p+j, p}(u)
    """
    ders = np.zeros((n_derivees + 1, p + 1), dtype=float)
    ndu = np.zeros((p + 1, p + 1), dtype=float)
    ndu[0, 0] = 1.0
    
    left = np.zeros(p + 1, dtype=float)
    right = np.zeros(p + 1, dtype=float)
    
    for j in range(1, p + 1):
        left[j] = u - U[i + 1 - j]
        right[j] = U[i + j] - u
        saved = 0.0
        for r in range(j):
            denom = right[r + 1] + left[j - r]
            ndu[j, r] = denom
            if denom != 0.0:
                temp = ndu[r, j - 1] / denom
                ndu[r, j] = saved + right[r + 1] * temp
                saved = left[j - r] * temp
            else:
                ndu[r, j] = saved
                saved = 0.0
        ndu[j, j] = saved
        
    # Ordre 0 (valeur des fonctions)
    for j in range(p + 1):
        ders[0, j] = ndu[j, p]
        
    # Dérivées
    a = np.zeros((2, p + 1), dtype=float)
    for r in range(p + 1):
        s1 = 0
        s2 = 1
        a[0, 0] = 1.0
        for k in range(1, n_derivees + 1):
            d = 0.0
            rk = r - k
            pk = p - k
            if r >= k:
                a[s2, 0] = a[s1, 0] / ndu[pk + 1, rk]
                d = a[s2, 0] * ndu[rk, pk]
            j1 = 1 if rk >= -1 else -rk
            j2 = k - 1 if (r - 1) <= pk else p - r
            for j in range(j1, j2 + 1):
                a[s2, j] = (a[s1, j] - a[s1, j - 1]) / ndu[pk + 1, rk + j]
                d += a[s2, j] * ndu[rk + j, pk]
            if r <= pk:
                a[s2, k] = -a[s1, k - 1] / ndu[pk + 1, r]
                d += a[s2, k] * ndu[r, pk]
            ders[k, r] = d
            s1, s2 = s2, s1
            
    # Multiplication par les coefficients factoriels
    r_fac = float(p)
    for k in range(1, n_derivees + 1):
        for j in range(p + 1):
            ders[k, j] *= r_fac
        r_fac *= (p - k)
        
    return ders


def fonctions_base_et_derivees_2d(i_span, j_span, xi, eta, p_xi, p_eta, U_xi, U_eta):
    """
    Calcule les fonctions de base 2D et leurs dérivées premières partielles (par rapport à xi et eta)
    par produit tensoriel :
        R_{a, b}(xi, eta) = N_a(xi) * M_b(eta)
        dR/dxi  = dN_a/dxi * M_b
        dR/deta = N_a * dM_b/deta
        
    Paramètres :
        i_span, j_span (int) : Indices d'intervalle en xi et eta
        xi, eta (float) : Coordonnées paramétriques
        p_xi, p_eta (int) : Degrés polynomiaux
        U_xi, U_eta (ndarray) : Vecteurs de nœuds
        
    Retourne :
        R (ndarray de taille (p_xi + 1)*(p_eta + 1)) : Valeurs des fonctions de base 2D
        dR_dxi (ndarray de même taille) : Dérivées partielles par rapport à xi
        dR_deta (ndarray de même taille) : Dérivées partielles par rapport à eta
    """
    # Calcul 1D avec dérivées d'ordre 1
    ders_xi = derivees_fonctions_base_1d(i_span, xi, p_xi, 1, U_xi)
    ders_eta = derivees_fonctions_base_1d(j_span, eta, p_eta, 1, U_eta)
    
    n_loc = (p_xi + 1) * (p_eta + 1)
    R = np.zeros(n_loc, dtype=float)
    dR_dxi = np.zeros(n_loc, dtype=float)
    dR_deta = np.zeros(n_loc, dtype=float)
    
    loc_idx = 0
    for b in range(p_eta + 1):
        M_b = ders_eta[0, b]
        dM_b = ders_eta[1, b]
        for a in range(p_xi + 1):
            N_a = ders_xi[0, a]
            dN_a = ders_xi[1, a]
            
            R[loc_idx] = N_a * M_b
            dR_dxi[loc_idx] = dN_a * M_b
            dR_deta[loc_idx] = N_a * dM_b
            loc_idx += 1
            
    return R, dR_dxi, dR_deta
