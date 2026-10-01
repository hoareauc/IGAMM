"""
Module assemblage.py
Assemblage procédural des matrices globales de rigidité et de masse pour l'acoustique 2D en IGA.
Formulation variationnelle de Helmholtz pour cavité acoustique :
    K_{ij} = int_Omega grad(R_i) . grad(R_j) dOmega
    M_{ij} = int_Omega R_i * R_j dOmega
Matrices retournées au format creux CSR (Compressed Sparse Row) de SciPy.
Approche 100% procédurale (fonctions pures, sans POO).
"""

import numpy as np
from scipy.sparse import coo_matrix

from .bspline import fonctions_base_et_derivees_2d
from .quadrature import quadrature_gauss_2d, mapper_quadrature_vers_element
from .geometrie import evaluer_jacobienne_et_gradients_physiques


def assembler_systeme_acoustique_2d(geo, n_gauss_xi=None, n_gauss_eta=None):
    """
    Assemble les matrices globales de rigidité K et de masse M pour l'acoustique 2D.
    
    Conditions aux limites :
        Parois rigides (conditions de Neumann homogènes dp/dn = 0 sur tout le bord).
        Les intégrales de bord s'annulent naturellement dans la forme faible,
        aucun terme de surface n'est donc requis pour des parois parfaitement rigides.
        
    Paramètres :
        geo (dict) : Description de la géométrie IGA créée par `creer_geometrie_rectangle`
        n_gauss_xi (int, optionnel) : Nombre de points de Gauss selon xi par élément (défaut p_xi + 1)
        n_gauss_eta (int, optionnel) : Nombre de points de Gauss selon eta par élément (défaut p_eta + 1)
        
    Retourne :
        K (csr_matrix de taille (n_dofs, n_dofs)) : Matrice globale de rigidité
        M (csr_matrix de taille (n_dofs, n_dofs)) : Matrice globale de masse
    """
    p_xi = geo['p_xi']
    p_eta = geo['p_eta']
    U_xi = geo['U_xi']
    U_eta = geo['U_eta']
    n_ctrl_xi = geo['n_ctrl_xi']
    n_dofs = geo['n_dofs']
    elements = geo['elements']
    
    # Ordre de quadrature par élément (p + 1 par direction est la règle standard exacte)
    if n_gauss_xi is None:
        n_gauss_xi = p_xi + 1
    if n_gauss_eta is None:
        n_gauss_eta = p_eta + 1
        
    # Points et poids de Gauss de référence sur [-1, 1]^2
    pts_canon, wts_canon = quadrature_gauss_2d(n_gauss_xi, n_gauss_eta)
    n_q_pts = len(wts_canon)
    
    n_loc = (p_xi + 1) * (p_eta + 1)
    
    # Listes pour assemblage creux (format triplets COO)
    lignes = []
    colonnes = []
    valeurs_K = []
    valeurs_M = []
    
    # Boucle sur tous les éléments actifs (spans de mesure non nulle)
    for elem in elements:
        i_span = elem['i_span']
        j_span = elem['j_span']
        xi_min, xi_max = elem['xi_lim']
        eta_min, eta_max = elem['eta_lim']
        
        # Mapping des points de quadrature sur le span courant
        pts_elem, wts_elem = mapper_quadrature_vers_element(
            xi_min, xi_max, eta_min, eta_max, pts_canon, wts_canon
        )
        
        # Matrices élémentaires
        K_elem = np.zeros((n_loc, n_loc), dtype=float)
        M_elem = np.zeros((n_loc, n_loc), dtype=float)
        
        # Intégration numérique de Gauss sur l'élément
        for q in range(n_q_pts):
            xi_q = pts_elem[q, 0]
            eta_q = pts_elem[q, 1]
            w_q = wts_elem[q]
            
            # Fonctions de base et dérivées paramétriques 2D
            R_loc, dR_dxi, dR_deta = fonctions_base_et_derivees_2d(
                i_span, j_span, xi_q, eta_q, p_xi, p_eta, U_xi, U_eta
            )
            
            # Gradients physiques et jacobienne
            det_J, dR_dx, dR_dy = evaluer_jacobienne_et_gradients_physiques(
                geo, i_span, j_span, dR_dxi, dR_deta
            )
            
            # Élément de surface physique différentiel dOmega = det(J) * w_q
            dOmega = det_J * w_q
            
            # Rigidité K : grad(R_a) . grad(R_b) = dR_dx[a]*dR_dx[b] + dR_dy[a]*dR_dy[b]
            K_elem += (np.outer(dR_dx, dR_dx) + np.outer(dR_dy, dR_dy)) * dOmega
            
            # Masse M : R_a * R_b
            M_elem += np.outer(R_loc, R_loc) * dOmega
            
        # Table de localisation des degrés de liberté locaux -> globaux
        indices_globaux = np.zeros(n_loc, dtype=int)
        loc_idx = 0
        for b in range(p_eta + 1):
            j_glob = j_span - p_eta + b
            for a in range(p_xi + 1):
                i_glob = i_span - p_xi + a
                indices_globaux[loc_idx] = j_glob * n_ctrl_xi + i_glob
                loc_idx += 1
                
        # Ajout aux listes COO
        for r in range(n_loc):
            row_g = indices_globaux[r]
            for c in range(n_loc):
                col_g = indices_globaux[c]
                lignes.append(row_g)
                colonnes.append(col_g)
                valeurs_K.append(K_elem[r, c])
                valeurs_M.append(M_elem[r, c])
                
    # Conversion en matrices CSR
    K_coo = coo_matrix((valeurs_K, (lignes, colonnes)), shape=(n_dofs, n_dofs), dtype=float)
    M_coo = coo_matrix((valeurs_M, (lignes, colonnes)), shape=(n_dofs, n_dofs), dtype=float)
    
    K = K_coo.tocsr()
    M = M_coo.tocsr()
    
    # Assurer la symétrie exacte (suppression d'éventuels bruits numériques machine)
    K = 0.5 * (K + K.transpose())
    M = 0.5 * (M + M.transpose())
    
    return K, M
