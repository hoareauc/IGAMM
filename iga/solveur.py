"""
Module solveur.py
Résolution du problème aux valeurs propres généralisé pour l'acoustique 2D en IGA.
Formulation :
    K * p = lambda * M * p
    lambda = (omega / c)^2 = k^2
    f = (c * sqrt(lambda)) / (2 * pi)
Contient également la solution analytique exacte pour la cavité rectangulaire 2D à parois rigides.
Approche 100% procédurale (fonctions pures, sans POO).
"""

import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as ssla


def resoudre_modes_acoustiques_2d(K, M, num_modes=10, c=343.0, shift=0.0):
    """
    Résout le problème aux valeurs propres généralisé K * p = lambda * M * p.
    
    Paramètres :
        K (csr_matrix ou ndarray) : Matrice de rigidité globale
        M (csr_matrix ou ndarray) : Matrice de masse globale
        num_modes (int) : Nombre de modes propres souhaités (valeurs et vecteurs)
        c (float) : Célérité du son dans le fluide (m/s, défaut 343.0 pour l'air à 20°C)
        shift (float) : Décalage spectral pour la méthode shift-invert si solveur creux utilisé
        
    Retourne :
        frequences (ndarray de taille num_modes) : Fréquences propres en Hertz, triées par ordre croissant
        modes (ndarray de forme (n_dofs, num_modes)) : Vecteurs propres normalisés par rapport à la masse (p^T M p = 1)
        valeurs_propres (ndarray de taille num_modes) : Valeurs propres lambda = k^2
    """
    n_dofs = K.shape[0]
    num_modes = min(num_modes, n_dofs)
    
    # Choix de la méthode selon la taille du système
    # Pour n_dofs <= 1500, le solveur dense eigh est très rapide, exact et ne manque aucun mode nul ou proche
    if n_dofs <= 1500:
        K_dense = K.toarray() if hasattr(K, 'toarray') else np.asarray(K)
        M_dense = M.toarray() if hasattr(M, 'toarray') else np.asarray(M)
        
        # scipy.linalg.eigh résout K v = w M v pour matrices symétriques
        w, v = sla.eigh(K_dense, M_dense)
    else:
        # Pour les grands systèmes, méthode de Krylov / Lanczos avec shift-invert
        # Un léger shift négatif (ex: -1e-4) évite la singularité due au mode acoustique rigide à fréquence 0
        sigma = -1e-4 if shift == 0.0 else shift
        w, v = ssla.eigsh(K, k=num_modes, M=M, sigma=sigma, which='LM')
        
    # Filtrer d'éventuels bruits numériques négatifs autour de 0
    w = np.maximum(w.real, 0.0)
    
    # Tri par valeurs propres croissantes
    idx_tri = np.argsort(w)[:num_modes]
    valeurs_propres = w[idx_tri]
    modes = v[:, idx_tri].real
    
    # Fréquences propres acoustiques f = c * sqrt(lambda) / (2 * pi)
    frequences = (c * np.sqrt(valeurs_propres)) / (2.0 * np.pi)
    
    # Normalisation par rapport à la matrice de masse : p_i^T M p_i = 1
    for i in range(num_modes):
        phi_i = modes[:, i]
        m_norm = np.dot(phi_i, M.dot(phi_i))
        if m_norm > 1e-15:
            modes[:, i] = phi_i / np.sqrt(m_norm)
            
    return frequences, modes, valeurs_propres


def frequences_analytiques_cavite_rectangulaire(Lx, Ly, c=343.0, max_m=8, max_n=8):
    """
    Calcule les fréquences propres analytiques exactes d'une cavité acoustique rectangulaire 2D
    à parois rigides (conditions de Neumann homogènes) :
        p_{m, n}(x, y) = cos(m * pi * x / Lx) * cos(n * pi * y / Ly)
        k_{m, n}^2 = (m * pi / Lx)^2 + (n * pi / Ly)^2
        f_{m, n} = (c / 2) * sqrt( (m / Lx)^2 + (n / Ly)^2 )
        
    Paramètres :
        Lx (float) : Longueur selon x (m)
        Ly (float) : Hauteur selon y (m)
        c (float) : Célérité du son (m/s)
        max_m (int) : Ordre modal maximal selon x
        max_n (int) : Ordre modal maximal selon y
        
    Retourne :
        frequences_triees (ndarray) : Fréquences analytiques triées par ordre croissant
        ordres_modaux (list of tuple) : Paires (m, n) correspondantes
    """
    modes_analytiques = []
    
    for m in range(max_m + 1):
        for n in range(max_n + 1):
            f_mn = 0.5 * c * np.sqrt((m / Lx)**2 + (n / Ly)**2)
            modes_analytiques.append((f_mn, (m, n)))
            
    # Tri selon la fréquence
    modes_analytiques.sort(key=lambda item: item[0])
    
    frequences_triees = np.array([item[0] for item in modes_analytiques])
    ordres_modaux = [item[1] for item in modes_analytiques]
    
    return frequences_triees, ordres_modaux


def frequences_analytiques_cavite_circulaire(R=1.0, c=343.0, max_m=10, max_n=6):
    """
    Calcule les fréquences propres analytiques exactes d'une cavité acoustique circulaire 2D (rayon R)
    à parois rigides (conditions de Neumann homogènes dp/dr = 0 à r = R).
    
    Les solutions de l'équation de Helmholtz en coordonnées polaires sont :
        p_{m, n}(r, theta) = J_m(k_{m, n} * r) * [cos(m * theta) ou sin(m * theta)]
        k_{m, n} = alpha'_{m, n} / R
        f_{m, n} = (c * alpha'_{m, n}) / (2 * pi * R)
        où alpha'_{m, n} est le n-ième zéro positif de la dérivée de la fonction de Bessel J'_m(x) = 0.
        
    Paramètres :
        R (float) : Rayon de la cavité circulaire (m)
        c (float) : Célérité acoustique (m/s)
        max_m (int) : Ordre azimutal maximal m
        max_n (int) : Ordre radial maximal n
        
    Retourne :
        frequences_triees (ndarray) : Fréquences propres analytiques (Hz) triées par ordre croissant
        ordres_modaux (list of tuple) : Paires (m, n) correspondantes (avec multiplicité 2 pour m > 0)
        zeros_bessel (list of float) : Valeurs des zéros alpha'_{m, n}
    """
    import scipy.special as sp
    
    modes_circulaires = []
    # Mode constant (m=0, n=0) : fréquence nulle
    modes_circulaires.append((0.0, (0, 0), 0.0))
    
    for m in range(max_m + 1):
        # sp.jnp_zeros(m, nt) retourne les nt premiers zéros positifs de J'_m(x)
        zeros = sp.jnp_zeros(m, max_n)
        for n_idx, z in enumerate(zeros):
            f_mn = (c * z) / (2.0 * np.pi * R)
            n_mode = n_idx + 1
            # Pour m = 0, le mode est simple (axisymétrique).
            # Pour m >= 1, le mode a une multiplicité 2 (solutions en cos(m*theta) et sin(m*theta)).
            mult = 1 if m == 0 else 2
            for _ in range(mult):
                modes_circulaires.append((f_mn, (m, n_mode), float(z)))
                
    modes_circulaires.sort(key=lambda item: item[0])
    
    frequences_triees = np.array([item[0] for item in modes_circulaires])
    ordres_modaux = [item[1] for item in modes_circulaires]
    zeros_bessel = [item[2] for item in modes_circulaires]
    
    return frequences_triees, ordres_modaux, zeros_bessel

