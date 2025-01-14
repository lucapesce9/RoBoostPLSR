import numpy as np
import pandas as pd

def F_weight(x, cw):
    """
    Calcola i pesi per gli elementi in x usando la funzione di ponderazione Tukey biquadrata.

    Parametri:
    ----------
    x : array-like
        Residui o valori per cui calcolare i pesi.
    cw : float
        Costante di ponderazione.

    Returns:
    --------
    numpy.ndarray
        Pesi calcolati per ogni elemento in x.
    """
    x = np.asarray(x).flatten()
    if cw == np.inf:
        return np.ones_like(x)

    s = np.median(np.abs(x - np.median(x)))  # MAD (Median Absolute Deviation)
    if s == 0:
        s = 1e-8  # Evita divisione per zero se la deviazione è zero
    x = x / (cw * s)

    w = np.zeros_like(x)
    mask = np.abs(x) <= 1
    w[mask] = (1 - x[mask] ** 2) ** 2
    return w

def to_matrix(X, row=False, prefix_colname="x"):
    """
    Converte l'input in una matrice pandas DataFrame.

    Parametri:
    ----------
    X : array-like
        Input da convertire.
    row : bool, opzionale
        Se True, X viene considerato come un singolo campione (riga).
    prefix_colname : str, opzionale
        Prefisso per i nomi delle colonne.

    Returns:
    --------
    pandas.DataFrame
        Matrice convertita.
    """
    if isinstance(X, pd.DataFrame):
        return X.copy()
    elif isinstance(X, pd.Series):
        X = X.to_frame().T if row else X.to_frame()
    else:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1) if row else X.reshape(-1, 1)
        X = pd.DataFrame(X)

    # Assegna nomi alle colonne se non presenti o sono numerici
    if X.columns.dtype == 'int64' or X.columns.dtype == 'float64':
        X.columns = [f"{prefix_colname}{i+1}" for i in range(X.shape[1])]
    return X

def xmean(X, weights=None):
    """
    Calcola la media ponderata delle colonne di X.

    Parametri:
    ----------
    X : pandas.DataFrame
        Dati di input.
    weights : array-like, opzionale
        Pesi per calcolare la media ponderata.

    Returns:
    --------
    numpy.ndarray
        Media ponderata delle colonne di X.
    """
    X = X.values  # Shape: (n_samples, n_features)
    n = X.shape[0]

    if weights is None:
        weights = np.ones(n) / n
    else:
        weights = np.asarray(weights).flatten()
        weights_sum = np.sum(weights)
        if weights_sum == 0:
            raise ValueError("La somma dei pesi non può essere zero.")
        weights = weights / weights_sum  # Normalizza i pesi

    mean = weights @ X
    return mean

def center(X, center):
    """
    Centra le colonne di X sottraendo la media specificata.

    Parametri:
    ----------
    X : pandas.DataFrame
        Dati di input.
    center : array-like
        Valori da sottrarre ad ogni colonna di X.

    Returns:
    --------
    pandas.DataFrame
        Dati centrati.
    """
    return X - center

def pls_nipalsw(X, Y, ncomp, weights=None):
    """
    Implementa l'algoritmo NIPALS PLS con pesi per una singola variabile di risposta.

    Parametri:
    ----------
    X : pandas.DataFrame o numpy.ndarray
        Variabili esplicative.
    Y : pandas.DataFrame o numpy.ndarray
        Variabile di risposta.
    ncomp : int
        Numero di componenti latenti.
    weights : array-like, opzionale
        Pesi per le osservazioni.

    Returns:
    --------
    dict
        Contiene T, P, W, C, X, Y, weights.
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(Y, pd.DataFrame):
        Y = Y.values.flatten()
    else:
        Y = np.asarray(Y).flatten()

    n, p = X.shape

    if weights is None:
        weights = np.ones(n)
    else:
        weights = np.asarray(weights).flatten()
        weights_sum = np.sum(weights)
        if weights_sum == 0:
            print("La somma dei pesi non può essere zero.")
            return None
            #raise ValueError("La somma dei pesi non può essere zero.")
        weights = weights / weights_sum  # Normalizza i pesi

    T = np.zeros((n, ncomp))
    P = np.zeros((p, ncomp))
    W = np.zeros((p, ncomp))
    C = np.zeros(ncomp)  # Poiché Y ha una sola variabile

    for a in range(ncomp):
        # Calcolo di w
        w = X.T.dot(weights * Y)  # Shape: (p,)
        w_norm = np.linalg.norm(w)
        if w_norm == 0:
            print("Il vettore w ha norma zero.")
            return None
            #raise ValueError("Il vettore w ha norma zero.")
        w /= w_norm  # Normalizza w

        # Calcolo di t
        t = X.dot(w)  # Shape: (n,)
        t_w = weights * t  # Shape: (n,)
        t_t_w = np.dot(t, t_w)
        if t_t_w == 0:
            print("Il denominatore t.T * w è zero.")
            return None
            #raise ValueError("Il denominatore t.T * w è zero.")

        # Calcolo di c
        c = (weights * Y).dot(t) / t_t_w  # Scalar

        # Calcolo di p
        p_vec = X.T.dot(t_w) / t_t_w  # Shape: (p,)

        # Aggiornamento di X e Y
        X = X - np.outer(t, p_vec)  # Shape: (n, p)
        Y = Y - c * t               # Shape: (n,)

        # Memorizzazione dei risultati
        T[:, a] = t
        P[:, a] = p_vec
        W[:, a] = w
        C[a] = c

    result = {
        'T': T,
        'P': P,
        'W': W,
        'C': C,
        'X': X,  # Aggiornato X (numpy array)
        'Y': Y,  # Aggiornato Y (numpy array)
        'weights': weights  # Array numpy
    }
    return result
