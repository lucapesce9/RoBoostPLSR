# roboost_plsr.py
import numpy as np
import pandas as pd

# Import the helper functions from utils.py
from .utils import (F_weight, to_matrix, xmean, center, pls_nipalsw)

class RoBoostPLSR:
    def __init__(self, ncomp, niter=50, gamma=np.inf, beta=np.inf, alpha=np.inf, th=1 - 1e-12):
        self.ncomp = ncomp
        self.niter = niter
        self.gamma = gamma
        self.beta = beta
        self.alpha = alpha
        self.th = th
        self.fm = {}
        self.T = None
        self.W = None
        self.P = None
        self.C = None
        self.R = None
        self.B = None  # Coefficienti di regressione
        self.crit = {}
        self.hyp_para = {'alpha': alpha, 'beta': beta, 'gamma': gamma}
        self.xmeans = None
        self.ymeans = None
        self.xmeans_init = None  # Medie iniziali di X
        self.ymeans_init = None  # Media iniziale di Y
        self.fit_successful = False  # Flag per indicare se il fit è stato completato con successo
    def fit(self, X, Y):
        X = to_matrix(X, row=False, prefix_colname="x")
        Y = to_matrix(Y, row=False, prefix_colname="y")

        if Y.shape[1] != 1:
            raise ValueError("Il modello supporta solo una singola variabile di risposta.")

        n, p = X.shape

        nam = [f"comp{i+1}" for i in range(self.ncomp)]
        self.T = pd.DataFrame(np.zeros((n, self.ncomp)), columns=nam, index=X.index)
        self.W = pd.DataFrame(np.zeros((p, self.ncomp)), columns=nam, index=X.columns)
        self.P = pd.DataFrame(np.zeros((p, self.ncomp)), columns=nam, index=X.columns)
        self.C = pd.Series(np.zeros(self.ncomp), index=nam)

        self.crit = {
            'r': pd.Series(np.zeros(self.ncomp), index=nam),
            'r1': pd.Series(np.zeros(self.ncomp), index=nam),
            'l': pd.Series(np.zeros(self.ncomp), index=nam)
        }

        a = 1
        fm = {}
        Y1 = Y.copy()
        X1 = X.copy()

        while a <= self.ncomp:
            cor = 0
            f = 1
            if a == 1:
                d = np.ones(n)
            else:
                # Dopo la prima iterazione, X e Y sono numpy arrays
                # Converti X e Y in DataFrame per mantenere gli indici
                X = pd.DataFrame(X, index=X1.index, columns=X1.columns)
                Y = pd.DataFrame(Y, index=Y1.index, columns=Y1.columns)
            flag = True
            while cor < self.th:
                if a == 1:
                    self.xmeans = xmean(X1, weights=d)
                    if flag:
                      self.xmeans_init = self.xmeans.copy()  # Memorizza la media iniziale di X
                    X_centered = center(X1, center=self.xmeans)
                    self.ymeans = xmean(Y1, weights=d)[0]
                    if flag:  # Assicuriamoci che sia uno scalare
                      self.ymeans_init = self.ymeans  # Memorizza la media iniziale di Y
                    #print("Media iniziale di X:", self.xmeans_init)
                    #print("Media iniziale di Y:", self.ymeans_init)
                    Y_centered = center(Y1, center=self.ymeans)
                    flag = False
                else:
                    self.xmeans = np.zeros(X.shape[1])
                    self.ymeans = 0.0
                    X_centered = center(X, center=self.xmeans)
                    Y_centered = center(Y, center=self.ymeans)

                # Esegui PLS NIPALS con i dati centrati e i pesi correnti
                result = pls_nipalsw(X_centered.values, Y_centered.values, ncomp=1, weights=d)
                if result is None:
                    print(f"Skipping component {a} due to zero norm in w.")
                    return None  # Segnala che il fit non è riuscito
                fm[a] = result
                fm[a]['xmeans'] = self.xmeans
                fm[a]['ymeans'] = self.ymeans

                # Calcola ry utilizzando i residui Y
                ry = result['Y']
                ry_w = F_weight(ry, cw=self.beta)
                r = ry_w / ry_w.sum()

                # Calcola r1 utilizzando i residui X
                r1 = np.sqrt(np.sum(result['X'] ** 2, axis=1))
                rx = r1.copy()
                r1_w = F_weight(r1, cw=self.alpha)
                if r1_w.sum() == 0:
                    print("La somma dei pesi r1 è zero.")
                    return None
                    #raise ValueError("La somma dei pesi r1 è zero.")
                r1 = r1_w / r1_w.sum()

                # Calcola l utilizzando i punteggi T
                l = result['T'][:, 0]
                rl = l.copy()
                l_w = F_weight(l, cw=self.gamma)
                if l_w.sum() == 0:
                    print("La somma dei pesi l è zero.")
                    return None
                    #raise ValueError("La somma dei pesi l è zero.")
                l = l_w / l_w.sum()

                # Calcola i nuovi pesi d
                d = r * r1 * l
                if d.sum() == 0:
                    print("La somma dei nuovi pesi d è zero.")
                    return None
                    #raise ValueError("La somma dei nuovi pesi d è zero.")
                d = d / d.sum()

                # Calcola u
                u = result['C'][0]

                # Calcola q_val
                numerator = np.sum(d * result['T'][:, 0] * u)
                denominator = np.sum(result['T'][:, 0] * u * result['T'][:, 0])
                if denominator == 0:
                    print("Il denominatore nel calcolo di q_val è zero.")
                    return None
                    #raise ValueError("Il denominatore nel calcolo di q_val è zero.")
                q_val = numerator / denominator

                if f > 1:
                    cor = min(q_val, q1) / max(q_val, q1)
                if f > self.niter:
                    cor = np.inf
                f += 1
                q1 = q_val

            # Calcola i pesi finali
            beta_w = F_weight(result['Y'], cw=self.beta)
            alpha_w = F_weight(rx, cw=self.alpha)
            gamma_w = F_weight(rl, cw=self.gamma)
            fm[a]['list_w'] = [alpha_w, beta_w, gamma_w]

            # Aggiorna X e Y per la prossima iterazione
            X = result['X']
            Y = result['Y']

            # Memorizza i risultati
            self.T.iloc[:, a - 1] = result['T'][:, 0]
            self.W.iloc[:, a - 1] = result['W'][:, 0]
            self.P.iloc[:, a - 1] = result['P'][:, 0]
            self.C.iloc[a - 1] = result['C'][0]

            # Memorizza i criteri
            self.crit['r'].iloc[a - 1] = np.median(np.abs(ry))
            self.crit['r1'].iloc[a - 1] = np.median(np.abs(rx))
            self.crit['l'].iloc[a - 1] = np.median(np.abs(rl))

            a += 1

        # Calcola la matrice R
        try:
            self.R = self.W.values.dot(np.linalg.pinv(self.P.values.T.dot(self.W.values)))
        except np.linalg.LinAlgError as e:
            print(f"Errore nella calcolo della matrice R: {e}")
            return None
            #raise ValueError(f"Errore nella computazione della matrice R: {e}")

        # Calcola i coefficienti di regressione B
        self.B = self.R.dot(self.C.values)  # Shape: (p,)

        # Memorizza il modello
        self.fm = fm
        self.fit_successful = True
        return self

    def predict(self, Xu):
        """
        Effettua previsioni utilizzando il modello RoBoost-PLSR addestrato.

        Parametri:
        ----------
        Xu : pandas.DataFrame
            Dati di input per la previsione.

        Returns:
        --------
        numpy.ndarray
            Previsioni.
        """
        if not isinstance(self.fm, dict):
            raise TypeError("Il modello non è stato addestrato ancora.")

        # Centra i dati di input utilizzando la media iniziale del modello
        Xu_centered = center(to_matrix(Xu), self.xmeans_init).values  # Shape: (m, p)

        # Calcola le previsioni utilizzando i coefficienti di regressione
        Y_pred = Xu_centered.dot(self.B) + self.ymeans_init  # Shape: (m,)

        return Y_pred
