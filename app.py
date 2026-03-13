import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

from scipy.stats import norm, t, skew, kurtosis, genpareto, chi2
from arch import arch_model
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Progiciel VaR",
    page_icon="📉",
    layout="wide"
)

# =========================================================
# FONCTIONS UTILITAIRES
# =========================================================
@st.cache_data
def telecharger_donnees(tickers, date_debut, date_fin):
    data = yf.download(
        tickers,
        start=date_debut,
        end=date_fin,
        auto_adjust=True,
        progress=False
    )
    return data


def extraire_prix_cloture(data):
    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            prix = data["Close"].copy()
        else:
            prix = data.copy()
    else:
        if "Close" in data.columns:
            prix = data[["Close"]].copy()
            prix.columns = ["Actif"]
        else:
            prix = data.copy()

    if isinstance(prix, pd.Series):
        prix = prix.to_frame()

    prix = prix.dropna(axis=1, how="all")
    return prix


def calculer_rendements(prix):
    if prix.empty:
        return pd.DataFrame()
    return prix.pct_change(fill_method=None).dropna(how="all")


def calculer_rendement_portefeuille(rendements, poids_vecteur):
    rp = rendements.dot(poids_vecteur)
    return pd.Series(rp).dropna()


# =========================================================
# VAR CLASSIQUES
# =========================================================
def var_historique(rp, alpha):
    rp = pd.Series(rp).dropna()
    if len(rp) == 0:
        return np.nan
    return -np.quantile(rp, alpha)


def var_normale(rp, alpha):
    rp = pd.Series(rp).dropna()
    if len(rp) == 0:
        return np.nan
    mu = rp.mean()
    sigma = rp.std(ddof=1)
    z = norm.ppf(alpha)
    return -(mu + sigma * z)


def var_student(rp, alpha, df=8):
    rp = pd.Series(rp).dropna()
    if len(rp) == 0:
        return np.nan
    mu = rp.mean()
    sigma = rp.std(ddof=1)
    q = t.ppf(alpha, df)
    facteur = np.sqrt((df - 2) / df)
    return -(mu + sigma * facteur * q)


def var_cornish_fisher(rp, alpha):
    rp = pd.Series(rp).dropna()
    if len(rp) < 10:
        return np.nan

    mu = rp.mean()
    sigma = rp.std(ddof=1)
    s = skew(rp, bias=False)
    k = kurtosis(rp, fisher=True, bias=False)
    z = norm.ppf(alpha)

    z_cf = (
        z
        + (1 / 6) * (z**2 - 1) * s
        + (1 / 24) * (z**3 - 3 * z) * k
        - (1 / 36) * (2 * z**3 - 5 * z) * (s**2)
    )

    return -(mu + sigma * z_cf)


# =========================================================
# RISKMETRICS / EWMA
# =========================================================
def ewma_volatilite(rp, lam=0.94):
    rp = pd.Series(rp).dropna()
    if len(rp) < 2:
        return np.nan

    sigma2 = rp.var(ddof=1)
    for r in rp:
        sigma2 = lam * sigma2 + (1 - lam) * (r ** 2)

    return np.sqrt(sigma2)


def var_riskmetrics(rp, alpha, lam=0.94):
    rp = pd.Series(rp).dropna()
    if len(rp) < 2:
        return np.nan, np.nan

    sigma_ewma = ewma_volatilite(rp, lam=lam)
    z = norm.ppf(alpha)
    mu = 0.0
    return -(mu + sigma_ewma * z), sigma_ewma


# =========================================================
# GARCH
# =========================================================
def fit_garch_normal(rp):
    rp = pd.Series(rp).dropna()
    if len(rp) < 80:
        return None

    rp_pct = 100 * rp
    model = arch_model(rp_pct, vol="Garch", p=1, q=1, mean="Constant", dist="normal")
    res = model.fit(disp="off")
    return res


def var_garch_normale(rp, alpha):
    rp = pd.Series(rp).dropna()
    res = fit_garch_normal(rp)
    if res is None:
        return np.nan, np.nan

    forecast = res.forecast(horizon=1, reindex=False)

    mu_forecast = forecast.mean.iloc[-1, 0] / 100
    var_forecast = forecast.variance.iloc[-1, 0] / (100 ** 2)
    sigma_forecast = np.sqrt(var_forecast)

    z = norm.ppf(alpha)
    var_g = -(mu_forecast + sigma_forecast * z)

    return var_g, sigma_forecast


# =========================================================
# EVT (POT / GPD)
# =========================================================
def var_evt(rp, alpha, seuil_quantile=0.10):
    """
    On modélise la queue gauche des rendements.
    On travaille sur les pertes L = -rp.
    """
    rp = pd.Series(rp).dropna()
    if len(rp) < 100:
        return np.nan

    pertes = -rp
    u = np.quantile(pertes, 1 - seuil_quantile)  # seuil élevé sur les pertes
    exces = pertes[pertes > u] - u

    Nu = len(exces)
    n = len(pertes)

    if Nu < 20:
        return np.nan

    # Ajustement GPD des excès
    c, loc, scale = genpareto.fit(exces, floc=0)

    p = alpha  # probabilité de dépassement côté rendement, donc queue gauche
    # Formule POT
    if abs(c) < 1e-8:
        var = u + scale * np.log(Nu / (n * p))
    else:
        var = u + (scale / c) * (((Nu / (n * p)) ** c) - 1)

    return var


# =========================================================
# EVT-GARCH
# =========================================================
def var_evt_garch(rp, alpha, seuil_quantile=0.10):
    rp = pd.Series(rp).dropna()
    res = fit_garch_normal(rp)
    if res is None:
        return np.nan

    rp_pct = 100 * rp

    cond_vol_pct = pd.Series(res.conditional_volatility, index=rp.index)
    mu_hat_pct = res.params.get("mu", 0.0)

    # Résidus standardisés
    z = (rp_pct - mu_hat_pct) / cond_vol_pct
    z = z.dropna()

    if len(z) < 100:
        return np.nan

    pertes_z = -z
    u = np.quantile(pertes_z, 1 - seuil_quantile)
    exces = pertes_z[pertes_z > u] - u

    Nu = len(exces)
    n = len(pertes_z)

    if Nu < 20:
        return np.nan

    c, loc, scale = genpareto.fit(exces, floc=0)

    p = alpha
    if abs(c) < 1e-8:
        qz = u + scale * np.log(Nu / (n * p))
    else:
        qz = u + (scale / c) * (((Nu / (n * p)) ** c) - 1)

    # Prévision GARCH à 1 pas
    forecast = res.forecast(horizon=1, reindex=False)
    mu_forecast_pct = forecast.mean.iloc[-1, 0]
    sigma_forecast_pct = np.sqrt(forecast.variance.iloc[-1, 0])

    # VaR en pourcentage, puis retour en rendement décimal
    var_pct = -(mu_forecast_pct - sigma_forecast_pct * qz)
    return var_pct / 100


# =========================================================
# BACKTESTING
# =========================================================
def violations_var(rp, var_value):
    rp = pd.Series(rp).dropna()
    if np.isnan(var_value):
        return pd.Series(index=rp.index, dtype=float)
    return (rp < -var_value).astype(int)


def test_kupiec(rp, var_value, alpha):
    rp = pd.Series(rp).dropna()

    if len(rp) == 0 or np.isnan(var_value):
        return {
            "T": np.nan,
            "Violations": np.nan,
            "Taux observé": np.nan,
            "Taux théorique": alpha,
            "LR Kupiec": np.nan,
            "p-value": np.nan,
            "Conclusion": "Non calculable"
        }

    I = (rp < -var_value).astype(int)
    T = len(I)
    x = int(I.sum())
    p = alpha
    phat = x / T if T > 0 else np.nan

    # Cas limites numériques
    if phat == 0 or phat == 1:
        lr_uc = np.nan
        pvalue = np.nan
        conclusion = "Cas limite"
    else:
        num = ((1 - p) ** (T - x)) * (p ** x)
        den = ((1 - phat) ** (T - x)) * (phat ** x)
        lr_uc = -2 * np.log(num / den)
        pvalue = 1 - chi2.cdf(lr_uc, df=1)
        conclusion = "Acceptée" if pvalue > 0.05 else "Rejetée"

    return {
        "T": T,
        "Violations": x,
        "Taux observé": phat,
        "Taux théorique": p,
        "LR Kupiec": lr_uc,
        "p-value": pvalue,
        "Conclusion": conclusion
    }


def tracer_violations(rp, var_value, nom_methode):
    rp = pd.Series(rp).dropna()
    if len(rp) == 0 or np.isnan(var_value):
        return None

    viol = rp < -var_value

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rp.index, rp.values, label="Rendement portefeuille")
    ax.axhline(-var_value, linestyle="--", label=f"-VaR {nom_methode}")
    ax.scatter(rp.index[viol], rp[viol], marker="x", s=60, label="Violations")
    ax.set_title(f"Backtesting - {nom_methode}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


# =========================================================
# SESSION STATE
# =========================================================
if "prix" not in st.session_state:
    st.session_state["prix"] = None

if "rendements" not in st.session_state:
    st.session_state["rendements"] = None

if "portefeuille" not in st.session_state:
    st.session_state["portefeuille"] = None

if "vars_calculees" not in st.session_state:
    st.session_state["vars_calculees"] = None


# =========================================================
# MENU
# =========================================================
menu = st.sidebar.selectbox(
    "Navigation",
    ["Accueil", "Portefeuille", "VaR", "Backtesting", "Reporting"]
)

# =========================================================
# PAGE ACCUEIL
# =========================================================
if menu == "Accueil":
    st.title("Progiciel de Calcul, Comparaison et Backtesting de la VaR")

    st.markdown("""
    Cet outil permet de construire un portefeuille d’actifs financiers,
    d’estimer la Value at Risk selon plusieurs méthodes quantitatives,
    d’effectuer le backtesting des modèles de risque et de générer
    automatiquement des reportings Excel et PDF.
    """)

    st.subheader("Méthodes disponibles")
    st.markdown("""
    - VaR historique  
    - VaR paramétrique normale  
    - VaR Student  
    - VaR Cornish-Fisher  
    - VaR RiskMetrics  
    - VaR GARCH  
    - VaR EVT  
    - VaR EVT-GARCH  
    """)

    st.subheader("Équipe projet")
    st.markdown("""
    - Kopangoye Guénolé Wariol  
    - Adjagba Harlem Désir  
    - Ecclésiaste Gnargo  
    - Anta Mbaye  
    """)

# =========================================================
# PAGE PORTEFEUILLE
# =========================================================
elif menu == "Portefeuille":
    st.title("Création du portefeuille")

    actifs_disponibles = {
        "Apple": "AAPL",
        "Microsoft": "MSFT",
        "Airbus": "AIR.PA",
        "TotalEnergies": "TTE.PA",
        "LVMH": "MC.PA",
        "BNP Paribas": "BNP.PA",
        "Safran": "SAF.PA",
        "Tesla": "TSLA",
        "Amazon": "AMZN",
        "Nvidia": "NVDA"
    }

    actifs_choisis = st.multiselect(
        "Choisir les actifs",
        list(actifs_disponibles.keys()),
        default=["Apple", "Microsoft", "Airbus"]
    )

    col1, col2 = st.columns(2)
    with col1:
        date_debut = st.date_input("Date de début", value=pd.to_datetime("2023-01-01"))
    with col2:
        date_fin = st.date_input("Date de fin", value=pd.to_datetime("today"))

    if st.button("Télécharger les données et construire le portefeuille"):
        if not actifs_choisis:
            st.warning("Veuillez sélectionner au moins un actif.")
        elif date_debut >= date_fin:
            st.warning("La date de début doit être antérieure à la date de fin.")
        else:
            tickers = [actifs_disponibles[a] for a in actifs_choisis]
            data = telecharger_donnees(tickers, date_debut, date_fin)
            prix = extraire_prix_cloture(data)
            rendements = calculer_rendements(prix)

            st.session_state["prix"] = prix
            st.session_state["rendements"] = rendements
            st.session_state["portefeuille"] = None
            st.session_state["vars_calculees"] = None

    if st.session_state["prix"] is not None and st.session_state["rendements"] is not None:
        prix = st.session_state["prix"]
        rendements = st.session_state["rendements"]

        st.subheader("Prix de clôture")
        st.dataframe(prix.tail(), use_container_width=True)

        st.subheader("Rendements journaliers")
        st.dataframe(rendements.tail(), use_container_width=True)

        st.subheader("Statistiques descriptives")
        stats = pd.DataFrame({
            "Rendement moyen": rendements.mean(),
            "Volatilité": rendements.std(),
            "Minimum": rendements.min(),
            "Maximum": rendements.max()
        })
        stats.index.name = "Ticker"
        st.dataframe(stats, use_container_width=True)

        st.subheader("Saisie des poids du portefeuille")
        st.markdown("Les poids doivent être compris entre 0 et 1 et leur somme doit être égale à 1.")

        poids = {}
        cols = st.columns(len(rendements.columns))

        for i, actif in enumerate(rendements.columns):
            with cols[i]:
                poids[actif] = st.number_input(
                    f"Poids {actif}",
                    min_value=0.0,
                    max_value=1.0,
                    value=round(1 / len(rendements.columns), 2),
                    step=0.01,
                    key=f"poids_{actif}"
                )

        poids_vecteur = np.array([poids[a] for a in rendements.columns])
        somme_poids = poids_vecteur.sum()

        st.write(f"Somme des poids = {somme_poids:.4f}")

        if abs(somme_poids - 1) > 1e-6:
            st.warning("La somme des poids doit être exactement égale à 1.")
        else:
            rp = calculer_rendement_portefeuille(rendements, poids_vecteur)
            st.session_state["portefeuille"] = rp
            st.session_state["vars_calculees"] = None

            st.subheader("Rendement journalier du portefeuille")
            st.dataframe(rp.to_frame(name="Rendement Portefeuille").tail(), use_container_width=True)

            st.subheader("Évolution des prix")
            st.line_chart(prix)

            st.subheader("Évolution du rendement du portefeuille")
            st.line_chart(rp)

# =========================================================
# PAGE VAR
# =========================================================
elif menu == "VaR":
    st.title("Calcul des différentes VaR")

    rp = st.session_state["portefeuille"]

    if rp is None:
        st.warning("Veuillez d'abord construire le portefeuille dans l'onglet Portefeuille.")
    else:
        niveau = st.selectbox("Niveau de confiance", [95, 99])
        alpha = 1 - niveau / 100

        st.subheader("Rendements du portefeuille")
        st.dataframe(rp.to_frame(name="Rendement Portefeuille").tail(), use_container_width=True)

        mu = rp.mean()
        sigma = rp.std(ddof=1)
        s = skew(rp.dropna(), bias=False)
        k = kurtosis(rp.dropna(), fisher=True, bias=False)

        st.subheader("Paramètres statistiques du portefeuille")
        stats_portefeuille = pd.DataFrame({
            "Moyenne": [mu],
            "Volatilité": [sigma],
            "Skewness": [s],
            "Excès de kurtosis": [k]
        })
        st.dataframe(stats_portefeuille, use_container_width=True)

        var_hist = var_historique(rp, alpha)
        var_norm = var_normale(rp, alpha)
        var_stud = var_student(rp, alpha, df=8)
        var_cf = var_cornish_fisher(rp, alpha)
        var_rm, sigma_ewma = var_riskmetrics(rp, alpha, lam=0.94)
        var_garch, sigma_garch = var_garch_normale(rp, alpha)
        var_evt_val = var_evt(rp, alpha, seuil_quantile=0.10)
        var_evt_garch_val = var_evt_garch(rp, alpha, seuil_quantile=0.10)

        resultats_var = pd.DataFrame({
            "Méthode": [
                "Historique",
                "Normale",
                "Student (ddl=8)",
                "Cornish-Fisher",
                "RiskMetrics (EWMA)",
                "GARCH(1,1) normale",
                "EVT",
                "EVT-GARCH"
            ],
            f"VaR {niveau}%": [
                var_hist,
                var_norm,
                var_stud,
                var_cf,
                var_rm,
                var_garch,
                var_evt_val,
                var_evt_garch_val
            ]
        })

        st.subheader("Résultats des VaR")
        st.dataframe(resultats_var, use_container_width=True)

        st.subheader("Volatilités conditionnelles")
        volatilites = pd.DataFrame({
            "Mesure": ["Volatilité historique", "Volatilité EWMA", "Volatilité GARCH"],
            "Valeur": [sigma, sigma_ewma, sigma_garch]
        })
        st.dataframe(volatilites, use_container_width=True)

        st.session_state["vars_calculees"] = {
            "Historique": var_hist,
            "Normale": var_norm,
            "Student (ddl=8)": var_stud,
            "Cornish-Fisher": var_cf,
            "RiskMetrics (EWMA)": var_rm,
            "GARCH(1,1) normale": var_garch,
            "EVT": var_evt_val,
            "EVT-GARCH": var_evt_garch_val,
            "alpha": alpha
        }

# =========================================================
# PAGE BACKTESTING
# =========================================================
elif menu == "Backtesting":
    st.title("Backtesting des modèles de VaR")

    rp = st.session_state["portefeuille"]
    vars_calculees = st.session_state["vars_calculees"]

    if rp is None or vars_calculees is None:
        st.warning("Veuillez d'abord construire le portefeuille puis calculer les VaR.")
    else:
        alpha = vars_calculees["alpha"]

        methodes = [
            "Historique",
            "Normale",
            "Student (ddl=8)",
            "Cornish-Fisher",
            "RiskMetrics (EWMA)",
            "GARCH(1,1) normale",
            "EVT",
            "EVT-GARCH"
        ]

        resultats_backtesting = []
        for m in methodes:
            res = test_kupiec(rp, vars_calculees[m], alpha)
            res["Méthode"] = m
            resultats_backtesting.append(res)

        df_bt = pd.DataFrame(resultats_backtesting)[[
            "Méthode", "T", "Violations", "Taux observé", "Taux théorique",
            "LR Kupiec", "p-value", "Conclusion"
        ]]

        st.subheader("Test de Kupiec")
        st.dataframe(df_bt, use_container_width=True)

        methode_plot = st.selectbox("Choisir une méthode à visualiser", methodes)
        fig = tracer_violations(rp, vars_calculees[methode_plot], methode_plot)

        if fig is not None:
            st.subheader(f"Graphique des violations - {methode_plot}")
            st.pyplot(fig)
        else:
            st.info("Graphique non disponible pour cette méthode.")

# =========================================================
# PAGE REPORTING
# =========================================================
elif menu == "Reporting":
    st.title("Reporting")
    st.write("Module à venir.")

