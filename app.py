import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm, t, skew, kurtosis
from arch import arch_model

st.set_page_config(
    page_title="Progiciel VaR",
    page_icon="📉",
    layout="wide"
)

# =========================
# FONCTIONS UTILES
# =========================
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
    return rendements.dot(poids_vecteur)


# =========================
# VAR CLASSIQUES
# =========================
def var_historique(rp, alpha):
    return -np.quantile(rp, alpha)


def var_normale(rp, alpha):
    mu = rp.mean()
    sigma = rp.std(ddof=1)
    z = norm.ppf(alpha)
    return -(mu + sigma * z)


def var_student(rp, alpha, df=8):
    mu = rp.mean()
    sigma = rp.std(ddof=1)
    q = t.ppf(alpha, df)
    facteur = np.sqrt((df - 2) / df)
    return -(mu + sigma * facteur * q)


def var_cornish_fisher(rp, alpha):
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


# =========================
# VAR RISKMETRICS / EWMA
# =========================
def ewma_volatilite(rp, lam=0.94):
    rp = pd.Series(rp).dropna()
    if len(rp) < 2:
        return np.nan

    sigma2 = rp.var(ddof=1)

    for r in rp:
        sigma2 = lam * sigma2 + (1 - lam) * (r ** 2)

    return np.sqrt(sigma2)


def var_riskmetrics(rp, alpha, lam=0.94):
    sigma_ewma = ewma_volatilite(rp, lam=lam)
    z = norm.ppf(alpha)
    mu = 0.0
    return -(mu + sigma_ewma * z), sigma_ewma


# =========================
# VAR GARCH
# =========================
def var_garch_normale(rp, alpha):
    rp = pd.Series(rp).dropna()

    if len(rp) < 50:
        return np.nan, np.nan

    # On travaille souvent en pourcentage pour la stabilité numérique
    rp_pct = 100 * rp

    model = arch_model(rp_pct, vol="Garch", p=1, q=1, mean="Constant", dist="normal")
    res = model.fit(disp="off")

    forecast = res.forecast(horizon=1, reindex=False)

    mu_forecast = forecast.mean.iloc[-1, 0] / 100
    var_forecast = forecast.variance.iloc[-1, 0] / (100 ** 2)
    sigma_forecast = np.sqrt(var_forecast)

    z = norm.ppf(alpha)
    var_g = -(mu_forecast + sigma_forecast * z)

    return var_g, sigma_forecast


# =========================
# PLACEHOLDERS EVT
# =========================
def var_evt_placeholder():
    return np.nan


def var_evt_garch_placeholder():
    return np.nan


# =========================
# ETAT SESSION
# =========================
if "prix" not in st.session_state:
    st.session_state["prix"] = None

if "rendements" not in st.session_state:
    st.session_state["rendements"] = None

if "portefeuille" not in st.session_state:
    st.session_state["portefeuille"] = None


# =========================
# MENU
# =========================
menu = st.sidebar.selectbox(
    "Navigation",
    ["Accueil", "Portefeuille", "VaR", "Backtesting", "Reporting"]
)

# =========================
# PAGE ACCUEIL
# =========================
if menu == "Accueil":
    st.title("Progiciel de Calcul, Comparaison et Backtesting de la VaR")

    st.markdown("""
    Cet outil permet de construire un portefeuille d’actifs financiers,
    d’estimer la Value at Risk selon plusieurs méthodes quantitatives,
    d’effectuer le backtesting des modèles de risque et de générer
    automatiquement des reportings Excel et PDF.
    """)

    st.subheader("Objectifs du progiciel")
    st.markdown("""
    - Sélectionner des actifs financiers  
    - Construire un portefeuille  
    - Estimer la VaR selon plusieurs méthodes  
    - Comparer les résultats  
    - Réaliser le backtesting  
    - Générer un reporting Excel  
    - Produire un rapport PDF  
    """)

    st.subheader("Méthodes disponibles")
    st.markdown("""
    - VaR historique  
    - VaR paramétrique normale  
    - VaR Student  
    - VaR Cornish-Fisher  
    - VaR RiskMetrics  
    - VaR GARCH  
    - VaR TVE  
    - VaR TVE-GARCH  
    """)

    st.subheader("Équipe projet")
    st.markdown("""
    - Kopangoye Guénolé Wariol  
    - Adjagba Harlem Désir  
    - Ecclésiaste Gnargo  
    - Anta Mbaye  
    """)

# =========================
# PAGE PORTEFEUILLE
# =========================
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
        colonnes = st.columns(len(rendements.columns))

        for i, actif in enumerate(rendements.columns):
            with colonnes[i]:
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

            st.subheader("Rendement journalier du portefeuille")
            st.dataframe(rp.to_frame(name="Rendement Portefeuille").tail(), use_container_width=True)

            st.subheader("Évolution des prix")
            st.line_chart(prix)

            st.subheader("Évolution du rendement du portefeuille")
            st.line_chart(rp)

# =========================
# PAGE VAR
# =========================
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
        s = skew(rp, bias=False)
        k = kurtosis(rp, fisher=True, bias=False)

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
                var_evt_placeholder(),
                var_evt_garch_placeholder()
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

        st.info(
            "EVT et EVT-GARCH sont laissées en placeholder pour l’instant. "
            "On peut les ajouter ensuite proprement."
        )

# =========================
# PAGE BACKTESTING
# =========================
elif menu == "Backtesting":
    st.title("Backtesting des modèles de VaR")
    st.write("Module à venir.")

# =========================
# PAGE REPORTING
# =========================
elif menu == "Reporting":
    st.title("Reporting")
    st.write("Module à venir.")

