import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(
    page_title="Progiciel VaR",
    page_icon="📉",
    layout="wide"
)

# -----------------------------
# Fonctions utiles
# -----------------------------
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

    # Cas multi-index classique avec Yahoo Finance
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            prix = data["Close"].copy()
        else:
            # fallback si "Close" n'est pas trouvé
            prix = data.xs(data.columns.get_level_values(0)[0], axis=1, level=0).copy()
    else:
        # Cas simple 1 actif
        if "Close" in data.columns:
            prix = data[["Close"]].copy()
            prix.columns = ["Actif"]
        else:
            prix = data.copy()

    if isinstance(prix, pd.Series):
        prix = prix.to_frame()

    # Supprimer les colonnes entièrement vides
    prix = prix.dropna(axis=1, how="all")

    return prix


def calculer_rendements(prix):
    if prix.empty:
        return pd.DataFrame()
    rendements = prix.pct_change(fill_method=None).dropna(how="all")
    return rendements


# -----------------------------
# Menu latéral
# -----------------------------
menu = st.sidebar.selectbox(
    "Navigation",
    ["Accueil", "Portefeuille", "VaR", "Backtesting", "Reporting"]
)

# -----------------------------
# PAGE ACCUEIL
# -----------------------------
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
    - VaR paramétrique  
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

    st.subheader("Formation")
    st.markdown("""
    - Double diplôme M2 IFIM  
    - Ing 3 MACS – Mathématiques Appliquées au Calcul Scientifique  
    """)

# -----------------------------
# PAGE PORTEFEUILLE
# -----------------------------
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
        default=["BNP Paribas", "Microsoft", "TotalEnergies"]
    )

    col1, col2 = st.columns(2)
    with col1:
        date_debut = st.date_input("Date de début", value=pd.to_datetime("2023-01-01"))
    with col2:
        date_fin = st.date_input("Date de fin", value=pd.to_datetime("today"))

    if st.button("Télécharger les données"):
        if not actifs_choisis:
            st.warning("Veuillez sélectionner au moins un actif.")
        elif date_debut >= date_fin:
            st.warning("La date de début doit être antérieure à la date de fin.")
        else:
            tickers = [actifs_disponibles[a] for a in actifs_choisis]

            with st.spinner("Téléchargement des données Yahoo Finance..."):
                data = telecharger_donnees(tickers, date_debut, date_fin)

            prix = extraire_prix_cloture(data)

            if prix.empty:
                st.error("Aucune donnée exploitable n'a été récupérée.")
            else:
                st.success("Données téléchargées avec succès.")

                st.subheader("Prix de clôture")
                st.dataframe(prix.tail(), use_container_width=True)

                rendements = calculer_rendements(prix)

                st.subheader("Rendements journaliers")
                if rendements.empty:
                    st.warning("Les rendements sont vides. Vérifie la période choisie ou les données téléchargées.")
                else:
                    st.dataframe(rendements.tail(), use_container_width=True)

                st.subheader("Statistiques descriptives")
                if rendements.empty:
                    st.info("Impossible de calculer les statistiques descriptives car les rendements sont vides.")
                else:
                    stats = pd.DataFrame({
                        "Rendement moyen": rendements.mean(),
                        "Volatilité": rendements.std(),
                        "Minimum": rendements.min(),
                        "Maximum": rendements.max()
                    })
                    stats.index.name = "Ticker"
                    st.dataframe(stats, use_container_width=True)

                st.subheader("Évolution des prix")
                st.line_chart(prix)

                if not rendements.empty:
                    st.subheader("Évolution des rendements")
                    st.line_chart(rendements)

                # Diagnostic léger
                with st.expander("Voir le diagnostic des données"):
                    st.write("Aperçu des prix :")
                    st.dataframe(prix.head(), use_container_width=True)

                    st.write("Aperçu des rendements :")
                    st.dataframe(rendements.head(), use_container_width=True)

                    st.write("Dimensions des prix :", prix.shape)
                    st.write("Dimensions des rendements :", rendements.shape)

# -----------------------------
# PAGE VAR
# -----------------------------
elif menu == "VaR":
    st.title("Calcul de la Value at Risk")
    st.write("Module VaR à venir.")

# -----------------------------
# PAGE BACKTESTING
# -----------------------------
elif menu == "Backtesting":
    st.title("Backtesting des modèles de VaR")
    st.write("Module Backtesting à venir.")

# -----------------------------
# PAGE REPORTING
# -----------------------------
elif menu == "Reporting":
    st.title("Reporting")
    st.write("Module Reporting à venir.")
