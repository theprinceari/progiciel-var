import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(
    page_title="Progiciel VaR",
    page_icon="📉",
    layout="wide"
)

menu = st.sidebar.selectbox(
    "Navigation",
    ["Accueil", "Portefeuille", "VaR", "Backtesting", "Reporting"]
)

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

    st.subheader("Équipe projet")
    st.markdown("""
    - Kopangoye Guénolé Wariol
    - Adjagba Harlem Désir
    - Ecclésiaste Gnargo
    - Anta Mbaye
    """)

elif menu == "Portefeuille":

    st.title("Création du portefeuille")

    actifs_disponibles = {
        "Apple": "AAPL",
        "Microsoft": "MSFT",
        "Airbus": "AIR.PA",
        "TotalEnergies": "TTE.PA",
        "LVMH": "MC.PA",
        "BNP Paribas": "BNP.PA",
        "Safran": "SAF.PA"
    }

    actifs_choisis = st.multiselect(
        "Choisir les actifs",
        list(actifs_disponibles.keys())
    )

    date_debut = st.date_input("Date de début", value=pd.to_datetime("2023-01-01"))
    date_fin = st.date_input("Date de fin", value=pd.to_datetime("today"))

    if st.button("Télécharger les données"):
        if not actifs_choisis:
            st.warning("Veuillez sélectionner au moins un actif.")
        else:
            tickers = [actifs_disponibles[a] for a in actifs_choisis]

            data = yf.download(
                tickers,
                start=date_debut,
                end=date_fin,
                auto_adjust=True,
                progress=False
            )

            if data.empty:
                st.error("Aucune donnée récupérée.")
            else:
                if "Close" in data:
                    prix = data["Close"].copy()
                else:
                    prix = data.copy()

                if isinstance(prix, pd.Series):
                    prix = prix.to_frame()

                st.subheader("Prix de clôture")
                st.dataframe(prix.tail())

                rendements = prix.pct_change().dropna()

                st.subheader("Rendements journaliers")
                st.dataframe(rendements.tail())

                st.subheader("Statistiques descriptives")
                stats = pd.DataFrame({
                    "Rendement moyen": rendements.mean(),
                    "Volatilité": rendements.std()
                })
                st.dataframe(stats)

                st.line_chart(prix)

elif menu == "VaR":
    st.title("Calcul de la Value at Risk")
    st.write("Module VaR à venir.")

elif menu == "Backtesting":
    st.title("Backtesting des modèles de VaR")
    st.write("Module Backtesting à venir.")

elif menu == "Reporting":
    st.title("Reporting")
    st.write("Module Reporting à venir.")
