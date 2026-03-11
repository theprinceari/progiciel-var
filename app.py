import streamlit as st

st.set_page_config(
    page_title="Progiciel VaR",
    page_icon="📉",
    layout="wide"
)

# MENU LATERAL
menu = st.sidebar.selectbox(
    "Navigation",
    ["Accueil", "Portefeuille", "VaR", "Backtesting", "Reporting"]
)

# PAGE ACCUEIL
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

# PAGE PORTEFEUILLE
elif menu == "Portefeuille":

    st.title("Création du portefeuille")

    st.write("Sélectionnez les actifs et leurs poids.")

    actifs = st.multiselect(
        "Choisir les actifs",
        ["Apple", "Microsoft", "Airbus", "Total", "LVMH"]
    )

    if actifs:
        st.write("Actifs sélectionnés :", actifs)

# PAGE VAR
elif menu == "VaR":

    st.title("Calcul de la Value at Risk")

    niveau = st.selectbox(
        "Niveau de confiance",
        [95, 99]
    )

    st.write("Calcul de la VaR au niveau", niveau, "%")

# PAGE BACKTESTING
elif menu == "Backtesting":

    st.title("Backtesting des modèles de VaR")

    st.write("Analyse des violations de la VaR.")

# PAGE REPORTING
elif menu == "Reporting":

    st.title("Reporting")

    st.write("Génération du fichier Excel et du rapport PDF.")
