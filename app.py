import streamlit as st

st.set_page_config(
    page_title="Progiciel VaR",
    page_icon="📉",
    layout="wide"
)

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

if st.button("Commencer"):
    st.success("Passer à la page de création du portefeuille.")
