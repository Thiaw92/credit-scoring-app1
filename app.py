import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ──────────────────────────────────────────────
# VALEURS PAR DÉFAUT DU FORMULAIRE
# ──────────────────────────────────────────────
DEFAULTS = {
    "age": None,
    "revenu": None,
    "montant": None,
    "ratio": None,
    "score_banque": None,
    "rejets": None,
    "decouvert": None,
    "emploi": "CDI",
    "secteur": "Public",
    "situation": "Célibataire",
}

def init_defaults():
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

def reset_formulaire():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    for key in ["prediction", "proba", "score_final"]:
        st.session_state.pop(key, None)

# ──────────────────────────────────────────────
# CONFIG PAGE
# ──────────────────────────────────────────────
st.set_page_config(page_title="Credit App", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #3b2f2f; color: #f5d76e; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

init_defaults()

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
st.sidebar.image("assets/ISM-Dakar-3.png", width=200)

# ──────────────────────────────────────────────
# AUTHENTIFICATION
# ──────────────────────────────────────────────
USERS = {"admin": "1234", "oumar": "credit2026"}

def login():
    st.title("🔐 Connexion")
    username = st.text_input("Nom d'utilisateur")
    password = st.text_input("Mot de passe", type="password")
    if st.button("Se connecter"):
        if username in USERS and USERS[username] == password:
            st.session_state["logged_in"] = True
            st.success("Connexion réussie")
            st.rerun()
        else:
            st.error("Identifiants incorrects")

if not st.session_state["logged_in"]:
    login()
    st.stop()

# ──────────────────────────────────────────────
# BOUTONS SIDEBAR (après login)
# ──────────────────────────────────────────────
if st.sidebar.button("🚪 Se déconnecter"):
    st.session_state["logged_in"] = False
    st.rerun()

if st.sidebar.button("🗑️ Effacer le formulaire"):
    reset_formulaire()
    st.rerun()

if st.sidebar.button("🗑️ Vider historique"):
    st.session_state.history = []
    st.rerun()

# ──────────────────────────────────────────────
# TITRE & À PROPOS
# ──────────────────────────────────────────────
st.title("💳 Application de Scoring Crédit")
st.markdown("Évaluez le risque d'un client en quelques secondes")

st.markdown("""
## ℹ️ À propos de l'application
Cette application de **scoring crédit** permet d'évaluer le risque de défaut d'un client
à partir de ses informations financières et personnelles.

Elle utilise un modèle de machine learning entraîné sur des données bancaires pour :
- Estimer la probabilité de défaut
- Générer un score de crédit sur 1000
- Aider à la prise de décision (accord/refus de crédit)

⚠️ Ceci est un outil d'aide à la décision.
""")

# ──────────────────────────────────────────────
# CHARGEMENT MODÈLE
# ──────────────────────────────────────────────
model_path   = "output/credit_scoring_model.pkl"
columns_path = "output/columns.pkl"

if not os.path.exists(model_path):
    st.error("⚠️ Modèle introuvable. Lance train_model.py")
    st.stop()

pipeline      = joblib.load(model_path)
expected_cols = joblib.load(columns_path)

# ──────────────────────────────────────────────
# FORMULAIRE
# ──────────────────────────────────────────────
st.header("📋 Informations client")

col1, col2 = st.columns(2)

with col1:
    age     = st.number_input("Âge",                    value=st.session_state["age"],         key="age")
    revenu  = st.number_input("Revenu mensuel (FCFA)",   value=st.session_state["revenu"],       key="revenu",  placeholder="Ex: 200000")
    montant = st.number_input("Montant demandé (FCFA)",  value=st.session_state["montant"],      key="montant")
    ratio   = st.number_input("Ratio d'endettement (%)", value=st.session_state["ratio"],        key="ratio",   placeholder="Ex: 30.0")
    score   = st.number_input("Score interne banque",    value=st.session_state["score_banque"], key="score_banque")

with col2:
    rejets    = st.number_input("Nb rejets prélèvements",    value=st.session_state["rejets"],    key="rejets")
    decouvert = st.number_input("Nb découverts (12 mois)",   value=st.session_state["decouvert"], key="decouvert")
    emploi    = st.selectbox("Type emploi",                  ["CDI", "CDD", "Indépendant", "Sans emploi"],
                             index=["CDI", "CDD", "Indépendant", "Sans emploi"].index(st.session_state["emploi"]),
                             key="emploi")
    secteur   = st.selectbox("Secteur activité",             ["Public", "Privé", "Agriculture", "Commerce", "Autre"],
                             index=["Public", "Privé", "Agriculture", "Commerce", "Autre"].index(st.session_state["secteur"]),
                             key="secteur")
    situation = st.selectbox("Situation matrimoniale",       ["Célibataire", "Marié", "Divorcé", "Veuf"],
                             index=["Célibataire", "Marié", "Divorcé", "Veuf"].index(st.session_state["situation"]),
                             key="situation")

# ──────────────────────────────────────────────
# DATAFRAME
# ──────────────────────────────────────────────
input_data = pd.DataFrame([{
    "AGE":                   age,
    "REVENU_MENSUEL_FCFA":   revenu,
    "MONTANT_DEMANDE_FCFA":  montant,
    "RATIO_ENDETTEMENT":     ratio,
    "SCORE_INTERNE_BANQUE":  score,
    "NB_REJETS_PRELEVEMENT": rejets,
    "NB_DECOUVERT_12MOIS":   decouvert,
    "TYPE_EMPLOI":           emploi,
    "SECTEUR_ACTIVITE":      secteur,
    "SITUATION_MATRIMONIALE": situation,
}])
input_data = input_data.reindex(columns=expected_cols)

# ──────────────────────────────────────────────
# BOUTON PRÉDIRE
# ──────────────────────────────────────────────
if st.button("🔍 Prédire le risque"):
    prediction = pipeline.predict(input_data)[0]
    proba      = pipeline.predict_proba(input_data)[0][1]
    score_final = int((1 - proba) * 1000)

    st.session_state.prediction = prediction
    st.session_state.proba      = proba
    st.session_state.score_final = score_final

    st.session_state.history.append({
        "Âge":         age,
        "Revenu":      revenu,
        "Montant":     montant,
        "Score crédit": score_final,
        "Probabilité": proba,
        "Décision":    "Refusé" if prediction == 1 else "Accepté",
    })

# ──────────────────────────────────────────────
# RÉSULTAT
# ──────────────────────────────────────────────
if "prediction" in st.session_state:
    st.subheader("📊 Résultat")

    if st.session_state.prediction == 1:
        st.error("❌ Crédit Refusé")
    else:
        st.success("✅ Crédit Accordé")

    st.metric("Probabilité de défaut", f"{st.session_state.proba*100:.1f}%")
    st.metric("Score crédit",          f"{st.session_state.score_final}/1000")

# ──────────────────────────────────────────────
# DASHBOARD
# ──────────────────────────────────────────────
if "score_final" in st.session_state:
    st.subheader("📊 Dashboard")

    c1, c2, c3 = st.columns(3)
    c1.metric("Score crédit", st.session_state.score_final)
    c2.metric("Risque (%)",   f"{st.session_state.proba*100:.1f}%")
    c3.metric("Décision",     "Refusé" if st.session_state.prediction == 1 else "Accepté")

    fig, ax = plt.subplots()
    ax.bar(["Risque", "Score"], [st.session_state.proba * 100, st.session_state.score_final])
    st.pyplot(fig)

# ──────────────────────────────────────────────
# HISTORIQUE
# ──────────────────────────────────────────────
st.subheader("📜 Historique des prédictions")

if len(st.session_state.history) > 0:
    df_history = pd.DataFrame(st.session_state.history)
    st.dataframe(df_history)
    st.download_button(
        "📥 Télécharger CSV",
        df_history.to_csv(index=False),
        file_name="historique_predictions.csv",
        mime="text/csv",
    )
else:
    st.info("Aucune prédiction enregistrée pour le moment.")