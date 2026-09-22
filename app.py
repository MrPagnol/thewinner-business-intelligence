
import os
import io
import json
import math
import pandas as pd
import streamlit as st

st.set_page_config(page_title="TheWinner Business Intelligence", page_icon="📊", layout="wide")

st.title("TheWinner Business Intelligence")
st.caption("Transformer les données des PME en décisions intelligentes.")

# ----------------------------
# Data preparation
# ----------------------------
def normalize_columns(df):
    mapping = {}
    for c in df.columns:
        x = str(c).strip().lower().replace("é","e").replace("è","e").replace("ê","e")
        x = x.replace("à","a").replace("ù","u")
        mapping[c] = x.replace(" ", "_")
    return df.rename(columns=mapping)

def find_col(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None

def analyze(df):
    df = normalize_columns(df.copy())

    revenue = find_col(df, ["ca","chiffre_affaires","ventes","revenue","sales"])
    expenses = find_col(df, ["depenses","charges","expenses","couts","cout_total"])
    profit = find_col(df, ["benefice","profit","resultat_net"])
    cash = find_col(df, ["tresorerie","cash","cash_balance"])
    debt = find_col(df, ["dettes","dette","debt"])
    stock = find_col(df, ["stock","stocks","valeur_stock"])
    date = find_col(df, ["date","mois","month","periode"])

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    result = {"df": df, "columns": {
        "revenue": revenue, "expenses": expenses, "profit": profit,
        "cash": cash, "debt": debt, "stock": stock, "date": date
    }}

    if revenue:
        result["total_revenue"] = float(df[revenue].sum())
    else:
        result["total_revenue"] = None

    if expenses:
        result["total_expenses"] = float(df[expenses].sum())
    else:
        result["total_expenses"] = None

    if profit:
        result["total_profit"] = float(df[profit].sum())
    elif revenue and expenses:
        result["total_profit"] = float(df[revenue].sum() - df[expenses].sum())
    else:
        result["total_profit"] = None

    if result["total_revenue"] and result["total_revenue"] != 0 and result["total_profit"] is not None:
        result["margin"] = result["total_profit"] / result["total_revenue"]
    else:
        result["margin"] = None

    # Simple anomaly detection using IQR on numeric columns
    anomalies=[]
    for c in numeric_cols:
        s=df[c].dropna()
        if len(s) >= 5:
            q1,q3=s.quantile(.25),s.quantile(.75)
            iqr=q3-q1
            if iqr > 0:
                low, high=q1-1.5*iqr, q3+1.5*iqr
                hits=df[(df[c]<low)|(df[c]>high)]
                if len(hits):
                    anomalies.append({"metric":c, "count":int(len(hits)), "low":float(low), "high":float(high)})
    result["anomalies"]=anomalies
    return result

def build_report(r):
    c=r["columns"]
    lines=[]
    if r["total_revenue"] is not None:
        lines.append(f"Chiffre d'affaires total: {r['total_revenue']:,.2f}")
    if r["total_expenses"] is not None:
        lines.append(f"Dépenses totales: {r['total_expenses']:,.2f}")
    if r["total_profit"] is not None:
        lines.append(f"Résultat estimé/rapporté: {r['total_profit']:,.2f}")
    if r["margin"] is not None:
        lines.append(f"Marge: {r['margin']*100:.1f}%")

    alerts=[]
    if r["margin"] is not None and r["margin"] < .10:
        alerts.append("Marge inférieure à 10% : examiner les coûts, prix et produits.")
    if c["cash"] is not None and len(r["df"]):
        last=float(r["df"][c["cash"]].dropna().iloc[-1]) if r["df"][c["cash"]].dropna().any() else None
        if last is not None and last < 0:
            alerts.append("Trésorerie négative sur la dernière observation.")
    if c["debt"] is not None and r["total_revenue"] not in (None,0):
        debt=float(r["df"][c["debt"]].sum())
        if debt > r["total_revenue"]:
            alerts.append("Dette cumulée supérieure au chiffre d'affaires observé : analyser l'endettement.")
    if r["anomalies"]:
        alerts.append(f"{len(r['anomalies'])} indicateur(s) présentent des observations atypiques.")

    return lines, alerts

# ----------------------------
# UI
# ----------------------------
with st.sidebar:
    st.header("Données")
    uploaded=st.file_uploader("Importer un fichier CSV ou Excel", type=["csv","xlsx"])
    st.info("MVP : utilisez les colonnes CA/ventes, dépenses, bénéfice, trésorerie, dettes, stocks et date si disponibles.")

if not uploaded:
    st.subheader("Mode démonstration")
    demo=pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=12, freq="MS"),
        "ventes": [12000,13500,12800,14200,15000,14700,16100,15800,17200,18000,17600,19500],
        "depenses": [9500,10100,10400,11300,12100,11900,13300,13700,14500,15100,15300,16200],
        "tresorerie": [2500,3900,4300,5100,6100,7000,7600,8000,9200,10400,11000,13000],
        "dettes": [7000,6800,6500,6200,6000,5800,5500,5200,5000,4700,4300,4000],
        "stock": [5000,5200,5100,5600,5800,6100,6200,6500,6700,6900,7100,7300]
    })
    df=demo
else:
    if uploaded.name.lower().endswith(".csv"):
        df=pd.read_csv(uploaded)
    else:
        df=pd.read_excel(uploaded)

r=analyze(df)
lines, alerts=build_report(r)

st.subheader("Vue dirigeant")
m1,m2,m3,m4=st.columns(4)
m1.metric("Chiffre d'affaires", f"{r['total_revenue']:,.0f}" if r["total_revenue"] is not None else "N/D")
m2.metric("Dépenses", f"{r['total_expenses']:,.0f}" if r["total_expenses"] is not None else "N/D")
m3.metric("Résultat", f"{r['total_profit']:,.0f}" if r["total_profit"] is not None else "N/D")
m4.metric("Marge", f"{r['margin']*100:.1f}%" if r["margin"] is not None else "N/D")

st.divider()
left,right=st.columns(2)

with left:
    st.subheader("Évolution")
    d=r["df"]
    rev=r["columns"]["revenue"]
    exp=r["columns"]["expenses"]
    if rev and exp:
        chart=d[[rev,exp]].copy()
        chart.columns=["CA","Dépenses"]
        st.line_chart(chart)
    else:
        st.dataframe(d, use_container_width=True)

with right:
    st.subheader("Alertes")
    if alerts:
        for a in alerts:
            st.warning(a)
    else:
        st.success("Aucune alerte simple détectée.")

st.subheader("Rapport d'analyse")
for line in lines:
    st.write("•", line)

st.subheader("Anomalies détectées")
if r["anomalies"]:
    st.dataframe(pd.DataFrame(r["anomalies"]), use_container_width=True)
else:
    st.write("Aucune anomalie statistique simple détectée.")

st.divider()
st.subheader("Couche IA — prochaine étape")
st.write(
    "Le moteur actuel calcule et détecte. La prochaine couche connectera un modèle IA "
    "à ce contexte financier afin de générer une analyse narrative, des hypothèses de causes "
    "et des scénarios. Ne fournissez pas de données personnelles ou bancaires sensibles dans ce MVP."
)
