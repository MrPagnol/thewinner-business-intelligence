import pandas as pd
import streamlit as st
from openai import OpenAI

st.set_page_config(
    page_title="TheWinner Business Intelligence",
    page_icon="📊",
    layout="wide"
)

st.title("TheWinner Business Intelligence")
st.caption("Transformer les données des PME en décisions intelligentes.")


# ============================================================
# NORMALISATION DES DONNÉES
# ============================================================

def normalize_columns(df):
    mapping = {}

    for c in df.columns:
        x = str(c).strip().lower()

        replacements = {
            "é": "e",
            "è": "e",
            "ê": "e",
            "à": "a",
            "ù": "u"
        }

        for old, new in replacements.items():
            x = x.replace(old, new)

        mapping[c] = x.replace(" ", "_")

    return df.rename(columns=mapping)


def find_col(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None


# ============================================================
# MOTEUR FINANCIER THEWINNER
# ============================================================

def analyze(df):

    df = normalize_columns(df.copy())

    revenue = find_col(
        df,
        ["ca", "chiffre_affaires", "ventes", "revenue", "sales"]
    )

    expenses = find_col(
        df,
        ["depenses", "charges", "expenses", "couts", "cout_total"]
    )

    profit = find_col(
        df,
        ["benefice", "profit", "resultat_net"]
    )

    cash = find_col(
        df,
        ["tresorerie", "cash", "cash_balance"]
    )

    debt = find_col(
        df,
        ["dettes", "dette", "debt"]
    )

    stock = find_col(
        df,
        ["stock", "stocks", "valeur_stock"]
    )

    date = find_col(
        df,
        ["date", "mois", "month", "periode"]
    )

    numeric_cols = df.select_dtypes(
        include="number"
    ).columns.tolist()

    result = {
        "df": df,
        "columns": {
            "revenue": revenue,
            "expenses": expenses,
            "profit": profit,
            "cash": cash,
            "debt": debt,
            "stock": stock,
            "date": date
        }
    }

    # Chiffre d'affaires
    if revenue:
        result["total_revenue"] = float(
            df[revenue].sum()
        )
    else:
        result["total_revenue"] = None

    # Dépenses
    if expenses:
        result["total_expenses"] = float(
            df[expenses].sum()
        )
    else:
        result["total_expenses"] = None

    # Résultat
    if profit:
        result["total_profit"] = float(
            df[profit].sum()
        )

    elif revenue and expenses:
        result["total_profit"] = float(
            df[revenue].sum() -
            df[expenses].sum()
        )

    else:
        result["total_profit"] = None

    # Marge
    if (
        result["total_revenue"]
        and result["total_revenue"] != 0
        and result["total_profit"] is not None
    ):
        result["margin"] = (
            result["total_profit"] /
            result["total_revenue"]
        )
    else:
        result["margin"] = None

    # Anomalies
    anomalies = []

    for c in numeric_cols:

        s = df[c].dropna()

        if len(s) >= 5:

            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)

            iqr = q3 - q1

            if iqr > 0:

                low = q1 - 1.5 * iqr
                high = q3 + 1.5 * iqr

                hits = df[
                    (df[c] < low) |
                    (df[c] > high)
                ]

                if len(hits):

                    anomalies.append({
                        "metric": c,
                        "count": int(len(hits)),
                        "low": float(low),
                        "high": float(high)
                    })

    result["anomalies"] = anomalies

    return result


# ============================================================
# RAPPORT CLASSIQUE
# ============================================================

def build_report(r):

    c = r["columns"]

    lines = []

    if r["total_revenue"] is not None:
        lines.append(
            f"Chiffre d'affaires total : "
            f"{r['total_revenue']:,.2f}"
        )

    if r["total_expenses"] is not None:
        lines.append(
            f"Dépenses totales : "
            f"{r['total_expenses']:,.2f}"
        )

    if r["total_profit"] is not None:
        lines.append(
            f"Résultat : "
            f"{r['total_profit']:,.2f}"
        )

    if r["margin"] is not None:
        lines.append(
            f"Marge : "
            f"{r['margin'] * 100:.1f}%"
        )

    alerts = []

    if (
        r["margin"] is not None
        and r["margin"] < 0.10
    ):
        alerts.append(
            "Marge inférieure à 10% : "
            "examiner les coûts, les prix et les produits."
        )

    if c["cash"] is not None:

        values = r["df"][c["cash"]].dropna()

        if len(values):

            last = float(values.iloc[-1])

            if last < 0:

                alerts.append(
                    "Trésorerie négative sur la dernière observation."
                )

    if (
        c["debt"] is not None
        and r["total_revenue"] not in (None, 0)
    ):

        debt = float(
            r["df"][c["debt"]].sum()
        )

        if debt > r["total_revenue"]:

            alerts.append(
                "Dette cumulée supérieure "
                "au chiffre d'affaires observé."
            )

    if r["anomalies"]:

        alerts.append(
            f"{len(r['anomalies'])} indicateur(s) "
            "présentent des observations atypiques."
        )

    return lines, alerts


# ============================================================
# CONTEXTE POUR L'IA
# ============================================================

def build_ai_context(r):

    df = r["df"]
    c = r["columns"]

    context = {
        "chiffre_affaires_total": r["total_revenue"],
        "depenses_totales": r["total_expenses"],
        "resultat": r["total_profit"],
        "marge": r["margin"],
        "anomalies": r["anomalies"]
    }

    # Dernière trésorerie
    if c["cash"]:

        values = df[c["cash"]].dropna()

        if len(values):
            context["tresorerie_derniere_observation"] = float(
                values.iloc[-1]
            )

    # Dette
    if c["debt"]:

        context["dettes_totales"] = float(
            df[c["debt"]].sum()
        )

    # Stock
    if c["stock"]:

        values = df[c["stock"]].dropna()

        if len(values):
            context["stock_derniere_observation"] = float(
                values.iloc[-1]
            )

    # Evolution CA
    if c["revenue"] and len(df) >= 2:

        first = float(df[c["revenue"]].iloc[0])
        last = float(df[c["revenue"]].iloc[-1])

        if first != 0:

            context["evolution_ca"] = (
                (last - first) / first
            )

    # Evolution dépenses
    if c["expenses"] and len(df) >= 2:

        first = float(df[c["expenses"]].iloc[0])
        last = float(df[c["expenses"]].iloc[-1])

        if first != 0:

            context["evolution_depenses"] = (
                (last - first) / first
            )

    return context


# ============================================================
# THEWINNER AI
# ============================================================

def run_ai_analysis(r):

    if "OPENAI_API_KEY" not in st.secrets:

        st.error(
            "OPENAI_API_KEY n'est pas configurée "
            "dans Streamlit Secrets."
        )

        return None

    client = OpenAI(
        api_key=st.secrets["OPENAI_API_KEY"]
    )

    context = build_ai_context(r)

    instructions = """
Tu es TheWinner AI, un analyste financier spécialisé
dans l'analyse des petites et moyennes entreprises.

Ta mission est d'aider un dirigeant à comprendre ses données.

Règles importantes :

1. Utilise uniquement les données fournies.
2. Ne fabrique jamais de chiffres.
3. Sépare clairement les faits des hypothèses.
4. Ne présente pas tes conclusions comme des certitudes
   lorsque les données sont insuffisantes.
5. Ne donne pas de conseil juridique ou fiscal définitif.
6. Ne promets jamais un résultat financier.

Structure obligatoirement ton analyse avec :

## 1. Diagnostic général

Explique en quelques lignes la situation financière.

## 2. Points positifs

Identifie les éléments favorables observables.

## 3. Risques

Identifie les principaux risques visibles.

## 4. Causes possibles

Donne des hypothèses et précise qu'elles doivent être vérifiées.

## 5. Priorités

Donne exactement 3 choses que le dirigeant devrait
examiner en priorité.

## 6. Questions à poser

Donne 3 questions supplémentaires qui permettraient
d'améliorer l'analyse.

Réponds en français.
Sois clair, professionnel et compréhensible par un dirigeant
qui n'est pas expert en finance.
"""

    prompt = f"""
Voici les données financières calculées par TheWinner Business Intelligence :

{context}

Analyse cette entreprise conformément à tes instructions.
"""

    try:

        response = client.responses.create(
            model="gpt-5.6-luna",
            instructions=instructions,
            input=prompt
        )

        return response.output_text

    except Exception as e:

        st.error(
            f"Erreur lors de l'analyse IA : {e}"
        )

        return None


# ============================================================
# DONNÉES
# ============================================================

with st.sidebar:

    st.header("Données")

    uploaded = st.file_uploader(
        "Importer un fichier CSV ou Excel",
        type=["csv", "xlsx"]
    )

    st.info(
        "Colonnes reconnues : CA, ventes, dépenses, "
        "bénéfice, trésorerie, dettes, stocks et date."
    )


# ============================================================
# MODE DÉMONSTRATION
# ============================================================

if not uploaded:

    st.subheader("Mode démonstration")

    demo = pd.DataFrame({

        "date": pd.date_range(
            "2026-01-01",
            periods=12,
            freq="MS"
        ),

        "ventes": [
            12000, 13500, 12800,
            14200, 15000, 14700,
            16100, 15800, 17200,
            18000, 17600, 19500
        ],

        "depenses": [
            9500, 10100, 10400,
            11300, 12100, 11900,
            13300, 13700, 14500,
            15100, 15300, 16200
        ],

        "tresorerie": [
            2500, 3900, 4300,
            5100, 6100, 7000,
            7600, 8000, 9200,
            10400, 11000, 13000
        ],

        "dettes": [
            7000, 6800, 6500,
            6200, 6000, 5800,
            5500, 5200, 5000,
            4700, 4300, 4000
        ],

        "stock": [
            5000, 5200, 5100,
            5600, 5800, 6100,
            6200, 6500, 6700,
            6900, 7100, 7300
        ]
    })

    df = demo

else:

    if uploaded.name.lower().endswith(".csv"):

        df = pd.read_csv(uploaded)

    else:

        df = pd.read_excel(uploaded)


# ============================================================
# ANALYSE
# ============================================================

r = analyze(df)

lines, alerts = build_report(r)


# ============================================================
# DASHBOARD
# ============================================================

st.subheader("Vue dirigeant")

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Chiffre d'affaires",
    f"{r['total_revenue']:,.0f}"
    if r["total_revenue"] is not None
    else "N/D"
)

m2.metric(
    "Dépenses",
    f"{r['total_expenses']:,.0f}"
    if r["total_expenses"] is not None
    else "N/D"
)

m3.metric(
    "Résultat",
    f"{r['total_profit']:,.0f}"
    if r["total_profit"] is not None
    else "N/D"
)

m4.metric(
    "Marge",
    f"{r['margin'] * 100:.1f}%"
    if r["margin"] is not None
    else "N/D"
)


# ============================================================
# GRAPHIQUE
# ============================================================

st.divider()

left, right = st.columns(2)

with left:

    st.subheader("Évolution")

    d = r["df"]

    rev = r["columns"]["revenue"]
    exp = r["columns"]["expenses"]

    if rev and exp:

        chart = d[[rev, exp]].copy()

        chart.columns = [
            "CA",
            "Dépenses"
        ]

        st.line_chart(chart)

    else:

        st.dataframe(
            d,
            use_container_width=True
        )


# ============================================================
# ALERTES
# ============================================================

with right:

    st.subheader("Alertes")

    if alerts:

        for alert in alerts:
            st.warning(alert)

    else:

        st.success(
            "Aucune alerte simple détectée."
        )


# ============================================================
# RAPPORT
# ============================================================

st.subheader("Rapport d'analyse")

for line in lines:

    st.write(
        "•",
        line
    )


# ============================================================
# ANOMALIES
# ============================================================

st.subheader("Anomalies détectées")

if r["anomalies"]:

    st.dataframe(
        pd.DataFrame(r["anomalies"]),
        use_container_width=True
    )

else:

    st.write(
        "Aucune anomalie statistique simple détectée."
    )


# ============================================================
# 🧠 THEWINNER AI
# ============================================================

st.divider()

st.subheader("🧠 TheWinner AI — Financial Analyst")

st.write(
    "L'IA analyse les indicateurs calculés par TheWinner "
    "et produit un diagnostic financier structuré."
)

if st.button(
    "🚀 Analyser cette entreprise avec TheWinner AI",
    type="primary"
):

    with st.spinner(
        "TheWinner AI analyse les données..."
    ):

        ai_result = run_ai_analysis(r)

    if ai_result:

        st.success(
            "Analyse IA terminée."
        )

        st.markdown(ai_result)
