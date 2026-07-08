"""
app.py
─────────────────────────────────────────────────────────
Application Streamlit du SAD Framework — version améliorée.

Améliorations v2 :
  - Thème professionnel avec CSS personnalisé
  - Filtres interactifs (date, catégorie, client)
  - Bouton d'ouverture du rapport EDA (eda_report.html)
  - Page Power BI avec bouton d'ouverture du fichier .pbix
  - Filtres appliqués sur toutes les sections
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import os
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="SAD Framework — Dashboard",
    layout="wide"
)

# ─────────────────────────────────────────────────────────
# CSS PERSONNALISÉ — thème professionnel bleu/sombre
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Fond sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1f2e;
    }
    [data-testid="stSidebar"] * {
        color: #e0e6f0 !important;
    }

    /* Titres principaux */
    h1 { color: #2563eb !important; }
    h2 { color: #1e40af !important; }
    h3 { color: #3b82f6 !important; }

    /* Métriques */
    [data-testid="metric-container"] {
        background-color: #f0f4ff;
        border: 1px solid #bfdbfe;
        border-radius: 10px;
        padding: 15px;
    }
    [data-testid="metric-container"] label {
        color: #1e40af !important;
        font-weight: 600;
    }

    /* Bouton principal */
    .stButton > button[kind="primary"] {
        background-color: #2563eb;
        border: none;
        border-radius: 8px;
        color: white;
        font-weight: 600;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #1d4ed8;
    }

    /* Boutons secondaires */
    .stButton > button {
        border-radius: 8px;
        border: 1px solid #2563eb;
        color: #2563eb;
        font-weight: 500;
    }

    /* Tabs */
    [data-testid="stTab"] {
        font-weight: 600;
    }

    /* Bandeau statut */
    .status-bar {
        background: linear-gradient(90deg, #1e40af, #2563eb);
        color: white;
        padding: 8px 16px;
        border-radius: 8px;
        font-size: 0.85em;
        margin-bottom: 16px;
    }

    /* Carte info */
    .info-card {
        background: #f8faff;
        border-left: 4px solid #2563eb;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# CHEMINS
# ─────────────────────────────────────────────────────────
PLOTS_DIR   = Path("outputs/plots")
OUTPUTS_DIR = Path("outputs")
EDA_REPORT  = OUTPUTS_DIR / "eda_report.html"


# ─────────────────────────────────────────────────────────
# CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_csv_safe(path):
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as e:
            st.warning(f"Erreur de lecture {path.name} : {e}")
            return None
    return None

def load_clean_data():
    return load_csv_safe(POWERBI_DIR / "clean_data.csv")

def load_predictions():
    return load_csv_safe(POWERBI_DIR / "predictions.csv")

def load_segments():
    return load_csv_safe(POWERBI_DIR / "segments.csv")

def load_anomalies():
    return load_csv_safe(POWERBI_DIR / "anomalies.csv")

def load_kpis():
    return load_csv_safe(POWERBI_DIR / "kpis.csv")

def get_pipeline_status():
    clean_path = POWERBI_DIR / "clean_data.csv"
    if not clean_path.exists():
        return "❌ Aucune donnée — pipeline jamais exécuté", None
    import datetime
    mtime = datetime.datetime.fromtimestamp(clean_path.stat().st_mtime)
    age   = datetime.datetime.now() - mtime
    age_str = f"{int(age.total_seconds() // 60)} min" \
              if age.total_seconds() < 3600 \
              else f"{age.total_seconds() / 3600:.1f}h"
    return f"✅ Dernière exécution il y a {age_str}", mtime


# ─────────────────────────────────────────────────────────
# FILTRES GLOBAUX — chargés une fois, réutilisés partout
# ─────────────────────────────────────────────────────────
def build_filters(df):
    """
    Construit les filtres dans la sidebar et retourne
    le DataFrame filtré.
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader(" Filtres")

    df_filtered = df.copy()

    # Filtre date
    # Filtre date
    if "date_principale" in df.columns:
        df["date_principale"] = pd.to_datetime(
            df["date_principale"], errors="coerce"
        )
        df_filtered["date_principale"] = pd.to_datetime(
            df_filtered["date_principale"], errors="coerce"
        )
        date_min = df["date_principale"].min()
        date_max = df["date_principale"].max()

        if pd.notna(date_min) and pd.notna(date_max):
            date_range = st.sidebar.date_input(
                "Période",
                value=(date_min.date(), date_max.date()),
                min_value=date_min.date(),
                max_value=date_max.date()
            )
            if len(date_range) == 2:
                df_filtered = df_filtered[
                    (df_filtered["date_principale"] >= pd.Timestamp(date_range[0])) &
                    (df_filtered["date_principale"] <= pd.Timestamp(date_range[1]))
                ]

    # Filtre catégorie
    if "categorie_principale" in df.columns:
        categories = sorted(df["categorie_principale"].dropna().unique())
        selected_cats = st.sidebar.multiselect(
            "Catégorie",
            options=categories,
            default=[]
        )
        if selected_cats:
            df_filtered = df_filtered[
                df_filtered["categorie_principale"].isin(selected_cats)
            ]

    # Filtre client
    if "identifiant_entite" in df.columns:
        clients = sorted(df["identifiant_entite"].dropna().unique())
        selected_clients = st.sidebar.multiselect(
            "Client",
            options=clients,
            default=[]
        )
        if selected_clients:
            df_filtered = df_filtered[
                df_filtered["identifiant_entite"].isin(selected_clients)
            ]

    n_filtrees = len(df_filtered)
    n_total    = len(df)
    st.sidebar.caption(
        f" {n_filtrees:,} / {n_total:,} lignes affichées"
    )

    return df_filtered


# ─────────────────────────────────────────────────────────
# SIDEBAR — navigation + contrôle pipeline
# ─────────────────────────────────────────────────────────
st.sidebar.title(" SAD Framework")

section = st.sidebar.radio(
    "Navigation",
    [
        " Vue d'ensemble",
        " Monitoring technique",
        " Dashboard business",
        " Rapport EDA",
        " Power BI"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader(" Contrôle du pipeline")

available_datasets = [
    d.name for d in Path("outputs/powerbi").iterdir()
    if d.is_dir()
] if Path("outputs/powerbi").exists() else ["lbl_2024"]

dataset_id_input = st.sidebar.selectbox(
    "📦 Dataset actif",
    options=available_datasets,
    index=0
)
POWERBI_DIR = Path(f"outputs/powerbi/{dataset_id_input}")

if st.sidebar.button(" Relancer le pipeline", type="primary"):
    with st.spinner(f"Pipeline en cours pour '{dataset_id_input}'..."):
        try:
            from pipeline import run_pipeline
            df_res, results = run_pipeline(dataset_id=dataset_id_input)
            if df_res is not None:
                st.sidebar.success("✅ Pipeline terminé")
                st.cache_data.clear()
                st.rerun()
            else:
                st.sidebar.error("❌ Pipeline arrêté")
        except Exception as e:
            st.sidebar.error(f"❌ Erreur : {e}")

status_text, last_run = get_pipeline_status()
st.sidebar.markdown("---")
st.sidebar.caption(status_text)
st.sidebar.caption("SAD Framework — v2.0")


# ─────────────────────────────────────────────────────────
# SECTION 1 — VUE D'ENSEMBLE
# ─────────────────────────────────────────────────────────
if section == " Vue d'ensemble":
    st.title(" SAD Framework — Vue d'ensemble")
    st.markdown(
        f'<div class="status-bar">{status_text}</div>',
        unsafe_allow_html=True
    )

    df_clean = load_clean_data()

    if df_clean is None:
        st.info("Aucune donnée disponible. Lance le pipeline depuis"
                " le menu de gauche.")
    else:
        df_f = build_filters(df_clean)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Lignes", f"{len(df_f):,}")
        col2.metric("Colonnes", df_f.shape[1])

        if "valeur_principale" in df_f.columns:
            col3.metric(
                "Valeur totale",
                f"{df_f['valeur_principale'].sum():,.0f}"
            )
            col4.metric(
                "Valeur moyenne",
                f"{df_f['valeur_principale'].mean():,.2f}"
            )

        st.markdown("---")
        st.subheader("Aperçu des données nettoyées")
        st.dataframe(df_f.head(20), use_container_width=True)


# ─────────────────────────────────────────────────────────
# SECTION 2 — MONITORING TECHNIQUE
# ─────────────────────────────────────────────────────────
elif section == " Monitoring technique":
    st.title(" Monitoring technique du pipeline")

    df_clean = load_clean_data()

    if df_clean is not None:
        df_f = build_filters(df_clean)
    else:
        df_f = None

    tab1, tab2, tab3 = st.tabs(
        [" Données nettoyées", " Graphiques EDA", " Qualité"]
    )

    with tab1:
        if df_f is not None:
            st.write(f"**{len(df_f):,} lignes** × "
                     f"**{df_f.shape[1]} colonnes**")
            st.dataframe(df_f, use_container_width=True, height=500)
            csv_export = df_f.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                " Télécharger les données filtrées (CSV)",
                csv_export,
                "clean_data_filtre.csv",
                "text/csv"
            )
        else:
            st.info("Aucune donnée disponible.")

    with tab2:
        if PLOTS_DIR.exists():
            plot_files = sorted(PLOTS_DIR.glob("*.png"))
            if plot_files:
                cols = st.columns(2)
                for i, plot_path in enumerate(plot_files):
                    with cols[i % 2]:
                        st.image(
                            str(plot_path),
                            caption=plot_path.stem,
                            use_container_width=True
                        )
            else:
                st.info("Aucun graphique généré pour le moment.")
        else:
            st.info("Le dossier outputs/plots/ n'existe pas encore.")

    with tab3:
        if df_f is not None:
            st.subheader("Valeurs manquantes")
            missing = df_f.isnull().sum()
            missing = missing[missing > 0].sort_values(ascending=False)
            if len(missing) > 0:
                st.bar_chart(missing)
            else:
                st.success("✅ Aucune valeur manquante")

            st.subheader("Types de colonnes")
            dtypes_df = pd.DataFrame({
                "colonne": df_f.dtypes.index,
                "type":    df_f.dtypes.values.astype(str)
            })
            st.dataframe(dtypes_df, use_container_width=True)
        else:
            st.info("Aucune donnée disponible.")


# ─────────────────────────────────────────────────────────
# SECTION 3 — DASHBOARD BUSINESS
# ─────────────────────────────────────────────────────────
elif section == " Dashboard business":
    st.title(" Dashboard business — Résultats ML")

    df_clean = load_clean_data()
    if df_clean is not None:
        df_f = build_filters(df_clean)
    else:
        df_f = None

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        " KPIs",
        " Prédictions RF",
        " Segments K-Means",
        " Anomalies Z-score",
        " Optimisation"        
    ])

    with tab1:
        df_kpis = load_kpis()
        if df_kpis is not None:
            for _, row in df_kpis.iterrows():
                cols = st.columns(4)
                cols[0].metric(f"{row['colonne']} — Total",
                               f"{row['total']:,.0f}")
                cols[1].metric(f"{row['colonne']} — Moyenne",
                               f"{row['moyenne']:,.2f}")
                cols[2].metric(f"{row['colonne']} — Max",
                               f"{row['max']:,.0f}")
                cols[3].metric(f"{row['colonne']} — Min",
                               f"{row['min']:,.0f}")
            st.markdown("---")
            st.dataframe(df_kpis, use_container_width=True)
        else:
            st.info("Aucun KPI disponible — relance le pipeline.")

    with tab2:
        df_pred = load_predictions()
        if df_pred is not None:
            # Appliquer filtre client si disponible
            if df_f is not None and "identifiant_entite" in df_pred.columns \
                    and "identifiant_entite" in df_f.columns:
                clients_f = df_f["identifiant_entite"].unique()
                df_pred = df_pred[
                    df_pred["identifiant_entite"].isin(clients_f)
                ]

            st.write(f"**{len(df_pred):,} prédictions**")

            if "rf_prediction" in df_pred.columns:
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Distribution prédictions (0=sans remise / 1=avec remise)")
                    dist = df_pred["rf_prediction"].value_counts()
                    st.bar_chart(dist)
                with col2:
                    if "rf_probabilite" in df_pred.columns:
                        st.write("Distribution des probabilités de remise")
                        st.bar_chart(
                            df_pred["rf_probabilite"]
                            .round(1).value_counts().sort_index()
                        )

            st.dataframe(
                df_pred.sort_values("rf_probabilite", ascending=False)
                if "rf_probabilite" in df_pred.columns else df_pred,
                use_container_width=True, height=400
            )
        else:
            st.info("Aucune prédiction disponible — relance le pipeline.")

    with tab3:
        df_seg = load_segments()
        if df_seg is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Clients par segment**")
                if "nb_clients" in df_seg.columns:
                    st.bar_chart(
                        df_seg.set_index("kmeans_segment")["nb_clients"]
                    )
            with col2:
                st.write("**Valeur totale par segment**")
                if "valeur_totale" in df_seg.columns:
                    st.bar_chart(
                        df_seg.set_index("kmeans_segment")["valeur_totale"]
                    )
            st.markdown("---")
            st.dataframe(df_seg, use_container_width=True)
        else:
            st.info("Aucun segment disponible — relance le pipeline.")

    with tab4:
        df_anom = load_anomalies()
        if df_anom is not None:
            st.markdown(
                f'<div class="info-card">🔴 <b>{len(df_anom):,} anomalies</b>'
                f' détectées par Z-score</div>',
                unsafe_allow_html=True
            )
            st.dataframe(df_anom, use_container_width=True, height=400)

            csv_anom = df_anom.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                " Télécharger les anomalies",
                csv_anom,
                "anomalies_export.csv",
                "text/csv"
            )
        else:
            st.success("✅ Aucune anomalie détectée.")

    with tab5:
        df_optim = load_csv_safe(POWERBI_DIR / "optimization.csv")

        if df_optim is not None:

            # ── KPIs globaux ───────────────────────────────
            economie_totale = df_optim["economie_totale"].iloc[0]
            economie_pct    = df_optim["economie_pct"].iloc[0]
            cout_actuel     = df_optim["cout_actuel"].sum()
            cout_optimal    = df_optim["cout_optimal"].sum()

            col1, col2, col3 = st.columns(3)
            col1.metric(" Coût actuel",
                        f"{cout_actuel:,.0f}")
            col2.metric(" Coût optimal",
                        f"{cout_optimal:,.0f}")
            col3.metric(" Économie potentielle",
                        f"{economie_totale:,.0f}",
                        f"{economie_pct:.1f}%")

            st.markdown("---")

            # ── Graphique volume actuel vs optimal ─────────
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Volume actuel vs optimal par catégorie**")
                df_chart = df_optim.set_index("categorie")[
                    ["volume_actuel", "volume_optimal"]
                ]
                st.bar_chart(df_chart)

            with col2:
                st.write("**Économie par catégorie**")
                st.bar_chart(
                    df_optim.set_index("categorie")["economie"]
                )

            st.markdown("---")

            # ── Tableau détaillé ───────────────────────────
            st.write("**Allocation recommandée par catégorie**")

            # Ajouter colonne direction visuelle
            df_display = df_optim.copy()
            df_display["direction"] = df_display["variation_pct"].apply(
                lambda x: "▲ Augmenter" if x > 0
                          else "▼ Réduire" if x < 0
                          else "= Maintenir"
            )

            st.dataframe(
                df_display[[
                    "categorie", "prix_moyen",
                    "volume_actuel", "volume_optimal",
                    "variation_pct", "direction",
                    "economie"
                ]].rename(columns={
                    "categorie":      "Catégorie",
                    "prix_moyen":     "Prix moyen",
                    "volume_actuel":  "Volume actuel",
                    "volume_optimal": "Volume optimal",
                    "variation_pct":  "Variation (%)",
                    "direction":      "Action",
                    "economie":       "Économie"
                }),
                use_container_width=True
            )

            # ── Téléchargement ─────────────────────────────
            csv_optim = df_optim.to_csv(
                index=False
            ).encode("utf-8-sig")
            st.download_button(
                "⬇ Télécharger les recommandations",
                csv_optim,
                "optimization_export.csv",
                "text/csv"
            )

        else:
            st.info("Aucun résultat d'optimisation disponible"
                    " — relance le pipeline.")


# ─────────────────────────────────────────────────────────
# SECTION 4 — RAPPORT EDA
# ─────────────────────────────────────────────────────────
elif section == " Rapport EDA":
    st.title(" Rapport EDA — ydata-profiling")

    st.markdown(
        '<div class="info-card">Le rapport EDA est généré automatiquement'
        ' par <b>ydata-profiling</b> après chaque exécution du pipeline.'
        ' Il contient les statistiques descriptives, distributions,'
        ' corrélations et alertes sur toutes les colonnes.</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    if EDA_REPORT.exists():
        col1, col2 = st.columns([1, 2])

        with col1:
            st.success(f"✅ Rapport disponible")
            st.caption(f"Fichier : {EDA_REPORT}")

            import datetime
            mtime = datetime.datetime.fromtimestamp(
                EDA_REPORT.stat().st_mtime
            )
            st.caption(f"Généré le : {mtime.strftime('%d/%m/%Y à %H:%M')}")

            # Bouton d'ouverture dans le navigateur
            if st.button(" Ouvrir le rapport EDA dans le navigateur",
                          type="primary"):
                try:
                    abs_path = str(EDA_REPORT.resolve())
                    url = f"file:///{abs_path.replace(os.sep, '/')}"
                    import webbrowser
                    webbrowser.open(url)
                    st.success("✅ Rapport ouvert dans ton navigateur par défaut")
                except Exception as e:
                    st.error(f"❌ Impossible d'ouvrir : {e}")
                    st.code(str(EDA_REPORT.resolve()))

            st.markdown("---")

            # Bouton de téléchargement en secours
            with open(EDA_REPORT, "rb") as f:
                st.download_button(
                    " Télécharger le rapport HTML",
                    f.read(),
                    "eda_report.html",
                    "text/html"
                )

        with col2:
            st.markdown("### Contenu du rapport")
            st.markdown("""
            Le rapport ydata-profiling contient :

            - **Vue d'ensemble** — nombre de variables, lignes,
              valeurs manquantes, doublons
            - **Variables** — statistiques par colonne (min, max,
              moyenne, distribution, valeurs fréquentes)
            - **Interactions** — corrélations entre variables
            - **Valeurs manquantes** — carte visuelle des NaN
            - **Alertes** — colonnes constantes, très corrélées,
              avec beaucoup de valeurs manquantes
            """)
    else:
        st.warning("⚠️ Le rapport EDA n'a pas encore été généré.")
        st.markdown(
            '<div class="info-card">Lance le pipeline depuis le menu'
            ' de gauche — <b>eda.py</b> génère automatiquement'
            ' <code>outputs/eda_report.html</code> à chaque exécution.'
            '</div>',
            unsafe_allow_html=True
        )
        st.markdown("**Chemin attendu :** `outputs/eda_report.html`")


# ─────────────────────────────────────────────────────────
# SECTION 5 — POWER BI
# ─────────────────────────────────────────────────────────
elif section == " Power BI":
    st.title(" Power BI — Dashboard")

    st.markdown(
        '<div class="info-card">Power BI lit les fichiers CSV générés'
        ' par le pipeline dans <b>outputs/powerbi/</b>.'
        ' Ouvre le fichier .pbix avec Power BI Desktop pour voir'
        ' le dashboard mis à jour.</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    # ── Trouver les fichiers .pbix disponibles ─────────────
    pbix_files = list(Path(".").glob("**/*.pbix"))

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Fichiers Power BI")

        if pbix_files:
            for pbix in pbix_files:
                st.markdown(f" `{pbix.name}`")
                if st.button(f" Ouvrir {pbix.name}", key=str(pbix)):
                    try:
                        os.startfile(str(pbix.resolve()))
                        st.success(f"✅ Ouverture de {pbix.name}...")
                    except Exception as e:
                        st.error(f"❌ Impossible d'ouvrir : {e}")
                        st.code(str(pbix.resolve()))
        else:
            st.warning("⚠️ Aucun fichier .pbix trouvé dans le projet.")
            st.markdown(
                "Place ton fichier `.pbix` dans le dossier"
                " `sad_framework/` et relance cette page."
            )
        st.caption(
            " Pour connecter Power BI : Obtenir les données"
            " → Texte/CSV → pointer sur outputs/powerbi/"
        )

# ─────────────────────────────────────────────────────────
# FOOTER SIDEBAR
# ─────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.caption("SAD Framework — Streamlit v2.0")