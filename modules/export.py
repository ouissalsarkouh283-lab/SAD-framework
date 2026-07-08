import pandas as pd
import json
from pathlib import Path

# ── Export principal ───────────────────────────────────────
def export(df, schema_path="schema.json",
           output_path=None, dataset_id="default"):
    if output_path is None:
        output_path = f"outputs/powerbi/{dataset_id}/"

    print("\n" + "═"*55)
    print("       EXPORT — SAD Framework")
    print("═"*55)

    # Créer le dossier si absent
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # ── 1. Export clean_data ───────────────────────────────
    clean_path = f"{output_path}clean_data.csv"
    df.to_csv(clean_path, index=False, encoding="utf-8")
    print(f"\n  ✅ clean_data.csv exporté")
    print(f"     {len(df)} lignes, {len(df.columns)} colonnes")

    # ── 2. Export prédictions Random Forest ───────────────
    rf_cols = ["rf_prediction", "rf_probabilite"]
    rf_available = [c for c in rf_cols if c in df.columns]

    if rf_available:
        base_cols = [c for c in ["identifiant_unique","identifiant_entite","date_principale","valeur_principale","categorie_principale"] if c in df.columns]

        df_rf = df[base_cols + rf_available].copy()
        rf_path = f"{output_path}predictions.csv"
        df_rf.to_csv(rf_path, index=False, encoding="utf-8")
        print(f"\n  ✅ predictions.csv exporté")
        print(f"     {len(df_rf)} lignes")
    else:
        print(f"\n  ⚠️  predictions.csv — RF non disponible")

    # ── 3. Export segments K-Means ─────────────────────────
    if "kmeans_segment" in df.columns:
        base_cols = [c for c in ["identifiant_entite","valeur_principale","quantite_principale","kmeans_segment"] if c in df.columns]

        df_seg = df[base_cols].copy()

        # Résumé par segment
        seg_summary = df_seg.groupby("kmeans_segment").agg(
            nb_clients=("identifiant_entite", "nunique"),
            valeur_totale=("valeur_principale", "sum"),
            valeur_moyenne=("valeur_principale", "mean"),
            qte_moyenne=("quantite_principale", "mean")
        ).round(2).reset_index()

        seg_path = f"{output_path}segments.csv"
        seg_summary.to_csv(seg_path, index=False,
                           encoding="utf-8")
        print(f"\n  ✅ segments.csv exporté")
        print(f"     {len(seg_summary)} segments")
    else:
        print(f"\n  ⚠️  segments.csv — K-Means non disponible")

    # ── 4. Export anomalies Z-score ────────────────────────
    anomalie_cols = [c for c in df.columns if c.startswith("anomalie_")]

    if anomalie_cols:
        # Garder seulement les lignes avec au moins
        # une anomalie détectée
        mask = df[anomalie_cols].any(axis=1)
        base_cols = [c for c in ["identifiant_unique","identifiant_entite","date_principale","valeur_principale","taux"] if c in df.columns]

        zscore_cols = [c for c in df.columns
                       if c.startswith("zscore_")]

        df_anom = df[mask][base_cols + anomalie_cols + zscore_cols].copy()

        anom_path = f"{output_path}anomalies.csv"
        df_anom.to_csv(anom_path, index=False,
                       encoding="utf-8")
        print(f"\n  ✅ anomalies.csv exporté")
        print(f"     {len(df_anom)} anomalies détectées")
    else:
        print(f"\n  ⚠️  anomalies.csv — Z-score non disponible")

    # ── 5. Export KPIs ─────────────────────────────────────
    kpis = {}

    numeric_cols = [c for c in ["valeur_principale","quantite_principale","taux"] if c in df.columns]

    for col in numeric_cols:
        kpis[col] = {
            "total":   round(df[col].sum(), 2),
            "moyenne": round(df[col].mean(), 2),
            "max":     round(df[col].max(), 2),
            "min":     round(df[col].min(), 2),
            "nb_lignes": len(df)
        }

    if kpis:
        df_kpis = pd.DataFrame(kpis).T.reset_index()
        df_kpis.columns = ["colonne", "total", "moyenne",
                           "max", "min", "nb_lignes"]
        kpis_path = f"{output_path}kpis.csv"
        df_kpis.to_csv(kpis_path, index=False,
                       encoding="utf-8")
        print(f"\n  ✅ kpis.csv exporté")
    
     # ── Export optimisation ────────────────────────────────
    optim_src = Path("outputs/temp/optimization.csv")
    if optim_src.exists():
        import shutil
        shutil.copy(optim_src, f"{output_path}optimization.csv")
        print(f"\n  ✅ optimization.csv exporté")
    else:
        print(f"\n  ⚠️  optimization.csv — non disponible")
    return output_path