import pandas as pd
import json
from pathlib import Path
from ydata_profiling import ProfileReport

# ── EDA principale ─────────────────────────────────────────
def eda(df, detected_schema, output_path="outputs/eda_report.html"):

    print("\n" + "═"*55)
    print("       EDA — SAD Framework")
    print("═"*55)

    date_cols        = detected_schema["date_columns"]
    numeric_cols     = detected_schema["numeric_columns"]
    categorical_cols = detected_schema["categorical_columns"]

    # ── 1. Statistiques descriptives ──────────────────────
    print("\n   Statistiques descriptives :\n")

    if numeric_cols:
        print(df[numeric_cols].describe().round(3).to_string())

    # ── 2. Valeurs manquantes ─────────────────────────────
    print("\n\n   Valeurs manquantes :\n")
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        print("     Aucune valeur manquante")
    else:
        for col, count in missing.items():
            pct = (count / len(df)) * 100 #pourcentage de valeures manquantes
            print(f"     • {col} : {count} ({pct:.1f}%)")

    # ── 3. Distributions catégorielles ───────────────────
    print("\n   Distributions catégorielles :\n")
    for col in categorical_cols:
        if col in df.columns:
            top5 = df[col].value_counts().head(5)
            print(f"     {col} :")
            for val, count in top5.items():
                print(f"       • {val} : {count}")

    # ── 4. Corrélations numériques ────────────────────────
    if len(numeric_cols) >= 2:
        print("\n   Corrélations :\n")
        corr = df[numeric_cols].corr().round(2)
        print(corr.to_string())

    # ── 5. Rapport ydata-profiling ────────────────────────
    print("\n\n  Génération rapport ydata-profiling...")

    # Crée le dossier outputs si absent
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    profile = ProfileReport(
        df,
        title="SAD Framework — EDA Report",
        explorative=True,
        minimal=False
    )
    profile.to_file(output_path)
    print(f"    Rapport sauvegardé → {output_path}")

    print("\n" + "═"*55 + "\n")

    return profile