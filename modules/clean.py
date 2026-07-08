import pandas as pd
import numpy as np
import json

# ── Chargement du schema.json ──────────────────────────────
def load_schema(schema_path="schema.json"):
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)

# ── Nettoyage des colonnes dates ───────────────────────────
def clean_dates(df, date_columns):
    print("\n   Nettoyage colonnes dates...")
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            # Extraire année et mois
            df[f"{col}_annee"] = df[col].dt.year
            df[f"{col}_mois"]  = df[col].dt.month
            print(f"      {col} → converti + année/mois extraits")
    return df

# ── Nettoyage des colonnes numériques ─────────────────────
def clean_numerics(df, numeric_columns):
    print("\n   Nettoyage colonnes numériques...")
    for col in numeric_columns:
        if col in df.columns:
            # Convertir en numérique si besoin
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # Remplir manquants avec médiane
            mediane = df[col].median()
            df[col] = df[col].fillna(mediane)
            # Supprimer valeurs négatives impossibles
            df[col] = df[col].clip(lower=0)
            print(f"      {col} → médiane={mediane:.2f}, "
                  f"négatifs supprimés")
    return df

# ── Nettoyage des colonnes catégorielles ──────────────────
def clean_categoricals(df, categorical_columns):
    print("\n   Nettoyage colonnes catégorielles...")
    for col in categorical_columns:
        if col in df.columns:
            # Remplir manquants avec mode
            mode = df[col].mode()[0]
            df[col] = df[col].fillna(mode)
            # Supprimer espaces inutiles
            df[col] = df[col].str.strip()
            # Mettre en majuscule première lettre
            df[col] = df[col].str.title()
            print(f"      {col} → mode='{mode}', "
                  f"espaces nettoyés")
    return df

# ── Suppression des doublons ───────────────────────────────
def remove_duplicates(df):
    avant = len(df)
    df = df.drop_duplicates()
    apres = len(df)
    supprimes = avant - apres
    print(f"\n    Doublons supprimés : {supprimes}")
    return df

# ── Suppression des colonnes vides ───────────────────────────────
def remove_useless_columns(df, threshold=0.90):
    avant = len(df.columns)

    # Colonnes constantes
    constantes = [col for col in df.columns
                  if df[col].nunique() <= 1]

    # Colonnes trop vides (> 90% manquants)
    trop_vides = [col for col in df.columns
                  if df[col].isnull().mean() > threshold]

    # Colonnes à supprimer
    a_supprimer = list(set(constantes + trop_vides))

    df = df.drop(columns=a_supprimer)

    print(f"\n  🗑️  Colonnes supprimées : {len(a_supprimer)}")
    print(f"     Constantes  : {len(constantes)}")
    print(f"     Trop vides  : {len(trop_vides)}")
    print(f"     Colonnes restantes : {len(df.columns)}")

    return df
# ── Rapport outliers (sans correction) ────────────────────
def report_outliers(df, numeric_cols):
    print("\n  📊 Rapport outliers détectés :")
    print("     (non corrigés — traités par Z-score)\n")

    for col in numeric_cols:
        if col not in df.columns:
            continue

        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1

        limite_basse = Q1 - 1.5 * IQR
        limite_haute = Q3 + 1.5 * IQR

        outliers = (
            (df[col] < limite_basse) |
            (df[col] > limite_haute)
        ).sum()

        pct = (outliers / len(df)) * 100

        if outliers > 0:
            print(f"     ⚠️  {col} : "
                  f"{outliers} outliers "
                  f"({pct:.2f}%)")
            print(f"        Zone normale : "
                  f"[{limite_basse:.2f} → "
                  f"{limite_haute:.2f}]")
        else:
            print(f"     ✅ {col} : aucun outlier")

    return df

# ── Nettoyage principal ────────────────────────────────────
def clean(df, detected_schema, schema_path="schema.json"):

    print("\n" + "═"*55)
    print("       NETTOYAGE — SAD Framework")
    print("═"*55)

    date_cols        = detected_schema["date_columns"]
    numeric_cols     = detected_schema["numeric_columns"]
    categorical_cols = detected_schema["categorical_columns"]

    # Rapport avant nettoyage
    print(f"\n  Avant nettoyage :")
    print(f"     Lignes        : {len(df)}")
    print(f"     Colonnes      : {len(df.columns)}")
    print(f"     Valeurs nulles: {df.isnull().sum().sum()}")

    # Nettoyage dans l'ordre
    df = remove_duplicates(df)
    df = remove_useless_columns(df)
    df = clean_dates(df, date_cols)
    df = clean_numerics(df, numeric_cols)
    df = report_outliers(df, numeric_cols)  
    df = clean_categoricals(df, categorical_cols)

    # Rapport après nettoyage
    print(f"\n  Après nettoyage :")
    print(f"     Lignes        : {len(df)}")
    print(f"     Colonnes      : {len(df.columns)}")
    print(f"     Valeurs nulles: {df.isnull().sum().sum()}")

    print("\n  Nettoyage terminé")
    print("\n" + "═"*55 + "\n")

# ── Sauvegarde data/clean/ ────────────────────────────
    import os
    os.makedirs("data/clean", exist_ok=True)
    df.to_csv("data/clean/clean_data.csv", index=False, encoding="utf-8-sig")
    print(f"  💾 data/clean/clean_data.csv → {len(df)} lignes sauvegardées\n")
    return df
