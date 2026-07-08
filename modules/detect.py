import pandas as pd
import json
from pathlib import Path

# ── Chargement du schema.json ──────────────────────────────
def load_schema(schema_path="schema.json"):
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)

# ── Détection du schéma confirmé ──────────────────────────
def detect(df, active_models, schema_path="schema.json"):
    schema      = load_schema(schema_path)
    required    = schema["pipeline"]["required"]
    recommended = schema["pipeline"]["recommended"]
    optional    = schema["pipeline"]["optional"]

    detected_schema = {
        "date_columns":        [],
        "numeric_columns":     [],
        "categorical_columns": [],
        "string_columns":      [],
        "active_models":       active_models
    }

    print("\n" + "═"*55)
    print("       DÉTECTION — SAD Framework")
    print("═"*55)

    # ── 1. Classer chaque colonne par type ─────────────────
    all_cols = required + recommended + optional

    for col in all_cols:
        name = col["name"]
        typ  = col["type"]

        if name not in df.columns:
            continue

        if typ == "date":
            detected_schema["date_columns"].append(name)
        elif typ == "numeric":
            detected_schema["numeric_columns"].append(name)
        elif typ == "categorical":
            detected_schema["categorical_columns"].append(name)
        elif typ == "string":
            detected_schema["string_columns"].append(name)

    # ── 2. Afficher le schéma détecté ──────────────────────
    print("\n Schéma détecté :\n")
    print(f"   Colonnes date         : "
          f"{detected_schema['date_columns']}")
    print(f"   Colonnes numériques   : "
          f"{detected_schema['numeric_columns']}")
    print(f"   Colonnes catégorielles: "
          f"{detected_schema['categorical_columns']}")
    print(f"   Modèles actifs        : "
          f"{detected_schema['active_models']}")

    print("\n" + "═"*55 + "\n")

    return df, detected_schema