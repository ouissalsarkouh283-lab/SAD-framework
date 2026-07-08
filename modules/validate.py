import json
import pandas as pd
from pathlib import Path

# ─ Chargement du schema.json ─
def load_schema(schema_path="schema.json"):
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)

# ─ Vérification des types ─
def check_type(series, expected_type):
    if expected_type == "date":
        try:
            pd.to_datetime(series)
            return True
        except:
            return False
    elif expected_type == "numeric":
        return pd.api.types.is_numeric_dtype(series)
    elif expected_type == "categorical":
        return pd.api.types.is_object_dtype(series)
    elif expected_type == "string":
        return pd.api.types.is_object_dtype(series)
    return False

# ─ Validation principale ─
def validate(df, schema_path="schema.json"):
    schema   = load_schema(schema_path)
    required = schema["pipeline"]["required"]
    recommended = schema["pipeline"]["recommended"]
    models   = schema["models"]

    errors   = []
    warnings = []
    info     = []
    active_models = []

    print("\n" + "═"*55)
    print("       VALIDATION — SAD Framework")
    print("═"*55)

    # ─ 1. Colonnes obligatoires ─
    print("\n Vérification colonnes obligatoires...")
    for col in required:
        name = col["name"]
        typ  = col["type"]

        if name not in df.columns:
            errors.append(f"Colonne manquante : '{name}' ({typ} requis)")
        else:
            if not check_type(df[name], typ):
                errors.append(
                    f"Type incorrect : '{name}' "
                    f"→ trouvé '{df[name].dtype}', "
                    f"attendu '{typ}'"
                )
            else:
                print(f"   {name}")

    # ─ 2. Colonnes recommandées ─
    print("\n Vérification colonnes recommandées...")
    for col in recommended:
        name = col["name"]
        typ  = col["type"]

        if name not in df.columns:
            warnings.append(f"Colonne recommandée absente : '{name}'")
            print(f"    {name} → absente")
        else:
            print(f"   {name}")

    # ─ 3. Colonnes par modèle ML ─
    print("\n Vérification modèles ML...")
    for model_name, model_info in models.items():
        required_cols = model_info["requires"]
        missing = [c for c in required_cols if c not in df.columns]

        if missing:
            info.append(
                f"Modèle '{model_name}' désactivé "
                f"→ colonnes manquantes : {missing}"
            )
            print(f"    {model_name} → désactivé")
        else:
            active_models.append(model_name)
            print(f"   {model_name} → activé")

    # ─ Rapport final ─
    print("\n" + "═"*55)

    if errors:
        print("\n VALIDATION ÉCHOUÉE\n")
        print("Colonnes obligatoires manquantes ou incorrectes :")
        for e in errors:
            print(f"   • {e}")
        print("\nColonnes trouvées dans votre dataset :")
        print(f"   {list(df.columns)}")
        print("\nACTION : Corrigez votre dataset dans DBeaver")
        print("         et relancez la validation.")
        print("\n" + "═"*55 + "\n")
        return False, []

    else:
        print("\n VALIDATION RÉUSSIE\n")

        if warnings:
            print("  Avertissements :")
            for w in warnings:
                print(f"   • {w}")

        if info:
            print("\n Modèles ML :")
            for i in info:
                print(f"   • {i}")

        print(f"\n Modèles actifs : {active_models}")
        print("\n" + "═"*55 + "\n")
        return True, active_models
