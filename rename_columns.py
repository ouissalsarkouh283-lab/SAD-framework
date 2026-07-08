import json
import pandas as pd

# ── Chargement du mapping ──────────────────────────────────
def load_mapping(mapping_path="column_mapping.json"):
    with open(mapping_path, "r", encoding="utf-8") as f:
        return json.load(f)

# ── Renommage des colonnes ─────────────────────────────────
def rename_columns(df, mapping_path="column_mapping.json"):
    config = load_mapping(mapping_path)
    mapping = config["columns"] 
    
    print("\n" + "═"*55)
    print("       RENOMMAGE — SAD Framework")
    print("═"*55)

    renamed_count = 0
    for original, standard in mapping.items():
        if original in df.columns:
            print(f"  ✅ {original} → {standard}")
            renamed_count += 1
        else:
            print(f"  ⚠️  {original} → absente du dataset")

    df = df.rename(columns=mapping)

    print(f"\n  📊 {renamed_count}/{len(mapping)} colonnes renommées")
    print("\n" + "═"*55 + "\n")

    return df