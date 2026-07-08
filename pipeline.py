"""
pipeline.py
─────────────────────────────────────────────────────────
Orchestrateur principal du SAD Framework.
Lit les données brutes depuis SurrealDB (raw_data), applique
la chaîne complète de traitement déjà validée (rename_columns
→ validate → detect → clean → models), puis écrit :
  - les données nettoyées dans clean_data
  - les résultats ML (segments, anomalies, prédictions) dans results
Stratégie de mise à jour : "vider et réécrire" — à chaque
exécution, les anciennes lignes de ce dataset_id dans
clean_data et results sont supprimées puis remplacées par
les nouvelles.
"""
import pandas as pd
from db_config import get_connection
from rename_columns import rename_columns
from modules.validate import validate
from modules.detect import detect
from modules.clean import clean
from modules.models import run_models
from modules.export import export


# ── Paramètres à changer selon le dataset traité ────────────
MAPPING_PATH = "column_mapping.json"
SCHEMA_PATH  = "schema.json"

def _get_latest_dataset_id():
    """
    Récupère automatiquement le dataset_id le plus récemment
    ingéré dans raw_data, basé sur ingested_at.
    """
    try:
        db = get_connection()
        result = db.query(
            "SELECT dataset_id, ingested_at FROM raw_data "
            "ORDER BY ingested_at DESC LIMIT 1;"
        )
        records = []
        if isinstance(result, list) and len(result) > 0:
            first = result[0]
            if isinstance(first, dict) and "result" not in first:
                records = result
            elif isinstance(first, dict) and "result" in first:
                records = first["result"] or []

        if records and len(records) > 0:
            dataset_id = records[0].get("dataset_id")
            ingested_at = records[0].get("ingested_at")
            print(f"     ✅ dataset_id détecté automatiquement : "
                  f"{dataset_id} (ingéré le {ingested_at})")
            return dataset_id
        else:
            print("     ❌ Aucun dataset_id trouvé dans raw_data")
            return None
    except Exception as e:
        print(f"     ❌ Erreur détection dataset_id : {e}")
        return None
    
def _records_from_raw(dataset_id):
    db = get_connection()
    result = db.query(
        "SELECT * FROM raw_data WHERE dataset_id = $dataset_id;",
        {"dataset_id": dataset_id}
    )

    if isinstance(result, list) and len(result) > 0:
        first = result[0]
        if isinstance(first, dict) and "result" not in first:
            return result
        if isinstance(first, dict) and "result" in first:
            return first["result"] or []
    return []

def _overwrite_table(table_name, dataset_id, records):
    """
    Supprime les anciennes lignes de ce dataset_id dans `table_name`,
    puis insère les nouvelles. Stratégie "vider et réécrire".
    """
    db = get_connection()

    try:
        db.query(
            f"DELETE {table_name} WHERE dataset_id = $dataset_id;",
            {"dataset_id": dataset_id}
        )
    except Exception:
        print(f"     ℹ️  Table '{table_name}' inexistante"
              f" (première écriture)")

    if not records:
        print(f"     ⚠️  Aucune ligne à insérer dans {table_name}")
        return

    batch_size = 500
    total = len(records)
    inserted = 0

    for i in range(0, total, batch_size):
        batch = records[i:i + batch_size]
        db.insert(table_name, batch)
        inserted += len(batch)
        print(f"     {inserted}/{total} lignes écrites dans"
              f" {table_name}...", end="\r")

    print(f"\n     ✅ {inserted} lignes dans {table_name}")
def run_pipeline(dataset_id=None,
                  mapping_path=MAPPING_PATH,
                  schema_path=SCHEMA_PATH):
    
    # ── Détection automatique du dataset_id ──────────────
    if dataset_id is None:
        print(f"\n  🔍 Détection automatique du dataset_id...")
        dataset_id = _get_latest_dataset_id()
        if dataset_id is None:
            print("     ❌ Impossible de détecter le dataset_id"
                  " — pipeline annulé")
            return None, None

    print("\n" + "█" * 55)
    print(f"   PIPELINE COMPLET — dataset_id = {dataset_id}")
    print("█" * 55)

    print("\n" + "█" * 55)
    print(f"   PIPELINE COMPLET — dataset_id = {dataset_id}")
    print("█" * 55)
    # ── 1. Lire les données brutes depuis SurrealDB ────────
    print(f"\n  📥 Lecture de raw_data (dataset_id={dataset_id})...")
    records = _records_from_raw(dataset_id)
    if not records:
        print(f"     ❌ Aucune ligne trouvée pour ce dataset_id"
              f" — pipeline annulé")
        return None, None
    df = pd.DataFrame(records)
    # Colonnes internes SurrealDB à retirer avant traitement
    for col in ["id", "dataset_id", "ingested_at"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    print(f"     {df.shape[0]} lignes, {df.shape[1]} colonnes"
          f" chargées depuis SurrealDB")
    # ── 2. Renommer les colonnes ─────────────────────────────
    df = rename_columns(df, mapping_path=mapping_path)
    # ── 3. Valider la structure ──────────────────────────────
    success, active_models = validate(df)
    if not success:
        print("\n  ❌ Validation échouée — pipeline arrêté")
        return None, None
    # ── 4. Détecter les types ────────────────────────────────
    df, detected_schema = detect(df, active_models, schema_path=schema_path)
    # ── 5. Nettoyer ───────────────────────────────────────────
    df = clean(df, detected_schema)
    print(f"\n  ✅ Données prêtes : {df.shape[0]} lignes,"
          f" {df.shape[1]} colonnes")
    # ── 6. Écrire les données nettoyées dans clean_data ─────
    print(f"\n  📤 Écriture dans clean_data...")
    df_clean_export = df.copy()
    df_clean_export["dataset_id"] = dataset_id
    clean_records = df_clean_export.where(
        pd.notnull(df_clean_export), None
    ).to_dict(orient="records")
    _overwrite_table("clean_data", dataset_id, clean_records)
    # ── 7. Exécuter les modèles ML ───────────────────────────
    df, results = run_models(
        df, detected_schema,
        schema_path=schema_path,
        mapping_path=mapping_path
    )
    # ── 8. Écrire les résultats ML dans results ─────────────
    print(f"\n  📤 Écriture dans results...")
    df_results_export = df.copy()
    df_results_export["dataset_id"] = dataset_id
    results_records = df_results_export.where(
        pd.notnull(df_results_export), None
    ).to_dict(orient="records")

    _overwrite_table("results", dataset_id, results_records)
    print("\n" + "█" * 55)
    print("   ✅ PIPELINE TERMINÉ")
    print("█" * 55 + "\n")

    # ── 9. Export Power BI ───────────────────────────────────
    print(f"\n  📤 Export Power BI...")
    export(df, schema_path=schema_path, dataset_id=dataset_id)

    return df, results
if __name__ == "__main__":
    run_pipeline()