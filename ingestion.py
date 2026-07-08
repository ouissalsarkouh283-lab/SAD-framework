"""
ingestion.py
─────────────────────────────────────────────────────────
Ingestion unique d'un fichier CSV brut dans la table
SurrealDB `raw_data`.

Principe :
- raw_data n'est JAMAIS modifiée après ingestion (lecture seule)
- Chaque ligne du CSV devient un enregistrement SurrealDB
- Un identifiant de batch (dataset_id) permet de tracer
  de quel dataset / quelle ingestion provient chaque ligne
"""

import pandas as pd
from datetime import datetime, timezone

from db_config import get_connection


def _extract_count(query_result):
    """
    Extrait le nombre trouvé par une requête `... GROUP ALL`,
    en gérant plusieurs structures de retour possibles selon
    la version du SDK SurrealDB utilisée.
    """
    if not query_result:
        return 0

    first = query_result[0]

    # Cas 1 — déjà dépouillé : {"count": N}
    if isinstance(first, dict) and "count" in first:
        return first["count"]

    # Cas 2 — ancienne enveloppe : {"result": [{"count": N}], ...}
    if isinstance(first, dict) and "result" in first:
        inner = first["result"]
        if isinstance(inner, list) and len(inner) > 0:
            return inner[0].get("count", 0)

    return 0


def ingest_csv(csv_path, dataset_id, encoding="utf-8", batch_size=500):
    print("\n" + "═" * 55)
    print("       INGESTION — SAD Framework")
    print("═" * 55)

    # ── 1. Charger le CSV ────────────────────────────────────
    print(f"\n  📂 Lecture de {csv_path}...")
    df = pd.read_csv(csv_path, encoding=encoding, low_memory=False)
    print(f"     {df.shape[0]} lignes, {df.shape[1]} colonnes")

    df = df.where(pd.notnull(df), None)

    # ── 2. Vérifier si ce dataset_id a déjà été ingéré ──────
    db = get_connection()

    try:
        existing = db.query(
            "SELECT count() FROM raw_data WHERE dataset_id = $dataset_id GROUP ALL;",
            {"dataset_id": dataset_id}
        )
        count = _extract_count(existing)
        print(f"     🔍 Lignes déjà présentes pour ce dataset_id : {count}")
        already_ingested = count > 0

    except Exception as e:
        print(f"\n     ℹ️  Table 'raw_data' inexistante ou requête"
              f" impossible ({type(e).__name__}) — traité comme"
              f" première ingestion")
        already_ingested = False

    if already_ingested:
        print(f"\n     ⚠️  Le dataset_id '{dataset_id}' existe déjà"
              f" dans raw_data.")
        print(f"       ❌ Ingestion annulée.Risque des doublons.")
        return False

    # ── 3. Préparer les enregistrements ─────────────────────
    timestamp = datetime.now(timezone.utc).isoformat()
    records = df.to_dict(orient="records")

    for record in records:
        record["dataset_id"]  = dataset_id
        record["ingested_at"] = timestamp

    print(f"\n  📥 Insertion dans raw_data...")
    print(f"     dataset_id : {dataset_id}")

    # ── 4. Insertion par lots ───────────────────────────────
    total = len(records)
    inserted = 0

    for i in range(0, total, batch_size):
        batch = records[i:i + batch_size]
        db.insert("raw_data", batch)
        inserted += len(batch)
        print(f"     {inserted}/{total} lignes insérées...", end="\r")

    print(f"\n     ✅ {inserted} lignes insérées dans raw_data")

    print("\n" + "═" * 55)
    print("       ✅ INGESTION TERMINÉE")
    print("═" * 55 + "\n")

    return True


if __name__ == "__main__":
    ingest_csv(csv_path="data/raw/lbl.csv",dataset_id="lbl_2024",encoding="utf-8")