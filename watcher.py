"""
watcher.py
─────────────────────────────────────────────────────────
Surveillance temps réel de la table raw_data via SurrealDB.

MODE PRODUCTION :
  - Détecte les nouvelles lignes insérées dans raw_data
  - Surveille TOUS les dataset_id en parallèle
  - DEBOUNCE de 15s : attend 15s sans nouvelle ligne pour un
    dataset_id avant de relancer le pipeline (évite de relancer
    50 fois si 50 lignes arrivent d'un coup)
  - Si plusieurs dataset_id reçoivent des lignes en même temps,
    chacun déclenche son propre run_pipeline() indépendamment
"""

import time
from datetime import datetime
from threading import Thread, Lock

from db_config import get_connection
from pipeline import run_pipeline

DEBOUNCE_SECONDS = 15


def format_timestamp():
    return datetime.now().strftime("%H:%M:%S")


# ── État partagé entre threads ──────────────────────────────
# Pour chaque dataset_id : timestamp de la dernière ligne détectée
last_seen = {}
lock = Lock()


def handle_new_record(record):
    """
    Appelée à chaque nouvelle ligne détectée dans raw_data.
    Met à jour le timestamp du dataset_id concerné — le
    déclenchement réel se fait dans debounce_worker().
    """
    dataset_id  = record.get("dataset_id", "INCONNU")
    ingested_at = record.get("ingested_at", "?")

    print(f"  🟢 [{format_timestamp()}] Nouvelle ligne détectée")
    print(f"     dataset_id   : {dataset_id}")
    print(f"     ingested_at  : {ingested_at}")

    with lock:
        last_seen[dataset_id] = time.time()


def debounce_worker():
    """
    Tourne en parallèle de la boucle de détection.
    Vérifie toutes les 2 secondes si un dataset_id est
    "calme" depuis DEBOUNCE_SECONDS — si oui, déclenche
    le pipeline pour ce dataset_id et l'enlève de la liste
    d'attente.
    """
    triggered_at = {}  # évite de relancer plusieurs fois le même calme

    while True:
        time.sleep(2)
        now = time.time()

        with lock:
            candidates = list(last_seen.items())

        for dataset_id, last_time in candidates:
            elapsed = now - last_time

            if elapsed >= DEBOUNCE_SECONDS:
                # Déjà déclenché pour ce timestamp précis ?
                if triggered_at.get(dataset_id) == last_time:
                    continue

                print(f"\n  ⏱️  [{format_timestamp()}] Debounce"
                      f" écoulé pour '{dataset_id}'"
                      f" ({elapsed:.0f}s sans nouvelle ligne)")
                print(f"  🚀 Lancement automatique du pipeline"
                      f" — dataset_id={dataset_id}\n")

                try:
                    run_pipeline(dataset_id=dataset_id)
                except Exception as e:
                    print(f"\n  ❌ Erreur pendant l'exécution"
                          f" du pipeline pour '{dataset_id}'"
                          f" : {e}\n")

                triggered_at[dataset_id] = last_time

                with lock:
                    # Retirer seulement si aucune nouvelle ligne
                    # n'est arrivée entre-temps
                    if last_seen.get(dataset_id) == last_time:
                        del last_seen[dataset_id]


def watch_raw_data(poll_interval=3):
    """
    Surveille raw_data en continu (polling) et lance le
    debounce_worker en parallèle dans un thread séparé.
    """
    print("\n" + "═" * 55)
    print("       WATCHER — SAD Framework (mode PRODUCTION)")
    print("═" * 55)
    print(f"\n  👀 Surveillance de raw_data toutes les"
          f" {poll_interval}s")
    print(f"     Tous les dataset_id sont surveillés")
    print(f"     Debounce : {DEBOUNCE_SECONDS}s avant relance"
          f" automatique du pipeline")
    print(f"     Chaque dataset_id est traité indépendamment\n")

    db = get_connection()

    # ── Démarrer le worker de debounce en arrière-plan ──────
    worker = Thread(target=debounce_worker, daemon=True)
    worker.start()

    # ── Initialiser avec les ID déjà connus ────────────────
    try:
        existing = db.query("SELECT id FROM raw_data;")
        known_ids = _extract_ids(existing)
        print(f"  📌 {len(known_ids)} lignes déjà présentes"
              f" (ignorées au démarrage)\n")
    except Exception:
        known_ids = set()
        print(f"  📌 Table raw_data vide ou inexistante"
              f" — démarrage à zéro\n")

    # ── Boucle de polling (détection) ──────────────────────
    try:
        while True:
            try:
                result = db.query("SELECT * FROM raw_data;")
                records = _extract_records(result)

                new_records = [
                    r for r in records
                    if str(r.get("id")) not in known_ids
                ]

                for record in new_records:
                    handle_new_record(record)
                    known_ids.add(str(record.get("id")))

                if not new_records:
                    print(f"  ⏳ [{format_timestamp()}]"
                          f" Aucune nouvelle ligne"
                          f" ({len(known_ids)} connues)",
                          end="\r")

            except Exception as e:
                print(f"\n  ⚠️  Erreur pendant la"
                      f" surveillance : {e}")

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print("\n\n  🛑 Watcher arrêté manuellement"
              " (Ctrl+C)\n")


# ── Fonctions utilitaires d'extraction ──────────────────────
def _extract_records(query_result):
    """
    Gère les différents formats de retour possibles du SDK
    SurrealDB (même logique que dans pipeline.py).
    """
    if isinstance(query_result, list) and len(query_result) > 0:
        first = query_result[0]
        if isinstance(first, dict) and "result" not in first:
            return query_result
        if isinstance(first, dict) and "result" in first:
            return first["result"] or []
    return []


def _extract_ids(query_result):
    records = _extract_records(query_result)
    return set(str(r.get("id")) for r in records if "id" in r)


if __name__ == "__main__":
    watch_raw_data(poll_interval=3)