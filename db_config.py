"""
db_config.py
─────────────────────────────────────────────────────────
Configuration centralisée de connexion à SurrealDB.
Toute la base de code (ingestion.py, pipeline.py, watcher.py)
importe get_connection() depuis ce fichier — un seul endroit
à modifier si l'adresse ou les identifiants changent.
"""
from surrealdb import Surreal
SURREAL_CONFIG = {
    "url": "http://127.0.0.1:8000",   
    "namespace": "sad_framework",
    "database":  "main",
    "username":  "root",
    "password":  "root",
}
def get_connection():
    """
    Ouvre une connexion SurrealDB prête à l'emploi
    (signin + sélection namespace/database déjà faits).
    """
    db = Surreal(SURREAL_CONFIG["url"])
    db.signin({
        "username": SURREAL_CONFIG["username"],
        "password": SURREAL_CONFIG["password"],
    })
    db.use(SURREAL_CONFIG["namespace"], SURREAL_CONFIG["database"])
    return db
def test_connection():
    """Petit test de fumée — à lancer une fois pour vérifier que tout fonctionne."""
    try:
        db = get_connection()
        result = db.query("RETURN 1 + 1;")
        print("✅ Connexion SurrealDB réussie")
        print(f"   Test de requête : {result}")
        return True
    except Exception as e:
        print(f"❌ Connexion SurrealDB échouée : {e}")
        return False
if __name__ == "__main__":
    test_connection()