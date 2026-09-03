# SAD Framework
## Systeme Automatise d'Analyse de Donnees — Machine Learning

Framework generique de traitement et d'analyse de donnees d'entreprise, integrant un pipeline automatise, 4 modeles de machine learning, un systeme de surveillance temps reel et un dashboard interactif.

---

## Objectif

Le SAD Framework permet de transformer n'importe quelle base de donnees d'entreprise en un pipeline complet d'analyse et de machine learning, reutilisable sans modifier le code pour differentes entreprises et datasets.

Le seul fichier qui change entre deux entreprises est `column_mapping.json` — tout le reste est automatique.

Teste sur :
- `lbl_2024` — Distribution de pieces automobiles (40 022 lignes, 67 colonnes)
- `auto_sales_2024` — Ventes automobiles internationales (2 747 lignes, 20 colonnes)

---

## Architecture

```
BDD Entreprise (noms originaux : datebl, puart, qte...)
        |
column_mapping.json  (traduction des noms — une seule fois par entreprise)
        |
ingestion.py  →  SurrealDB (raw_data)
        |
watcher.py  (surveillance temps reel — debounce 15s)
        |
pipeline.py orchestre :
  rename_columns → validate → detect → clean → models → export
        |
SurrealDB (clean_data + results) + outputs/powerbi/
        |
Streamlit (dashboard technique) + Power BI (dashboard business)
```

---

## Modeles ML integres

| Modele | Type | Utilite |
|--------|------|---------|
| Random Forest | Supervise | Prediction binaire (ex: remise oui/non, commande forte valeur) |
| K-Means | Non supervise | Segmentation clients (Elbow Method automatique) |
| Z-score | Statistique | Detection d'anomalies et fraudes |
| scipy.linprog | Optimisation | Allocation optimale des ressources par categorie |

---

## Stack technologique

| Composant | Technologie |
|-----------|-------------|
| Langage | Python 3.11 |
| Base de donnees | SurrealDB (surrealkv — persistant) |
| Machine Learning | scikit-learn, scipy |
| MLOps | MLflow 3.14.0 (SQLite backend) |
| Dashboard technique | Streamlit |
| Dashboard business | Power BI Desktop |
| Analyse exploratoire | ydata-profiling |
| Exploration BDD | DBeaver |

---

## Structure du projet

```
sad_framework/
|
|-- schema.json              # Template universel — fixe, ne change jamais
|-- column_mapping.json      # Mapping colonnes — specifique a chaque entreprise
|-- db_config.py             # Connexion centralisee SurrealDB
|-- ingestion.py             # CSV → SurrealDB raw_data
|-- pipeline.py              # Orchestrateur principal (9 etapes)
|-- watcher.py               # Surveillance temps reel + debounce 15s
|-- rename_columns.py        # Renommage automatique des colonnes
|-- app.py                   # Dashboard Streamlit
|
|-- modules/
|   |-- validate.py          # Validation structure et types
|   |-- detect.py            # Detection automatique des types
|   |-- clean.py             # Nettoyage adaptatif
|   |-- eda.py               # Analyse exploratoire (ydata-profiling)
|   |-- visualize.py         # Graphiques automatiques
|   |-- models.py            # 4 modeles ML + MLflow tracking
|   └-- export.py            # Export CSV pour Power BI
|
|-- data/
|   |-- raw/                 # Fichiers CSV originaux
|   └-- clean/               # Donnees nettoyees
|
└-- outputs/
    |-- plots/               # Graphiques generes
    |-- powerbi/
    |   |-- lbl_2024/        # CSV exports dataset 1
    |   └-- auto_sales_2024/ # CSV exports dataset 2
    └-- mlflow.db            # Historique des runs MLflow
```

---

## Installation

### Prerequis
- Python 3.11+
- SurrealDB (https://surrealdb.com/install)
- Power BI Desktop (optionnel)

### Installation des dependances

```bash
pip install -r requirements.txt
```

---

## Lancement

### 1. Demarrer SurrealDB
```bash
surreal start --user root --pass root --bind 127.0.0.1:8000 surrealkv:sad_db.db
```

### 2. Ingerer les donnees
```bash
python ingestion.py
```

### 3. Lancer le pipeline
```bash
python pipeline.py
```

### 4. Lancer le watcher (temps reel)
```bash
python watcher.py
```

### 5. Lancer le dashboard Streamlit
```bash
python -m streamlit run app.py
```

### 6. Lancer l'interface MLflow
```bash
python -m mlflow ui --backend-store-uri sqlite:///outputs/mlflow.db
```

---

## Utiliser un nouveau dataset

Pour adapter le framework a une nouvelle entreprise, deux etapes seulement :

### Etape 1 — Mettre a jour column_mapping.json
```json
{
  "dataset_id": "nouveau_dataset_2024",
  "columns": {
    "col_originale": "date_principale",
    "col_originale_2": "valeur_principale",
    "col_originale_3": "quantite_principale",
    "col_originale_4": "categorie_principale",
    "col_originale_5": "identifiant_entite",
    "col_originale_6": "identifiant_unique"
  },
  "target_definition": {
    "source_column": "valeur_principale",
    "rule": "greater_than",
    "value": 1000
  }
}
```

### Etape 2 — Ingerer et lancer
```bash
python ingestion.py
python pipeline.py
```

Le pipeline detecte automatiquement le dernier dataset_id ingere — plus besoin de modifier pipeline.py.

---

## Outputs generes

Apres chaque execution du pipeline, les fichiers suivants sont generes dans `outputs/powerbi/{dataset_id}/` :

| Fichier | Contenu |
|---------|---------|
| clean_data.csv | Donnees nettoyees completes |
| predictions.csv | Predictions Random Forest |
| segments.csv | Segments clients K-Means |
| anomalies.csv | Anomalies detectees Z-score |
| kpis.csv | KPIs resumes |
| optimization.csv | Recommandations d'optimisation des couts |

---

## MLOps — MLflow

Chaque entrainement de modele est automatiquement trace dans MLflow :

```bash
python -m mlflow ui --backend-store-uri sqlite:///outputs/mlflow.db
# Ouvrir : http://localhost:5000
```

Metriques trackees :
- Random Forest : accuracy, auc_roc, importance par feature
- K-Means : silhouette, davies_bouldin, inertie
- Z-score : total_anomalies, taux_global
- Optimisation : cout_actuel, cout_optimal, economie_pct

---

## Dashboard Streamlit

```bash
python -m streamlit run app.py
# Ouvrir : http://localhost:8501
```

5 sections disponibles :
- Vue d'ensemble — KPIs + apercu des donnees avec filtres interactifs
- Monitoring technique — donnees nettoyees, graphiques EDA, qualite
- Dashboard business — predictions, segments, anomalies, KPIs, optimisation
- Rapport EDA — ouverture du rapport ydata-profiling
- Power BI — acces aux fichiers CSV et .pbix

---

## Watcher — Surveillance temps reel

Le watcher surveille raw_data en continu (polling toutes les 3 secondes) et declenche automatiquement le pipeline apres un debounce de 15 secondes sans nouvelle donnee :

```
Nouvelle donnee inseree dans raw_data
        |
watcher.py detecte en moins de 3s
        |
Attente de 15s (debounce) sans nouvelle ligne
        |
pipeline.py relance automatiquement
        |
outputs/powerbi/ mis a jour
```

Chaque dataset_id est traite independamment via threading.

---

## Ce qui ne change jamais

```
schema.json      — Universel, valable pour toutes les entreprises
pipeline.py      — Fixe, orchestre toujours les memes 9 etapes
tous les modules — Generiques, s'adaptent automatiquement
watcher.py       — Surveille tous les datasets automatiquement
app.py           — S'adapte au dataset selectionne
```

---

## Perspectives d'evolution

- Migration du watcher vers une vraie Live Query SurrealDB (WebSocket async)
- Integration de la detection de derive des donnees (Evidently)
- CI/CD via GitHub Actions
- Migration Streamlit vers lecture directe SurrealDB en temps reel
- Extension a d'autres secteurs d'activite

---

## Auteur

[Ouissal SARKOUH]

Stage d'été — [LeaderTec Engineering]

Juin — Juillet 2026

Encadrant : [Nakti Bilel]
