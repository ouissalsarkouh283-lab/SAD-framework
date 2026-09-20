import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.optimize import linprog
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             silhouette_score, davies_bouldin_score)

import mlflow
import mlflow.sklearn

# ── Configuration MLflow ───────────────────────────────────
mlflow.set_tracking_uri("sqlite:///outputs/mlflow.db")
mlflow.set_experiment("SAD_Framework")


# ── Chargement du schema.json ──────────────────────────────
def load_schema(schema_path="schema.json"):
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_mapping(mapping_path="column_mapping.json"):
    with open(mapping_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Modèle 1 — Random Forest ──────────────────────────────
def run_random_forest(df, schema, mapping_config):
    print("\n  🌲 Random Forest...")

    config   = schema["models"]["random_forest"]
    target   = config["target"]

    # ── 1. Construire la cible depuis target_definition ────
    target_def = mapping_config.get("target_definition")
    if not target_def:
        print(f"     ❌ Aucune 'target_definition' dans le mapping")
        return df, None

    source = target_def["source_column"]
    rule   = target_def["rule"]
    value  = target_def["value"]

    if source not in df.columns:
        print(f"     ❌ Colonne source '{source}' absente")
        return df, None

    if rule == "greater_than":
        df[target] = (df[source] > value).astype(int)
    elif rule == "equals":
        df[target] = (df[source] == value).astype(int)
    elif rule == "less_than":
        df[target] = (df[source] < value).astype(int)
    else:
        print(f"     ❌ Règle '{rule}' inconnue")
        return df, None

    print(f"     ✅ Cible '{target}' créée depuis '{source}'"
          f" (règle: {rule} {value})")

    distribution = df[target].value_counts()
    print(f"     Distribution :")
    for val, count in distribution.items():
        print(f"       {val} : {count}")

    if distribution.shape[0] < 2:
        print(f"     ❌ Cible '{target}' n'a qu'une seule classe"
              f" → modèle annulé")
        return df, None

    # ── 2. Préparer les features ───────────────────────────
    features  = config["features"]
    available = [f for f in features if f in df.columns]
    print(f"     Features utilisées : {available}")

    if len(available) < 1:
        print(f"     ❌ Pas de features disponibles")
        return df, None

    df_model = df[available + [target]].copy()
    for col in df_model.select_dtypes(include="object"):
        df_model[col] = df_model[col].astype("category").cat.codes

    df_model = df_model.dropna()

    for col in ["valeur_principale", "quantite_principale"]:
        if col in df_model.columns:
            df_model[col] = np.log1p(df_model[col])

    X = df_model[available]
    y = df_model[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"     Train : {len(X_train)} lignes")
    print(f"     Test  : {len(X_test)} lignes")

    # ── 3. Entraînement + MLflow ───────────────────────────
    run_name = f"RandomForest_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}"

    with mlflow.start_run(run_name=run_name):

        # Paramètres
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("n_lignes",     len(df))
        mlflow.log_param("n_features",   len(available))
        mlflow.log_param("features",     str(available))
        mlflow.log_param("target",       target)
        mlflow.log_param("test_size",    0.2)

        # Entraînement
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        print(f"     ✅ Modèle entraîné")

        # Évaluation
        y_pred   = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        auc      = roc_auc_score(
            y_test, rf.predict_proba(X_test)[:, 1]
        )

        print(f"\n     📊 Rapport d'évaluation :\n")
        print(classification_report(y_test, y_pred))
        print(f"\n     📊 Métriques globales :")
        print(f"       Accuracy : {accuracy:.2%}")
        print(f"       AUC-ROC  : {auc:.3f}")
        print(f"       → AUC > 0.7 = bon modèle")
        print(f"       → AUC > 0.9 = excellent")

        # ← MLflow : métriques
        mlflow.log_metric("accuracy", round(accuracy, 4))
        mlflow.log_metric("auc_roc",  round(auc, 4))
      
        # Générer et sauvegarder la matrice de confusion
        disp = ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred,
            display_labels=["Sans remise (0)", "Avec remise (1)"],
            cmap="Blues"
        )
        disp.ax_.set_title("Matrice de confusion — Random Forest\nlbl_2024")
        plt.tight_layout()
        plt.savefig("outputs/plots/confusion_matrix_rf.png", dpi=150)
        plt.close()
        print("     ✅ Matrice de confusion sauvegardée → outputs/plots/confusion_matrix_rf.png")


        # ← MLflow : feature importance
        importances = dict(zip(available, rf.feature_importances_))
        importances = dict(sorted(
            importances.items(), key=lambda x: x[1], reverse=True
        ))
        print(f"\n     📊 Feature importance :")
        for feat, imp in importances.items():
            print(f"       {feat} : {imp:.1%}")
            mlflow.log_metric(f"importance_{feat}", round(imp, 4))

        # ← MLflow : sauvegarder le modèle
        mlflow.sklearn.log_model(rf, "random_forest_model")
        print(f"     ✅ Run MLflow enregistré — {run_name}")

    # ── 4. Prédictions sur tout le dataset ────────────────
    df.loc[df_model.index, "rf_prediction"]  = rf.predict(X)
    df.loc[df_model.index, "rf_probabilite"] = (
        rf.predict_proba(X)[:, 1].round(2)
        if rf.predict_proba(X).shape[1] > 1 else None
    )

    print(f"\n     ✅ Random Forest terminé")
    return df, rf


# ── Modèle 2 — K-Means ────────────────────────────────────
def run_kmeans(df, schema):
    print("\n  🔵 K-Means...")

    config   = schema["models"]["kmeans"]
    features = config.get("features", config["requires"])

    available = [f for f in features if f in df.columns]
    if len(available) < 2:
        print("     ❌ Pas assez de colonnes disponibles")
        return df, None

    print(f"     Features utilisées : {available}")

    # ── 1. Préparer les données ────────────────────────────
    if "identifiant_entite" in df.columns:
        df_model = df.groupby("identifiant_entite")[available].sum()

        if "identifiant_unique" in df.columns:
            df_model["nb_commandes"] = df.groupby(
                "identifiant_entite"
            )["identifiant_unique"].count()
            available_extended = available + ["nb_commandes"]
        else:
            available_extended = available

        df_model = df_model.dropna()
        print("     ✅ Groupé par client")
        print(f"     Nombre de clients : {len(df_model)}")
    else:
        df_model           = df[available].copy()
        available_extended = available
        df_model           = df_model.dropna()

    for col in available_extended:
        if col in df_model.columns:
            df_model[col] = np.log1p(df_model[col])

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df_model[available_extended])

    # ── 2. Elbow Method ────────────────────────────────────
    print("     📊 Elbow Method en cours...")
    inertias = []
    k_range  = range(2, 11)

    for k in k_range:
        km_test = KMeans(n_clusters=k, random_state=42, n_init=10)
        km_test.fit(X_scaled)
        inertias.append(km_test.inertia_)

    diffs  = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
    diffs2 = [diffs[i] - diffs[i+1]       for i in range(len(diffs)-1)]
    best_k = k_range[diffs2.index(max(diffs2)) + 1]

    n_clusters_config = config.get("n_clusters", "auto")
    if n_clusters_config == "auto":
        n_clusters = best_k
        print(f"     Mode AUTO → k Elbow : {n_clusters}")
    else:
        n_clusters = int(n_clusters_config)
        print(f"     Mode MANUEL → k fixé : {n_clusters}")
        print(f"     (Elbow suggère : {best_k})")

    # Graphique elbow
    Path("outputs/plots").mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(k_range, inertias, marker="o", color="steelblue")
    plt.axvline(x=best_k,     color="red",   linestyle="--",
                label=f"Suggestion Elbow = {best_k}")
    plt.axvline(x=n_clusters, color="green", linestyle=":",
                label=f"k utilisé = {n_clusters}")
    plt.title("Elbow Method — K-Means")
    plt.xlabel("Nombre de clusters (k)")
    plt.ylabel("Inertie")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/plots/elbow.png", dpi=150)
    plt.close()
    print("     ✅ Graphique elbow sauvegardé")

    # ── 3. Entraînement + MLflow ───────────────────────────
    run_name = f"KMeans_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}"

    with mlflow.start_run(run_name=run_name):

        # Paramètres
        mlflow.log_param("n_clusters",    n_clusters)
        mlflow.log_param("best_k_elbow",  best_k)
        mlflow.log_param("n_clients",     len(df_model))
        mlflow.log_param("features",      str(available_extended))

        # Entraînement
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        km.fit(X_scaled)

        silhouette     = silhouette_score(X_scaled, km.labels_)
        davies_bouldin = davies_bouldin_score(X_scaled, km.labels_)

        print(f"\n     📊 Métriques clustering :")
        print(f"       Silhouette     : {silhouette:.3f}")
        print(f"       Davies-Bouldin : {davies_bouldin:.3f}")
        print(f"       → Silhouette > 0.5 = bon")
        print(f"       → Davies-Bouldin < 1 = bon")

        # ← MLflow : métriques
        mlflow.log_metric("silhouette",     round(silhouette, 4))
        mlflow.log_metric("davies_bouldin", round(davies_bouldin, 4))
        mlflow.log_metric("inertie",        round(km.inertia_, 2))

        # ← MLflow : graphique elbow
        mlflow.log_artifact("outputs/plots/elbow.png")

        # ← MLflow : sauvegarder le modèle
        mlflow.sklearn.log_model(km, "kmeans_model")
        print(f"     ✅ Run MLflow enregistré — {run_name}")

    # ── 4. Ajouter segments au dataset ────────────────────
    if "identifiant_entite" in df.columns:
        segment_map = dict(zip(df_model.index, km.labels_))
        df["kmeans_segment"] = df["identifiant_entite"].map(segment_map)
        print("     ✅ Segments assignés par client")
    else:
        df.loc[df_model.index, "kmeans_segment"] = km.labels_

    # ── 5. Analyse des segments ───────────────────────────
    print(f"\n     📊 Analyse des segments clients :\n")
    for i in range(n_clusters):
        segment = df[df["kmeans_segment"] == i]

        if "identifiant_entite" in df.columns:
            nb_clients = segment["identifiant_entite"].nunique()
        else:
            nb_clients = len(segment)

        print(f"     Segment {i} :")
        print(f"       Nombre clients  : {nb_clients}")

        if "valeur_principale" in df.columns:
            print(f"       Valeur totale   : "
                  f"{segment['valeur_principale'].sum():,.2f}")
            print(f"       Valeur moyenne  : "
                  f"{segment['valeur_principale'].mean():,.2f}")

        if "quantite_principale" in df.columns:
            print(f"       Qté moyenne     : "
                  f"{segment['quantite_principale'].mean():,.2f}")

        if "nb_commandes" in df_model.columns:
            nb_cmd = df_model.loc[
                df_model.index.isin(segment["identifiant_entite"]),
                "nb_commandes"
            ].mean() if "identifiant_entite" in df.columns else 0
            print(f"       Nb commandes moy: {nb_cmd:.0f}")

    print(f"\n     ✅ K-Means terminé")
    return df, km


# ── Modèle 3 — Z-score ────────────────────────────────────
def run_zscore(df, schema):
    print("\n  📉 Z-score — Détection d'anomalies...")

    config    = schema["models"]["zscore"]
    features  = config.get("features",
                [config.get("feature", "valeur_principale")])
    threshold = config.get("threshold", 3)

    print(f"     Colonnes analysées : {features}")
    print(f"     Seuil Z-score      : {threshold}")

    run_name = f"Zscore_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}"
    total_anomalies = 0

    with mlflow.start_run(run_name=run_name):

        # Paramètres
        mlflow.log_param("threshold", threshold)
        mlflow.log_param("features",  str(features))

        for feature in features:
            if feature not in df.columns:
                print(f"     ❌ '{feature}' absente")
                continue

            col_data   = df[feature].dropna()
            moyenne    = col_data.mean()
            ecart_type = col_data.std()

            if ecart_type == 0:
                print(f"     ⚠️  '{feature}' constante → ignorée")
                continue

            df[f"zscore_{feature}"]   = (df[feature] - moyenne) / ecart_type
            df[f"anomalie_{feature}"] = (
                df[f"zscore_{feature}"].abs() > threshold
            ).astype(int)

            anomalies  = df[f"anomalie_{feature}"].sum()
            pct        = (anomalies / len(df)) * 100
            seuil_min  = moyenne - threshold * ecart_type
            seuil_max  = moyenne + threshold * ecart_type
            taux_observe = anomalies / len(df)

            total_anomalies += anomalies

            print(f"\n     📊 Rapport — {feature} :\n")
            print(f"       Moyenne      : {moyenne:.2f}")
            print(f"       Ecart-type   : {ecart_type:.2f}")
            print(f"       Zone normale : [{seuil_min:.2f} → {seuil_max:.2f}]")
            print(f"       Anomalies    : {anomalies} ({pct:.2f}%)")
            print(f"\n     📊 Métriques Z-score :")
            print(f"       Taux attendu : 0.30%")
            print(f"       Taux observé : {taux_observe:.2%}")

            if taux_observe <= 0.05:
                print(f"       ✅ Taux cohérent")
            else:
                print(f"       ⚠️  Beaucoup d'anomalies → vérifier le seuil")

            # ← MLflow : métriques par feature
            mlflow.log_metric(f"anomalies_{feature}", int(anomalies))
            mlflow.log_metric(f"taux_{feature}",      round(taux_observe, 4))

            # Top 5
            top5 = (
                df[df[f"anomalie_{feature}"] == 1]
                [[feature, f"zscore_{feature}"]]
                .sort_values(f"zscore_{feature}", ascending=False)
                .head(5)
            )
            if len(top5) > 0:
                print(f"\n     🔴 Top 5 anomalies ({feature}) :\n")
                print(top5.to_string(index=False))

        # ← MLflow : total global
        mlflow.log_metric("total_anomalies", int(total_anomalies))
        mlflow.log_metric("taux_global",
                          round(total_anomalies / len(df), 4))
        print(f"     ✅ Run MLflow enregistré — {run_name}")

    print(f"\n     ✅ Z-score terminé")
    return df


# ── Modèle 4 — Optimisation ───────────────────────────────
def run_optimization(df, schema):
    print("\n  📐 Optimisation des coûts...")

    config   = schema["models"]["optimization"]
    features = config["requires"]

    available = [f for f in features if f in df.columns]
    if "categorie_principale" not in df.columns or len(available) < 2:
        print("     ❌ Colonnes insuffisantes pour l'optimisation")
        return df

    stats_df = df.groupby("categorie_principale").agg(
        prix_moyen=("valeur_principale",   "mean"),
        volume_historique=("quantite_principale", "sum")
    ).dropna()

    if len(stats_df) < 2:
        print("     ❌ Pas assez de catégories distinctes")
        return df

    categories  = stats_df.index.tolist()
    prix        = stats_df["prix_moyen"].values
    volume_hist = stats_df["volume_historique"].values

    print(f"     Catégories analysées : {len(categories)}")

    volume_total_requis = volume_hist.sum() * 0.8
    A_ineq = [[-1] * len(categories)]
    b_ineq = [-volume_total_requis]
    bounds = [(vol * 0.1, vol * 1.5) for vol in volume_hist]

    result = linprog(prix, A_ub=A_ineq, b_ub=b_ineq,
                     bounds=bounds, method="highs")

    if not result.success:
        print(f"     ❌ Optimisation échouée : {result.message}")
        return df

    cout_optimal  = result.fun
    cout_actuel   = (prix * volume_hist).sum()
    economie      = cout_actuel - cout_optimal
    economie_pct  = (economie / cout_actuel) * 100 if cout_actuel > 0 else 0

    print(f"\n     📊 Résultat de l'optimisation :\n")
    print(f"       Coût actuel    : {cout_actuel:,.2f}")
    print(f"       Coût optimal   : {cout_optimal:,.2f}")
    print(f"       Économie       : {economie:,.2f} ({economie_pct:.1f}%)")

    print(f"\n     📊 Allocation recommandée par catégorie :\n")
    for cat, qte_opt, qte_act, px in zip(
        categories, result.x, volume_hist, prix
    ):
        variation = ((qte_opt - qte_act) / qte_act * 100) if qte_act > 0 else 0
        signe = "▲" if variation > 0 else "▼" if variation < 0 else "="
        print(f"       {cat} : {qte_act:.0f} → {qte_opt:.0f}"
              f" ({signe} {abs(variation):.1f}%) — prix moyen {px:.2f}")

    # ← MLflow
    run_name = f"Optimization_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("n_categories",   len(categories))
        mlflow.log_metric("cout_actuel",   round(cout_actuel,  2))
        mlflow.log_metric("cout_optimal",  round(cout_optimal, 2))
        mlflow.log_metric("economie",      round(economie,     2))
        mlflow.log_metric("economie_pct",  round(economie_pct, 2))
        print(f"     ✅ Run MLflow enregistré — {run_name}")

    print(f"\n     ✅ Optimisation terminée")
    df_optim = pd.DataFrame({
        "categorie":        categories,
        "prix_moyen":       prix.round(2),
        "volume_actuel":    volume_hist.round(0),
        "volume_optimal":   result.x.round(0),
        "variation_pct":    [
            round(((qte_opt - qte_act) / qte_act * 100), 1)
            if qte_act > 0 else 0
            for qte_opt, qte_act in zip(result.x, volume_hist)
        ],
        "cout_actuel":      (prix * volume_hist).round(2),
        "cout_optimal":     (prix * result.x).round(2),
        "economie":         ((prix * volume_hist) - (prix * result.x)).round(2)
    })

    # Ajouter résumé global
    df_optim["economie_totale"]  = round(economie, 2)
    df_optim["economie_pct"]     = round(economie_pct, 2)

    os.makedirs("outputs/temp", exist_ok=True)
    df_optim.to_csv(
        "outputs/temp/optimization.csv",
        index=False, encoding="utf-8-sig"
    )
    print("     ✅ outputs/powerbi/optimization.csv sauvegardé")

    print(f"\n     ✅ Optimisation terminée")
    return df

# ── Fonction principale ────────────────────────────────────
def run_models(df, detected_schema, schema_path="schema.json",
               mapping_path="column_mapping.json"):

    schema         = load_schema(schema_path)
    mapping_config = load_mapping(mapping_path)
    active_models  = detected_schema["active_models"]
    results        = {}

    print("\n" + "═"*55)
    print("       MODÈLES ML — SAD Framework")
    print("═"*55)
    print(f"\n  Modèles actifs : {active_models}\n")

    if "random_forest" in active_models:
        df, rf = run_random_forest(df, schema, mapping_config)
        results["random_forest"] = rf

    if "kmeans" in active_models:
        df, km = run_kmeans(df, schema)
        results["kmeans"] = km

    if "zscore" in active_models:
        df = run_zscore(df, schema)
        results["zscore"] = True

    if "optimization" in active_models:
        df = run_optimization(df, schema)
        results["optimization"] = True

    print("\n" + "═"*55)
    print("       ✅ TOUS LES MODÈLES TERMINÉS")
    print("═"*55 + "\n")

    return df, results
