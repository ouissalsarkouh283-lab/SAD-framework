import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── Configuration graphiques ───────────────────────────────
sns.set_theme(style="whitegrid")
COLORS = sns.color_palette("husl", 10)
OUTPUT = "outputs/plots/"

def save_plot(filename):
    Path(OUTPUT).mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT}{filename}", dpi=150)
    plt.close()
    print(f"     ✅ {filename} sauvegardé")

def plot_trends(df, date_cols, numeric_cols):
    print("\n  📅 Tendances temporelles...")

    for date_col in date_cols:
        col_mois  = f"{date_col}_mois"
        col_annee = f"{date_col}_annee"

        if col_mois not in df.columns:
            continue

        for num_col in numeric_cols:
            if num_col not in df.columns:
                continue

            fig, ax = plt.subplots(figsize=(12, 5))

            trend = df.groupby(
                [col_annee, col_mois]
            )[num_col].sum().reset_index()

            trend["periode"] = (
                trend[col_annee].astype(str) + "-" +
                trend[col_mois].astype(str).str.zfill(2)
            )

            ax.plot(
                trend["periode"],
                trend[num_col],
                marker="o",
                color=COLORS[0],
                linewidth=2
            )
            ax.set_title(
                f"Tendance {num_col} par {date_col}",
                fontsize=14
            )
            ax.set_xlabel("Période")
            ax.set_ylabel(num_col)
            plt.xticks(rotation=45)

            save_plot(f"trend_{num_col}_{date_col}.png")

def plot_top10(df, categorical_cols, numeric_cols):
    print("\n  🏆 Top 10 catégories...")

    for cat_col in categorical_cols:
        if cat_col not in df.columns:
            continue

        for num_col in numeric_cols:
            if num_col not in df.columns:
                continue

            top10 = (df.groupby(cat_col)[num_col].sum().sort_values(ascending=False).head(10).reset_index())

            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.barh(
                top10[cat_col],
                top10[num_col],
                color=COLORS
            )
            ax.set_title(
                f"Top 10 {cat_col} par {num_col}",
                fontsize=14
            )
            ax.set_xlabel(num_col)
            ax.invert_yaxis()

            for bar, val in zip(bars, top10[num_col]):
                ax.text(
                    bar.get_width() * 1.01,
                    bar.get_y() + bar.get_height()/2,
                    f"{val:,.0f}",
                    va="center",
                    fontsize=9
                )

            save_plot(f"top10_{cat_col}_{num_col}.png")
    
def plot_kpis(df, numeric_cols):
    print("\n  💰 KPIs...")

    kpis = {}
    for col in numeric_cols:
        if col in df.columns:
            kpis[col] = {
                "total":  df[col].sum(),
                "moyen":  df[col].mean(),
                "max":    df[col].max(),
                "min":    df[col].min()
            }

    if not kpis:
        return

    fig, axes = plt.subplots(1, len(kpis), figsize=(6 * len(kpis), 5))

    if len(kpis) == 1:
        axes = [axes]

    for ax, (col, vals) in zip(axes, kpis.items()):
        labels = list(vals.keys())
        values = list(vals.values())
        bars = ax.bar(labels, values, color=COLORS[:4])
        ax.set_title(f"KPIs — {col}", fontsize=13)
        ax.set_ylabel("Valeur")

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() * 1.01,
                f"{val:,.0f}",
                ha="center",
                fontsize=9
            )

    save_plot("kpis.png")

def plot_distributions(df, numeric_cols):
    print("\n  📊 Distributions...")

    for col in numeric_cols:
        if col not in df.columns:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Histogramme
        axes[0].hist( df[col].dropna(),bins=20,color=COLORS[1], edgecolor="white")
        axes[0].set_title(f"Distribution — {col}")
        axes[0].set_xlabel(col)
        axes[0].set_ylabel("Fréquence")

        # Boxplot
        axes[1].boxplot(
            df[col].dropna(),
            patch_artist=True,
            boxprops=dict(facecolor=COLORS[2])
        )
        axes[1].set_title(f"Boxplot — {col}")
        axes[1].set_ylabel(col)

        save_plot(f"distribution_{col}.png")

def visualize(df, detected_schema):

    print("\n" + "═"*55)
    print("       VISUALISATION — SAD Framework")
    print("═"*55)

    date_cols        = detected_schema["date_columns"]
    numeric_cols     = detected_schema["numeric_columns"]
    categorical_cols = detected_schema["categorical_columns"]

    plot_trends(df, date_cols, numeric_cols)
    plot_top10(df, categorical_cols, numeric_cols)
    plot_kpis(df, numeric_cols)
    plot_distributions(df, numeric_cols)

    print(f"\n  📁 Tous les graphiques → {OUTPUT}")
    print("\n" + "═"*55 + "\n")