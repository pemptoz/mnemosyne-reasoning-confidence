"""
plot_quadrant.py

Visualisation des résultats post-hoc de mReasoner pour le dataset E2.

Le fichier mental_models_E2.csv contient deux estimations distinctes :

    number_models_generated_int
        Nombre de modèles générés avec les paramètres ajustés sur les
        réponses intuitives.

    number_models_generated_ref
        Nombre de modèles générés avec les paramètres ajustés sur les
        réponses réfléchies.

Aucune classification Système 1 / Système 2 n'est utilisée.

Fichiers produits :
    plots_E2/quadrant_phases_E2.png
    plots_E2/cognitive_space_3d_E2.png
    plots_E2/crossed_projections_E2.png
    plots_E2/phase_comparison_E2.png
    plots_E2/conflict_analysis_E2.png
    plots_E2/confidence_phase_analysis_E2.png

    plots_E2/subject_summary_E2.csv
    plots_E2/conflict_summary_E2.csv

Dans les graphiques construits au niveau participant :
    un point = un participant.
"""

import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans


# ======================================================================
# CONFIGURATION
# ======================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# On remonte 4 niveaux depuis src/analysis/confidence/E2 -> racine du repo
REPO_ROOT = os.path.abspath(
    os.path.join(
        SCRIPT_DIR,
        "..",
        "..",
        "..",
        "..",
    )
)

INPUT_FILE = os.path.join(
    REPO_ROOT,
    "results",
    "tables",
    "mental_models",
    "mental_models_E2_2.csv",
)

OUTPUT_DIRECTORY = os.path.join(
    REPO_ROOT,
    "results",
    "analysis",
    "confidence",
    "E2",
    "plots_E2_2",
)

os.makedirs(
    OUTPUT_DIRECTORY,
    exist_ok=True,
)


# ----------------------------------------------------------------------
# Fichiers graphiques
# ----------------------------------------------------------------------

QUADRANT_OUTPUT_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "quadrant_phases_E2_2.png",
)

MODEL_PHASE_COMPARISON_OUTPUT_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "model_count_phase_comparison_E2_2.png",
)


THREE_DIMENSIONAL_OUTPUT_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "cognitive_space_3d_E2_2.png",
)

CROSSED_PROJECTIONS_OUTPUT_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "crossed_projections_E2_2.png",
)

PHASE_COMPARISON_OUTPUT_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "phase_comparison_E2_2.png",
)

CONFLICT_OUTPUT_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "conflict_analysis_E2_2.png",
)

CONFIDENCE_PHASE_OUTPUT_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "confidence_phase_analysis_E2_2.png",
)


# ----------------------------------------------------------------------
# Fichiers CSV
# ----------------------------------------------------------------------

SUBJECT_SUMMARY_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "subject_summary_E2_2.csv",
)

MODEL_PHASE_COMPARISON_SUMMARY_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "model_count_phase_comparison_E2_2.csv",
)


CONFLICT_SUMMARY_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "conflict_summary_E2_2.csv",
)

GROUPED_MEDIANS_STATISTICS_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "grouped_medians_statistics_E2_2.csv",
)


GROUPED_MEDIANS_OUTPUT_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "grouped_medians_E2_2.png",
)

GROUPED_MEDIANS_REF_OUTPUT_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "grouped_medians_ref_E2_2.png",
)


# ----------------------------------------------------------------------
# Affichage
# ----------------------------------------------------------------------

DPI = 300
SHOW_FIGURES = True

ANNOTATE_SUBJECTS = False
MAX_ANNOTATED_SUBJECTS = 40



# ----------------------------------------------------------------------
# Groupes de nombre moyen de modèles
# ----------------------------------------------------------------------



MODEL_GROUP_COLORS = {
    "< 2,6 modèles": "#3b82f6",
    "2,6 à < 3 modèles": "#f59e0b",
    "≥ 3 modèles": "#ef4444",
}


# ----------------------------------------------------------------------
# Conflit
# ----------------------------------------------------------------------

CONFLICT_LABELS = {
    0: "Sans conflit",
    1: "Conflit",
}


sns.set_theme(
    style="whitegrid",
    context="notebook",
)


# ======================================================================
# OUTILS
# ======================================================================

def normalize_subject_id(value):
    """
    Normalise un identifiant participant.

    Exemples :
        63873   -> "63873"
        63873.0 -> "63873"
    """
    if pd.isna(value):
        return None

    normalized = str(value).strip()

    if not normalized:
        return None

    try:
        numeric = float(normalized)

        if numeric.is_integer():
            return str(int(numeric))

    except (
        TypeError,
        ValueError,
    ):
        pass

    return normalized


def ensure_numeric(dataframe, columns):
    """
    Convertit les colonnes indiquées en valeurs numériques.
    """
    dataframe = dataframe.copy()

    for column in columns:
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

    return dataframe


def save_figure(figure, output_file):
    """
    Sauvegarde une figure.
    """
    output_directory = (
        os.path.dirname(output_file)
        or "."
    )

    os.makedirs(
        output_directory,
        exist_ok=True,
    )

    figure.savefig(
        output_file,
        dpi=DPI,
        bbox_inches="tight",
    )

    print(
        "Graphique enregistré :",
        output_file,
    )


def standard_error(series):
    """
    Calcule l'erreur standard de la moyenne.
    """
    clean_series = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if len(clean_series) <= 1:
        return 0.0

    return float(
        clean_series.std(ddof=1)
        / np.sqrt(len(clean_series))
    )


def safe_correlation(
    dataframe,
    x_column,
    y_column,
):
    """
    Calcule une corrélation de Pearson si elle est définie.
    """
    missing_columns = [
        column
        for column in [
            x_column,
            y_column,
        ]
        if column not in dataframe.columns
    ]

    if missing_columns:
        warnings.warn(
            "Impossible de calculer la corrélation : "
            f"colonnes absentes {missing_columns}"
        )

        return np.nan

    subset = (
        dataframe[[
            x_column,
            y_column,
        ]]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    if len(subset) < 2:
        return np.nan

    if (
        subset[x_column].nunique() < 2
        or subset[y_column].nunique() < 2
    ):
        return np.nan

    return subset[x_column].corr(
        subset[y_column],
        method="pearson",
    )


def format_correlation(value):
    """
    Formate une corrélation pour l'affichage dans le terminal.
    """
    if pd.isna(value):
        return "non définie"

    return f"{value:.4f}"


def add_model_group(dataframe, models_column, output_column="model_group"):
    """
    Regroupe les participants en 3 clusters à l'aide de K-Means.
    Génère des étiquettes dynamiques basées sur le centre du cluster.
    """
    dataframe = dataframe.copy()

    if models_column not in dataframe.columns:
        raise KeyError(f"Colonne absente : {models_column}")

    # Isoler les valeurs valides pour le clustering
    valid_mask = dataframe[models_column].notna()
    X = dataframe.loc[valid_mask, [models_column]].values

    # K-Means avec 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    # Trier les clusters par leur centre (du plus petit au plus grand)
    centers = kmeans.cluster_centers_.flatten()
    sorted_indices = np.argsort(centers)
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_indices)}
    sorted_labels = np.vectorize(label_mapping.get)(labels)

    # Création des labels dynamiques (ex: "Groupe 1 (moy ~ 2.05)")
    dynamic_labels = []
    for i in range(3):
        center_val = centers[sorted_indices[i]]
        dynamic_labels.append(f"Gr. {i+1} (moy ~ {center_val:.2f})")

    # Assigner au dataframe
    dataframe.loc[valid_mask, output_column] = [dynamic_labels[lbl] for lbl in sorted_labels]
    
    # Convertir en variable catégorielle ordonnée
    dataframe[output_column] = pd.Categorical(
        dataframe[output_column],
        categories=dynamic_labels,
        ordered=True
    )

    return dataframe, dynamic_labels


# ======================================================================
# CHARGEMENT DES DONNÉES
# ======================================================================

def load_detailed_data():
    """
    Charge mental_models_E2.csv.

    Le fichier doit contenir deux comptages :

        number_models_generated_int
            paramètres ajustés sur les réponses intuitives ;

        number_models_generated_ref
            paramètres ajustés sur les réponses réfléchies.
    """
    if not os.path.isfile(
        INPUT_FILE
    ):
        raise FileNotFoundError(
            "Le fichier d'entrée est introuvable : "
            f"{INPUT_FILE}"
        )

    dataframe = pd.read_csv(
        INPUT_FILE
    )

    print(
        "Fichier utilisé :",
        INPUT_FILE,
    )

    print(
        "Colonnes disponibles :",
        list(dataframe.columns),
    )

    required_columns = {
        "subject_id",
        "sequence",
        "task_type",
        "conflict",
        "correct_int",
        "correct_ref",
        "for_int",
        "for_ref",
        "rt_int",
        "rt_ref",
        "number_models_generated_int",
        "number_models_generated_ref",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise KeyError(
            "Colonnes absentes de mental_models_E2.csv : "
            f"{sorted(missing_columns)}"
        )

    dataframe = dataframe.copy()

    dataframe["subject_id"] = (
        dataframe["subject_id"]
        .apply(normalize_subject_id)
        .astype("string")
    )

    dataframe["task_type"] = (
        dataframe["task_type"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    numeric_columns = [
        "sequence",
        "validity",
        "believability",
        "conflict",
        "correct_int",
        "correct_ref",
        "for_int",
        "for_ref",
        "rt_int",
        "rt_ref",
        "response_changed",
        "accuracy_gain",
        "for_change",
        "rt_change",

        "number_models_generated_int",
        "std_models_generated_int",
        "minimum_models_generated_int",
        "maximum_models_generated_int",
        "total_simulation_count_int",

        "number_models_generated_ref",
        "std_models_generated_ref",
        "minimum_models_generated_ref",
        "maximum_models_generated_ref",
        "total_simulation_count_ref",

        "models_change",
    ]

    dataframe = ensure_numeric(
        dataframe,
        numeric_columns,
    )

    dataframe = (
        dataframe
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna(
            subset=[
                "subject_id",
                "sequence",
                "task_type",
                "correct_int",
                "correct_ref",
                "for_int",
                "for_ref",
                "number_models_generated_int",
                "number_models_generated_ref",
            ]
        )
        .copy()
    )

    # Vérification des colonnes de correction.
    for column in [
        "correct_int",
        "correct_ref",
    ]:
        invalid_values = (
            dataframe.loc[
                dataframe[column].notna()
                & ~dataframe[column].isin(
                    [
                        0,
                        1,
                    ]
                ),
                column,
            ]
            .unique()
            .tolist()
        )

        if invalid_values:
            raise ValueError(
                f"La colonne {column} contient des valeurs "
                f"non binaires : {invalid_values}"
            )

    dataframe["correct_int"] = (
        dataframe["correct_int"]
        .astype(int)
    )

    dataframe["correct_ref"] = (
        dataframe["correct_ref"]
        .astype(int)
    )

    # Recalcul de sécurité si la colonne n'existe pas.
    if "models_change" not in dataframe.columns:
        dataframe["models_change"] = (
            dataframe[
                "number_models_generated_ref"
            ]
            - dataframe[
                "number_models_generated_int"
            ]
        )

    dataframe["conflict_label"] = (
        dataframe["conflict"]
        .map(CONFLICT_LABELS)
        .fillna("Inconnu")
    )

    print(
        "Nombre d'essais :",
        len(dataframe),
    )

    print(
        "Nombre de participants :",
        dataframe["subject_id"].nunique(),
    )

    print(
        "Précision intuitive globale :",
        round(
            dataframe["correct_int"].mean()
            * 100,
            4,
        ),
        "%",
    )

    print(
        "Précision réfléchie globale :",
        round(
            dataframe["correct_ref"].mean()
            * 100,
            4,
        ),
        "%",
    )

    print(
        "Nombre moyen de modèles — ajustement intuitif :",
        round(
            dataframe[
                "number_models_generated_int"
            ].mean(),
            4,
        ),
    )

    print(
        "Nombre moyen de modèles — ajustement réfléchi :",
        round(
            dataframe[
                "number_models_generated_ref"
            ].mean(),
            4,
        ),
    )

    print(
        "Différence moyenne réfléchi − intuitif :",
        round(
            dataframe["models_change"].mean(),
            4,
        ),
    )

    return dataframe


# ======================================================================
# RÉSUMÉ PAR PARTICIPANT
# ======================================================================

def build_subject_summary(dataframe):
    """
    Produit une ligne par participant.

    Chaque participant possède deux estimations :

        mean_number_models_generated_int
        mean_number_models_generated_ref
    """
    subject_summary = (
        dataframe
        .groupby(
            "subject_id",
            as_index=False,
        )
        .agg(
            intuitive_accuracy=(
                "correct_int",
                "mean",
            ),

            reflective_accuracy=(
                "correct_ref",
                "mean",
            ),

            mean_for_int=(
                "for_int",
                "mean",
            ),

            mean_for_ref=(
                "for_ref",
                "mean",
            ),

            mean_rt_int=(
                "rt_int",
                "mean",
            ),

            mean_rt_ref=(
                "rt_ref",
                "mean",
            ),

            # Modèles sous l'ajustement intuitif.
            mean_number_models_generated_int=(
                "number_models_generated_int",
                "mean",
            ),

            median_number_models_generated_int=(
                "number_models_generated_int",
                "median",
            ),

            std_number_models_generated_int=(
                "number_models_generated_int",
                "std",
            ),

            minimum_number_models_generated_int=(
                "minimum_models_generated_int",
                "min",
            ),

            maximum_number_models_generated_int=(
                "maximum_models_generated_int",
                "max",
            ),

            # Modèles sous l'ajustement réfléchi.
            mean_number_models_generated_ref=(
                "number_models_generated_ref",
                "mean",
            ),

            median_number_models_generated_ref=(
                "number_models_generated_ref",
                "median",
            ),

            std_number_models_generated_ref=(
                "number_models_generated_ref",
                "std",
            ),

            minimum_number_models_generated_ref=(
                "minimum_models_generated_ref",
                "min",
            ),

            maximum_number_models_generated_ref=(
                "maximum_models_generated_ref",
                "max",
            ),

            mean_models_change=(
                "models_change",
                "mean",
            ),

            response_change_rate=(
                "response_changed",
                "mean",
            ),

            mean_accuracy_gain=(
                "accuracy_gain",
                "mean",
            ),

            mean_confidence_change=(
                "for_change",
                "mean",
            ),

            mean_rt_change=(
                "rt_change",
                "mean",
            ),

            number_of_trials=(
                "sequence",
                "count",
            ),

            number_of_task_types=(
                "task_type",
                "nunique",
            ),
        )
    )

    for column in [
        "intuitive_accuracy",
        "reflective_accuracy",
        "response_change_rate",
    ]:
        subject_summary[column] *= 100

    subject_summary, _ = add_model_group(
        subject_summary, "mean_number_models_generated_int", "model_group_int"
    )
    subject_summary, _ = add_model_group(
        subject_summary, "mean_number_models_generated_ref", "model_group_ref"
    )

    subject_summary = (
        subject_summary
        .sort_values(
            by="subject_id"
        )
        .reset_index(
            drop=True
        )
    )

    subject_summary.to_csv(
        SUBJECT_SUMMARY_FILE,
        index=False,
    )

    print(
        "Résumé par participant :",
        SUBJECT_SUMMARY_FILE,
    )

    return subject_summary


# ======================================================================
# 1. QUADRANTS INTUITIF ET RÉFLÉCHI
# ======================================================================

def create_quadrant_plot(dataframe):
    """
    Crée deux quadrants :

        - confiance et précision intuitives ;
        - confiance et précision réfléchies.

    Chaque point représente un participant.
    """
    required_columns = [
        "subject_id",
        "mean_for_int",
        "mean_for_ref",
        "intuitive_accuracy",
        "reflective_accuracy",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise KeyError(
            "Colonnes manquantes pour les quadrants : "
            f"{missing_columns}"
        )

    plot_data = (
        dataframe[
            required_columns
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .copy()
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(18, 8),
        sharex=True,
        sharey=True,
    )

    panels = [
        {
            "axis": axes[0],
            "confidence_column": "mean_for_int",
            "accuracy_column": "intuitive_accuracy",
            "title": "Cas intuitif",
            "x_label": "Confiance intuitive",
            "y_label":
                "Réponses intuitives correctes (%)",
            "point_color": "#2563eb",
        },
        {
            "axis": axes[1],
            "confidence_column": "mean_for_ref",
            "accuracy_column": "reflective_accuracy",
            "title": "Cas réfléchi",
            "x_label": "Confiance réfléchie",
            "y_label":
                "Réponses réfléchies correctes (%)",
            "point_color": "#8b5cf6",
        },
    ]

    for panel in panels:
        axis = panel["axis"]

        confidence_column = (
            panel["confidence_column"]
        )

        accuracy_column = (
            panel["accuracy_column"]
        )

        panel_data = (
            plot_data[[
                "subject_id",
                confidence_column,
                accuracy_column,
            ]]
            .dropna()
            .copy()
        )

        if panel_data.empty:
            axis.set_title(
                f"{panel['title']}\nAucune donnée",
                fontweight="bold",
            )

            continue

        median_confidence = float(
            panel_data[
                confidence_column
            ].median()
        )

        median_accuracy = float(
            panel_data[
                accuracy_column
            ].median()
        )

        axis.scatter(
            panel_data[confidence_column],
            panel_data[accuracy_column],
            color=panel["point_color"],
            s=75,
            alpha=0.70,
            edgecolors="white",
            linewidths=0.6,
        )

        axis.axvline(
            median_confidence,
            color="#374151",
            linestyle="--",
            linewidth=1.7,
            label=(
                "Médiane de confiance : "
                f"{median_confidence:.1f}"
            ),
        )

        axis.axhline(
            median_accuracy,
            color="#6b7280",
            linestyle="--",
            linewidth=1.7,
            label=(
                "Médiane de précision : "
                f"{median_accuracy:.1f} %"
            ),
        )

        axis.scatter(
            median_confidence,
            median_accuracy,
            marker="D",
            s=120,
            color="black",
            edgecolors="white",
            linewidths=0.7,
            zorder=10,
        )

        if ANNOTATE_SUBJECTS:
            for _, row in panel_data.head(
                MAX_ANNOTATED_SUBJECTS
            ).iterrows():
                axis.annotate(
                    str(row["subject_id"]),
                    (
                        row[confidence_column],
                        row[accuracy_column],
                    ),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=6,
                    alpha=0.7,
                )

        axis.set_title(
            panel["title"],
            fontsize=16,
            fontweight="bold",
            pad=14,
        )

        axis.set_xlabel(
            panel["x_label"]
        )

        axis.set_ylabel(
            panel["y_label"]
        )

        axis.set_xlim(
            0,
            100,
        )

        axis.set_ylim(
            0,
            100,
        )

        axis.grid(
            True,
            linestyle="--",
            alpha=0.4,
        )

        axis.legend(
            fontsize=9,
            loc="best",
        )

    figure.suptitle(
        "Confiance et précision selon la phase de réponse",
        fontsize=18,
        fontweight="bold",
    )

    figure.tight_layout(
        rect=[
            0,
            0,
            1,
            0.94,
        ]
    )

    save_figure(
        figure,
        QUADRANT_OUTPUT_FILE,
    )


# ======================================================================
# 2. ESPACES COGNITIFS 3D
# ======================================================================

def create_three_dimensional_plot(dataframe):
    """
    Crée deux espaces 3D :

        intuitif :
            confiance intuitive,
            modèles issus de l'ajustement intuitif,
            précision intuitive ;

        réfléchi :
            confiance réfléchie,
            modèles issus de l'ajustement réfléchi,
            précision réfléchie.
    """
    required_columns = [
        "mean_for_int",
        "mean_for_ref",
        "intuitive_accuracy",
        "reflective_accuracy",
        "mean_number_models_generated_int",
        "mean_number_models_generated_ref",
    ]

    plot_data = (
        dataframe[
            required_columns
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    if plot_data.empty:
        warnings.warn(
            "Aucune donnée disponible pour les graphiques 3D."
        )

        return

    figure = plt.figure(
        figsize=(18, 8)
    )

    axis_int = figure.add_subplot(
        121,
        projection="3d",
    )

    axis_ref = figure.add_subplot(
        122,
        projection="3d",
    )

    # Phase intuitive.
    axis_int.scatter(
        plot_data["mean_for_int"],
        plot_data[
            "mean_number_models_generated_int"
        ],
        plot_data["intuitive_accuracy"],
        color="#2563eb",
        s=45,
        alpha=0.72,
    )

    axis_int.set_title(
        "Cas intuitif",
        fontweight="bold",
    )

    axis_int.set_xlabel(
        "Confiance intuitive",
        labelpad=10,
    )

    axis_int.set_ylabel(
        "Nombre moyen de modèles",
        labelpad=10,
    )

    axis_int.set_zlabel(
        "Précision intuitive (%)",
        labelpad=10,
    )

    axis_int.set_xlim(
        0,
        100,
    )

    axis_int.set_zlim(
        0,
        100,
    )

    # Phase réfléchie.
    axis_ref.scatter(
        plot_data["mean_for_ref"],
        plot_data[
            "mean_number_models_generated_ref"
        ],
        plot_data["reflective_accuracy"],
        color="#8b5cf6",
        s=45,
        alpha=0.72,
    )

    axis_ref.set_title(
        "Cas réfléchi",
        fontweight="bold",
    )

    axis_ref.set_xlabel(
        "Confiance réfléchie",
        labelpad=10,
    )

    axis_ref.set_ylabel(
        "Nombre moyen de modèles",
        labelpad=10,
    )

    axis_ref.set_zlabel(
        "Précision réfléchie (%)",
        labelpad=10,
    )

    axis_ref.set_xlim(
        0,
        100,
    )

    axis_ref.set_zlim(
        0,
        100,
    )

    figure.suptitle(
        "Espaces cognitifs 3D : confiance, modèles et précision",
        fontsize=16,
        fontweight="bold",
    )

    figure.tight_layout(
        rect=[
            0,
            0,
            1,
            0.94,
        ]
    )

    save_figure(
        figure,
        THREE_DIMENSIONAL_OUTPUT_FILE,
    )


# ======================================================================
# 3. PROJECTIONS CROISÉES
# ======================================================================

def add_simple_scatter(
    axis,
    dataframe,
    x_column,
    y_column,
    title,
    x_label,
    y_label,
    color,
):
    """
    Ajoute un nuage de points sans régression et sans corrélation
    affichée sur le graphique.
    """
    plot_data = (
        dataframe[[
            x_column,
            y_column,
        ]]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    axis.scatter(
        plot_data[x_column],
        plot_data[y_column],
        color=color,
        s=48,
        alpha=0.68,
        edgecolors="white",
        linewidths=0.5,
    )

    axis.set_title(
        title,
        fontweight="bold",
    )

    axis.set_xlabel(
        x_label
    )

    axis.set_ylabel(
        y_label
    )

    axis.grid(
        True,
        linestyle="--",
        alpha=0.4,
    )


def create_crossed_projections(dataframe):
    """
    Crée six projections sans régression.

    La ligne intuitive utilise les modèles obtenus avec les paramètres
    ajustés aux réponses intuitives.

    La ligne réfléchie utilise les modèles obtenus avec les paramètres
    ajustés aux réponses réfléchies.
    """
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(19, 11),
    )

    figure.suptitle(
        "Projections croisées : confiance, modèles et précision",
        fontsize=18,
        fontweight="bold",
    )

    # ------------------------------------------------------------------
    # Phase intuitive
    # ------------------------------------------------------------------

    add_simple_scatter(
        axis=axes[0, 0],
        dataframe=dataframe,
        x_column="mean_for_int",
        y_column=(
            "mean_number_models_generated_int"
        ),
        title="Confiance et nombre de modèles",
        x_label="Confiance intuitive",
        y_label=(
            "Nombre moyen de modèles — intuitif"
        ),
        color="#6366f1",
    )

    add_simple_scatter(
        axis=axes[0, 1],
        dataframe=dataframe,
        x_column="mean_for_int",
        y_column="intuitive_accuracy",
        title="Confiance et précision",
        x_label="Confiance intuitive",
        y_label="Précision intuitive (%)",
        color="#10b981",
    )

    add_simple_scatter(
        axis=axes[0, 2],
        dataframe=dataframe,
        x_column=(
            "mean_number_models_generated_int"
        ),
        y_column="intuitive_accuracy",
        title="Nombre de modèles et précision",
        x_label=(
            "Nombre moyen de modèles — intuitif"
        ),
        y_label="Précision intuitive (%)",
        color="#f59e0b",
    )

    axes[0, 0].text(
        -0.28,
        0.5,
        "Cas intuitif",
        transform=axes[0, 0].transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=15,
        fontweight="bold",
        color="#1d4ed8",
    )

    # ------------------------------------------------------------------
    # Phase réfléchie
    # ------------------------------------------------------------------

    add_simple_scatter(
        axis=axes[1, 0],
        dataframe=dataframe,
        x_column="mean_for_ref",
        y_column=(
            "mean_number_models_generated_ref"
        ),
        title="Confiance et nombre de modèles",
        x_label="Confiance réfléchie",
        y_label=(
            "Nombre moyen de modèles — réfléchi"
        ),
        color="#6366f1",
    )

    add_simple_scatter(
        axis=axes[1, 1],
        dataframe=dataframe,
        x_column="mean_for_ref",
        y_column="reflective_accuracy",
        title="Confiance et précision",
        x_label="Confiance réfléchie",
        y_label="Précision réfléchie (%)",
        color="#10b981",
    )

    add_simple_scatter(
        axis=axes[1, 2],
        dataframe=dataframe,
        x_column=(
            "mean_number_models_generated_ref"
        ),
        y_column="reflective_accuracy",
        title="Nombre de modèles et précision",
        x_label=(
            "Nombre moyen de modèles — réfléchi"
        ),
        y_label="Précision réfléchie (%)",
        color="#f59e0b",
    )

    axes[1, 0].text(
        -0.28,
        0.5,
        "Cas réfléchi",
        transform=axes[1, 0].transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=15,
        fontweight="bold",
        color="#7e22ce",
    )

    # Échelles de confiance.
    axes[0, 0].set_xlim(0, 100)
    axes[0, 1].set_xlim(0, 100)
    axes[1, 0].set_xlim(0, 100)
    axes[1, 1].set_xlim(0, 100)

    # Échelles de précision.
    axes[0, 1].set_ylim(0, 100)
    axes[0, 2].set_ylim(0, 100)
    axes[1, 1].set_ylim(0, 100)
    axes[1, 2].set_ylim(0, 100)

    figure.tight_layout(
        rect=[
            0.03,
            0,
            1,
            0.95,
        ]
    )

    save_figure(
        figure,
        CROSSED_PROJECTIONS_OUTPUT_FILE,
    )


# ======================================================================
# 4. MÉDIANES GLOBALES PAR GROUPE DE MODÈLES
# ======================================================================

def create_grouped_medians_plot_ref_only(dataframe):
    """
    Répartit les participants en trois groupes selon leur nombre 
    de modèles générés (uniquement dans le cas réfléchi).

    Pour chaque participant, les mesures utilisées sont strictement :
        - confiance réfléchie
        - précision réfléchie
        - nombre de modèles issus de l'ajustement réfléchi
    """
    required_columns = [
        "subject_id",
        "mean_for_ref",
        "reflective_accuracy",
        "mean_number_models_generated_ref",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
            raise KeyError(
                "Colonnes manquantes pour le graphique réfléchi "
                f"par groupes de modèles : {missing_columns}"
            )
        
    plot_data = dataframe[required_columns].copy()

    numeric_columns = [
        "mean_for_ref",
        "reflective_accuracy",
        "mean_number_models_generated_ref",
    ]

    for column in numeric_columns:
        plot_data[column] = pd.to_numeric(
            plot_data[column],
            errors="coerce",
        )

    plot_data = (
        plot_data
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=numeric_columns)
        .copy()
    )

    if plot_data.empty:
        raise ValueError(
            "Aucune donnée valide pour générer le graphique réfléchi."
        )

    # ==============================================================
    # MESURES STRICTEMENT RÉFLÉCHIES
    # ==============================================================

    plot_data["confidence"] = plot_data["mean_for_ref"]
    plot_data["accuracy"] = plot_data["reflective_accuracy"]
    plot_data["number_models_generated"] = plot_data["mean_number_models_generated_ref"]

    # ==============================================================
    # GROUPES DE NOMBRE DE MODÈLES
    # ==============================================================

    plot_data, dynamic_labels = add_model_group(
            plot_data, "number_models_generated", "model_group"
        )

    # ==============================================================
    # FIGURE
    # ==============================================================

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(20, 6.5),
        sharex=True,
        sharey=True,
    )

    figure.suptitle(
        "Confiance et précision réfléchies selon le nombre "
        "de modèles générés (Système 2)",
        fontsize=17,
        fontweight="bold",
        y=0.99,
    )

    dynamic_colors = ["#3b82f6", "#f59e0b", "#ef4444"]
    
    for axis, group_label, color in zip(axes, dynamic_labels, dynamic_colors):
        group_data = plot_data.loc[
            plot_data["model_group"] == group_label
        ].copy()

        if group_data.empty:
            axis.set_title(
                f"{group_label}\nAucun participant",
                fontweight="bold",
            )
            axis.text(
                0.5, 0.5, "Aucune donnée",
                transform=axis.transAxes,
                ha="center", va="center",
                fontsize=13, color="gray",
            )
            axis.set_xlabel("Confiance réfléchie (%)")
            axis.grid(True, linestyle="--", alpha=0.4)
            continue

        participant_count = len(group_data)
        median_confidence = float(group_data["confidence"].median())
        median_accuracy = float(group_data["accuracy"].median())
        mean_models = float(group_data["number_models_generated"].mean())

        # Participants
        axis.scatter(
            group_data["confidence"],
            group_data["accuracy"],
            s=75,
            color=color,
            alpha=0.65,
            edgecolors="white",
            linewidths=0.7,
            label="Participants",
        )

        # Médianes
        axis.axvline(
            median_confidence,
            color="#111827", linestyle="--", linewidth=2, alpha=0.9,
            label=f"Médiane confiance : {median_confidence:.1f} %",
        )
        axis.axhline(
            median_accuracy,
            color="#6b21a8", linestyle="--", linewidth=2, alpha=0.9,
            label=f"Médiane précision : {median_accuracy:.1f} %",
        )

        # Intersection
        axis.scatter(
            [median_confidence], [median_accuracy],
            marker="D", s=150, color="black", edgecolors="white",
            linewidths=1.2, zorder=10, label="Intersection des médianes",
        )

        axis.annotate(
            f"Médianes\nC = {median_confidence:.1f} %\nP = {median_accuracy:.1f} %",
            xy=(median_confidence, median_accuracy),
            xytext=(10, 10), textcoords="offset points",
            fontsize=9, fontweight="bold",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "black", "alpha": 0.85},
        )

        axis.set_title(
            f"{group_label}\nn = {participant_count} | moyenne modèles = {mean_models:.2f}",
            fontsize=13, fontweight="bold",
        )
        axis.set_xlabel("Confiance réfléchie (%)")
        axis.grid(True, linestyle="--", alpha=0.4)
        axis.legend(loc="best", fontsize=8, framealpha=0.9)

        if ANNOTATE_SUBJECTS:
            for _, row in group_data.head(MAX_ANNOTATED_SUBJECTS).iterrows():
                axis.annotate(
                    str(row["subject_id"]),
                    (row["confidence"], row["accuracy"]),
                    xytext=(3, 3), textcoords="offset points",
                    fontsize=6, alpha=0.65,
                )

    axes[0].set_ylabel("Précision réfléchie (%)")

    if plot_data["confidence"].min() >= 0 and plot_data["confidence"].max() <= 100:
        for axis in axes: axis.set_xlim(0, 100)
    if plot_data["accuracy"].min() >= 0 and plot_data["accuracy"].max() <= 100:
        for axis in axes: axis.set_ylim(0, 100)

    figure.tight_layout(rect=[0, 0, 1, 0.92])
    save_figure(figure, GROUPED_MEDIANS_REF_OUTPUT_FILE)

def create_grouped_medians_plot(dataframe):
    """
    Répartit les participants en trois groupes selon leur nombre global
    moyen de modèles générés.

    Cette représentation ne sépare pas les cas intuitif et réfléchi.

    Pour chaque participant, les mesures globales sont définies par :

        confiance globale =
            (confiance intuitive + confiance réfléchie) / 2

        précision globale =
            (précision intuitive + précision réfléchie) / 2

        nombre global de modèles =
            (
                nombre moyen de modèles avec ajustement intuitif
                + nombre moyen de modèles avec ajustement réfléchi
            ) / 2


    Dans chaque panneau :

        - chaque point représente un participant ;
        - X représente la confiance globale ;
        - Y représente la précision globale ;
        - la ligne verticale représente la médiane de confiance ;
        - la ligne horizontale représente la médiane de précision ;
        - le losange noir représente l'intersection des médianes.
    """
    required_columns = [
        "subject_id",
        "mean_for_int",
        "mean_for_ref",
        "intuitive_accuracy",
        "reflective_accuracy",
        "mean_number_models_generated_int",
        "mean_number_models_generated_ref",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise KeyError(
            "Colonnes manquantes pour le graphique global "
            f"par groupes de modèles : {missing_columns}"
        )

    plot_data = dataframe[
        required_columns
    ].copy()

    numeric_columns = [
        "mean_for_int",
        "mean_for_ref",
        "intuitive_accuracy",
        "reflective_accuracy",
        "mean_number_models_generated_int",
        "mean_number_models_generated_ref",
    ]

    for column in numeric_columns:
        plot_data[column] = pd.to_numeric(
            plot_data[column],
            errors="coerce",
        )

    if plot_data.empty:
        raise ValueError(
            "Aucune donnée valide pour générer le graphique global "
            "par groupes de modèles."
        )

    # ==============================================================
    # MESURES GLOBALES, INDÉPENDANTES DE LA PHASE
    # ==============================================================

    plot_data["confidence"] = (
        plot_data["mean_for_int"]
        + plot_data["mean_for_ref"]
    ) / 2.0

    plot_data["accuracy"] = (
        plot_data["intuitive_accuracy"]
        + plot_data["reflective_accuracy"]
    ) / 2.0

    plot_data["number_models_generated"] = (
        plot_data["mean_number_models_generated_int"]
        + plot_data["mean_number_models_generated_ref"]
    ) / 2.0

    # ==============================================================
    # GROUPES DE NOMBRE DE MODÈLES
    # ==============================================================

    plot_data, dynamic_labels = add_model_group(
        plot_data, "number_models_generated", "model_group"
    )

    # ==============================================================
    # FIGURE
    # ==============================================================

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(20, 6.5),
        sharex=True,
        sharey=True,
    )

    figure.suptitle(
        "Confiance et précision globales selon le nombre "
        "moyen de modèles générés",
        fontsize=17,
        fontweight="bold",
        y=0.99,
    )

    group_statistics = []

    dynamic_colors = ["#3b82f6", "#f59e0b", "#ef4444"]
    
    for axis, group_label, color in zip(axes, dynamic_labels, dynamic_colors):
        group_data = plot_data.loc[
            plot_data["model_group"] == group_label
        ].copy()

        if group_data.empty:
            group_statistics.append({
                "group": group_label,
                "number_of_subjects": 0,
                "median_confidence": np.nan,
                "mean_confidence": np.nan,
                "median_accuracy": np.nan,
                "mean_accuracy": np.nan,
                "median_models": np.nan,
                "mean_models": np.nan,
            })

            axis.set_title(
                f"{group_label}\nAucun participant",
                fontweight="bold",
            )

            axis.text(
                0.5,
                0.5,
                "Aucune donnée",
                transform=axis.transAxes,
                ha="center",
                va="center",
                fontsize=13,
                color="gray",
            )

            axis.set_xlabel(
                "Confiance globale moyenne (%)"
            )

            axis.grid(
                True,
                linestyle="--",
                alpha=0.4,
            )

            continue

        participant_count = len(
            group_data
        )

        median_confidence = float(
            group_data["confidence"].median()
        )

        mean_confidence = float(
            group_data["confidence"].mean()
        )

        median_accuracy = float(
            group_data["accuracy"].median()
        )

        mean_accuracy = float(
            group_data["accuracy"].mean()
        )

        median_models = float(
            group_data[
                "number_models_generated"
            ].median()
        )

        mean_models = float(
            group_data[
                "number_models_generated"
            ].mean()
        )

        group_statistics.append({
            "group": group_label,
            "number_of_subjects": participant_count,
            "median_confidence": median_confidence,
            "mean_confidence": mean_confidence,
            "median_accuracy": median_accuracy,
            "mean_accuracy": mean_accuracy,
            "median_models": median_models,
            "mean_models": mean_models,
        })

        # ==========================================================
        # PARTICIPANTS
        # ==========================================================

        axis.scatter(
            group_data["confidence"],
            group_data["accuracy"],
            s=75,
            color=color,
            alpha=0.65,
            edgecolors="white",
            linewidths=0.7,
            label="Participants",
        )

        # ==========================================================
        # MÉDIANE DE CONFIANCE
        # ==========================================================

        axis.axvline(
            median_confidence,
            color="#111827",
            linestyle="--",
            linewidth=2,
            alpha=0.9,
            label=(
                "Médiane confiance : "
                f"{median_confidence:.1f} %"
            ),
        )

        # ==========================================================
        # MÉDIANE DE PRÉCISION
        # ==========================================================

        axis.axhline(
            median_accuracy,
            color="#6b21a8",
            linestyle="--",
            linewidth=2,
            alpha=0.9,
            label=(
                "Médiane précision : "
                f"{median_accuracy:.1f} %"
            ),
        )

        # ==========================================================
        # INTERSECTION DES MÉDIANES
        # ==========================================================

        axis.scatter(
            [median_confidence],
            [median_accuracy],
            marker="D",
            s=150,
            color="black",
            edgecolors="white",
            linewidths=1.2,
            zorder=10,
            label="Intersection des médianes",
        )

        axis.annotate(
            (
                f"Médianes\n"
                f"C = {median_confidence:.1f} %\n"
                f"P = {median_accuracy:.1f} %"
            ),
            xy=(
                median_confidence,
                median_accuracy,
            ),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "black",
                "alpha": 0.85,
            },
        )

        axis.set_title(
            (
                f"{group_label}\n"
                f"n = {participant_count} | "
                f"moyenne modèles = {mean_models:.2f}"
            ),
            fontsize=13,
            fontweight="bold",
        )

        axis.set_xlabel(
            "Confiance globale moyenne (%)"
        )

        axis.grid(
            True,
            linestyle="--",
            alpha=0.4,
        )

        axis.legend(
            loc="best",
            fontsize=8,
            framealpha=0.9,
        )

        # ==========================================================
        # IDENTIFIANTS FACULTATIFS
        # ==========================================================

        if ANNOTATE_SUBJECTS:
            rows_to_annotate = group_data.head(
                MAX_ANNOTATED_SUBJECTS
            )

            for _, row in rows_to_annotate.iterrows():
                axis.annotate(
                    str(row["subject_id"]),
                    (
                        row["confidence"],
                        row["accuracy"],
                    ),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=6,
                    alpha=0.65,
                )

    axes[0].set_ylabel(
        "Précision globale moyenne (%)"
    )

    # Échelles identiques pour les trois panneaux.
    if (
        plot_data["confidence"].min() >= 0
        and plot_data["confidence"].max() <= 100
    ):
        for axis in axes:
            axis.set_xlim(
                0,
                100,
            )

    if (
        plot_data["accuracy"].min() >= 0
        and plot_data["accuracy"].max() <= 100
    ):
        for axis in axes:
            axis.set_ylim(
                0,
                100,
            )

    figure.tight_layout(
        rect=[
            0,
            0,
            1,
            0.92,
        ]
    )

    save_figure(
        figure,
        GROUPED_MEDIANS_OUTPUT_FILE,
    )

    # ==============================================================
    # STATISTIQUES TERMINAL
    # ==============================================================

    print(
        "\nStatistiques globales des groupes de modèles :"
    )

    for statistics in group_statistics:
        if statistics["number_of_subjects"] == 0:
            print(
                f"  {statistics['group']} : "
                "aucun participant"
            )

            continue

        print(
            f"  {statistics['group']} : "
            f"n={statistics['number_of_subjects']}, "
            f"médiane confiance="
            f"{statistics['median_confidence']:.2f} %, "
            f"médiane précision="
            f"{statistics['median_accuracy']:.2f} %, "
            f"moyenne modèles="
            f"{statistics['mean_models']:.4f}"
        )

    # ==============================================================
    # CSV DES STATISTIQUES
    # ==============================================================

    statistics_dataframe = pd.DataFrame(
        group_statistics
    )

    statistics_dataframe.to_csv(
        GROUPED_MEDIANS_STATISTICS_FILE,
        index=False,
    )

    print(
        "Statistiques des groupes enregistrées dans :",
        GROUPED_MEDIANS_STATISTICS_FILE,
    )


# ======================================================================
# 4. COMPARAISON DES DEUX PHASES
# ======================================================================

def create_phase_comparison(dataframe):
    """
    Compare les phases intuitive et réfléchie sur :

        1. la précision ;
        2. la confiance ;
        3. le temps de réponse ;
        4. le nombre de modèles générés.
    """
    required_columns = [
        "subject_id",
        "correct_int",
        "correct_ref",
        "for_int",
        "for_ref",
        "rt_int",
        "rt_ref",
        "number_models_generated_int",
        "number_models_generated_ref",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise KeyError(
            "Colonnes manquantes pour la comparaison des phases : "
            f"{missing_columns}"
        )

    plot_data = (
        dataframe[
            required_columns
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna(
            subset=[
                "subject_id",
                "correct_int",
                "correct_ref",
                "for_int",
                "for_ref",
                "number_models_generated_int",
                "number_models_generated_ref",
            ]
        )
        .copy()
    )

    if plot_data.empty:
        warnings.warn(
            "Aucune donnée disponible pour comparer les phases."
        )

        return

    # Une ligne par participant.
    subject_phase_summary = (
        plot_data
        .groupby(
            "subject_id",
            as_index=False,
        )
        .agg(
            intuitive_accuracy=(
                "correct_int",
                "mean",
            ),

            reflective_accuracy=(
                "correct_ref",
                "mean",
            ),

            intuitive_confidence=(
                "for_int",
                "mean",
            ),

            reflective_confidence=(
                "for_ref",
                "mean",
            ),

            intuitive_rt=(
                "rt_int",
                "mean",
            ),

            reflective_rt=(
                "rt_ref",
                "mean",
            ),

            intuitive_models=(
                "number_models_generated_int",
                "mean",
            ),

            reflective_models=(
                "number_models_generated_ref",
                "mean",
            ),
        )
    )

    subject_phase_summary[
        "intuitive_accuracy"
    ] *= 100

    subject_phase_summary[
        "reflective_accuracy"
    ] *= 100

    phase_order = [
        "Intuitif",
        "Réfléchi",
    ]

    phase_colors = [
        "#2563eb",
        "#8b5cf6",
    ]

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10),
    )

    # ------------------------------------------------------------------
    # Précision
    # ------------------------------------------------------------------

    accuracy_means = [
        subject_phase_summary[
            "intuitive_accuracy"
        ].mean(),

        subject_phase_summary[
            "reflective_accuracy"
        ].mean(),
    ]

    accuracy_errors = [
        standard_error(
            subject_phase_summary[
                "intuitive_accuracy"
            ]
        ),

        standard_error(
            subject_phase_summary[
                "reflective_accuracy"
            ]
        ),
    ]

    axes[0, 0].bar(
        phase_order,
        accuracy_means,
        yerr=accuracy_errors,
        capsize=5,
        color=phase_colors,
        alpha=0.8,
        edgecolor="white",
    )

    axes[0, 0].set_title(
        "Précision selon la phase",
        fontweight="bold",
    )

    axes[0, 0].set_ylabel(
        "Réponses correctes (%)"
    )

    axes[0, 0].set_ylim(
        0,
        100,
    )

    # ------------------------------------------------------------------
    # Confiance
    # ------------------------------------------------------------------

    confidence_means = [
        subject_phase_summary[
            "intuitive_confidence"
        ].mean(),

        subject_phase_summary[
            "reflective_confidence"
        ].mean(),
    ]

    confidence_errors = [
        standard_error(
            subject_phase_summary[
                "intuitive_confidence"
            ]
        ),

        standard_error(
            subject_phase_summary[
                "reflective_confidence"
            ]
        ),
    ]

    axes[0, 1].bar(
        phase_order,
        confidence_means,
        yerr=confidence_errors,
        capsize=5,
        color=phase_colors,
        alpha=0.8,
        edgecolor="white",
    )

    axes[0, 1].set_title(
        "Confiance selon la phase",
        fontweight="bold",
    )

    axes[0, 1].set_ylabel(
        "Confiance"
    )

    axes[0, 1].set_ylim(
        0,
        100,
    )

    # ------------------------------------------------------------------
    # Temps de réponse
    # ------------------------------------------------------------------

    rt_data = subject_phase_summary.dropna(
        subset=[
            "intuitive_rt",
            "reflective_rt",
        ]
    )

    rt_means = [
        rt_data[
            "intuitive_rt"
        ].mean(),

        rt_data[
            "reflective_rt"
        ].mean(),
    ]

    rt_errors = [
        standard_error(
            rt_data[
                "intuitive_rt"
            ]
        ),

        standard_error(
            rt_data[
                "reflective_rt"
            ]
        ),
    ]

    axes[1, 0].bar(
        phase_order,
        rt_means,
        yerr=rt_errors,
        capsize=5,
        color=phase_colors,
        alpha=0.8,
        edgecolor="white",
    )

    axes[1, 0].set_title(
        "Temps de réponse selon la phase",
        fontweight="bold",
    )

    axes[1, 0].set_ylabel(
        "Temps de réponse moyen"
    )

    # ------------------------------------------------------------------
    # Nombre de modèles
    # ------------------------------------------------------------------

    model_means = [
        subject_phase_summary[
            "intuitive_models"
        ].mean(),

        subject_phase_summary[
            "reflective_models"
        ].mean(),
    ]

    model_errors = [
        standard_error(
            subject_phase_summary[
                "intuitive_models"
            ]
        ),

        standard_error(
            subject_phase_summary[
                "reflective_models"
            ]
        ),
    ]

    axes[1, 1].bar(
        phase_order,
        model_means,
        yerr=model_errors,
        capsize=5,
        color=phase_colors,
        alpha=0.8,
        edgecolor="white",
    )

    axes[1, 1].set_title(
        "Nombre de modèles selon l'ajustement",
        fontweight="bold",
    )

    axes[1, 1].set_ylabel(
        "Nombre moyen de modèles générés"
    )

    for axis in axes.flatten():
        axis.grid(
            axis="y",
            linestyle="--",
            alpha=0.4,
        )

    figure.suptitle(
        "Comparaison des phases intuitive et réfléchie",
        fontsize=17,
        fontweight="bold",
    )

    figure.tight_layout(
        rect=[
            0,
            0,
            1,
            0.95,
        ]
    )

    save_figure(
        figure,
        PHASE_COMPARISON_OUTPUT_FILE,
    )


# ======================================================================
# COMPARAISON DU NOMBRE DE MODÈLES ENTRE LES DEUX AJUSTEMENTS
# ======================================================================

def create_model_count_phase_comparison(subject_summary):
    """
    Compare, au niveau participant, le nombre moyen de modèles générés :

        - avec les paramètres ajustés sur les réponses intuitives ;
        - avec les paramètres ajustés sur les réponses réfléchies.

    Chaque participant contribue exactement une observation par
    ajustement.

    La différence est définie comme :

        modèles réfléchis - modèles intuitifs

    Une différence positive indique donc davantage de modèles générés
    avec l'ajustement réfléchi.
    """
    required_columns = [
        "subject_id",
        "mean_number_models_generated_int",
        "mean_number_models_generated_ref",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in subject_summary.columns
    ]

    if missing_columns:
        raise KeyError(
            "Colonnes manquantes pour comparer le nombre de modèles "
            f"entre les deux ajustements : {missing_columns}"
        )

    plot_data = (
        subject_summary[required_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna(
            subset=[
                "mean_number_models_generated_int",
                "mean_number_models_generated_ref",
            ]
        )
        .copy()
    )

    if plot_data.empty:
        warnings.warn(
            "Aucune donnée disponible pour comparer le nombre de "
            "modèles entre les ajustements intuitif et réfléchi."
        )
        return

    # Différence appariée pour chaque participant.
    plot_data["models_change"] = (
        plot_data["mean_number_models_generated_ref"]
        - plot_data["mean_number_models_generated_int"]
    )

    plot_data["more_models_ref"] = (
        plot_data["models_change"] > 0
    )

    plot_data["more_models_int"] = (
        plot_data["models_change"] < 0
    )

    plot_data["same_number_models"] = np.isclose(
        plot_data["models_change"],
        0.0,
        rtol=1e-9,
        atol=1e-9,
    )

    plot_data, dynamic_labels = add_model_group(
        plot_data, "mean_number_models_generated_ref", "model_group_ref"
    )

    # --------------------------------------------------------------
    # Statistiques descriptives
    # --------------------------------------------------------------

    number_of_subjects = len(
        plot_data
    )

    mean_int = float(
        plot_data[
            "mean_number_models_generated_int"
        ].mean()
    )

    mean_ref = float(
        plot_data[
            "mean_number_models_generated_ref"
        ].mean()
    )

    sem_int = standard_error(
        plot_data[
            "mean_number_models_generated_int"
        ]
    )

    sem_ref = standard_error(
        plot_data[
            "mean_number_models_generated_ref"
        ]
    )

    mean_change = float(
        plot_data["models_change"].mean()
    )

    median_change = float(
        plot_data["models_change"].median()
    )

    sem_change = standard_error(
        plot_data["models_change"]
    )

    number_more_ref = int(
        (
            plot_data["models_change"] > 0
        ).sum()
    )

    number_more_int = int(
        (
            plot_data["models_change"] < 0
        ).sum()
    )

    number_equal = int(
        plot_data[
            "same_number_models"
        ].sum()
    )

    percentage_more_ref = (
        100.0
        * number_more_ref
        / number_of_subjects
    )

    percentage_more_int = (
        100.0
        * number_more_int
        / number_of_subjects
    )

    percentage_equal = (
        100.0
        * number_equal
        / number_of_subjects
    )

    # --------------------------------------------------------------
    # Sauvegarde des différences individuelles
    # --------------------------------------------------------------

    plot_data = (
        plot_data
        .sort_values(
            by="models_change"
        )
        .reset_index(
            drop=True
        )
    )

    plot_data.to_csv(
        MODEL_PHASE_COMPARISON_SUMMARY_FILE,
        index=False,
    )

    print(
        "Comparaison individuelle du nombre de modèles :",
        MODEL_PHASE_COMPARISON_SUMMARY_FILE,
    )

    # --------------------------------------------------------------
    # Création de la figure
    # --------------------------------------------------------------

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(16, 7),
    )

    intuitive_color = "#2563eb"
    reflective_color = "#8b5cf6"

    x_int = 0
    x_ref = 1

    # ==============================================================
    # PANNEAU 1 : COMPARAISON APPARIÉE
    # ==============================================================

    for _, row in plot_data.iterrows():
        intuitive_value = row[
            "mean_number_models_generated_int"
        ]

        reflective_value = row[
            "mean_number_models_generated_ref"
        ]

        change = row["models_change"]

        if change > 0:
            line_color = "#16a34a"

        elif change < 0:
            line_color = "#dc2626"

        else:
            line_color = "#6b7280"

        axes[0].plot(
            [
                x_int,
                x_ref,
            ],
            [
                intuitive_value,
                reflective_value,
            ],
            color=line_color,
            linewidth=0.9,
            alpha=0.30,
            zorder=1,
        )

    dynamic_colors = ["#3b82f6", "#f59e0b", "#ef4444"]
    
    for group_label, color in zip(dynamic_labels, dynamic_colors):
        group_mask = plot_data["model_group_ref"] == group_label
        axes[0].scatter(
            np.full(group_mask.sum(), x_ref),
            plot_data.loc[group_mask, "mean_number_models_generated_ref"],
            color=color,
            s=45,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
            label=f"Réfléchi : {group_label}",
            zorder=3,
        )

    axes[0].scatter(
        np.full(
            number_of_subjects,
            x_ref,
        ),
        plot_data[
            "mean_number_models_generated_ref"
        ],
        color=reflective_color,
        s=38,
        alpha=0.65,
        edgecolors="white",
        linewidths=0.4,
        label="Ajustement réfléchi",
        zorder=2,
    )

    # Moyennes générales et erreurs standards.
    axes[0].errorbar(
        x_int,
        mean_int,
        yerr=sem_int,
        fmt="D",
        markersize=10,
        color="#111827",
        markerfacecolor=intuitive_color,
        markeredgecolor="white",
        markeredgewidth=0.8,
        capsize=6,
        linewidth=2,
        zorder=10,
    )

    axes[0].errorbar(
        x_ref,
        mean_ref,
        yerr=sem_ref,
        fmt="D",
        markersize=10,
        color="#111827",
        markerfacecolor=reflective_color,
        markeredgecolor="white",
        markeredgewidth=0.8,
        capsize=6,
        linewidth=2,
        zorder=10,
    )

    axes[0].set_xticks([
        x_int,
        x_ref,
    ])

    axes[0].set_xticklabels([
        "Ajustement\nintuitif",
        "Ajustement\nréfléchi",
    ])

    axes[0].set_xlim(
        -0.35,
        1.35,
    )

    axes[0].set_title(
        "Comparaison appariée par participant",
        fontsize=14,
        fontweight="bold",
    )

    axes[0].set_ylabel(
        "Nombre moyen de modèles générés"
    )

    axes[0].grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    axes[0].legend(
        loc="best",
        fontsize=9,
    )

    means_text = (
        f"Moyenne intuitive : {mean_int:.3f}\n"
        f"Moyenne réfléchie : {mean_ref:.3f}\n"
        f"Différence moyenne : {mean_change:+.3f}"
    )

    axes[0].text(
        0.03,
        0.97,
        means_text,
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "edgecolor": "#9ca3af",
            "alpha": 0.9,
        },
    )

    # ==============================================================
    # PANNEAU 2 : DISTRIBUTION DES DIFFÉRENCES
    # ==============================================================

    axes[1].hist(
        plot_data["models_change"],
        bins="auto",
        color="#14b8a6",
        alpha=0.78,
        edgecolor="white",
        linewidth=0.8,
    )

    axes[1].axvline(
        0,
        color="#111827",
        linestyle="--",
        linewidth=1.8,
        label="Aucune différence",
    )

    axes[1].axvline(
        mean_change,
        color="#dc2626",
        linestyle="-",
        linewidth=2.2,
        label=(
            "Différence moyenne : "
            f"{mean_change:+.3f}"
        ),
    )

    axes[1].axvline(
        median_change,
        color="#f59e0b",
        linestyle=":",
        linewidth=2.2,
        label=(
            "Différence médiane : "
            f"{median_change:+.3f}"
        ),
    )

    axes[1].set_title(
        "Distribution des différences individuelles",
        fontsize=14,
        fontweight="bold",
    )

    axes[1].set_xlabel(
        "Nombre de modèles : réfléchi − intuitif"
    )

    axes[1].set_ylabel(
        "Nombre de participants"
    )

    axes[1].grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    axes[1].legend(
        loc="best",
        fontsize=9,
    )

    comparison_text = (
        f"Réfléchi > intuitif : "
        f"{number_more_ref}/{number_of_subjects} "
        f"({percentage_more_ref:.1f} %)\n"
        f"Réfléchi < intuitif : "
        f"{number_more_int}/{number_of_subjects} "
        f"({percentage_more_int:.1f} %)\n"
        f"Égalité : "
        f"{number_equal}/{number_of_subjects} "
        f"({percentage_equal:.1f} %)\n"
        f"SEM de la différence : {sem_change:.3f}"
    )

    axes[1].text(
        0.03,
        0.97,
        comparison_text,
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "edgecolor": "#9ca3af",
            "alpha": 0.9,
        },
    )

    figure.suptitle(
        "Nombre de modèles avec les ajustements intuitif et réfléchi",
        fontsize=17,
        fontweight="bold",
    )

    figure.tight_layout(
        rect=[
            0,
            0,
            1,
            0.94,
        ]
    )

    save_figure(
        figure,
        MODEL_PHASE_COMPARISON_OUTPUT_FILE,
    )

    # --------------------------------------------------------------
    # Résumé dans le terminal
    # --------------------------------------------------------------

    print(
        "\nComparaison du nombre de modèles au niveau participant :"
    )

    print(
        "  Nombre de participants :",
        number_of_subjects,
    )

    print(
        "  Moyenne avec ajustement intuitif :",
        round(
            mean_int,
            4,
        ),
    )

    print(
        "  Moyenne avec ajustement réfléchi :",
        round(
            mean_ref,
            4,
        ),
    )

    print(
        "  Différence moyenne réfléchi − intuitif :",
        round(
            mean_change,
            4,
        ),
    )

    print(
        "  Différence médiane réfléchi − intuitif :",
        round(
            median_change,
            4,
        ),
    )

    print(
        "  Participants avec davantage de modèles "
        "sous l'ajustement réfléchi :",
        f"{number_more_ref}/{number_of_subjects}",
        f"({percentage_more_ref:.2f} %)",
    )

    print(
        "  Participants avec moins de modèles "
        "sous l'ajustement réfléchi :",
        f"{number_more_int}/{number_of_subjects}",
        f"({percentage_more_int:.2f} %)",
    )

    print(
        "  Participants sans différence :",
        f"{number_equal}/{number_of_subjects}",
        f"({percentage_equal:.2f} %)",
    )


# ======================================================================
# 5. ANALYSE DU CONFLIT
# ======================================================================

def create_conflict_analysis(dataframe):
    """
    Compare les essais avec et sans conflit sur :

        1. le nombre de modèles intuitif et réfléchi ;
        2. la précision intuitive et réfléchie ;
        3. la confiance intuitive et réfléchie.
    """
    required_columns = [
        "subject_id",
        "sequence",
        "conflict",
        "number_models_generated_int",
        "number_models_generated_ref",
        "correct_int",
        "correct_ref",
        "for_int",
        "for_ref",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise KeyError(
            "Colonnes manquantes pour l'analyse du conflit : "
            f"{missing_columns}"
        )

    plot_data = (
        dataframe[
            required_columns
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
        .copy()
    )

    plot_data = plot_data.loc[
        plot_data["conflict"].isin(
            [
                0,
                1,
            ]
        )
    ].copy()

    if plot_data.empty:
        warnings.warn(
            "Aucune donnée disponible pour l'analyse du conflit."
        )

        return

    plot_data["conflict_label"] = (
        plot_data["conflict"]
        .map(CONFLICT_LABELS)
    )

    conflict_order = [
        "Sans conflit",
        "Conflit",
    ]

    # Une ligne par participant et condition de conflit.
    subject_conflict_summary = (
        plot_data
        .groupby([
            "subject_id",
            "conflict_label",
            ],
            as_index=False,
        )
        .agg(
            mean_models_int=(
                "number_models_generated_int",
                "mean",
            ),

            mean_models_ref=(
                "number_models_generated_ref",
                "mean",
            ),

            intuitive_accuracy=(
                "correct_int",
                "mean",
            ),

            reflective_accuracy=(
                "correct_ref",
                "mean",
            ),

            mean_confidence_intuitive=(
                "for_int",
                "mean",
            ),

            mean_confidence_reflective=(
                "for_ref",
                "mean",
            ),

            number_of_trials=(
                "sequence",
                "count",
            ),
        )
    )

    subject_conflict_summary[
        "intuitive_accuracy"
    ] *= 100

    subject_conflict_summary[
        "reflective_accuracy"
    ] *= 100

    summary = (
        subject_conflict_summary
        .groupby(
            "conflict_label",
            as_index=False,
        )
        .agg(
            mean_models_int=(
                "mean_models_int",
                "mean",
            ),

            sem_models_int=(
                "mean_models_int",
                standard_error,
            ),

            mean_models_ref=(
                "mean_models_ref",
                "mean",
            ),

            sem_models_ref=(
                "mean_models_ref",
                standard_error,
            ),

            mean_intuitive_accuracy=(
                "intuitive_accuracy",
                "mean",
            ),

            sem_intuitive_accuracy=(
                "intuitive_accuracy",
                standard_error,
            ),

            mean_reflective_accuracy=(
                "reflective_accuracy",
                "mean",
            ),

            sem_reflective_accuracy=(
                "reflective_accuracy",
                standard_error,
            ),

            mean_confidence_intuitive=(
                "mean_confidence_intuitive",
                "mean",
            ),

            sem_confidence_intuitive=(
                "mean_confidence_intuitive",
                standard_error,
            ),

            mean_confidence_reflective=(
                "mean_confidence_reflective",
                "mean",
            ),

            sem_confidence_reflective=(
                "mean_confidence_reflective",
                standard_error,
            ),

            number_of_subjects=(
                "subject_id",
                "nunique",
            ),

            number_of_trials=(
                "number_of_trials",
                "sum",
            ),
        )
    )

    summary = (
        summary
        .set_index(
            "conflict_label"
        )
        .reindex(
            conflict_order
        )
        .reset_index()
    )

    summary.to_csv(
        CONFLICT_SUMMARY_FILE,
        index=False,
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5.8),
    )

    x_positions = np.arange(
        len(conflict_order)
    )

    width = 0.36

    # ------------------------------------------------------------------
    # Nombre de modèles
    # ------------------------------------------------------------------

    axes[0].bar(
        x_positions - width / 2,
        summary["mean_models_int"],
        width,
        yerr=summary["sem_models_int"],
        capsize=4,
        label="Ajustement intuitif",
        color="#2563eb",
        alpha=0.8,
        edgecolor="white",
    )

    axes[0].bar(
        x_positions + width / 2,
        summary["mean_models_ref"],
        width,
        yerr=summary["sem_models_ref"],
        capsize=4,
        label="Ajustement réfléchi",
        color="#8b5cf6",
        alpha=0.8,
        edgecolor="white",
    )

    axes[0].set_xticks(
        x_positions
    )

    axes[0].set_xticklabels(
        conflict_order
    )

    axes[0].set_title(
        "Nombre de modèles",
        fontweight="bold",
    )

    axes[0].set_ylabel(
        "Nombre moyen de modèles générés"
    )

    axes[0].legend()

    # ------------------------------------------------------------------
    # Précision
    # ------------------------------------------------------------------

    axes[1].bar(
        x_positions - width / 2,
        summary["mean_intuitive_accuracy"],
        width,
        yerr=summary["sem_intuitive_accuracy"],
        capsize=4,
        label="Phase intuitive",
        color="#2563eb",
        alpha=0.8,
        edgecolor="white",
    )

    axes[1].bar(
        x_positions + width / 2,
        summary["mean_reflective_accuracy"],
        width,
        yerr=summary["sem_reflective_accuracy"],
        capsize=4,
        label="Phase réfléchie",
        color="#8b5cf6",
        alpha=0.8,
        edgecolor="white",
    )

    axes[1].set_xticks(
        x_positions
    )

    axes[1].set_xticklabels(
        conflict_order
    )

    axes[1].set_title(
        "Précision",
        fontweight="bold",
    )

    axes[1].set_ylabel(
        "Réponses correctes (%)"
    )

    axes[1].set_ylim(
        0,
        100,
    )

    axes[1].legend()

    # ------------------------------------------------------------------
    # Confiance
    # ------------------------------------------------------------------

    axes[2].bar(
        x_positions - width / 2,
        summary["mean_confidence_intuitive"],
        width,
        yerr=summary["sem_confidence_intuitive"],
        capsize=4,
        label="Phase intuitive",
        color="#22c55e",
        alpha=0.8,
        edgecolor="white",
    )

    axes[2].bar(
        x_positions + width / 2,
        summary["mean_confidence_reflective"],
        width,
        yerr=summary["sem_confidence_reflective"],
        capsize=4,
        label="Phase réfléchie",
        color="#8b5cf6",
        alpha=0.8,
        edgecolor="white",
    )

    axes[2].set_xticks(
        x_positions
    )

    axes[2].set_xticklabels(
        conflict_order
    )

    axes[2].set_title(
        "Confiance",
        fontweight="bold",
    )

    axes[2].set_ylabel(
        "Confiance"
    )

    axes[2].set_ylim(
        0,
        100,
    )

    axes[2].legend()

    for axis in axes:
        axis.grid(
            axis="y",
            linestyle="--",
            alpha=0.4,
        )

    figure.suptitle(
        "Effet du conflit sur les modèles, la précision et la confiance",
        fontsize=16,
        fontweight="bold",
    )

    figure.tight_layout(
        rect=[
            0,
            0,
            1,
            0.93,
        ]
    )

    save_figure(
        figure,
        CONFLICT_OUTPUT_FILE,
    )

    print(
        "Résumé de l'analyse du conflit :",
        CONFLICT_SUMMARY_FILE,
    )


# ======================================================================
# 6. CONFIANCE SELON LES GROUPES DE MODÈLES
# ======================================================================

def create_confidence_phase_analysis(dataframe):
    """
    Compare la confiance intuitive et réfléchie selon le nombre moyen
    de modèles obtenu dans l'ajustement correspondant.

    Pour la phase intuitive :
        groupe calculé avec number_models_generated_int.

    Pour la phase réfléchie :
        groupe calculé avec number_models_generated_ref.
    """
    subject_data = (
        dataframe
        .groupby(
            "subject_id",
            as_index=False,
        )
        .agg(
            mean_models_int=(
                "number_models_generated_int",
                "mean",
            ),

            mean_models_ref=(
                "number_models_generated_ref",
                "mean",
            ),

            mean_for_int=(
                "for_int",
                "mean",
            ),

            mean_for_ref=(
                "for_ref",
                "mean",
            ),
        )
    )

    intuitive_data = (
        subject_data[
            [
                "subject_id",
                "mean_models_int",
                "mean_for_int",
        ]]
        .rename(
            columns={
                "mean_models_int":
                    "mean_models",

                "mean_for_int":
                    "confidence",
            }
        )
        .copy()
    )

    intuitive_data["phase"] = (
        "Intuitive"
    )

    reflective_data = (
        subject_data[[
            "subject_id",
            "mean_models_ref",
            "mean_for_ref",
        ]]
        .rename(
            columns={
                "mean_models_ref":
                    "mean_models",

                "mean_for_ref":
                    "confidence",
            }
        )
        .copy()
    )

    reflective_data["phase"] = (
        "Réfléchie"
    )

    long_data = pd.concat([
        intuitive_data,
        reflective_data,
        ],
        ignore_index=True,
    )

    long_data, dynamic_labels = add_model_group(
            dataframe=long_data,
            models_column="mean_models",
            output_column="model_group",
        )

    summary = (
        long_data
        .groupby(
            [
                "model_group",
                "phase",
            ],
            observed=True,
            as_index=False,
        )
        .agg(
            mean_confidence=(
                "confidence",
                "mean",
            ),

            sem_confidence=(
                "confidence",
                standard_error,
            ),

            number_of_subjects=(
                "subject_id",
                "nunique",
            ),
        )
    )

    figure, axis = plt.subplots(
        figsize=(11, 7)
    )

    x_positions = np.arange(
        len(dynamic_labels)
    )

    width = 0.36

    phase_settings = [
        (
            "Intuitive",
            "#22c55e",
        ),
        (
            "Réfléchie",
            "#8b5cf6",
        ),
    ]

    for phase_index, (
        phase,
        color,
    ) in enumerate(
        phase_settings
    ):
        phase_summary = (
            summary.loc[
                summary["phase"] == phase
            ]
            .set_index(
                "model_group"
            )
            .reindex(
                dynamic_labels
            )
        )

        offset = (
            -width / 2
            if phase_index == 0
            else width / 2
        )

        axis.bar(
            x_positions + offset,
            phase_summary[
                "mean_confidence"
            ],
            width,
            yerr=phase_summary[
                "sem_confidence"
            ],
            capsize=5,
            color=color,
            alpha=0.8,
            label=phase,
        )

    axis.set_xticks(
        x_positions
    )

    axis.set_xticklabels(
        dynamic_labels
    )

    axis.set_title(
        "Confiance selon le nombre de modèles et la phase",
        fontsize=15,
        fontweight="bold",
    )

    axis.set_xlabel(
        "Groupe de modèles"
    )

    axis.set_ylabel(
        "Confiance moyenne"
    )

    axis.set_ylim(
        0,
        100,
    )

    axis.legend(
        title="Phase"
    )

    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    figure.tight_layout()

    save_figure(
        figure,
        CONFIDENCE_PHASE_OUTPUT_FILE,
    )


# ======================================================================
# STATISTIQUES TERMINAL
# ======================================================================

def print_correlations(subject_summary):
    """
    Affiche les corrélations correspondant aux deux ajustements.

    Les corrélations sont uniquement affichées dans le terminal.
    Elles ne sont pas ajoutées aux graphiques.
    """
    correlations = [
        (
            "Confiance intuitive ↔ précision intuitive",
            "mean_for_int",
            "intuitive_accuracy",
        ),
        (
            "Confiance réfléchie ↔ précision réfléchie",
            "mean_for_ref",
            "reflective_accuracy",
        ),
        (
            "Confiance intuitive ↔ modèles intuitifs",
            "mean_for_int",
            "mean_number_models_generated_int",
        ),
        (
            "Confiance réfléchie ↔ modèles réfléchis",
            "mean_for_ref",
            "mean_number_models_generated_ref",
        ),
        (
            "Modèles intuitifs ↔ précision intuitive",
            "mean_number_models_generated_int",
            "intuitive_accuracy",
        ),
        (
            "Modèles réfléchis ↔ précision réfléchie",
            "mean_number_models_generated_ref",
            "reflective_accuracy",
        ),
        (
            "Modèles intuitifs ↔ modèles réfléchis",
            "mean_number_models_generated_int",
            "mean_number_models_generated_ref",
        ),
    ]

    print(
        "\nCorrélations de Pearson au niveau participant :"
    )

    for label, x_column, y_column in correlations:
        correlation = safe_correlation(
            subject_summary,
            x_column,
            y_column,
        )

        print(
            f"  {label} : "
            f"{format_correlation(correlation)}"
        )


# ======================================================================
# PROGRAMME PRINCIPAL
# ======================================================================

def main():
    print("=" * 80)
    print(
        "ANALYSE MREASONER — "
        "AJUSTEMENTS INTUITIF ET RÉFLÉCHI — DATASET E2"
    )
    print("=" * 80)

    try:
        detailed_data = (
            load_detailed_data()
        )

        subject_summary = (
            build_subject_summary(
                detailed_data
            )
        )

        print(
            "\nAperçu du résumé participant :"
        )

        print(
            subject_summary
            .head(10)
            .to_string(index=False)
        )

        print_correlations(
            subject_summary
        )

        print(
            "\n1. Génération des quadrants..."
        )

        create_quadrant_plot(
            subject_summary
        )

        print(
            "\n2. Génération des espaces 3D..."
        )

        create_three_dimensional_plot(
            subject_summary
        )

        print(
            "\n3. Génération des projections croisées..."
        )

        create_crossed_projections(
            subject_summary
        )

        print(
            "\n4. Comparaison des phases intuitive et réfléchie..."
        )

        create_phase_comparison(
            detailed_data
        )

        print(
            "\n5. Analyse des essais avec et sans conflit..."
        )

        create_conflict_analysis(
            detailed_data
        )

        print(
            "\n6. Analyse de la confiance selon le nombre de modèles..."
        )

        create_confidence_phase_analysis(
            detailed_data
        )

        print(
            "\n7. Comparaison du nombre de modèles entre les ajustements..."
        )

        create_model_count_phase_comparison(
            subject_summary
        )

        print(
            "\n8. Génération des médianes globales par groupe de modèles..."
        )

        create_grouped_medians_plot(
            subject_summary
        )

        print(
            "\n9. Génération des médianes par groupe de modèles (Cas Réfléchi uniquement)..."
        )

        create_grouped_medians_plot_ref_only(
            subject_summary
        )


    except (
        FileNotFoundError,
        KeyError,
        ValueError,
        RuntimeError,
        pd.errors.ParserError,
    ) as error:
        print(
            "\nERREUR :",
            error,
        )

        sys.exit(1)

    print("\n" + "=" * 80)
    print("ANALYSE TERMINÉE")
    print("=" * 80)

    print(
        "Dossier des résultats :",
        OUTPUT_DIRECTORY,
    )

    print(
        "Nombre de participants :",
        subject_summary[
            "subject_id"
        ].nunique(),
    )

    print(
        "Nombre d'essais :",
        len(detailed_data),
    )

    print(
        "\nNombre moyen de modèles — ajustement intuitif :",
        round(
            detailed_data[
                "number_models_generated_int"
            ].mean(),
            4,
        ),
    )

    print(
        "Nombre moyen de modèles — ajustement réfléchi :",
        round(
            detailed_data[
                "number_models_generated_ref"
            ].mean(),
            4,
        ),
    )

    print(
        "Différence moyenne réfléchi − intuitif :",
        round(
            detailed_data[
                "models_change"
            ].mean(),
            4,
        ),
    )

    print(
        "\nPrécision intuitive globale :",
        round(
            detailed_data[
                "correct_int"
            ].mean() * 100,
            4,
        ),
        "%",
    )

    print(
        "Précision réfléchie globale :",
        round(
            detailed_data[
                "correct_ref"
            ].mean() * 100,
            4,
        ),
        "%",
    )

    print(
        "\nGroupes de modèles — ajustement intuitif :"
    )

    print(
        subject_summary[
            "model_group_int"
        ]

        .value_counts(
            dropna=False,
            sort=False,
        )
        .to_string()
    )

    print(
        "\nGroupes de modèles — ajustement réfléchi :"
    )

    print(
        subject_summary[
            "model_group_ref"
        ]
        .value_counts(
            dropna=False,
            sort=False,
        )
        .to_string()
    )

    if SHOW_FIGURES:
        plt.show()

    else:
        plt.close("all")


if __name__ == "__main__":
    main() 