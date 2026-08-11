import ast
import json
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ======================================================================
# CONFIGURATION
# ======================================================================

BASE_DIR = os.path.dirname(
    os.path.abspath('../../../..')
)

# Fichier contenant le nombre de modèles par participant et par tâche
MODELS_FILE = os.path.join(
    BASE_DIR,
    "results",
    "tables",
    "mental_models",
    "mental_models_count_E1.csv",
)

# Le script essaie de retrouver automatiquement le fichier humain depuis
# benchmark_E1.json. Si cela ne fonctionne pas, indique ici son chemin.
#
# Exemple :
#
# HUMAN_DATA_FILE = os.path.join(
#     BASE_DIR,
#     "data",
#     "E1.csv",
# )
#
HUMAN_DATA_FILE = None

BENCHMARK_FILE = os.path.join(
    BASE_DIR,
    "config",
    "benchmarks",
    "benchmark_E1.json",
)

# Fichier agrégé produit par ce script.
MERGED_OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "results",
    "analysis",
    "confidence",
    "E1",
    "quadrant_data_by_subject.csv",
)

# Figures produites.
QUADRANT_OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "results",
    "analysis",
    "confidence",
    "E1",
    "quadrant_confidence_accuracy_models.png",
)

THREE_DIMENSIONAL_OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "results",
    "analysis",
    "confidence",
    "E1",
    "analyse_3d_confidence_accuracy_models.png",
)

TWO_DIMENSIONAL_OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "results",
    "analysis",
    "confidence",
    "E1",
    "analyse_2d_confidence_accuracy_models.png",
)

GROUPED_MEDIANS_OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "results",
    "analysis",
    "confidence",
    "E1",
    "grouped_medians_confidence_accuracy.png",
)


# ----------------------------------------------------------------------
# Réponses correctes
# ----------------------------------------------------------------------
#
# Le script utilise en priorité :
#
# 1. une colonne is_correct / correct / accuracy si elle existe ;
# 2. une colonne correct_response / expected_response si elle existe ;
# 3. sinon ce dictionnaire.
#
# Les identifiants doivent correspondre à la colonne "task" de
# mental_models_count_E1.csv.
#
# IMPORTANT :
# vérifie les réponses correctes pour tes quatre tâches.
#
CORRECT_RESPONSES_BY_TASK = {
    # Exemple seulement : à valider selon ton expérience E1.
    1: "No",
    2: "Yes",
    3: "No",
    4: "No",
}

# Si la base humaine possède directement une colonne indiquant si la
# réponse est correcte, le dictionnaire précédent ne sera pas utilisé.


# ----------------------------------------------------------------------
# Configuration de la confiance
# ----------------------------------------------------------------------
#
# Le script détecte automatiquement des colonnes appelées par exemple :
#
#     confidence
#     confidence_rating
#     certainty
#     confiance
#
# Si les valeurs de confiance sont comprises entre 0 et 1, elles seront
# converties en pourcentage.
#
# Si elles sont comprises entre 1 et 5, 1 et 7, etc., elles seront
# normalisées entre 0 et 100 selon le minimum et le maximum observés.
#
NORMALIZE_CONFIDENCE_TO_PERCENTAGE = True


# ----------------------------------------------------------------------
# Affichage
# ----------------------------------------------------------------------

SHOW_FIGURES = True
DPI = 300

# Affiche l'identifiant du participant à côté de chaque point du
# graphique quadrant. Pour une grande base, il vaut mieux laisser False.
ANNOTATE_SUBJECTS = False

# Nombre maximal de participants annotés si ANNOTATE_SUBJECTS = True.
MAX_ANNOTATED_SUBJECTS = 50

# Seuil utilisé pour afficher un avertissement de fusion.
MINIMUM_MERGE_RATE = 0.80


# ======================================================================
# OUTILS DE DÉTECTION DES FICHIERS ET COLONNES
# ======================================================================

def find_column(dataframe, candidates, required=True):
    """
    Recherche une colonne sans tenir compte de la casse, des espaces
    ou des tirets.
    """

    def normalize_column_name(name):
        return (
            str(name)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

    normalized_columns = {
        normalize_column_name(column): column
        for column in dataframe.columns
    }

    for candidate in candidates:
        normalized_candidate = normalize_column_name(
            candidate
        )

        if normalized_candidate in normalized_columns:
            return normalized_columns[
                normalized_candidate
            ]

    if required:
        raise KeyError(
            "Aucune colonne trouvée parmi "
            f"{candidates!r}.\n"
            "Colonnes disponibles : "
            f"{list(dataframe.columns)!r}"
        )

    return None


def find_dataset_in_object(value):
    """
    Recherche récursivement une clé data.test dans un objet JSON.
    """
    if isinstance(value, dict):
        if "data.test" in value:
            return value["data.test"]

        data_section = value.get("data")

        if (
            isinstance(data_section, dict)
            and "test" in data_section
        ):
            return data_section["test"]

        for nested_value in value.values():
            result = find_dataset_in_object(
                nested_value
            )

            if result is not None:
                return result

    elif isinstance(value, list):
        for element in value:
            result = find_dataset_in_object(
                element
            )

            if result is not None:
                return result

    return None


def detect_human_data_file():
    """
    Détecte automatiquement le fichier humain utilisé par
    benchmark_E1.json.
    """
    if HUMAN_DATA_FILE is not None:
        path = os.path.abspath(
            os.path.expanduser(HUMAN_DATA_FILE)
        )

        if not os.path.isfile(path):
            raise FileNotFoundError(
                "Le fichier HUMAN_DATA_FILE est introuvable : "
                f"{path}"
            )

        return path

    if not os.path.isfile(BENCHMARK_FILE):
        raise FileNotFoundError(
            "benchmark_E1.json est introuvable. "
            "Indique manuellement HUMAN_DATA_FILE en haut du script."
        )

    with open(
        BENCHMARK_FILE,
        "r",
        encoding="utf-8",
    ) as input_file:
        benchmark_data = json.load(
            input_file
        )

    candidate = find_dataset_in_object(
        benchmark_data
    )

    if candidate is None:
        raise KeyError(
            "Impossible de retrouver data.test dans "
            "benchmark_E1.json. Indique manuellement "
            "HUMAN_DATA_FILE en haut du script."
        )

    if isinstance(candidate, list):
        if len(candidate) != 1:
            raise ValueError(
                "Plusieurs fichiers de test ont été trouvés : "
                f"{candidate!r}. Indique manuellement "
                "HUMAN_DATA_FILE."
            )

        candidate = candidate[0]

    candidate = os.path.expanduser(
        str(candidate)
    )

    candidate_paths = []

    if os.path.isabs(candidate):
        candidate_paths.append(candidate)

    else:
        # Le chemin peut être relatif au benchmark.
        candidate_paths.append(
            os.path.join(
                BASE_DIR,
                candidate,
            )
        )

        # Il peut aussi être relatif au modèle.
        candidate_paths.append(
            os.path.join(
                BASE_DIR,
                "models",
                "pymreasoner_2",
                candidate,
            )
        )

    for path in candidate_paths:
        absolute_path = os.path.abspath(
            path
        )

        if os.path.isfile(absolute_path):
            return absolute_path

    raise FileNotFoundError(
        "Le fichier humain indiqué dans benchmark_E1.json "
        f"est introuvable : {candidate!r}.\n"
        f"Chemins essayés : {candidate_paths!r}"
    )


# ======================================================================
# NORMALISATION DES IDENTIFIANTS
# ======================================================================

def normalize_subject_id(value):
    """
    Normalise les identifiants afin que 63873 et 63873.0 correspondent.
    """
    if pd.isna(value):
        return None

    value_as_string = str(value).strip()

    if not value_as_string:
        return None

    try:
        numeric_value = float(
            value_as_string
        )

        if numeric_value.is_integer():
            return str(
                int(numeric_value)
            )

    except ValueError:
        pass

    return value_as_string


def unwrap_singleton(value):
    """
    Retire récursivement les listes unitaires.

    Exemples :
        [['Yes']] -> 'Yes'
        ['No']    -> 'No'
    """
    while isinstance(
        value,
        (list, tuple, np.ndarray),
    ):
        if np.size(value) != 1:
            break

        if isinstance(value, np.ndarray):
            value = value.reshape(-1)[0]
        else:
            value = value[0]

    return value


def parse_possible_collection(value):
    """
    Essaie de convertir une chaîne comme "[['Yes']]" en liste Python.
    """
    if not isinstance(value, str):
        return value

    stripped = value.strip()

    if not stripped:
        return value

    if not (
        stripped.startswith("[")
        or stripped.startswith("(")
        or stripped.startswith("{")
    ):
        return value

    # D'abord JSON.
    try:
        return json.loads(
            stripped
        )
    except json.JSONDecodeError:
        pass

    # Puis format Python.
    try:
        return ast.literal_eval(
            stripped
        )
    except (
        ValueError,
        SyntaxError,
    ):
        return value


def normalize_response(value):
    """
    Normalise une réponse vers Yes ou No si possible.
    """
    value = parse_possible_collection(
        value
    )

    value = unwrap_singleton(
        value
    )

    normalized = str(
        value
    ).strip().lower()

    if normalized in {
        "yes",
        "y",
        "oui",
        "true",
        "1",
    }:
        return "Yes"

    if normalized in {
        "no",
        "n",
        "non",
        "false",
        "0",
        "nvc",
    }:
        return "No"

    return str(value).strip()


def normalize_task_id(value):
    """
    Normalise l'identifiant d'une tâche vers un entier compris entre 1 et 4.

    Exemples :
        1       -> 1
        1.0     -> 1
        "1"     -> 1
        "1.0"   -> 1

    Les valeurs non numériques ne sont pas utilisées directement :
    les tâches textuelles sont traitées séparément par
    infer_task_id_from_text().
    """
    if pd.isna(value):
        return np.nan

    value_as_string = str(value).strip()

    if not value_as_string:
        return np.nan

    try:
        numeric_value = float(value_as_string)
    except (ValueError, TypeError):
        return np.nan

    if not numeric_value.is_integer():
        return np.nan

    task_id = int(numeric_value)

    if task_id not in {1, 2, 3, 4}:
        return np.nan

    return task_id



# ======================================================================
# IDENTIFICATION DES QUATRE TÂCHES
# ======================================================================

def normalize_task_text(value):
    """
    Transforme une tâche en texte canonique.
    """
    value = parse_possible_collection(
        value
    )

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, (list, tuple)):
        parts = []

        for element in value:
            element = unwrap_singleton(
                element
            )

            if isinstance(
                element,
                (list, tuple),
            ):
                element = " ".join(
                    str(part)
                    for part in element
                )

            element = str(
                element
            ).strip()

            if element:
                parts.append(element)

        return " / ".join(parts)

    return (
        str(value)
        .strip()
        .replace("  ", " ")
    )


def canonical_task_key(value):
    """
    Produit une représentation canonique d'une tâche textuelle.
    """
    normalized = normalize_task_text(value)

    normalized = (
        normalized
        .lower()
        .replace(";", " ")
        .replace("/", " ")
        .replace(",", " ")
        .replace(".", " ")
    )

    # Remplace tous les espaces multiples par un seul espace.
    return " ".join(normalized.split())



# Les quatre tâches trouvées précédemment dans ton cache.
KNOWN_TASKS = {
    canonical_task_key(
        "All B are C / No A are C"
    ): 1,

    canonical_task_key(
        "All B are C / All A are B"
    ): 2,

    canonical_task_key(
        "All B are C / All A are C"
    ): 3,

    canonical_task_key(
        "All B are C / No A are B"
    ): 4,
}


def infer_task_id_from_text(value):
    """
    Identifie l'une des quatre tâches à partir de ses prémisses.
    """
    if pd.isna(value):
        return np.nan

    normalized = canonical_task_key(value)

    task_id = KNOWN_TASKS.get(normalized)

    if task_id is None:
        return np.nan

    return int(task_id)



def add_task_id_to_human_data(dataframe):
    """
    Ajoute une colonne task_id normalisée à la base humaine.
    """
    task_id_column = find_column(
        dataframe,
        [
            "task_id",
            "item_id",
            "problem_id",
            "task_number",
            "condition",
            "item_number",
        ],
        required=False,
    )

    if task_id_column is not None:
        dataframe["task_id_normalized"] = (
            dataframe[task_id_column]
            .apply(normalize_task_id)
        )

        recognized = dataframe[
            "task_id_normalized"
        ].notna().sum()

        if recognized > 0:
            print(
                "Colonne de tâche utilisée :",
                task_id_column,
            )

            return dataframe

    task_column = find_column(
        dataframe,
        [
            "task",
            "premises",
            "problem",
            "item",
        ],
        required=False,
    )

    if task_column is not None:
        dataframe["task_id_normalized"] = (
            dataframe[task_column]
            .apply(infer_task_id_from_text)
        )

        recognized = dataframe[
            "task_id_normalized"
        ].notna().sum()

        print(
            "Colonne textuelle de tâche utilisée :",
            task_column,
        )

        print(
            "Tâches reconnues :",
            recognized,
            "/",
            len(dataframe),
        )

        if recognized > 0:
            return dataframe

    premise_1_column = find_column(
        dataframe,
        [
            "premise_1",
            "premise1",
            "first_premise",
        ],
        required=False,
    )

    premise_2_column = find_column(
        dataframe,
        [
            "premise_2",
            "premise2",
            "second_premise",
        ],
        required=False,
    )

    if (
        premise_1_column is not None
        and premise_2_column is not None
    ):
        combined_tasks = (
            dataframe[premise_1_column]
            .astype(str)
            + " / "
            + dataframe[premise_2_column]
            .astype(str)
        )

        dataframe["task_id_normalized"] = (
            combined_tasks.apply(
                infer_task_id_from_text
            )
        )

        recognized = dataframe[
            "task_id_normalized"
        ].notna().sum()

        print(
            "Tâches reconnues depuis deux colonnes de prémisses :",
            recognized,
            "/",
            len(dataframe),
        )

        if recognized > 0:
            return dataframe

    raise KeyError(
        "Impossible d'identifier les tâches dans la base humaine.\n"
        "Le fichier doit contenir soit une colonne task_id, soit une "
        "colonne task, soit premise_1 et premise_2."
    )


# ======================================================================
# CONFIANCE
# ======================================================================

def normalize_confidence(series):
    """
    Convertit la confiance en échelle 0-100.
    """
    numeric_series = pd.to_numeric(
        series,
        errors="coerce",
    )

    valid_values = numeric_series.dropna()

    if valid_values.empty:
        raise ValueError(
            "La colonne de confiance ne contient aucune "
            "valeur numérique."
        )

    minimum = float(
        valid_values.min()
    )

    maximum = float(
        valid_values.max()
    )

    print(
        f"Échelle de confiance détectée : "
        f"minimum={minimum}, maximum={maximum}"
    )

    if not NORMALIZE_CONFIDENCE_TO_PERCENTAGE:
        return numeric_series

    # Déjà une proportion entre 0 et 1.
    if minimum >= 0 and maximum <= 1:
        print(
            "Confiance convertie de [0, 1] vers [0, 100]."
        )

        return numeric_series * 100

    # Déjà un pourcentage.
    if minimum >= 0 and maximum <= 100:
        # Une échelle entière 1-5 ou 1-7 ne doit pas être interprétée
        # directement comme un pourcentage.
        unique_values = sorted(
            valid_values.unique()
        )

        looks_like_small_rating_scale = (
            maximum <= 10
            and len(unique_values) <= 10
        )

        if not looks_like_small_rating_scale:
            print(
                "La confiance semble déjà exprimée sur 100."
            )

            return numeric_series

    # Échelle de Likert ou autre échelle bornée.
    if np.isclose(
        maximum,
        minimum,
    ):
        warnings.warn(
            "Toutes les valeurs de confiance sont identiques. "
            "Elles seront placées à 50 %."
        )

        return pd.Series(
            np.where(
                numeric_series.notna(),
                50.0,
                np.nan,
            ),
            index=series.index,
        )

    print(
        "Confiance normalisée par min-max vers [0, 100]."
    )

    return (
        100
        * (numeric_series - minimum)
        / (maximum - minimum)
    )


# ======================================================================
# CALCUL DE L'ACCURACY
# ======================================================================

def normalize_boolean_correctness(value):
    """
    Convertit une colonne indiquant la correction en 0 ou 1.
    """
    if pd.isna(value):
        return np.nan

    if isinstance(
        value,
        (bool, np.bool_),
    ):
        return int(value)

    normalized = str(
        value
    ).strip().lower()

    if normalized in {
        "1",
        "true",
        "correct",
        "yes",
        "y",
        "oui",
    }:
        return 1

    if normalized in {
        "0",
        "false",
        "incorrect",
        "wrong",
        "no",
        "n",
        "non",
    }:
        return 0

    try:
        numeric_value = float(
            normalized
        )

        if numeric_value in {0, 1}:
            return int(
                numeric_value
            )

    except ValueError:
        pass

    return np.nan


def add_correctness_to_human_data(dataframe):
    """
    Ajoute une colonne is_correct.

    Priorité :
        1. colonne de correction existante ;
        2. comparaison réponse / réponse correcte ;
        3. dictionnaire CORRECT_RESPONSES_BY_TASK.
    """
    correctness_column = find_column(
        dataframe,
        [
            "is_correct",
            "correct",
            "correctness",
            "accuracy",
            "is_right",
            "score",
        ],
        required=False,
    )

    if correctness_column is not None:
        candidate_correctness = (
            dataframe[correctness_column]
            .apply(
                normalize_boolean_correctness
            )
        )

        if candidate_correctness.notna().any():
            dataframe["is_correct"] = (
                candidate_correctness
            )

            print(
                "Correction lue depuis la colonne :",
                correctness_column,
            )

            return dataframe

    response_column = find_column(
        dataframe,
        [
            "response",
            "answer",
            "participant_response",
            "choice",
        ],
        required=True,
    )

    correct_response_column = find_column(
        dataframe,
        [
            "correct_response",
            "expected_response",
            "correct_answer",
            "solution",
            "ground_truth",
            "truth",
        ],
        required=False,
    )

    dataframe[
        "response_normalized"
    ] = dataframe[
        response_column
    ].apply(
        normalize_response
    )

    print(
        "Colonne de réponse humaine utilisée :",
        response_column,
    )

    if correct_response_column is not None:
        dataframe[
            "correct_response_normalized"
        ] = dataframe[
            correct_response_column
        ].apply(
            normalize_response
        )

        dataframe["is_correct"] = (
            dataframe["response_normalized"]
            == dataframe[
                "correct_response_normalized"
            ]
        ).astype(int)

        print(
            "Réponse correcte lue depuis la colonne :",
            correct_response_column,
        )

        return dataframe

    print(
        "Aucune colonne de réponse correcte trouvée."
    )

    print(
        "Utilisation de CORRECT_RESPONSES_BY_TASK :",
        CORRECT_RESPONSES_BY_TASK,
    )

    dataframe[
        "correct_response_normalized"
    ] = dataframe[
        "task_id_normalized"
    ].map(
        CORRECT_RESPONSES_BY_TASK
    ).apply(
        lambda value: (
            normalize_response(value)
            if pd.notna(value)
            else np.nan
        )
    )

    missing_correct_answers = dataframe[
        "correct_response_normalized"
    ].isna()

    if missing_correct_answers.any():
        missing_tasks = sorted(
            dataframe.loc[
                missing_correct_answers,
                "task_id_normalized",
            ]
            .dropna()
            .unique()
            .tolist(),
            key=str,
        )

        raise ValueError(
            "Certaines tâches n'ont pas de réponse correcte "
            "dans CORRECT_RESPONSES_BY_TASK : "
            f"{missing_tasks!r}"
        )

    dataframe["is_correct"] = (
        dataframe["response_normalized"]
        == dataframe["correct_response_normalized"]
    ).astype(int)

    return_dataframe = dataframe
    return dataframe


# ======================================================================
# GRAPHIQUE PAR GROUPES DE NOMBRE DE MODÈLES
# ======================================================================

def create_grouped_medians_plot(dataframe):
    """
    Répartit les participants en trois groupes selon leur nombre moyen
    de modèles générés :

        Groupe 1 : moins de 2.6 modèles
        Groupe 2 : de 2.6 inclus à 3.0 exclu
        Groupe 3 : 3.0 modèles ou plus

    Pour chaque groupe :
        - chaque point représente un participant ;
        - l'axe X représente la confiance moyenne ;
        - l'axe Y représente le pourcentage de bonnes réponses ;
        - la ligne verticale représente la médiane de confiance ;
        - la ligne horizontale représente la médiane d'accuracy ;
        - le losange noir représente l'intersection des deux médianes.
    """
    plot_data = dataframe.copy()

    # Vérification et conversion des colonnes.
    required_columns = [
        "subject_id",
        "confidence",
        "accuracy",
        "number_models_generated",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in plot_data.columns
    ]

    if missing_columns:
        raise KeyError(
            "Colonnes manquantes pour le graphique groupé : "
            f"{missing_columns}"
        )

    for column in [
        "confidence",
        "accuracy",
        "number_models_generated",
    ]:
        plot_data[column] = pd.to_numeric(
            plot_data[column],
            errors="coerce",
        )

    plot_data = (
        plot_data
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna(
            subset=[
                "confidence",
                "accuracy",
                "number_models_generated",
            ]
        )
        .copy()
    )

    if plot_data.empty:
        raise ValueError(
            "Aucune donnée valide pour générer "
            "le graphique par groupes."
        )

    # Les bornes [2.6, 3.0[ comprennent toutes les valeurs allant
    # de 2.6 à 2.9..., sans laisser de trou entre les groupes.
    group_labels = [
        "< 2,6 modèles",
        "2,6 à < 3 modèles",
        "≥ 3 modèles",
    ]

    plot_data["model_group"] = pd.cut(
        plot_data["number_models_generated"],
        bins=[
            -np.inf,
            2.6,
            3.0,
            np.inf,
        ],
        labels=group_labels,
        right=False,
        include_lowest=True,
    )

    # Conserve l'ordre défini ci-dessus.
    plot_data["model_group"] = pd.Categorical(
        plot_data["model_group"],
        categories=group_labels,
        ordered=True,
    )

    # Couleurs propres aux trois groupes.
    group_colors = {
        "< 2,6 modèles": "#3b82f6",
        "2,6 à < 3 modèles": "#f59e0b",
        "≥ 3 modèles": "#ef4444",
    }

    sns.set_theme(
        style="whitegrid",
        context="notebook",
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(20, 6.5),
        sharex=True,
        sharey=True,
    )

    figure.suptitle(
        "Confiance et performance selon le nombre "
        "de modèles mentaux générés",
        fontsize=17,
        fontweight="bold",
        y=0.99,
    )

    group_statistics = []

    for axis, group_label in zip(
        axes,
        group_labels,
    ):
        group_data = plot_data.loc[
            plot_data["model_group"] == group_label
        ].copy()

        if group_data.empty:
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
                "Confiance moyenne (%)"
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

        median_accuracy = float(
            group_data["accuracy"].median()
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
            "median_accuracy": median_accuracy,
            "mean_models": mean_models,
        })

        # Tous les participants du groupe.
        axis.scatter(
            group_data["confidence"],
            group_data["accuracy"],
            s=75,
            color=group_colors[group_label],
            alpha=0.65,
            edgecolors="white",
            linewidths=0.7,
            label="Participants",
        )

        # Médiane de confiance.
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

        # Médiane du pourcentage de bonnes réponses.
        axis.axhline(
            median_accuracy,
            color="#6b21a8",
            linestyle="--",
            linewidth=2,
            alpha=0.9,
            label=(
                "Médiane bonnes réponses : "
                f"{median_accuracy:.1f} %"
            ),
        )

        # Intersection des deux médianes.
        axis.scatter(
            [median_confidence],
            [median_accuracy],
            marker="D",
            s=150,
            color="black",
            edgecolors="white",
            linewidths=1.2,
            zorder=10,
            label="Point médian",
        )

        # Annotation du point médian.
        axis.annotate(
            (
                f"Médianes\n"
                f"C = {median_confidence:.1f} %\n"
                f"BR = {median_accuracy:.1f} %"
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
            "Confiance moyenne (%)"
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

        # Annotation facultative des participants.
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
        "Pourcentage de bonnes réponses (%)"
    )

    # Échelles identiques pour faciliter la comparaison.
    if (
        plot_data["confidence"].min() >= 0
        and plot_data["confidence"].max() <= 100
    ):
        for axis in axes:
            axis.set_xlim(0, 100)

    if (
        plot_data["accuracy"].min() >= 0
        and plot_data["accuracy"].max() <= 100
    ):
        for axis in axes:
            axis.set_ylim(0, 100)

    figure.tight_layout(
        rect=[
            0,
            0,
            1,
            0.92,
        ]
    )

    output_directory = (
        os.path.dirname(
            GROUPED_MEDIANS_OUTPUT_FILE
        )
        or "."
    )

    os.makedirs(
        output_directory,
        exist_ok=True,
    )

    figure.savefig(
        GROUPED_MEDIANS_OUTPUT_FILE,
        dpi=DPI,
        bbox_inches="tight",
    )

    print(
        "Graphique des médianes par groupe enregistré dans :",
        GROUPED_MEDIANS_OUTPUT_FILE,
    )

    # Affiche également les statistiques dans le terminal.
    print(
        "\nStatistiques des groupes de modèles :"
    )

    for statistics in group_statistics:
        print(
            f"  {statistics['group']} : "
            f"n={statistics['number_of_subjects']}, "
            f"médiane confiance="
            f"{statistics['median_confidence']:.2f} %, "
            f"médiane bonnes réponses="
            f"{statistics['median_accuracy']:.2f} %, "
            f"moyenne modèles="
            f"{statistics['mean_models']:.4f}"
        )

    # Enregistre les statistiques utilisées dans un CSV.
    statistics_file = os.path.join(
        os.path.dirname(
            GROUPED_MEDIANS_OUTPUT_FILE
        ),
        "grouped_medians_statistics.csv",
    )

    pd.DataFrame(
        group_statistics
    ).to_csv(
        statistics_file,
        index=False,
    )

    print(
        "Statistiques des groupes enregistrées dans :",
        statistics_file,
    )


# ======================================================================
# CHARGEMENT ET AGRÉGATION
# ======================================================================

def load_and_prepare_human_data():
    """
    Charge la base humaine et calcule confiance et accuracy.
    """
    human_data_file = detect_human_data_file()

    print(
        "Fichier humain utilisé :",
        human_data_file,
    )

    dataframe = pd.read_csv(
        human_data_file
    )

    print(
        "Colonnes de la base humaine :",
        list(dataframe.columns),
    )

    subject_column = find_column(
        dataframe,
        [
            "subject_id",
            "subject",
            "subj_id",
            "participant_id",
            "participant",
            "person_id",
            "id",
        ],
        required=True,
    )

    confidence_column = find_column(
        dataframe,
        [
            "confidence",
            "confidence_rating",
            "confidence_score",
            "certainty",
            "certainty_rating",
            "confiance",
            "rating",
        ],
        required=True,
    )

    print(
        "Colonne participant utilisée :",
        subject_column,
    )

    print(
        "Colonne confiance utilisée :",
        confidence_column,
    )

    dataframe[
        "subject_id_normalized"
    ] = dataframe[
        subject_column
    ].apply(
        normalize_subject_id
    )

    dataframe = add_task_id_to_human_data(
        dataframe
    )

    dataframe[
        "confidence_percentage"
    ] = normalize_confidence(
        dataframe[confidence_column]
    )

    dataframe = add_correctness_to_human_data(
        dataframe
    )

    # Retire les lignes inutilisables.
    before_drop = len(
        dataframe
    )

    dataframe = dataframe.dropna(
        subset=[
            "subject_id_normalized",
            "task_id_normalized",
            "confidence_percentage",
            "is_correct",
        ]
    ).copy()

    dropped = (
        before_drop - len(dataframe)
    )

    if dropped:
        print(
            "Lignes humaines ignorées car incomplètes :",
            dropped,
        )

    # Une ligne agrégée par participant et tâche.
    by_subject_task = (
        dataframe
        .groupby(
            [
                "subject_id_normalized",
                "task_id_normalized",
            ],
            as_index=False,
        )
        .agg(
            confidence=(
                "confidence_percentage",
                "mean",
            ),
            is_correct=(
                "is_correct",
                "mean",
            ),
            number_of_human_trials=(
                "is_correct",
                "size",
            ),
        )
        .rename(
            columns={
                "subject_id_normalized":
                    "subject_id",
                "task_id_normalized":
                    "task",
            }
        )
    )

    return by_subject_task


def load_and_prepare_model_data():
    """
    Charge le nombre de modèles par participant et tâche.
    """
    if not os.path.isfile(
        MODELS_FILE
    ):
        raise FileNotFoundError(
            "Le fichier du nombre de modèles est introuvable : "
            f"{MODELS_FILE}"
        )

    dataframe = pd.read_csv(
        MODELS_FILE
    )

    print(
        "Fichier de modèles utilisé :",
        MODELS_FILE,
    )

    print(
        "Colonnes du fichier de modèles :",
        list(dataframe.columns),
    )

    subject_column = find_column(
        dataframe,
        [
            "subject_id",
            "subject",
            "subj_id",
            "participant_id",
            "participant",
            "id",
        ],
        required=True,
    )

    task_column = find_column(
        dataframe,
        [
            "task",
            "task_id",
            "item_id",
            "task_number",
        ],
        required=True,
    )

    models_column = find_column(
        dataframe,
        [
            "number_models_generated",
            "number_of_models_generated",
            "models_generated",
            "number_models",
            "mental_models",
        ],
        required=True,
    )

    dataframe["subject_id"] = (
        dataframe[subject_column]
        .apply(normalize_subject_id)
    )

    dataframe["task"] = (
        dataframe[task_column]
        .apply(normalize_task_id)
    )

    dataframe[
        "number_models_generated"
    ] = pd.to_numeric(
        dataframe[models_column],
        errors="coerce",
    )

    dataframe = dataframe.dropna(
        subset=[
            "subject_id",
            "task",
            "number_models_generated",
        ]
    ).copy()

    # Sécurité en cas de doublons.
    dataframe = (
        dataframe
        .groupby(
            [
                "subject_id",
                "task",
            ],
            as_index=False,
        )
        .agg(
            number_models_generated=(
                "number_models_generated",
                "mean",
            )
        )
    )

    return dataframe


def merge_and_aggregate():
    """
    Fusionne données humaines et données mReasoner, puis agrège par
    participant.
    """
    human_data = load_and_prepare_human_data()
    model_data = load_and_prepare_model_data()

    print(
        "\nNombre de lignes participant × tâche humaines :",
        len(human_data),
    )

    print(
        "Nombre de lignes participant × tâche mReasoner :",
        len(model_data),
    )

    merged_by_task = pd.merge(
        human_data,
        model_data,
        on=[
            "subject_id",
            "task",
        ],
        how="inner",
        validate="one_to_one",
    )

    expected_maximum = min(
        len(human_data),
        len(model_data),
    )

    merge_rate = (
        len(merged_by_task)
        / expected_maximum
        if expected_maximum > 0
        else 0
    )

    print(
        "Lignes fusionnées :",
        len(merged_by_task),
    )

    print(
        f"Taux de fusion approximatif : "
        f"{merge_rate * 100:.2f} %"
    )

    if merge_rate < MINIMUM_MERGE_RATE:
        warnings.warn(
            "Le taux de fusion est faible. Vérifie que les identifiants "
            "des participants et des tâches correspondent entre les "
            "deux fichiers."
        )

        human_keys = set(
            zip(
                human_data["subject_id"],
                human_data["task"],
            )
        )

        model_keys = set(
            zip(
                model_data["subject_id"],
                model_data["task"],
            )
        )

        print(
            "Exemples absents du fichier mReasoner :",
            list(human_keys - model_keys)[:10],
        )

        print(
            "Exemples absents de la base humaine :",
            list(model_keys - human_keys)[:10],
        )

    if merged_by_task.empty:
        raise RuntimeError(
            "La fusion entre les données humaines et mReasoner "
            "n'a produit aucune ligne."
        )

    # Agrégation pondérée naturellement par les tâches fusionnées.
    by_subject = (
        merged_by_task
        .groupby(
            "subject_id",
            as_index=False,
        )
        .agg(
            confidence=(
                "confidence",
                "mean",
            ),
            accuracy=(
                "is_correct",
                "mean",
            ),
            number_models_generated=(
                "number_models_generated",
                "mean",
            ),
            number_of_tasks=(
                "task",
                "nunique",
            ),
            number_of_human_trials=(
                "number_of_human_trials",
                "sum",
            ),
        )
    )

    by_subject["accuracy"] = (
        by_subject["accuracy"]
        * 100
    )

    by_subject["confidence"] = (
        by_subject["confidence"]
        .round(4)
    )

    by_subject["accuracy"] = (
        by_subject["accuracy"]
        .round(4)
    )

    by_subject[
        "number_models_generated"
    ] = (
        by_subject[
            "number_models_generated"
        ]
        .round(4)
    )

    by_subject = by_subject.sort_values(
        by="subject_id"
    )

    by_subject.to_csv(
        MERGED_OUTPUT_FILE,
        index=False,
    )

    print(
        "\nDonnées agrégées enregistrées dans :",
        MERGED_OUTPUT_FILE,
    )

    print(
        "Nombre de participants fusionnés :",
        len(by_subject),
    )

    return merged_by_task, by_subject


# ======================================================================
# STATISTIQUES
# ======================================================================

def safe_correlation(dataframe, x_column, y_column):
    """
    Calcule une corrélation de Pearson si elle est définie.
    """
    subset = dataframe[
        [
            x_column,
            y_column,
        ]
    ].dropna()

    if len(subset) < 2:
        return np.nan

    if (
        subset[x_column].nunique() < 2
        or subset[y_column].nunique() < 2
    ):
        return np.nan

    return subset[
        x_column
    ].corr(
        subset[y_column],
        method="pearson",
    )


def print_correlations(dataframe):
    """
    Affiche les corrélations entre les trois variables.
    """
    confidence_accuracy = safe_correlation(
        dataframe,
        "confidence",
        "accuracy",
    )

    confidence_models = safe_correlation(
        dataframe,
        "confidence",
        "number_models_generated",
    )

    accuracy_models = safe_correlation(
        dataframe,
        "accuracy",
        "number_models_generated",
    )

    print("\nCorrélations de Pearson :")

    print(
        "  Confiance ↔ Accuracy :",
        (
            f"{confidence_accuracy:.4f}"
            if pd.notna(confidence_accuracy)
            else "non définie"
        ),
    )

    print(
        "  Confiance ↔ Modèles :",
        (
            f"{confidence_models:.4f}"
            if pd.notna(confidence_models)
            else "non définie"
        ),
    )

    print(
        "  Accuracy ↔ Modèles :",
        (
            f"{accuracy_models:.4f}"
            if pd.notna(accuracy_models)
            else "non définie"
        ),
    )


# ======================================================================
# GRAPHIQUE QUADRANT
# ======================================================================

def scale_point_sizes(values):
    """
    Convertit le nombre de modèles en tailles de points lisibles.
    """
    values = np.asarray(
        values,
        dtype=float,
    )

    minimum = np.nanmin(
        values
    )

    maximum = np.nanmax(
        values
    )

    if np.isclose(
        minimum,
        maximum,
    ):
        return np.full(
            len(values),
            180.0,
        )

    normalized = (
        (values - minimum)
        / (maximum - minimum)
    )

    return (
        80
        + normalized * 420
    )


def create_quadrant_plot(dataframe):
    """
    Crée le graphique principal :
        X = confiance
        Y = accuracy
        taille/couleur = modèles générés
    """
    sns.set_theme(
        style="whitegrid",
        context="talk",
    )

    figure, axis = plt.subplots(
        figsize=(13, 9)
    )

    confidence_threshold = float(
        dataframe["confidence"].median()
    )

    accuracy_threshold = float(
        dataframe["accuracy"].median()
    )

    point_sizes = scale_point_sizes(
        dataframe[
            "number_models_generated"
        ]
    )

    scatter = axis.scatter(
        dataframe["confidence"],
        dataframe["accuracy"],
        s=point_sizes,
        c=dataframe[
            "number_models_generated"
        ],
        cmap="viridis",
        alpha=0.78,
        edgecolors="white",
        linewidths=0.8,
    )

    axis.axvline(
        confidence_threshold,
        color="#374151",
        linestyle="--",
        linewidth=1.5,
        alpha=0.85,
        label=(
            "Médiane confiance "
            f"({confidence_threshold:.1f})"
        ),
    )

    axis.axhline(
        accuracy_threshold,
        color="#6b7280",
        linestyle="--",
        linewidth=1.5,
        alpha=0.85,
        label=(
            "Médiane bonnes réponses "
            f"({accuracy_threshold:.1f} %)"
        ),
    )

    x_min, x_max = axis.get_xlim()
    y_min, y_max = axis.get_ylim()

    x_padding = (
        (x_max - x_min) * 0.025
    )

    y_padding = (
        (y_max - y_min) * 0.035
    )

    # Titres des quadrants.
    axis.text(
        x_min + x_padding,
        y_max - y_padding,
        "Faible confiance\nBonne performance",
        ha="left",
        va="top",
        fontsize=11,
        color="#047857",
        fontweight="bold",
        bbox={
            "facecolor": "white",
            "alpha": 0.65,
            "edgecolor": "none",
        },
    )

    axis.text(
        x_max - x_padding,
        y_max - y_padding,
        "Forte confiance\nBonne performance",
        ha="right",
        va="top",
        fontsize=11,
        color="#047857",
        fontweight="bold",
        bbox={
            "facecolor": "white",
            "alpha": 0.65,
            "edgecolor": "none",
        },
    )

    axis.text(
        x_min + x_padding,
        y_min + y_padding,
        "Faible confiance\nFaible performance",
        ha="left",
        va="bottom",
        fontsize=11,
        color="#b91c1c",
        fontweight="bold",
        bbox={
            "facecolor": "white",
            "alpha": 0.65,
            "edgecolor": "none",
        },
    )

    axis.text(
        x_max - x_padding,
        y_min + y_padding,
        "Forte confiance\nFaible performance",
        ha="right",
        va="bottom",
        fontsize=11,
        color="#b91c1c",
        fontweight="bold",
        bbox={
            "facecolor": "white",
            "alpha": 0.65,
            "edgecolor": "none",
        },
    )

    if ANNOTATE_SUBJECTS:
        rows_to_annotate = dataframe.head(
            MAX_ANNOTATED_SUBJECTS
        )

        for _, row in rows_to_annotate.iterrows():
            axis.annotate(
                str(row["subject_id"]),
                (
                    row["confidence"],
                    row["accuracy"],
                ),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                alpha=0.75,
            )

    colorbar = figure.colorbar(
        scatter,
        ax=axis,
        pad=0.02,
    )

    colorbar.set_label(
        "Nombre moyen de modèles générés",
        fontsize=12,
    )

    axis.set_title(
        "Confiance, performance et nombre de modèles mentaux",
        fontsize=17,
        fontweight="bold",
        pad=18,
    )

    axis.set_xlabel(
        "Confiance moyenne (%)"
    )

    axis.set_ylabel(
        "Pourcentage de bonnes réponses (%)"
    )

    # Garde une échelle cohérente si les données sont en pourcentage.
    if (
        dataframe["confidence"].min() >= 0
        and dataframe["confidence"].max() <= 100
    ):
        axis.set_xlim(
            max(0, x_min),
            min(100, x_max),
        )

    if (
        dataframe["accuracy"].min() >= 0
        and dataframe["accuracy"].max() <= 100
    ):
        axis.set_ylim(
            max(0, y_min),
            min(100, y_max),
        )

    axis.legend(
        loc="best",
        fontsize=9,
    )

    axis.grid(
        True,
        linestyle="--",
        alpha=0.45,
    )

    figure.tight_layout()

    figure.savefig(
        QUADRANT_OUTPUT_FILE,
        dpi=DPI,
        bbox_inches="tight",
    )

    print(
        "Graphique quadrant enregistré dans :",
        QUADRANT_OUTPUT_FILE,
    )


# ======================================================================
# GRAPHIQUE 3D
# ======================================================================

def create_three_dimensional_plot(dataframe):
    """
    Crée une représentation 3D simple.

    Chaque point représente un participant :
        X = confiance moyenne ;
        Y = nombre moyen de modèles générés ;
        Z = pourcentage de bonnes réponses.

    Tous les points ont la même couleur et la même taille.
    """
    plot_data = (
        dataframe[
            [
                "confidence",
                "number_models_generated",
                "accuracy",
            ]
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
        .copy()
    )

    if plot_data.empty:
        raise ValueError(
            "Aucune donnée valide pour générer le graphique 3D."
        )

    figure = plt.figure(
        figsize=(11, 9)
    )

    axis = figure.add_subplot(
        111,
        projection="3d",
    )

    # Tous les points ont la même couleur et la même taille.
    axis.scatter(
        plot_data["confidence"],
        plot_data["number_models_generated"],
        plot_data["accuracy"],
        color="#2563eb",
        s=45,
        alpha=0.75,
    )

    axis.set_title(
        "Espace cognitif 3D : confiance, modèles et performance",
        fontsize=15,
        fontweight="bold",
        pad=20,
    )

    axis.set_xlabel(
        "Confiance moyenne (%)",
        labelpad=10,
    )

    axis.set_ylabel(
        "Nombre moyen de modèles",
        labelpad=10,
    )

    axis.set_zlabel(
        "Bonnes réponses (%)",
        labelpad=10,
    )

    # Échelles en pourcentage lorsque les données sont comprises
    # entre 0 et 100.
    if (
        plot_data["confidence"].min() >= 0
        and plot_data["confidence"].max() <= 100
    ):
        axis.set_xlim(0, 100)

    if (
        plot_data["accuracy"].min() >= 0
        and plot_data["accuracy"].max() <= 100
    ):
        axis.set_zlim(0, 100)

    axis.grid(True)

    figure.tight_layout()

    output_directory = (
        os.path.dirname(
            THREE_DIMENSIONAL_OUTPUT_FILE
        )
        or "."
    )

    os.makedirs(
        output_directory,
        exist_ok=True,
    )

    figure.savefig(
        THREE_DIMENSIONAL_OUTPUT_FILE,
        dpi=DPI,
        bbox_inches="tight",
    )

    print(
        "Graphique 3D enregistré dans :",
        THREE_DIMENSIONAL_OUTPUT_FILE,
    )



# ======================================================================
# PROJECTIONS 2D
# ======================================================================

def add_scatter_plot(
    axis,
    dataframe,
    x_column,
    y_column,
    title,
    x_label,
    y_label,
    scatter_color,
):
    """
    Ajoute une projection 2D simple sans régression linéaire.

    La corrélation de Pearson reste affichée dans le titre, mais aucune
    droite de régression n'est tracée.
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
        .copy()
    )

    axis.scatter(
        plot_data[x_column],
        plot_data[y_column],
        color=scatter_color,
        alpha=0.68,
        s=48,
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
        alpha=0.45,
    )



def create_two_dimensional_plots(dataframe):
    """
    Crée trois projections croisées sous forme de nuages de points.

    Aucune droite de régression n'est affichée.
    """
    sns.set_theme(
        style="whitegrid",
        context="notebook",
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(19, 5.8),
    )

    figure.suptitle(
        "Projections croisées : confiance, performance et modèles",
        fontsize=17,
        fontweight="bold",
    )

    # Confiance et nombre de modèles.
    add_scatter_plot(
        axis=axes[0],
        dataframe=dataframe,
        x_column="confidence",
        y_column="number_models_generated",
        title="Confiance vs modèles mReasoner",
        x_label="Confiance moyenne (%)",
        y_label="Nombre moyen de modèles",
        scatter_color="#6366f1",
    )

    # Confiance et performance.
    add_scatter_plot(
        axis=axes[1],
        dataframe=dataframe,
        x_column="confidence",
        y_column="accuracy",
        title="Confiance vs performance",
        x_label="Confiance moyenne (%)",
        y_label="Bonnes réponses (%)",
        scatter_color="#10b981",
    )

    # Nombre de modèles et performance.
    add_scatter_plot(
        axis=axes[2],
        dataframe=dataframe,
        x_column="number_models_generated",
        y_column="accuracy",
        title="Modèles mReasoner vs performance",
        x_label="Nombre moyen de modèles",
        y_label="Bonnes réponses (%)",
        scatter_color="#f59e0b",
    )

    # Échelles en pourcentage pour les graphiques concernés.
    if (
        dataframe["confidence"].min() >= 0
        and dataframe["confidence"].max() <= 100
    ):
        axes[0].set_xlim(0, 100)
        axes[1].set_xlim(0, 100)

    if (
        dataframe["accuracy"].min() >= 0
        and dataframe["accuracy"].max() <= 100
    ):
        axes[1].set_ylim(0, 100)
        axes[2].set_ylim(0, 100)

    figure.tight_layout(
        rect=[
            0,
            0,
            1,
            0.91,
        ]
    )

    output_directory = (
        os.path.dirname(
            TWO_DIMENSIONAL_OUTPUT_FILE
        )
        or "."
    )

    os.makedirs(
        output_directory,
        exist_ok=True,
    )

    figure.savefig(
        TWO_DIMENSIONAL_OUTPUT_FILE,
        dpi=DPI,
        bbox_inches="tight",
    )

    print(
        "Projections 2D enregistrées dans :",
        TWO_DIMENSIONAL_OUTPUT_FILE,
    )



# ======================================================================
# PROGRAMME PRINCIPAL
# ======================================================================

def main():
    print("=" * 80)
    print("ANALYSE CONFIANCE × PERFORMANCE × MODÈLES MREASONER")
    print("=" * 80)

    try:
        merged_by_task, by_subject = (
            merge_and_aggregate()
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

    print("\nAperçu des données par participant :")

    print(
        by_subject.head(10).to_string(
            index=False
        )
    )

    print_correlations(
        by_subject
    )

    print("\nGénération du graphique quadrant...")
    create_quadrant_plot(
        by_subject
    )

    print("Génération du graphique 3D...")
    create_three_dimensional_plot(
        by_subject
    )

    print("Génération des projections 2D...")
    create_two_dimensional_plots(
        by_subject
    )

    print(
    "Génération du graphique des médianes par groupe..."
    )

    create_grouped_medians_plot(
        by_subject
    )


    print("\n" + "=" * 80)
    print("ANALYSE TERMINÉE")
    print("=" * 80)

    print(
        "Données fusionnées :",
        MERGED_OUTPUT_FILE,
    )

    print(
        "Graphique quadrant :",
        QUADRANT_OUTPUT_FILE,
    )

    print(
        "Graphique 3D :",
        THREE_DIMENSIONAL_OUTPUT_FILE,
    )

    print(
        "Projections 2D :",
        TWO_DIMENSIONAL_OUTPUT_FILE,
    )

    print(
        "Graphique par groupes :",
        GROUPED_MEDIANS_OUTPUT_FILE,
    )


    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()