"""
convert_dataset_E2.py

Convertit E2_syllogismData_full.csv au format CCOBRA.

Corrections principales :
    - utilise bien le fichier E2 ;
    - conserve les essais Conflict ET No-conflict ;
    - True signifie correct (1), False signifie incorrect (0) ;
    - ne supprime pas un essai uniquement parce que la confiance ou le
      temps de réponse est manquant ;
    - utilise response_ref comme réponse cible pour l'ajustement CCOBRA ;
    - affiche des contrôles détaillés avant et après conversion.
"""

import os
import sys

import numpy as np
import pandas as pd
from pathlib import Path

# ======================================================================
# CONFIGURATION
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


INPUT_FILE = str(
    PROJECT_ROOT
    / "data"
    / "raw"
    / "E4_syllogismData_full.csv"
)

OUTPUT_FILE_INT = str(
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dataset_ccobra_E2_int.csv"
)

OUTPUT_FILE_REF = str(
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dataset_ccobra_E2_ref.csv"
)



TASK_MAPPING = {
    "MP": "All B are C/All A are B",
    "MT": "All B are C/No A are C",
    "AC": "All B are C/All A are C",
    "DA": "All B are C/No A are B",
}


# ======================================================================
# NORMALISATION GÉNÉRALE
# ======================================================================

def normalize_subject_id(value):
    """
    Normalise l'identifiant du participant.

    Exemples :
        69594     -> "69594"
        69594.0   -> "69594"
    """
    if pd.isna(value):
        return np.nan

    normalized = str(value).strip()

    if not normalized:
        return np.nan

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


def normalize_numeric(series):
    """
    Convertit une série vers un format numérique.
    """
    return pd.to_numeric(
        series,
        errors="coerce",
    )


# ======================================================================
# NORMALISATION DES TÂCHES
# ======================================================================

def normalize_task_type(value):
    """
    Normalise le type de tâche vers MP, MT, AC ou DA.
    """
    if pd.isna(value):
        return np.nan

    task_type = str(value).strip().upper()

    if task_type not in TASK_MAPPING:
        return np.nan

    return task_type


def translate_task(value):
    """
    Transforme MP, MT, AC ou DA en prémisses formelles.
    """
    task_type = normalize_task_type(value)

    if pd.isna(task_type):
        return np.nan

    return TASK_MAPPING[task_type]


# ======================================================================
# NORMALISATION DES RÉPONSES
# ======================================================================

def normalize_yes_no(value):
    """
    Normalise une réponse humaine vers Yes ou No.
    """
    if pd.isna(value):
        return np.nan

    if isinstance(value, (bool, np.bool_)):
        return "Yes" if value else "No"

    normalized = str(value).strip().lower()

    yes_values = {
        "1",
        "1.0",
        "yes",
        "y",
        "true",
        "valid",
        "follows",
        "oui",
    }

    no_values = {
        "0",
        "0.0",
        "no",
        "n",
        "false",
        "invalid",
        "does not follow",
        "doesn't follow",
        "nvc",
        "non",
    }

    if normalized in yes_values:
        return "Yes"

    if normalized in no_values:
        return "No"

    return np.nan


def normalize_correctness(value):
    """
    Normalise les colonnes correct_int et correct_ref.

        True / correct   -> 1
        False / incorrect -> 0

    Cette fonction ne doit pas inverser les valeurs.
    """
    if pd.isna(value):
        return np.nan

    if isinstance(value, (bool, np.bool_)):
        return int(value)

    if isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        numeric = float(value)

        if numeric == 1:
            return 1

        if numeric == 0:
            return 0

    normalized = str(value).strip().lower()

    correct_values = {
        "1",
        "1.0",
        "true",
        "correct",
        "correcte",
        "yes",
        "y",
        "oui",
    }

    incorrect_values = {
        "0",
        "0.0",
        "false",
        "incorrect",
        "incorrecte",
        "wrong",
        "no",
        "n",
        "non",
    }

    if normalized in correct_values:
        return 1

    if normalized in incorrect_values:
        return 0

    return np.nan


# ======================================================================
# NORMALISATION DES VARIABLES EXPÉRIMENTALES
# ======================================================================

def normalize_validity(value):
    """
    Normalise la validité logique.

        Valid   -> 1
        Invalid -> 0
    """
    if pd.isna(value):
        return np.nan

    if isinstance(value, (bool, np.bool_)):
        return int(value)

    if isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        numeric = float(value)

        if numeric == 1:
            return 1

        if numeric == 0:
            return 0

    normalized = str(value).strip().lower()

    if normalized in {
        "1",
        "1.0",
        "true",
        "valid",
        "yes",
        "y",
    }:
        return 1

    if normalized in {
        "0",
        "0.0",
        "false",
        "invalid",
        "no",
        "n",
    }:
        return 0

    return np.nan


def normalize_believability(value):
    """
    Normalise la crédibilité.

        Believable   -> 1
        Unbelievable -> 0
    """
    if pd.isna(value):
        return np.nan

    if isinstance(value, (bool, np.bool_)):
        return int(value)

    if isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        numeric = float(value)

        if numeric == 1:
            return 1

        if numeric == 0:
            return 0

    normalized = str(value).strip().lower()

    if normalized in {
        "1",
        "1.0",
        "true",
        "believable",
        "believed",
        "yes",
        "y",
    }:
        return 1

    if normalized in {
        "0",
        "0.0",
        "false",
        "unbelievable",
        "not believable",
        "no",
        "n",
    }:
        return 0

    return np.nan


def normalize_conflict(value):
    """
    Normalise la présence d'un conflit.

        Conflict    -> 1
        No-conflict -> 0
    """
    if pd.isna(value):
        return np.nan

    if isinstance(value, (bool, np.bool_)):
        return int(value)

    if isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        numeric = float(value)

        if numeric == 1:
            return 1

        if numeric == 0:
            return 0

    normalized = (
        str(value)
        .strip()
        .lower()
        .replace("_", "-")
    )

    if normalized in {
        "1",
        "1.0",
        "true",
        "conflict",
        "conflicting",
        "yes",
        "y",
    }:
        return 1

    if normalized in {
        "0",
        "0.0",
        "false",
        "no-conflict",
        "no conflict",
        "non-conflict",
        "nonconflict",
        "non-conflicting",
        "no",
        "n",
    }:
        return 0

    return np.nan


# ======================================================================
# NORMALISATION DE LA CONFIANCE
# ======================================================================

def normalize_for_percentage(series):
    """
    Normalise une mesure de confiance vers une échelle 0-100.

    FOR signifie Feeling of Rightness, c'est-à-dire la confiance
    du participant dans sa réponse.
    """
    numeric = normalize_numeric(series)
    valid = numeric.dropna()

    if valid.empty:
        return numeric

    minimum = float(valid.min())
    maximum = float(valid.max())

    print(
        f"Échelle de confiance détectée pour {series.name} : "
        f"minimum={minimum}, maximum={maximum}"
    )

    # Proportion comprise entre 0 et 1.
    if minimum >= 0 and maximum <= 1:
        return numeric * 100

    # Déjà exprimée sur une échelle allant jusqu'à 100.
    if minimum >= 0 and maximum <= 100:
        # Petite échelle de Likert, par exemple 1-5 ou 1-7.
        if maximum <= 10 and valid.nunique() <= 11:
            if np.isclose(minimum, maximum):
                return pd.Series(
                    np.where(
                        numeric.notna(),
                        50.0,
                        np.nan,
                    ),
                    index=numeric.index,
                )

            return (
                100
                * (numeric - minimum)
                / (maximum - minimum)
            )

        return numeric

    # Échelle inhabituelle : normalisation min-max.
    if np.isclose(minimum, maximum):
        return pd.Series(
            np.where(
                numeric.notna(),
                50.0,
                np.nan,
            ),
            index=numeric.index,
        )

    return (
        100
        * (numeric - minimum)
        / (maximum - minimum)
    )


# ======================================================================
# CONTRÔLES
# ======================================================================

def print_raw_data_summary(dataframe):
    """
    Affiche un résumé du fichier brut avant conversion.
    """
    raw_correct_int = (
        dataframe["correct_int"]
        .apply(normalize_correctness)
    )

    raw_correct_ref = (
        dataframe["correct_ref"]
        .apply(normalize_correctness)
    )

    print("\n" + "=" * 80)
    print("CONTRÔLE DU FICHIER BRUT")
    print("=" * 80)

    print(
        "Nombre total de lignes :",
        len(dataframe),
    )

    print(
        "Nombre de participants :",
        dataframe["sona_id"].nunique(),
    )

    print(
        "Nombre de corrections utilisables :",
        raw_correct_int.notna().sum(),
    )

    print(
        "Précision intuitive brute :",
        round(
            raw_correct_int.mean() * 100,
            4,
        ),
        "%",
    )

    print(
        "Précision réfléchie brute :",
        round(
            raw_correct_ref.mean() * 100,
            4,
        ),
        "%",
    )

    print("\nRépartition brute du conflit :")

    print(
        dataframe["conflict"]
        .value_counts(
            dropna=False,
        )
        .to_string()
    )


def print_output_summary(output, phase_name):
    """
    Affiche un résumé du fichier CCOBRA produit pour une phase.

    phase_name doit valoir :
        "intuitive"
        ou
        "réfléchie"
    """
    print("\n" + "=" * 80)
    print(
        f"CONTRÔLE DU FICHIER CCOBRA — PHASE {phase_name.upper()}"
    )
    print("=" * 80)

    print(
        "Participants :",
        output["id"].nunique(),
    )

    print(
        "Essais :",
        len(output),
    )

    print(
        "\nRépartition des essais par conflit :"
    )

    conflict_counts = (
        output["conflict"]
        .map({
            0: "No-conflict",
            1: "Conflict",
        })
        .value_counts(
            dropna=False,
        )
    )

    print(
        conflict_counts.to_string()
    )

    print(
        "\nPrécision intuitive globale :",
        round(
            output["correct_int"].mean() * 100,
            4,
        ),
        "%",
    )

    print(
        "Précision réfléchie globale :",
        round(
            output["correct_ref"].mean() * 100,
            4,
        ),
        "%",
    )

    print("\nRéponses utilisées pour l'ajustement :")

    print(
        output["response"]
        .value_counts(
            dropna=False,
        )
        .to_string()
    )

    print("\nPrécision par conflit :")

    accuracy_by_conflict = (
        output
        .assign(
            conflict_label=output["conflict"].map({
                0: "No-conflict",
                1: "Conflict",
            })
        )
        .groupby(
            "conflict_label",
            as_index=True,
        )
        .agg(
            number_of_trials=(
                "sequence",
                "size",
            ),
            intuitive_accuracy=(
                "correct_int",
                "mean",
            ),
            reflective_accuracy=(
                "correct_ref",
                "mean",
            ),
        )
    )

    accuracy_by_conflict[
        "intuitive_accuracy"
    ] *= 100

    accuracy_by_conflict[
        "reflective_accuracy"
    ] *= 100

    print(
        accuracy_by_conflict.to_string()
    )

    print("\nRépartition des tâches :")

    print(
        output["task_type"]
        .value_counts(
            dropna=False,
        )
        .to_string()
    )

    print("\nAperçu :")

    columns_to_display = [
        "id",
        "sequence",
        "task_type",
        "task",
        "response",
        "response_int",
        "response_ref",
        "correct_int",
        "correct_ref",
        "conflict",
    ]

    print(
        output[
            columns_to_display
        ]
        .head(10)
        .to_string(index=False)
    )


# ======================================================================
# PROGRAMME PRINCIPAL
# ======================================================================

def main():
    input_path = os.path.abspath(
        INPUT_FILE
    )

    output_path_int = os.path.abspath(
        OUTPUT_FILE_INT
    )

    output_path_ref = os.path.abspath(
        OUTPUT_FILE_REF
    )

    print("=" * 80)
    print("CONVERSION DU DATASET E2 AU FORMAT CCOBRA")
    print("=" * 80)

    print(
        "Fichier d'entrée :",
        input_path,
    )

    print(
        "Fichier intuitif de sortie :",
        output_path_int,
    )

    print(
        "Fichier réfléchi de sortie :",
        output_path_ref,
    )


    if not os.path.isfile(input_path):
        print(
            "ERREUR — fichier introuvable :",
            input_path,
        )

        sys.exit(1)

    try:
        dataframe = pd.read_csv(
            input_path
        )

    except pd.errors.ParserError as error:
        print(
            "ERREUR lors de la lecture du CSV :",
            error,
        )

        sys.exit(1)

    required_columns = {
        "sona_id",
        "type",
        "validity",
        "believability",
        "conflict",
        "response_int",
        "response_ref",
        "correct_int",
        "correct_ref",
        "for_int",
        "for_ref",
        "rt_int",
        "rt_ref",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        print(
            "ERREUR — colonnes obligatoires absentes :",
            sorted(missing_columns),
        )

        print(
            "Colonnes disponibles :",
            list(dataframe.columns),
        )

        sys.exit(1)

    print_raw_data_summary(
        dataframe
    )

    # ------------------------------------------------------------------
    # Construction du fichier CCOBRA
    # ------------------------------------------------------------------

    output = pd.DataFrame(
        index=dataframe.index
    )

    output["id"] = (
        dataframe["sona_id"]
        .apply(normalize_subject_id)
    )

    if "trial_num" in dataframe.columns:
        output["sequence"] = normalize_numeric(
            dataframe["trial_num"]
        )

    elif "qnum" in dataframe.columns:
        output["sequence"] = normalize_numeric(
            dataframe["qnum"]
        )

    else:
        output["sequence"] = (
            dataframe
            .groupby(
                "sona_id"
            )
            .cumcount()
            + 1
        )

    output["domain"] = "conditional"
    output["response_type"] = "single-choice"

    output["task_type"] = (
        dataframe["type"]
        .apply(normalize_task_type)
    )

    output["task"] = (
        dataframe["type"]
        .apply(translate_task)
    )

    output["choices"] = "Yes/No"


    # Réponses des deux phases.
    output["response_int"] = (
        dataframe["response_int"]
        .apply(normalize_yes_no)
    )

    output["response_ref"] = (
        dataframe["response_ref"]
        .apply(normalize_yes_no)
    )

    # Correction des deux phases.
    # True est conservé comme 1 et False comme 0.
    output["correct_int"] = (
        dataframe["correct_int"]
        .apply(normalize_correctness)
    )

    output["correct_ref"] = (
        dataframe["correct_ref"]
        .apply(normalize_correctness)
    )

    # Confiance intuitive et réfléchie.
    output["for_int"] = normalize_for_percentage(
        dataframe["for_int"]
    )

    output["for_ref"] = normalize_for_percentage(
        dataframe["for_ref"]
    )

    # Temps de réponse.
    output["rt_int"] = normalize_numeric(
        dataframe["rt_int"]
    )

    output["rt_ref"] = normalize_numeric(
        dataframe["rt_ref"]
    )

    # Variables expérimentales.
    output["validity"] = (
        dataframe["validity"]
        .apply(normalize_validity)
    )

    output["believability"] = (
        dataframe["believability"]
        .apply(normalize_believability)
    )

    output["conflict"] = (
        dataframe["conflict"]
        .apply(normalize_conflict)
    )

    # Conservation facultative de colonnes supplémentaires.
    optional_columns = [
        "condition",
        "stimulus",
        "qnum",
        "total_qnum",
    ]

    for column in optional_columns:
        if column in dataframe.columns:
            output[column] = dataframe[column]

    # ------------------------------------------------------------------
    # Variations entre les phases
    # ------------------------------------------------------------------

    responses_available = (
        output["response_int"].notna()
        & output["response_ref"].notna()
    )

    output["response_changed"] = np.where(
        responses_available,
        (
            output["response_int"]
            != output["response_ref"]
        ).astype(int),
        np.nan,
    )

    output["accuracy_gain"] = (
        output["correct_ref"]
        - output["correct_int"]
    )

    output["for_change"] = (
        output["for_ref"]
        - output["for_int"]
    )

    output["rt_change"] = (
        output["rt_ref"]
        - output["rt_int"]
    )

    # ------------------------------------------------------------------
    # Contrôle des valeurs non reconnues
    # ------------------------------------------------------------------

    columns_to_check = [
        "id",
        "sequence",
        "task_type",
        "task",
        "response_int",
        "response_ref",
        "correct_int",
        "correct_ref",
        "validity",
        "believability",
        "conflict",
    ]

    print("\n" + "=" * 80)
    print("VALEURS MANQUANTES APRÈS NORMALISATION")
    print("=" * 80)

    print(
        output[columns_to_check]
        .isna()
        .sum()
        .to_string()
    )

    # ------------------------------------------------------------------
    # Suppression minimale
    # ------------------------------------------------------------------
    #
    # La confiance et les temps de réponse ne figurent volontairement
    # pas dans cette liste.
    #
    # Un essai n'est donc plus supprimé simplement parce que for_int,
    # for_ref, rt_int ou rt_ref est absent.
    # ------------------------------------------------------------------

    required_output = [
        "id",
        "sequence",
        "task_type",
        "task",
        "response_int",
        "response_ref",
        "correct_int",
        "correct_ref",
        "validity",
        "believability",
        "conflict",
    ]

    before_drop = len(output)

    output = (
        output
        .dropna(
            subset=required_output
        )
        .copy()
    )

    removed_rows = (
        before_drop - len(output)
    )

    print(
        "\nLignes supprimées car informations essentielles manquantes :",
        removed_rows,
    )

    # ------------------------------------------------------------------
    # Types finaux
    # ------------------------------------------------------------------

    output["sequence"] = (
        output["sequence"]
        .astype(int)
    )

    integer_columns = [
        "correct_int",
        "correct_ref",
        "validity",
        "believability",
        "conflict",
        "response_changed",
        "accuracy_gain",
    ]

    for column in integer_columns:
        if output[column].isna().any():
            output[column] = (
                output[column]
                .astype("Int64")
            )
        else:
            output[column] = (
                output[column]
                .astype(int)
            )

    # ------------------------------------------------------------------
    # Tri du tableau commun
    # ------------------------------------------------------------------

    output = (
        output
        .sort_values(
            by=[
                "id",
                "sequence",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # ------------------------------------------------------------------
    # Création des deux datasets CCOBRA
    # ------------------------------------------------------------------

    output_int = output.copy()
    output_ref = output.copy()

    # Pour l'ajustement intuitif, la réponse cible de CCOBRA est
    # la première réponse du participant.
    output_int["response"] = (
        output_int["response_int"]
    )

    # Pour l'ajustement réfléchi, la réponse cible de CCOBRA est
    # la réponse donnée après réflexion.
    output_ref["response"] = (
        output_ref["response_ref"]
    )

    # Vérification de sécurité.
    if output_int["response"].isna().any():
        raise ValueError(
            "Le dataset intuitif contient des réponses cibles "
            "manquantes."
        )

    if output_ref["response"].isna().any():
        raise ValueError(
            "Le dataset réfléchi contient des réponses cibles "
            "manquantes."
        )

    # ------------------------------------------------------------------
    # Ordre des colonnes
    # ------------------------------------------------------------------

    preferred_column_order = [
        "id",
        "sequence",
        "domain",
        "response_type",
        "task",
        "choices",
        "response",
        "task_type",
        "response_int",
        "response_ref",
        "correct_int",
        "correct_ref",
        "for_int",
        "for_ref",
        "rt_int",
        "rt_ref",
        "validity",
        "believability",
        "conflict",
        "response_changed",
        "accuracy_gain",
        "for_change",
        "rt_change",
        "condition",
        "stimulus",
        "qnum",
        "total_qnum",
    ]

    final_column_order = [
        column
        for column in preferred_column_order
        if column in output_int.columns
    ]

    remaining_columns = [
        column
        for column in output_int.columns
        if column not in final_column_order
    ]

    final_column_order.extend(
        remaining_columns
    )

    output_int = output_int[
        final_column_order
    ]

    output_ref = output_ref[
        final_column_order
    ]

    # ------------------------------------------------------------------
    # Création des dossiers de sortie
    # ------------------------------------------------------------------

    for output_path in [
        output_path_int,
        output_path_ref,
    ]:
        output_directory = (
            os.path.dirname(output_path)
            or "."
        )

        os.makedirs(
            output_directory,
            exist_ok=True,
        )

    # ------------------------------------------------------------------
    # Sauvegarde des deux datasets
    # ------------------------------------------------------------------

    output_int.to_csv(
        output_path_int,
        index=False,
    )

    output_ref.to_csv(
        output_path_ref,
        index=False,
    )

    # ------------------------------------------------------------------
    # Contrôle : les deux fichiers doivent contenir les mêmes essais
    # ------------------------------------------------------------------

    key_columns = [
        "id",
        "sequence",
        "task_type",
    ]

    if not output_int[
        key_columns
    ].equals(
        output_ref[
            key_columns
        ]
    ):
        raise RuntimeError(
            "Les datasets intuitif et réfléchi ne contiennent "
            "pas les mêmes essais."
        )

    different_target_responses = (
        output_int["response"]
        != output_ref["response"]
    ).sum()

    # ------------------------------------------------------------------
    # Résumés
    # ------------------------------------------------------------------

    print_output_summary(
        output_int,
        phase_name="intuitive",
    )

    print_output_summary(
        output_ref,
        phase_name="réfléchie",
    )

    print("\n" + "=" * 80)
    print("CONVERSION TERMINÉE")
    print("=" * 80)

    print(
        "Dataset intuitif créé :",
        output_path_int,
    )

    print(
        "Dataset réfléchi créé :",
        output_path_ref,
    )

    print(
        "Participants dans chaque dataset :",
        output_int["id"].nunique(),
    )

    print(
        "Essais dans chaque dataset :",
        len(output_int),
    )

    print(
        "Essais où la réponse change entre les deux phases :",
        different_target_responses,
    )

    print(
        "Taux de changement de réponse :",
        round(
            100
            * different_target_responses
            / len(output_int),
            4,
        ),
        "%",
    )

    print(
        "\nPrécision intuitive :",
        round(
            output_int["correct_int"].mean()
            * 100,
            4,
        ),
        "%",
    )

    print(
        "Précision réfléchie :",
        round(
            output_ref["correct_ref"].mean()
            * 100,
            4,
        ),
        "%",
    )



if __name__ == "__main__":
    main()
