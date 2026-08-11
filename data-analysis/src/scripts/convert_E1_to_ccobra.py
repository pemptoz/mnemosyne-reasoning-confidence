import os
import sys

import numpy as np
import pandas as pd
from pathlib import Path

# ======================================================================
# Configuration
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = str(
    PROJECT_ROOT
    / "data"
    / "raw"
    / "E3_syllogismData_full.csv"
)

OUTPUT_FILE = str(
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dataset_ccobra_E1.csv"
)



# La colonne "feelingR" correspond au jugement de confiance.
CONFIDENCE_COLUMN = "feelingR"

# True : transforme la confiance sur une échelle de 0 à 100.
# False : conserve l'échelle originale de feelingR.
NORMALIZE_CONFIDENCE_TO_PERCENTAGE = True


# ======================================================================
# Traduction des tâches
# ======================================================================

TASK_MAPPING = {
    "MP": "All B are C/All A are B",
    "MT": "All B are C/No A are C",
    "AC": "All B are C/All A are C",
    "DA": "All B are C/No A are B",
}


def translate_task(task_type):
    """
    Traduit le type expérimental en deux prémisses CCOBRA.
    """
    if pd.isna(task_type):
        return np.nan

    normalized_type = str(task_type).strip().upper()

    return TASK_MAPPING.get(
        normalized_type,
        np.nan,
    )


# ======================================================================
# Normalisation des réponses
# ======================================================================

def normalize_response(value):
    """
    Convertit une réponse expérimentale en Yes ou No.
    """
    if pd.isna(value):
        return np.nan

    if isinstance(value, (bool, np.bool_)):
        return "Yes" if value else "No"

    normalized = str(value).strip().lower()

    yes_values = {
        "yes",
        "y",
        "oui",
        "true",
        "1",
        "valid",
        "valid argument",
        "conclusion follows",
        "follows",
    }

    no_values = {
        "no",
        "n",
        "non",
        "false",
        "0",
        "invalid",
        "invalid argument",
        "nvc",
        "no valid conclusion",
        "conclusion does not follow",
        "does not follow",
    }

    if normalized in yes_values:
        return "Yes"

    if normalized in no_values:
        return "No"

    return np.nan


# ======================================================================
# Normalisation de la correction
# ======================================================================

def normalize_correctness(value):
    """
    Convertit la colonne 'correct' en :

        1 = réponse correcte
        0 = réponse incorrecte
    """
    if pd.isna(value):
        return np.nan

    if isinstance(value, (bool, np.bool_)):
        return int(value)

    if isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        if float(value) == 1:
            return 1

        if float(value) == 0:
            return 0

    normalized = str(value).strip().lower()

    correct_values = {
        "1",
        "1.0",
        "true",
        "t",
        "yes",
        "y",
        "correct",
        "right",
        "oui",
    }

    incorrect_values = {
        "0",
        "0.0",
        "false",
        "f",
        "no",
        "n",
        "incorrect",
        "wrong",
        "non",
    }

    if normalized in correct_values:
        return 1

    if normalized in incorrect_values:
        return 0

    return np.nan


# ======================================================================
# Conversion de la confiance
# ======================================================================

def confidence_to_numeric(series):
    """
    Convertit feelingR en valeurs numériques.

    Gère notamment :
        4
        "4"
        "75%"
        "4/5"
    """
    cleaned = (
        series
        .astype("string")
        .str.strip()
        .str.replace("%", "", regex=False)
    )

    numeric = pd.to_numeric(
        cleaned,
        errors="coerce",
    )

    fraction_mask = cleaned.str.match(
        r"^\s*-?\d+(?:\.\d+)?\s*/\s*-?\d+(?:\.\d+)?\s*$",
        na=False,
    )

    if fraction_mask.any():
        fractions = cleaned.loc[
            fraction_mask
        ].str.split(
            "/",
            n=1,
            expand=True,
        )

        numerators = pd.to_numeric(
            fractions[0],
            errors="coerce",
        )

        denominators = pd.to_numeric(
            fractions[1],
            errors="coerce",
        )

        valid_denominators = (
            denominators.notna()
            & (denominators != 0)
        )

        fraction_values = pd.Series(
            np.nan,
            index=fractions.index,
            dtype=float,
        )

        fraction_values.loc[
            valid_denominators
        ] = (
            numerators.loc[valid_denominators]
            / denominators.loc[valid_denominators]
        )

        numeric.loc[
            fraction_mask
        ] = fraction_values

    numeric.loc[
        series.isna()
    ] = np.nan

    return numeric.astype(float)


def normalize_confidence(series):
    """
    Convertit la confiance vers une échelle de 0 à 100.

    Règles :
        0 à 1   -> multiplication par 100
        0 à 100 -> conservation
        petite échelle, par exemple 1 à 5 ou 1 à 7
                 -> normalisation min-max
    """
    numeric = confidence_to_numeric(
        series
    )

    valid_values = numeric.dropna()

    if valid_values.empty:
        raise ValueError(
            "La colonne feelingR ne contient aucune valeur "
            "de confiance numérique exploitable."
        )

    minimum = float(valid_values.min())
    maximum = float(valid_values.max())

    print(
        "Échelle de confiance détectée : "
        f"minimum={minimum}, maximum={maximum}"
    )

    if not NORMALIZE_CONFIDENCE_TO_PERCENTAGE:
        return numeric

    # Proportions entre 0 et 1.
    if minimum >= 0 and maximum <= 1:
        print(
            "Conversion de la confiance de [0, 1] vers [0, 100]."
        )
        return numeric * 100

    unique_values = valid_values.unique()

    # Échelles discrètes de type 1-5, 1-7 ou 0-10.
    if (
        minimum >= 0
        and maximum <= 10
        and len(unique_values) <= 11
    ):
        if np.isclose(minimum, maximum):
            print(
                "Toutes les valeurs de confiance sont identiques ; "
                "elles sont converties en 50."
            )

            return pd.Series(
                np.where(
                    numeric.notna(),
                    50.0,
                    np.nan,
                ),
                index=numeric.index,
                dtype=float,
            )

        print(
            "Normalisation min-max de la confiance vers [0, 100]."
        )

        return (
            100.0
            * (numeric - minimum)
            / (maximum - minimum)
        )

    # Échelle déjà exprimée sur 100.
    if minimum >= 0 and maximum <= 100:
        print(
            "La confiance est déjà exprimée sur une échelle de 0 à 100."
        )
        return numeric

    if np.isclose(minimum, maximum):
        return pd.Series(
            np.where(
                numeric.notna(),
                50.0,
                np.nan,
            ),
            index=numeric.index,
            dtype=float,
        )

    print(
        "Échelle inhabituelle : normalisation min-max vers [0, 100]."
    )

    return (
        100.0
        * (numeric - minimum)
        / (maximum - minimum)
    )


# ======================================================================
# Programme principal
# ======================================================================

def main():
    input_path = os.path.abspath(
        os.path.expanduser(INPUT_FILE)
    )

    output_path = os.path.abspath(
        os.path.expanduser(OUTPUT_FILE)
    )

    print("1. Chargement des données brutes...")
    print("Fichier d'entrée :", input_path)

    if not os.path.isfile(input_path):
        print(
            f"Erreur : le fichier '{input_path}' est introuvable."
        )
        sys.exit(1)

    try:
        df_raw = pd.read_csv(
            input_path
        )
    except (
        pd.errors.ParserError,
        UnicodeDecodeError,
        OSError,
    ) as error:
        print(
            "Erreur pendant la lecture du fichier CSV :",
            error,
        )
        sys.exit(1)

    required_columns = {
        "sona_id",
        "type",
        "response",
        "correct",
        "trial_num",
        CONFIDENCE_COLUMN,
    }

    missing_columns = (
        required_columns
        - set(df_raw.columns)
    )

    if missing_columns:
        print(
            "Erreur : colonnes obligatoires absentes :",
            sorted(missing_columns),
        )

        print(
            "Colonnes disponibles :",
            list(df_raw.columns),
        )

        sys.exit(1)

    print(
        "Nombre de lignes brutes :",
        len(df_raw),
    )

    print(
        "Colonnes disponibles :",
        list(df_raw.columns),
    )

    # ------------------------------------------------------------------
    # Conversion des variables
    # ------------------------------------------------------------------

    print("\n2. Conversion des variables...")

    normalized_response = (
        df_raw["response"]
        .apply(normalize_response)
    )

    normalized_correctness = (
        df_raw["correct"]
        .apply(normalize_correctness)
    )

    try:
        normalized_confidence = normalize_confidence(
            df_raw[CONFIDENCE_COLUMN]
        )
    except ValueError as error:
        print(
            "Erreur pendant la conversion de la confiance :",
            error,
        )
        sys.exit(1)

    translated_tasks = (
        df_raw["type"]
        .apply(translate_task)
    )

    # ------------------------------------------------------------------
    # Construction du fichier enrichi
    # ------------------------------------------------------------------

    print("\n3. Construction du fichier CCOBRA enrichi...")

    df_ccobra = pd.DataFrame(
        index=df_raw.index
    )

    df_ccobra["id"] = (
        df_raw["sona_id"]
        .astype("string")
        .str.strip()
    )

    df_ccobra["sequence"] = pd.to_numeric(
        df_raw["trial_num"],
        errors="coerce",
    )

    df_ccobra["domain"] = "conditional"
    df_ccobra["response_type"] = "single-choice"

    df_ccobra["task"] = translated_tasks
    df_ccobra["choices"] = "Yes/No"
    df_ccobra["response"] = normalized_response

    # Confiance individuelle utilisée par plot_quadrant.py.
    df_ccobra["confidence"] = (
        normalized_confidence
    )

    # 1 si la réponse est correcte, 0 sinon.
    df_ccobra["is_correct"] = (
        normalized_correctness
    )

    # Informations expérimentales utiles.
    df_ccobra["task_type"] = (
        df_raw["type"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    df_ccobra["condition"] = (
        df_raw["condition"]
        if "condition" in df_raw.columns
        else np.nan
    )

    df_ccobra["validity"] = (
        df_raw["validity"]
        if "validity" in df_raw.columns
        else np.nan
    )

    df_ccobra["believability"] = (
        df_raw["believability"]
        if "believability" in df_raw.columns
        else np.nan
    )

    df_ccobra["conflict"] = (
        df_raw["conflict"]
        if "conflict" in df_raw.columns
        else np.nan
    )

    df_ccobra["stimulus"] = (
        df_raw["stimulus"]
        if "stimulus" in df_raw.columns
        else np.nan
    )

    df_ccobra["qnum"] = (
        df_raw["qnum"]
        if "qnum" in df_raw.columns
        else np.nan
    )

    df_ccobra["total_qnum"] = (
        df_raw["total_qnum"]
        if "total_qnum" in df_raw.columns
        else np.nan
    )

    df_ccobra["rt"] = pd.to_numeric(
        df_raw["rt"],
        errors="coerce",
    ) if "rt" in df_raw.columns else np.nan

    df_ccobra["logRT"] = pd.to_numeric(
        df_raw["logRT"],
        errors="coerce",
    ) if "logRT" in df_raw.columns else np.nan

    df_ccobra["rt_for"] = pd.to_numeric(
        df_raw["rt_for"],
        errors="coerce",
    ) if "rt_for" in df_raw.columns else np.nan

    df_ccobra["statementEval"] = (
        df_raw["statementEval"]
        if "statementEval" in df_raw.columns
        else np.nan
    )

    # ------------------------------------------------------------------
    # Diagnostic des valeurs non reconnues
    # ------------------------------------------------------------------

    unrecognized_response_mask = (
        df_raw["response"].notna()
        & df_ccobra["response"].isna()
    )

    if unrecognized_response_mask.any():
        examples = (
            df_raw.loc[
                unrecognized_response_mask,
                "response",
            ]
            .astype(str)
            .drop_duplicates()
            .head(10)
            .tolist()
        )

        print(
            "Attention : réponses non reconnues :",
            examples,
        )

    unrecognized_correctness_mask = (
        df_raw["correct"].notna()
        & df_ccobra["is_correct"].isna()
    )

    if unrecognized_correctness_mask.any():
        examples = (
            df_raw.loc[
                unrecognized_correctness_mask,
                "correct",
            ]
            .astype(str)
            .drop_duplicates()
            .head(10)
            .tolist()
        )

        print(
            "Attention : valeurs de correction non reconnues :",
            examples,
        )

    unknown_task_mask = (
        df_raw["type"].notna()
        & df_ccobra["task"].isna()
    )

    if unknown_task_mask.any():
        examples = (
            df_raw.loc[
                unknown_task_mask,
                "type",
            ]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

        print(
            "Attention : types de tâches non reconnus :",
            examples,
        )

    # ------------------------------------------------------------------
    # Nettoyage
    # ------------------------------------------------------------------

    initial_row_count = len(
        df_ccobra
    )

    df_ccobra = df_ccobra.dropna(
        subset=[
            "id",
            "sequence",
            "task",
            "response",
            "confidence",
            "is_correct",
        ]
    ).copy()

    removed_row_count = (
        initial_row_count
        - len(df_ccobra)
    )

    if removed_row_count > 0:
        print(
            f"{removed_row_count} ligne(s) supprimée(s), car une "
            "information obligatoire était absente ou invalide."
        )

    df_ccobra["sequence"] = (
        df_ccobra["sequence"]
        .astype(int)
    )

    df_ccobra["is_correct"] = (
        df_ccobra["is_correct"]
        .astype(int)
    )

    df_ccobra["confidence"] = (
        df_ccobra["confidence"]
        .astype(float)
        .clip(
            lower=0,
            upper=100,
        )
        .round(4)
    )

    df_ccobra = (
        df_ccobra
        .sort_values(
            by=[
                "id",
                "sequence",
            ]
        )
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------
    # Sauvegarde
    # ------------------------------------------------------------------

    output_directory = (
        os.path.dirname(output_path)
        or "."
    )

    os.makedirs(
        output_directory,
        exist_ok=True,
    )

    df_ccobra.to_csv(
        output_path,
        index=False,
    )

    # ------------------------------------------------------------------
    # Résumé
    # ------------------------------------------------------------------

    print("\n4. Fichier sauvegardé avec succès.")
    print("Fichier de sortie :", output_path)
    print("Nombre de lignes :", len(df_ccobra))

    print(
        "Nombre de participants :",
        df_ccobra["id"].nunique(),
    )

    print(
        "Colonnes produites :",
        list(df_ccobra.columns),
    )

    print("\nDistribution des tâches :")
    print(
        df_ccobra["task_type"]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )

    print("\nDistribution des réponses :")
    print(
        df_ccobra["response"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nPourcentage global de bonnes réponses :")
    print(
        f"{df_ccobra['is_correct'].mean() * 100:.2f} %"
    )

    print("\nStatistiques de confiance :")
    print(
        df_ccobra["confidence"]
        .describe()
        .to_string()
    )

    print("\nAperçu du fichier produit :")
    print(
        df_ccobra.head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
