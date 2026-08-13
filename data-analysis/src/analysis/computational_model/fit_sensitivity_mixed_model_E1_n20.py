#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fit_sensitivity_mixed_model_E1.py

Réalise les analyses de sensibilité du modèle cognitif E1 fondé
sur 20 simulations de MReasoner.

Fichier d'entrée
----------------
dataset_analysis_E1_n20.csv

Modèles comparés en ML
----------------------

1. Modèle de contrôle :

    confidence ~
        condition
        + sequence_c10

2. Modèle cognitif sans validité :

    confidence ~
        condition
        + sequence_c10
        + subject_accuracy_z
        + item_entropy_z
        + subject_mean_models_z
        + models_within_subject_z

3. Modèle cognitif avec validité :

    modèle cognitif sans validité
        + validity_binary

4. Modèle cognitif avec type de tâche :

    modèle cognitif sans validité
        + task_type

Le modèle avec validité correspond au modèle cognitif principal
ajusté dans fit_cognitive_mixed_model_E1_n20.py.

Ce script examine :

    - l'amélioration apportée par le bloc cognitif ;
    - l'amélioration apportée par la validité ;
    - l'amélioration apportée par le type de tâche ;
    - la contribution propre de chaque prédicteur du modèle avec
      validité au moyen de tests drop-one.

Les comparaisons de modèles sont effectuées en ML.

Le modèle alternatif avec task_type est également ajusté en REML
afin de présenter ses coefficients.

Effets aléatoires croisés
-------------------------
- intercept participant ;
- intercept item.

Fichiers produits
-----------------
sensitivity_mixed_model_E1_n20/
    global_likelihood_ratio_tests.csv
    drop_one_tests.csv
    model_fit_comparison.csv
    task_type_model_fixed_effects.csv
    task_type_model_REML_summary.txt
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = SCRIPT_DIR.parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "computational_model"
    / "dataset_analysis_E1_n20.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "analysis"
    / "computational_model"
    / "sensitivity_mixed_model_E1_n20"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------------------------------
# Fichiers produits
# --------------------------------------------------------------------------

GLOBAL_TESTS_FILE = (
    OUTPUT_DIR
    / "global_likelihood_ratio_tests.csv"
)

DROP_ONE_TESTS_FILE = (
    OUTPUT_DIR
    / "drop_one_tests.csv"
)

MODEL_FIT_COMPARISON_FILE = (
    OUTPUT_DIR
    / "model_fit_comparison.csv"
)

TASK_TYPE_FIXED_EFFECTS_FILE = (
    OUTPUT_DIR
    / "task_type_model_fixed_effects.csv"
)

TASK_TYPE_REML_SUMMARY_FILE = (
    OUTPUT_DIR
    / "task_type_model_REML_summary.txt"
)



# --------------------------------------------------------------------------
# Effets aléatoires croisés
# --------------------------------------------------------------------------

VC_FORMULA = {
    "item":
        "0 + C(item_id)",

    "subject":
        "0 + C(subject_id)",
}


# --------------------------------------------------------------------------
# Termes des modèles
# --------------------------------------------------------------------------

CONTROL_TERMS = [
    "C(condition, Treatment(reference='Neutral'))",
    "sequence_c10",
]

COGNITIVE_TERMS = [
    "subject_accuracy_z",
    "item_entropy_z",
    "subject_mean_models_z",
    "models_within_subject_z",
]

VALIDITY_TERM = (
    "validity_binary"
)

TASK_TYPE_TERM = (
    "C(task_type, Treatment(reference='AC'))"
)


# --------------------------------------------------------------------------
# Colonnes nécessaires
# --------------------------------------------------------------------------

ESSENTIAL_COLUMNS = [
    "confidence",
    "subject_id",
    "item_id",
    "condition",
    "sequence",
    "task_type",
    "validity_binary",
    "subject_accuracy",
    "item_entropy",
    "subject_mean_models",
    "models_within_subject",
]

STANDARDIZED_COLUMNS = [
    "subject_accuracy",
    "item_entropy",
    "subject_mean_models",
    "models_within_subject",
]


# --------------------------------------------------------------------------
# Optimisation
# --------------------------------------------------------------------------

OPTIMIZERS = [
    "lbfgs",
    "bfgs",
    "cg",
    "powell",
]

MAX_ITERATIONS = 3000

USE_SPARSE_MATRICES = True


# ============================================================================
# AFFICHAGE
# ============================================================================

def section(title):
    """Affiche un titre de section."""
    print("")
    print("=" * 80)
    print(title)
    print("=" * 80)


# ============================================================================
# OUTILS GÉNÉRAUX
# ============================================================================

def safe_float(value):
    """
    Convertit une valeur numérique en float.

    Retourne NaN si la conversion est impossible ou si la valeur
    n'est pas finie.
    """
    try:
        numeric = float(value)

    except (TypeError, ValueError):
        return np.nan

    if not np.isfinite(numeric):
        return np.nan

    return numeric


def normalize_identifier(value):
    """
    Normalise un identifiant participant ou item.

    Exemples :
        63873.0 -> "63873"
        63873   -> "63873"
    """
    if pd.isna(value):
        return pd.NA

    normalized = str(value).strip()

    if not normalized:
        return pd.NA

    try:
        numeric = float(normalized)

        if numeric.is_integer():
            return str(int(numeric))

    except (TypeError, ValueError):
        pass

    return normalized


def make_formula(terms):
    """Construit une formule statsmodels."""
    return (
        "confidence ~ "
        + " + ".join(terms)
    )


def standardize(series):
    """
    Standardise une variable :

        z = (x - moyenne) / écart-type

    L'écart-type empirique est calculé avec ddof=1.
    """
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    mean = float(
        numeric.mean()
    )

    standard_deviation = float(
        numeric.std(ddof=1)
    )

    if (
        not np.isfinite(standard_deviation)
        or standard_deviation <= 0
    ):
        raise ValueError(
            "Impossible de standardiser "
            f"{series.name} : "
            f"écart-type = {standard_deviation}"
        )

    standardized = (
        numeric - mean
    ) / standard_deviation

    return (
        standardized,
        mean,
        standard_deviation,
    )


# ============================================================================
# CHARGEMENT ET PRÉPARATION
# ============================================================================

def load_and_prepare_data():
    """
    Charge le dataset et reconstruit exactement les transformations
    utilisées dans le modèle cognitif principal.
    """
    section(
        "CHARGEMENT ET PRÉPARATION DES DONNÉES"
    )

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {DATA_FILE}"
        )

    data = pd.read_csv(
        DATA_FILE
    )

    print(
        "Fichier :",
        DATA_FILE,
    )

    print(
        "Nombre de lignes brutes :",
        len(data),
    )

    missing_columns = [
        column
        for column in ESSENTIAL_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise KeyError(
            "Colonnes nécessaires absentes : "
            + ", ".join(missing_columns)
        )

    data = data.copy()

    # ------------------------------------------------------------------
    # Filtrage des lignes complètes
    # ------------------------------------------------------------------

    if "analysis_complete" in data.columns:
        complete_mask = (
            data["analysis_complete"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({
                "true",
                "1",
                "1.0",
                "yes",
            })
        )

        before_filter = len(data)

        data = data.loc[
            complete_mask
        ].copy()

        print(
            "Lignes retirées car analysis_complete=False :",
            before_filter - len(data),
        )

    # ------------------------------------------------------------------
    # Conversion numérique
    # ------------------------------------------------------------------

    numeric_columns = [
        "confidence",
        "sequence",
        "validity_binary",
        "subject_accuracy",
        "item_entropy",
        "subject_mean_models",
        "models_within_subject",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data["subject_id"] = (
        data["subject_id"]
        .apply(normalize_identifier)
        .astype("string")
    )

    data["item_id"] = (
        data["item_id"]
        .apply(normalize_identifier)
        .astype("string")
    )

    data["condition"] = (
        data["condition"]
        .astype("string")
        .str.strip()
    )

    data["task_type"] = (
        data["task_type"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    before_drop = len(data)

    data = data.dropna(
        subset=ESSENTIAL_COLUMNS
    ).copy()

    print(
        "Lignes supprimées pour donnée essentielle manquante :",
        before_drop - len(data),
    )

    # ------------------------------------------------------------------
    # Contrôle de la confiance
    # ------------------------------------------------------------------

    invalid_confidence = (
        (data["confidence"] < 0)
        | (data["confidence"] > 100)
    )

    if invalid_confidence.any():
        raise ValueError(
            "Certaines valeurs de confiance sont hors de [0, 100]."
        )

    # ------------------------------------------------------------------
    # Contrôle des conditions
    # ------------------------------------------------------------------

    expected_conditions = {
        "Neutral",
        "Standard",
    }

    observed_conditions = set(
        data["condition"]
        .dropna()
        .unique()
    )

    if observed_conditions != expected_conditions:
        raise ValueError(
            "Conditions attendues : Neutral et Standard. "
            "Conditions observées : "
            f"{sorted(observed_conditions)}"
        )

    # ------------------------------------------------------------------
    # Contrôle des types de tâche
    # ------------------------------------------------------------------

    expected_task_types = {
        "AC",
        "DA",
        "MP",
        "MT",
    }

    observed_task_types = set(
        data["task_type"]
        .dropna()
        .unique()
    )

    if observed_task_types != expected_task_types:
        raise ValueError(
            "Types de tâche attendus : AC, DA, MP et MT. "
            "Types observés : "
            f"{sorted(observed_task_types)}"
        )

    # ------------------------------------------------------------------
    # Contrôle de validity_binary
    # ------------------------------------------------------------------

    validity_values = set(
        data["validity_binary"]
        .dropna()
        .unique()
    )

    if not validity_values.issubset({
        0,
        1,
        0.0,
        1.0,
    }):
        raise ValueError(
            "validity_binary doit uniquement contenir "
            "les valeurs 0 et 1. "
            f"Valeurs observées : {sorted(validity_values)}"
        )

    if validity_values != {
        0.0,
        1.0,
    }:
        raise ValueError(
            "Les deux niveaux de validité, 0 et 1, "
            "doivent être présents."
        )

    data["validity_binary"] = (
        data["validity_binary"]
        .astype(int)
    )

    # ------------------------------------------------------------------
    # Relation structurelle entre type de tâche et validité
    # ------------------------------------------------------------------

    validity_task_table = pd.crosstab(
        data["task_type"],
        data["validity_binary"],
    )

    print("")
    print(
        "Relation entre task_type et validity_binary :"
    )

    print(
        validity_task_table.to_string()
    )

    validity_counts_by_task = (
        data
        .groupby("task_type")[
        "validity_binary"]
        .nunique()
    )

    validity_constant_within_task = bool(
        (
            validity_counts_by_task == 1
        ).all()
    )

    print(
        "Validité constante à l'intérieur de chaque type :",
        validity_constant_within_task,
    )

    # ------------------------------------------------------------------
    # Centrage de la séquence
    # ------------------------------------------------------------------

    sequence_mean = float(
        data["sequence"].mean()
    )

    data["sequence_c10"] = (
        data["sequence"]
        - sequence_mean
    ) / 10.0

    # ------------------------------------------------------------------
    # Standardisation
    # ------------------------------------------------------------------

    standardization_rows = []

    for variable in STANDARDIZED_COLUMNS:
        standardized_name = (
            f"{variable}_z"
        )

        (
            data[standardized_name],
            variable_mean,
            variable_standard_deviation,
        ) = standardize(
            data[variable]
        )

        standardization_rows.append({
            "variable":
                variable,

            "standardized_variable":
                standardized_name,

            "mean":
                variable_mean,

            "standard_deviation":
                variable_standard_deviation,
        })

    standardization_table = pd.DataFrame(
        standardization_rows
    )

    # ------------------------------------------------------------------
    # Tri stable
    # ------------------------------------------------------------------

    data = (
        data
        .sort_values(
            by=[
                "subject_id",
                "item_id",
                "sequence",
            ]
        )
        .reset_index(drop=True)
    )

    print("")
    print(
        "Nombre de lignes utilisées :",
        len(data),
    )

    print(
        "Nombre de participants :",
        data["subject_id"].nunique(),
    )

    print(
        "Nombre d'items :",
        data["item_id"].nunique(),
    )

    print(
        "Types de tâche :",
        sorted(observed_task_types),
    )

    print(
        "Centre de la séquence :",
        sequence_mean,
    )

    print("")
    print(
        "Paramètres de standardisation :"
    )

    print(
        standardization_table
        .to_string(index=False)
    )

    return data


# ============================================================================
# CONSTRUCTION DU MODÈLE
# ============================================================================

def build_model(
    data,
    formula,
):
    """
    Construit un modèle mixte avec des intercepts aléatoires croisés
    pour les participants et les items.
    """
    model_data = data.copy()

    model_data["_global_group"] = (
        "all_observations"
    )

    return smf.mixedlm(
        formula=formula,
        data=model_data,
        groups=model_data[
            "_global_group"
        ],
        re_formula="0",
        vc_formula=VC_FORMULA,
        use_sparse=USE_SPARSE_MATRICES,
    )


# ============================================================================
# AJUSTEMENT
# ============================================================================

def fit_model(
    data,
    formula,
    model_name,
    reml=False,
):
    """
    Essaie plusieurs optimiseurs et conserve le premier ajustement
    convergé.
    """
    estimation = (
        "REML"
        if reml
        else "ML"
    )

    section(
        f"AJUSTEMENT — {model_name} — {estimation}"
    )

    print(
        "Formule :",
        formula,
    )

    last_error = None
    last_result = None

    for optimizer in OPTIMIZERS:
        print("")
        print(
            "Optimiseur essayé :",
            optimizer,
        )

        try:
            model = build_model(
                data=data,
                formula=formula,
            )

            with warnings.catch_warnings(
                record=True
            ) as caught_warnings:
                warnings.simplefilter(
                    "always"
                )

                result = model.fit(
                    reml=reml,
                    method=optimizer,
                    maxiter=MAX_ITERATIONS,
                    full_output=True,
                    disp=False,
                )

            last_result = result

            for warning in caught_warnings:
                print(
                    "Avertissement statsmodels :",
                    warning.message,
                )

            converged = bool(
                getattr(
                    result,
                    "converged",
                    False,
                )
            )

            print(
                "Convergence :",
                converged,
            )

            print(
                "Log-vraisemblance :",
                safe_float(result.llf),
            )

            if converged:
                print(
                    "Optimiseur retenu :",
                    optimizer,
                )

                return (
                    result,
                    optimizer,
                )

        except Exception as error:
            last_error = error

            print(
                "Échec avec",
                optimizer,
                ":",
                repr(error),
            )

    if last_result is not None:
        raise RuntimeError(
            f"{model_name} : un résultat a été obtenu, "
            "mais aucun optimiseur n'a signalé une "
            "convergence complète."
        )

    raise RuntimeError(
        f"Échec de tous les optimiseurs pour {model_name}. "
        f"Dernière erreur : {last_error!r}"
    )


# ============================================================================
# EXTRACTION DES COEFFICIENTS
# ============================================================================

def fixed_effects_table(result):
    """
    Extrait les effets fixes et calcule les tests de Wald
    bilatéraux.
    """
    parameter_names = list(
        result.fe_params.index
    )

    estimates = np.asarray(
        result.fe_params,
        dtype=float,
    )

    covariance_matrix = (
        result
        .cov_params()
        .loc[
            parameter_names,
            parameter_names,
        ]
    )

    standard_errors = np.sqrt(
        np.diag(
            covariance_matrix
        )
    )

    z_values = (
        estimates
        / standard_errors
    )

    p_values = (
        2
        * stats.norm.sf(
            np.abs(z_values)
        )
    )

    return pd.DataFrame({
        "parameter":
            parameter_names,

        "estimate":
            estimates,

        "standard_error":
            standard_errors,

        "z_value":
            z_values,

        "p_value":
            p_values,

        "ci_95_lower":
            estimates
            - 1.96
            * standard_errors,

        "ci_95_upper":
            estimates
            + 1.96
            * standard_errors,
    })


# ============================================================================
# COMPOSANTES DE VARIANCE
# ============================================================================

def get_variance_components(result):
    """
    Extrait les variances participant, item et résiduelle.
    """
    try:
        component_names = list(
            result.model.exog_vc.names
        )

    except (AttributeError, TypeError):
        component_names = []

    component_values = np.asarray(
        result.vcomp,
        dtype=float,
    )

    if not component_names:
        component_names = [
            f"component_{index}"
            for index in range(
                len(component_values)
            )
        ]

    component_map = {
        str(name).strip().lower():
            float(value)

        for name, value in zip(
            component_names,
            component_values,
        )
    }

    participant_variance = np.nan
    item_variance = np.nan

    for name, variance in component_map.items():
        if "subject" in name:
            participant_variance = variance

        elif "item" in name:
            item_variance = variance

    if (
        pd.isna(participant_variance)
        or pd.isna(item_variance)
    ):
        raise RuntimeError(
            "Impossible d'identifier les composantes "
            "participant et item. "
            f"Composantes trouvées : {component_map}"
        )

    return {
        "participant_variance":
            participant_variance,

        "item_variance":
            item_variance,

        "residual_variance":
            float(result.scale),
    }


# ============================================================================
# STATISTIQUES D'AJUSTEMENT
# ============================================================================

def create_fit_statistics(
    model_name,
    formula,
    result,
    optimizer,
):
    """
    Crée une ligne de statistiques pour un modèle ML.
    """
    variances = get_variance_components(
        result
    )

    return {
        "model":
            model_name,

        "formula":
            formula,

        "converged":
            bool(
                getattr(
                    result,
                    "converged",
                    False,
                )
            ),

        "optimizer":
            optimizer,

        "log_likelihood":
            safe_float(
                result.llf
            ),

        "aic":
            safe_float(
                result.aic
            ),

        "bic":
            safe_float(
                result.bic
            ),

        "number_of_estimated_parameters":
            int(
                len(result.params)
            ),

        "n_observations":
            int(
                result.nobs
            ),

        "participant_variance":
            variances[
                "participant_variance"
            ],

        "item_variance":
            variances[
                "item_variance"
            ],

        "residual_variance":
            variances[
                "residual_variance"
            ],
    }


# ============================================================================
# TEST DU RAPPORT DE VRAISEMBLANCE
# ============================================================================

def likelihood_ratio_test(
    reduced_result,
    full_result,
):
    """
    Compare deux modèles emboîtés ajustés en ML.

    LR = 2 × (logL_complet - logL_réduit)

    Les degrés de liberté correspondent à la différence du nombre
    de paramètres estimés.
    """
    likelihood_ratio = (
        2
        * (
            safe_float(
                full_result.llf
            )
            - safe_float(
                reduced_result.llf
            )
        )
    )

    degrees_of_freedom = (
        len(full_result.params)
        - len(reduced_result.params)
    )

    if degrees_of_freedom <= 0:
        p_value = np.nan

    else:
        p_value = stats.chi2.sf(
            likelihood_ratio,
            degrees_of_freedom,
        )

    return {
        "likelihood_ratio":
            likelihood_ratio,

        "degrees_of_freedom":
            degrees_of_freedom,

        "p_value":
            p_value,
    }


# ============================================================================
# RÉSUMÉ REML DU MODÈLE TASK_TYPE
# ============================================================================

def save_task_type_reml_summary(
    result,
    optimizer,
):
    """
    Sauvegarde le résumé du modèle alternatif avec task_type.
    """
    with open(
        TASK_TYPE_REML_SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as output_file:
        output_file.write(
            "MODÈLE COGNITIF ALTERNATIF AVEC TASK_TYPE — REML\n"
        )

        output_file.write(
            "=" * 80
        )

        output_file.write("\n\n")

        output_file.write(
            "FORMULE\n"
        )

        output_file.write(
            "-" * 80
        )

        output_file.write("\n")

        output_file.write(
            TASK_TYPE_FORMULA
        )

        output_file.write("\n\n")

        output_file.write(
            f"Optimiseur retenu : {optimizer}\n"
        )

        output_file.write(
            "Catégorie de référence de task_type : AC\n\n"
        )

        output_file.write(
            result.summary().as_text()
        )

        output_file.write("\n")


# ============================================================================
# NETTOYAGE DES ANCIENNES SORTIES
# ============================================================================

def remove_obsolete_output_files():
    """
    Supprime les anciennes sorties qui ne sont plus produites.
    """
    obsolete_filenames = [
        "standardization.csv",
        "validity_model_fixed_effects_REML.csv",
        "task_type_model_fixed_effects_REML.csv",
        "validity_model_REML_summary.txt",
        "global_likelihood_ratio_tests.csv",
        "drop_one_tests.csv",
        "model_fit_comparison.csv",
    ]

    for filename in obsolete_filenames:
        path = (
            OUTPUT_DIR
            / filename
        )

        if path.is_file():
            path.unlink()

            print(
                "Ancienne sortie supprimée :",
                path,
            )


# ============================================================================
# FORMULES
# ============================================================================

CONTROL_FORMULA = make_formula(
    CONTROL_TERMS
)

BASE_COGNITIVE_FORMULA = make_formula(
    CONTROL_TERMS
    + COGNITIVE_TERMS
)

VALIDITY_FORMULA = make_formula(
    CONTROL_TERMS
    + COGNITIVE_TERMS
    + [VALIDITY_TERM]
)

TASK_TYPE_FORMULA = make_formula(
    CONTROL_TERMS
    + COGNITIVE_TERMS
    + [TASK_TYPE_TERM]
)


# ============================================================================
# PROGRAMME PRINCIPAL
# ============================================================================

def main():
    section(
        "ANALYSES DE SENSIBILITÉ DU MODÈLE MIXTE E1"
    )

    remove_obsolete_output_files()

    try:
        # ==================================================================
        # 1. Données
        # ==================================================================

        data = load_and_prepare_data()

        section(
            "FORMULES"
        )

        print(
            "Contrôle :",
            CONTROL_FORMULA,
        )

        print(
            "Cognitif sans validité :",
            BASE_COGNITIVE_FORMULA,
        )

        print(
            "Cognitif avec validité :",
            VALIDITY_FORMULA,
        )

        print(
            "Cognitif avec task_type :",
            TASK_TYPE_FORMULA,
        )

        # ==================================================================
        # 2. Modèles ML
        # ==================================================================

        (
            control_ml,
            control_optimizer,
        ) = fit_model(
            data=data,
            formula=CONTROL_FORMULA,
            model_name="Contrôle",
            reml=False,
        )

        (
            base_cognitive_ml,
            base_optimizer,
        ) = fit_model(
            data=data,
            formula=BASE_COGNITIVE_FORMULA,
            model_name="Cognitif sans validité",
            reml=False,
        )

        (
            validity_ml,
            validity_optimizer,
        ) = fit_model(
            data=data,
            formula=VALIDITY_FORMULA,
            model_name="Cognitif avec validité",
            reml=False,
        )

        (
            task_type_ml,
            task_type_optimizer,
        ) = fit_model(
            data=data,
            formula=TASK_TYPE_FORMULA,
            model_name="Cognitif avec task_type",
            reml=False,
        )

        # ==================================================================
        # 3. Modèle task_type REML
        # ==================================================================

        (
            task_type_reml,
            task_type_reml_optimizer,
        ) = fit_model(
            data=data,
            formula=TASK_TYPE_FORMULA,
            model_name="Cognitif avec task_type",
            reml=True,
        )

        # ==================================================================
        # 4. Tests globaux
        # ==================================================================

        section(
            "TESTS GLOBAUX"
        )

        global_comparisons = [
            (
                "Control vs Cognitive_without_validity",
                control_ml,
                base_cognitive_ml,
            ),
            (
                "Cognitive_without_validity vs Validity",
                base_cognitive_ml,
                validity_ml,
            ),
            (
                "Cognitive_without_validity vs Task_type",
                base_cognitive_ml,
                task_type_ml,
            ),
        ]

        global_test_rows = []

        for (
            comparison_name,
            reduced_result,
            full_result,
        ) in global_comparisons:
            test = likelihood_ratio_test(
                reduced_result=
                    reduced_result,
                full_result=
                    full_result,
            )

            global_test_rows.append({
                "comparison":
                    comparison_name,

                **test,
            })

        global_tests = pd.DataFrame(
            global_test_rows
        )

        global_tests.to_csv(
            GLOBAL_TESTS_FILE,
            index=False,
        )

        print(
            global_tests
            .to_string(index=False)
        )

        # ==================================================================
        # 5. Tests drop-one
        # ==================================================================

        section(
            "TESTS DROP-ONE"
        )

        full_validity_terms = (
            CONTROL_TERMS
            + COGNITIVE_TERMS
            + [VALIDITY_TERM]
        )

        tested_terms = (
            COGNITIVE_TERMS
            + [VALIDITY_TERM]
        )

        drop_one_rows = []

        full_model_aic = safe_float(
            validity_ml.aic
        )

        for removed_term in tested_terms:
            reduced_terms = [
                term
                for term in full_validity_terms
                if term != removed_term
            ]

            reduced_formula = make_formula(
                reduced_terms
            )

            (
                reduced_result,
                reduced_optimizer,
            ) = fit_model(
                data=data,
                formula=reduced_formula,
                model_name=(
                    f"Modèle sans {removed_term}"
                ),
                reml=False,
            )

            test = likelihood_ratio_test(
                reduced_result=
                    reduced_result,
                full_result=
                    validity_ml,
            )

            reduced_aic = safe_float(
                reduced_result.aic
            )

            drop_one_rows.append({
                "removed_predictor":
                    removed_term,

                "likelihood_ratio":
                    test[
                        "likelihood_ratio"
                    ],

                "degrees_of_freedom":
                    test[
                        "degrees_of_freedom"
                    ],

                "p_value":
                    test[
                        "p_value"
                    ],

                "reduced_aic":
                    reduced_aic,

                "delta_aic_reduced_minus_full":
                    (
                        reduced_aic
                        - full_model_aic
                    ),
            })

            print("")
            print(
                "Prédicteur retiré :",
                removed_term,
            )

            print(
                "Optimiseur :",
                reduced_optimizer,
            )

            print(
                "LR :",
                test[
                    "likelihood_ratio"
                ],
            )

            print(
                "p :",
                test[
                    "p_value"
                ],
            )

        drop_one_table = pd.DataFrame(
            drop_one_rows
        )

        drop_one_table.to_csv(
            DROP_ONE_TESTS_FILE,
            index=False,
        )

        section(
            "RÉSULTATS DROP-ONE"
        )

        print(
            drop_one_table
            .to_string(index=False)
        )

        # ==================================================================
        # 6. Comparaison des ajustements ML
        # ==================================================================

        fit_rows = [
            create_fit_statistics(
                model_name="Control",
                formula=CONTROL_FORMULA,
                result=control_ml,
                optimizer=control_optimizer,
            ),
            create_fit_statistics(
                model_name=(
                    "Cognitive_without_validity"
                ),
                formula=
                    BASE_COGNITIVE_FORMULA,
                result=
                    base_cognitive_ml,
                optimizer=
                    base_optimizer,
            ),
            create_fit_statistics(
                model_name=(
                    "Cognitive_validity"
                ),
                formula=
                    VALIDITY_FORMULA,
                result=
                    validity_ml,
                optimizer=
                    validity_optimizer,
            ),
            create_fit_statistics(
                model_name=(
                    "Cognitive_task_type"
                ),
                formula=
                    TASK_TYPE_FORMULA,
                result=
                    task_type_ml,
                optimizer=
                    task_type_optimizer,
            ),
        ]

        fit_table = pd.DataFrame(
            fit_rows
        )

        fit_table.to_csv(
            MODEL_FIT_COMPARISON_FILE,
            index=False,
        )

        section(
            "COMPARAISON DES AJUSTEMENTS ML"
        )

        print(
            fit_table
            .to_string(index=False)
        )

        # ==================================================================
        # 7. Coefficients task_type REML
        # ==================================================================

        task_type_fixed_effects = (
            fixed_effects_table(
                task_type_reml
            )
        )

        task_type_fixed_effects.to_csv(
            TASK_TYPE_FIXED_EFFECTS_FILE,
            index=False,
        )

        save_task_type_reml_summary(
            result=task_type_reml,
            optimizer=
                task_type_reml_optimizer,
        )

        section(
            "EFFETS FIXES DU MODÈLE TASK_TYPE — REML"
        )

        print(
            task_type_fixed_effects
            .to_string(index=False)
        )

        # ==================================================================
        # 8. Résumé
        # ==================================================================

        section(
            "FICHIERS PRODUITS"
        )

        for output_file in [
            GLOBAL_TESTS_FILE,
            DROP_ONE_TESTS_FILE,
            MODEL_FIT_COMPARISON_FILE,
            TASK_TYPE_FIXED_EFFECTS_FILE,
            TASK_TYPE_REML_SUMMARY_FILE,
        ]:
            print(
                output_file
            )

        print("")
        print("=" * 80)
        print("ANALYSES DE SENSIBILITÉ TERMINÉES")
        print("=" * 80)

    except (
        FileNotFoundError,
        KeyError,
        ValueError,
        TypeError,
        RuntimeError,
        pd.errors.ParserError,
        pd.errors.MergeError,
    ) as error:
        section(
            "ERREUR"
        )

        print(
            type(error).__name__,
            ":",
            error,
        )

        sys.exit(1)


if __name__ == "__main__":
    main()
