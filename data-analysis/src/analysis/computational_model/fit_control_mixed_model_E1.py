"""
fit_control_mixed_model_E1.py

Ajuste et compare deux modèles linéaires mixtes pour la confiance
dans l'expérience E1.

Entrée :
    dataset_analysis_E1.csv

Modèle nul :
    confidence ~ 1
    + interception aléatoire participant
    + interception aléatoire item

Modèle de contrôle :
    confidence ~ condition + sequence_c10
    + interception aléatoire participant
    + interception aléatoire item

La condition Neutral est utilisée comme catégorie de référence.

La variable sequence_c10 est définie par :

    sequence_c10 = (sequence - moyenne(sequence)) / 10

Son coefficient représente donc l'évolution moyenne de confiance
associée à dix essais supplémentaires.

Estimations :
    - ML pour comparer le modèle nul et le modèle de contrôle ;
    - REML pour présenter les estimations finales du modèle de contrôle.

Fichiers produits dans control_mixed_model_E1/ :

    control_model_REML_summary.txt
    control_model_ML_summary.txt
    null_model_ML_summary.txt

    model_comparison.csv
    control_model_fixed_effects.csv
    control_model_variance_components.csv
    variance_comparison.csv
    control_model_fit_statistics.csv
    control_model_predictions.csv

    condition_adjusted_means.csv
    sequence_predictions.csv

    control_model_residuals_vs_fitted.png
    control_model_residual_distribution.png
    control_model_qqplot.png
    control_model_condition_effect.png
    control_model_sequence_effect.png
    control_model_variance_comparison.png

    control_model_results.json
    control_model_report.txt

Ce script ne contient pas encore les prédicteurs cognitifs principaux :
    - précision ;
    - entropie ;
    - nombre de modèles ;
    - validité.
"""

import json
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import seaborn as sns
import statsmodels
import statsmodels.formula.api as smf


# ======================================================================
# CONFIGURATION
# ======================================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

INPUT_FILE = os.path.join(
    SCRIPT_DIR,
    "dataset_analysis_E1.csv",
)

OUTPUT_DIRECTORY = os.path.join(
    SCRIPT_DIR,
    "control_mixed_model_E1",
)

os.makedirs(
    OUTPUT_DIRECTORY,
    exist_ok=True,
)


# ----------------------------------------------------------------------
# Résumés texte
# ----------------------------------------------------------------------

NULL_ML_SUMMARY_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "null_model_ML_summary.txt",
)

CONTROL_ML_SUMMARY_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "control_model_ML_summary.txt",
)

CONTROL_REML_SUMMARY_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "control_model_REML_summary.txt",
)

REPORT_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "control_model_report.txt",
)

JSON_RESULTS_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "control_model_results.json",
)


# ----------------------------------------------------------------------
# Fichiers CSV
# ----------------------------------------------------------------------

MODEL_COMPARISON_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "model_comparison.csv",
)

FIXED_EFFECTS_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "control_model_fixed_effects.csv",
)

VARIANCE_COMPONENTS_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "control_model_variance_components.csv",
)

VARIANCE_COMPARISON_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "variance_comparison.csv",
)

FIT_STATISTICS_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "control_model_fit_statistics.csv",
)

PREDICTIONS_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "control_model_predictions.csv",
)

CONDITION_MEANS_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "condition_adjusted_means.csv",
)

SEQUENCE_PREDICTIONS_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "sequence_predictions.csv",
)


# ----------------------------------------------------------------------
# Fichiers graphiques
# ----------------------------------------------------------------------

RESIDUALS_VS_FITTED_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "control_model_residuals_vs_fitted.png",
)

RESIDUAL_DISTRIBUTION_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "control_model_residual_distribution.png",
)

QQPLOT_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "control_model_qqplot.png",
)

CONDITION_EFFECT_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "control_model_condition_effect.png",
)

SEQUENCE_EFFECT_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "control_model_sequence_effect.png",
)

VARIANCE_COMPARISON_PLOT_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "control_model_variance_comparison.png",
)


# ----------------------------------------------------------------------
# Options
# ----------------------------------------------------------------------

# Catégorie de référence de la condition.
REFERENCE_CONDITION = "Neutral"

# Dix essais constituent une unité pour le coefficient de séquence.
SEQUENCE_SCALE = 10.0

# Matrices creuses pour les effets participant et item.
USE_SPARSE_MATRICES = True

MAX_ITERATIONS = 2000

OPTIMIZER_DISPLAY = False

OPTIMIZATION_METHODS = [
    "lbfgs",
    "bfgs",
    "cg",
    "powell",
]

DPI = 300

sns.set_theme(
    style="whitegrid",
    context="notebook",
)


# ======================================================================
# FORMULES
# ======================================================================

NULL_FORMULA = (
    "confidence ~ 1"
)

CONTROL_FORMULA = (
    "confidence ~ "
    "C(condition, Treatment(reference='Neutral')) "
    "+ sequence_c10"
)

VARIANCE_COMPONENT_FORMULAS = {
    "item": "0 + C(item_id)",
    "subject": "0 + C(subject_id)",
}


# ======================================================================
# RAPPORT
# ======================================================================

REPORT_LINES = []


def report_print(*values):
    """
    Affiche une ligne et l'ajoute au rapport final.
    """
    text = " ".join(
        str(value)
        for value in values
    )

    print(text)

    REPORT_LINES.append(
        text
    )


def report_section(title):
    """
    Affiche un titre de section.
    """
    separator = "=" * 80

    report_print("")
    report_print(separator)
    report_print(title)
    report_print(separator)


def save_report():
    """
    Sauvegarde le rapport terminal.
    """
    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as output_file:
        output_file.write(
            "\n".join(REPORT_LINES)
        )

        output_file.write("\n")

    print(
        "Rapport enregistré :",
        REPORT_FILE,
    )


# ======================================================================
# OUTILS GÉNÉRAUX
# ======================================================================

def normalize_identifier(value):
    """
    Normalise un identifiant numérique ou textuel.
    """
    if pd.isna(value):
        return pd.NA

    normalized = str(value).strip()

    if not normalized:
        return pd.NA

    try:
        numeric = float(
            normalized
        )

        if numeric.is_integer():
            return str(
                int(numeric)
            )

    except (
        TypeError,
        ValueError,
    ):
        pass

    return normalized


def normalize_condition(value):
    """
    Normalise la condition vers Standard ou Neutral.
    """
    if pd.isna(value):
        return pd.NA

    normalized = str(
        value
    ).strip().lower()

    if normalized == "standard":
        return "Standard"

    if normalized in {
        "neutral",
        "neutre",
    }:
        return "Neutral"

    return str(value).strip()


def normalize_analysis_complete(value):
    """
    Normalise la colonne analysis_complete.
    """
    if pd.isna(value):
        return False

    if isinstance(
        value,
        (bool, np.bool_),
    ):
        return bool(value)

    normalized = str(
        value
    ).strip().lower()

    return normalized in {
        "true",
        "1",
        "1.0",
        "yes",
        "oui",
    }


def safe_float(value):
    """
    Convertit une valeur en float utilisable dans un JSON.
    """
    try:
        numeric = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        return None

    if not np.isfinite(
        numeric
    ):
        return None

    return numeric


def save_figure(
    figure,
    output_file,
):
    """
    Sauvegarde une figure.
    """
    figure.savefig(
        output_file,
        dpi=DPI,
        bbox_inches="tight",
    )

    print(
        "Graphique enregistré :",
        output_file,
    )


def percentage_change(
    old_value,
    new_value,
):
    """
    Calcule le pourcentage de variation entre deux valeurs.

    Une valeur négative indique une diminution.
    """
    if (
        old_value is None
        or new_value is None
        or pd.isna(old_value)
        or pd.isna(new_value)
        or np.isclose(old_value, 0.0)
    ):
        return np.nan

    return float(
        100.0
        * (
            new_value
            - old_value
        )
        / old_value
    )


# ======================================================================
# CHARGEMENT DES DONNÉES
# ======================================================================

def load_analysis_data():
    """
    Charge et valide dataset_analysis_E1.csv.
    """
    report_section(
        "CHARGEMENT DES DONNÉES"
    )

    if not os.path.isfile(
        INPUT_FILE
    ):
        raise FileNotFoundError(
            "Fichier analytique introuvable : "
            f"{INPUT_FILE}"
        )

    dataframe = pd.read_csv(
        INPUT_FILE
    )

    report_print(
        "Fichier :",
        INPUT_FILE,
    )

    report_print(
        "Nombre de lignes brutes :",
        len(dataframe),
    )

    required_columns = {
        "subject_id",
        "item_id",
        "confidence",
        "condition",
        "sequence",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise KeyError(
            "Colonnes absentes du fichier analytique : "
            f"{sorted(missing_columns)}"
        )

    dataframe = dataframe.copy()

    dataframe["subject_id"] = (
        dataframe["subject_id"]
        .apply(normalize_identifier)
        .astype("string")
    )

    dataframe["item_id"] = (
        dataframe["item_id"]
        .apply(normalize_identifier)
        .astype("string")
    )

    dataframe["confidence"] = pd.to_numeric(
        dataframe["confidence"],
        errors="coerce",
    )

    dataframe["sequence"] = pd.to_numeric(
        dataframe["sequence"],
        errors="coerce",
    )

    dataframe["condition"] = (
        dataframe["condition"]
        .apply(normalize_condition)
        .astype("string")
    )

    if "analysis_complete" in dataframe.columns:
        dataframe[
            "analysis_complete"
        ] = (
            dataframe[
                "analysis_complete"
            ]
            .apply(
                normalize_analysis_complete
            )
        )

        before_filter = len(
            dataframe
        )

        dataframe = dataframe.loc[
            dataframe[
                "analysis_complete"
            ]
        ].copy()

        report_print(
            "Lignes retirées car analysis_complete=False :",
            before_filter - len(dataframe),
        )

    dataframe = dataframe.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    before_drop = len(
        dataframe
    )

    dataframe = (
        dataframe
        .dropna(
            subset=[
                "subject_id",
                "item_id",
                "confidence",
                "condition",
                "sequence",
            ]
        )
        .copy()
    )

    report_print(
        "Lignes retirées pour valeur essentielle manquante :",
        before_drop - len(dataframe),
    )

    invalid_confidence = dataframe.loc[
        (
            dataframe["confidence"] < 0
        )
        | (
            dataframe["confidence"] > 100
        )
    ]

    if not invalid_confidence.empty:
        raise ValueError(
            "Certaines confiances sont hors de [0, 100]."
        )

    observed_conditions = set(
        dataframe[
            "condition"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    expected_conditions = {
        "Neutral",
        "Standard",
    }

    if observed_conditions != expected_conditions:
        raise ValueError(
            "Les conditions attendues sont Neutral et Standard. "
            f"Conditions observées : {sorted(observed_conditions)}"
        )

    if (
        REFERENCE_CONDITION
        not in observed_conditions
    ):
        raise ValueError(
            "La condition de référence est absente : "
            f"{REFERENCE_CONDITION}"
        )

    # --------------------------------------------------------------
    # Centrage de la séquence
    # --------------------------------------------------------------

    sequence_mean = float(
        dataframe[
            "sequence"
        ].mean()
    )

    dataframe[
        "sequence_centered"
    ] = (
        dataframe[
            "sequence"
        ]
        - sequence_mean
    )

    dataframe[
        "sequence_c10"
    ] = (
        dataframe[
            "sequence_centered"
        ]
        / SEQUENCE_SCALE
    )

    # Groupe artificiel unique pour les effets croisés.
    dataframe[
        "_global_group"
    ] = "all_observations"

    dataframe = (
        dataframe
        .sort_values(
            by=[
                "subject_id",
                "sequence",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    report_print(
        "Nombre de lignes utilisées :",
        len(dataframe),
    )

    report_print(
        "Nombre de participants :",
        dataframe[
            "subject_id"
        ].nunique(),
    )

    report_print(
        "Nombre d'items :",
        dataframe[
            "item_id"
        ].nunique(),
    )

    report_print(
        "Moyenne de la séquence utilisée pour le centrage :",
        sequence_mean,
    )

    report_print(
        "Interprétation de sequence_c10 : "
        "une unité correspond à dix essais."
    )

    report_print(
        "Répartition des conditions par participant :"
    )

    condition_counts = (
        dataframe[[
            "subject_id",
            "condition",
        ]]
        .drop_duplicates()
        ["condition"]
        .value_counts()
    )

    for condition, count in condition_counts.items():
        report_print(
            f"  {condition} :",
            int(count),
            "participant(s)",
        )

    return (
        dataframe,
        sequence_mean,
    )


# ======================================================================
# CONSTRUCTION DES MODÈLES
# ======================================================================

def build_mixed_model(
    dataframe,
    formula,
    model_label,
):
    """
    Construit un modèle mixte avec effets croisés participant/item.
    """
    report_section(
        f"CONSTRUCTION — {model_label}"
    )

    report_print(
        "Formule :",
        formula,
    )

    report_print(
        "Effet aléatoire participant :",
        VARIANCE_COMPONENT_FORMULAS[
            "subject"
        ],
    )

    report_print(
        "Effet aléatoire item :",
        VARIANCE_COMPONENT_FORMULAS[
            "item"
        ],
    )

    model = smf.mixedlm(
        formula=formula,
        data=dataframe,
        groups=dataframe[
            "_global_group"
        ],
        re_formula="0",
        vc_formula=
            VARIANCE_COMPONENT_FORMULAS,
        use_sparse=USE_SPARSE_MATRICES,
    )

    return model


# ======================================================================
# AJUSTEMENT
# ======================================================================

def fit_with_fallback(
    model,
    reml,
    label,
):
    """
    Ajuste un modèle en essayant plusieurs optimiseurs.
    """
    report_section(
        f"AJUSTEMENT — {label}"
    )

    last_result = None
    last_error = None

    for method in OPTIMIZATION_METHODS:
        report_print(
            "Optimiseur essayé :",
            method,
        )

        try:
            with warnings.catch_warnings(
                record=True
            ) as caught_warnings:
                warnings.simplefilter(
                    "always"
                )

                result = model.fit(
                    reml=reml,
                    method=method,
                    maxiter=MAX_ITERATIONS,
                    disp=OPTIMIZER_DISPLAY,
                    full_output=True,
                )

            last_result = result

            for warning in caught_warnings:
                report_print(
                    "WARNING statsmodels :",
                    str(
                        warning.message
                    ),
                )

            converged = bool(
                getattr(
                    result,
                    "converged",
                    False,
                )
            )

            report_print(
                "Convergence :",
                converged,
            )

            report_print(
                "Log-vraisemblance :",
                safe_float(
                    result.llf
                ),
            )

            if converged:
                report_print(
                    "Ajustement convergé avec :",
                    method,
                )

                return result

        except Exception as error:
            last_error = error

            report_print(
                "Échec avec",
                method,
                ":",
                repr(error),
            )

    if last_result is not None:
        report_print(
            "ATTENTION : résultat retourné sans convergence complète."
        )

        return last_result

    raise RuntimeError(
        "Impossible d'ajuster le modèle. "
        f"Dernière erreur : {last_error!r}"
    )


# ======================================================================
# RÉSUMÉS
# ======================================================================

def save_model_summary(
    result,
    output_file,
    title,
):
    """
    Enregistre le résumé statsmodels.
    """
    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as output:
        output.write(
            title
        )

        output.write("\n")
        output.write("=" * 80)
        output.write("\n\n")

        output.write(
            result.summary().as_text()
        )

        output.write("\n")

    print(
        "Résumé enregistré :",
        output_file,
    )


def create_fixed_effects_table(result):
    """
    Produit le tableau des effets fixes.
    """
    confidence_intervals = (
        result.conf_int()
    )

    rows = []

    for parameter_name, estimate in (
        result.fe_params.items()
    ):
        standard_error = float(
            result.bse_fe[
                parameter_name
            ]
        )

        if standard_error > 0:
            z_value = (
                float(estimate)
                / standard_error
            )

            p_value = float(
                2.0
                * stats.norm.sf(
                    abs(z_value)
                )
            )

        else:
            z_value = np.nan
            p_value = np.nan

        if (
            parameter_name
            in confidence_intervals.index
        ):
            ci_lower = float(
                confidence_intervals.loc[
                    parameter_name,
                    0,
                ]
            )

            ci_upper = float(
                confidence_intervals.loc[
                    parameter_name,
                    1,
                ]
            )

        else:
            ci_lower = np.nan
            ci_upper = np.nan

        rows.append({
            "parameter":
                parameter_name,

            "estimate":
                float(estimate),

            "standard_error":
                standard_error,

            "z_value":
                z_value,

            "p_value":
                p_value,

            "ci_95_lower":
                ci_lower,

            "ci_95_upper":
                ci_upper,
        })

    return pd.DataFrame(
        rows
    )


# ======================================================================
# COMPOSANTES DE VARIANCE
# ======================================================================

def get_variance_component_map(result):
    """
    Associe les noms des composantes aux variances estimées.
    """
    try:
        component_names = list(
            result.model.exog_vc.names
        )

    except (
        AttributeError,
        TypeError,
    ):
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

    return {
        str(name).strip().lower():
            float(value)
        for name, value in zip(
            component_names,
            component_values,
        )
    }


def extract_variances(result):
    """
    Extrait les variances participant, item et résiduelle.
    """
    component_map = (
        get_variance_component_map(
            result
        )
    )

    subject_variance = np.nan
    item_variance = np.nan

    for name, variance in (
        component_map.items()
    ):
        if "subject" in name:
            subject_variance = variance

        elif "item" in name:
            item_variance = variance

    if (
        pd.isna(subject_variance)
        or pd.isna(item_variance)
    ):
        raise RuntimeError(
            "Impossible d'identifier les composantes de variance : "
            f"{component_map}"
        )

    residual_variance = float(
        result.scale
    )

    total_variance = (
        subject_variance
        + item_variance
        + residual_variance
    )

    return {
        "subject_variance":
            subject_variance,

        "item_variance":
            item_variance,

        "residual_variance":
            residual_variance,

        "total_variance":
            total_variance,

        "subject_icc":
            (
                subject_variance
                / total_variance
            ),

        "item_icc":
            (
                item_variance
                / total_variance
            ),

        "residual_proportion":
            (
                residual_variance
                / total_variance
            ),
    }


def create_variance_table(variances):
    """
    Produit un tableau des composantes de variance.
    """
    rows = []

    component_definitions = [
        (
            "Participant",
            "subject_variance",
        ),
        (
            "Item",
            "item_variance",
        ),
        (
            "Residual",
            "residual_variance",
        ),
    ]

    for label, key in component_definitions:
        variance = float(
            variances[key]
        )

        rows.append({
            "component":
                label,

            "variance":
                variance,

            "standard_deviation":
                float(
                    np.sqrt(
                        max(
                            variance,
                            0.0,
                        )
                    )
                ),

            "proportion_total_variance":
                (
                    variance
                    / variances[
                        "total_variance"
                    ]
                ),
        })

    rows.append({
        "component":
            "Total",

        "variance":
            variances[
                "total_variance"
            ],

        "standard_deviation":
            float(
                np.sqrt(
                    variances[
                        "total_variance"
                    ]
                )
            ),

        "proportion_total_variance":
            1.0,
    })

    return pd.DataFrame(
        rows
    )


# ======================================================================
# R² DU MODÈLE MIXTE
# ======================================================================

def compute_mixed_model_r2(
    dataframe,
    result,
    variances,
):
    """
    Calcule des R² descriptifs de type marginal et conditionnel.

    R² marginal :
        proportion de variance représentée par les effets fixes.

    R² conditionnel :
        proportion représentée par les effets fixes et aléatoires.
    """
    fixed_predictions = np.asarray(
        result.model.exog
        @ np.asarray(
            result.fe_params,
            dtype=float,
        ),
        dtype=float,
    )

    fixed_variance = float(
        np.var(
            fixed_predictions,
            ddof=0,
        )
    )

    denominator = (
        fixed_variance
        + variances[
            "subject_variance"
        ]
        + variances[
            "item_variance"
        ]
        + variances[
            "residual_variance"
        ]
    )

    if denominator <= 0:
        return {
            "fixed_effect_variance":
                fixed_variance,

            "marginal_r2":
                np.nan,

            "conditional_r2":
                np.nan,
        }

    marginal_r2 = (
        fixed_variance
        / denominator
    )

    conditional_r2 = (
        fixed_variance
        + variances[
            "subject_variance"
        ]
        + variances[
            "item_variance"
        ]
    ) / denominator

    return {
        "fixed_effect_variance":
            fixed_variance,

        "marginal_r2":
            marginal_r2,

        "conditional_r2":
            conditional_r2,
    }


# ======================================================================
# COMPARAISON DES MODÈLES ML
# ======================================================================

def compare_ml_models(
    null_result,
    control_result,
):
    """
    Compare le modèle nul et le modèle de contrôle en ML.
    """
    null_log_likelihood = float(
        null_result.llf
    )

    control_log_likelihood = float(
        control_result.llf
    )

    likelihood_ratio = (
        2.0
        * (
            control_log_likelihood
            - null_log_likelihood
        )
    )

    parameter_difference = (
        len(control_result.params)
        - len(null_result.params)
    )

    if parameter_difference <= 0:
        raise RuntimeError(
            "Le modèle de contrôle ne contient pas davantage "
            "de paramètres que le modèle nul."
        )

    likelihood_ratio_p = float(
        stats.chi2.sf(
            likelihood_ratio,
            parameter_difference,
        )
    )

    comparison = pd.DataFrame([
        {
            "model":
                "Null",

            "formula":
                NULL_FORMULA,

            "log_likelihood":
                null_log_likelihood,

            "aic":
                float(
                    null_result.aic
                ),

            "bic":
                float(
                    null_result.bic
                ),

            "number_of_estimated_parameters":
                len(
                    null_result.params
                ),

            "likelihood_ratio_vs_null":
                0.0,

            "degrees_of_freedom_difference":
                0,

            "likelihood_ratio_p_value":
                np.nan,
        },
        {
            "model":
                "Control",

            "formula":
                CONTROL_FORMULA,

            "log_likelihood":
                control_log_likelihood,

            "aic":
                float(
                    control_result.aic
                ),

            "bic":
                float(
                    control_result.bic
                ),

            "number_of_estimated_parameters":
                len(
                    control_result.params
                ),

            "likelihood_ratio_vs_null":
                likelihood_ratio,

            "degrees_of_freedom_difference":
                parameter_difference,

            "likelihood_ratio_p_value":
                likelihood_ratio_p,
        },
    ])

    statistics = {
        "likelihood_ratio":
            likelihood_ratio,

        "degrees_of_freedom":
            parameter_difference,

        "p_value":
            likelihood_ratio_p,

        "delta_aic_control_minus_null":
            (
                float(
                    control_result.aic
                )
                - float(
                    null_result.aic
                )
            ),

        "delta_bic_control_minus_null":
            (
                float(
                    control_result.bic
                )
                - float(
                    null_result.bic
                )
            ),
    }

    return (
        comparison,
        statistics,
    )


# ======================================================================
# COMPARAISON DES VARIANCES
# ======================================================================

def create_variance_comparison(
    null_variances,
    control_variances,
):
    """
    Compare les composantes de variance des deux modèles.
    """
    components = [
        (
            "Participant",
            "subject_variance",
        ),
        (
            "Item",
            "item_variance",
        ),
        (
            "Residual",
            "residual_variance",
        ),
    ]

    rows = []

    for label, key in components:
        null_value = float(
            null_variances[key]
        )

        control_value = float(
            control_variances[key]
        )

        rows.append({
            "component":
                label,

            "null_variance":
                null_value,

            "control_variance":
                control_value,

            "absolute_change":
                (
                    control_value
                    - null_value
                ),

            "percentage_change":
                percentage_change(
                    null_value,
                    control_value,
                ),

            "proportion_explained_relative_to_null":
                (
                    (
                        null_value
                        - control_value
                    )
                    / null_value
                    if not np.isclose(
                        null_value,
                        0.0,
                    )
                    else np.nan
                ),
        })

    return pd.DataFrame(
        rows
    )


# ======================================================================
# PRÉDICTIONS FIXES
# ======================================================================

def get_fixed_parameter(
    result,
    search_text,
):
    """
    Retrouve un coefficient par une partie de son nom.
    """
    search_text = search_text.lower()

    matches = [
        parameter
        for parameter in result.fe_params.index
        if search_text
        in parameter.lower()
    ]

    if len(matches) != 1:
        raise KeyError(
            "Impossible d'identifier un coefficient unique pour "
            f"{search_text!r}. Paramètres : "
            f"{list(result.fe_params.index)}"
        )

    return matches[0]


def create_condition_adjusted_means(
    result,
):
    """
    Calcule la confiance moyenne prédite à la séquence moyenne
    pour Neutral et Standard.
    """
    intercept = float(
        result.fe_params[
            "Intercept"
        ]
    )

    condition_parameter = (
        get_fixed_parameter(
            result,
            "[t.standard]",
        )
    )

    condition_effect = float(
        result.fe_params[
            condition_parameter
        ]
    )

    condition_standard_error = float(
        result.bse_fe[
            condition_parameter
        ]
    )

    neutral_mean = intercept

    standard_mean = (
        intercept
        + condition_effect
    )

    # L'intervalle autour de Neutral correspond directement à
    # l'interception.
    intercept_se = float(
        result.bse_fe[
            "Intercept"
        ]
    )

    neutral_ci = (
        neutral_mean
        - 1.96
        * intercept_se,
        neutral_mean
        + 1.96
        * intercept_se,
    )

    # La variance de la somme intercept + effet Standard est calculée
    # à partir de la matrice de covariance des effets fixes.
    covariance_matrix = (
        result.cov_params()
    )

    variance_standard_mean = (
        covariance_matrix.loc[
            "Intercept",
            "Intercept",
        ]
        + covariance_matrix.loc[
            condition_parameter,
            condition_parameter,
        ]
        + 2.0
        * covariance_matrix.loc[
            "Intercept",
            condition_parameter,
        ]
    )

    standard_mean_se = float(
        np.sqrt(
            max(
                variance_standard_mean,
                0.0,
            )
        )
    )

    standard_ci = (
        standard_mean
        - 1.96
        * standard_mean_se,
        standard_mean
        + 1.96
        * standard_mean_se,
    )

    return pd.DataFrame([
        {
            "condition":
                "Neutral",

            "adjusted_mean_confidence":
                neutral_mean,

            "standard_error":
                intercept_se,

            "ci_95_lower":
                neutral_ci[0],

            "ci_95_upper":
                neutral_ci[1],
        },
        {
            "condition":
                "Standard",

            "adjusted_mean_confidence":
                standard_mean,

            "standard_error":
                standard_mean_se,

            "ci_95_lower":
                standard_ci[0],

            "ci_95_upper":
                standard_ci[1],
        },
    ])


def create_sequence_predictions(
    result,
    sequence_mean,
):
    """
    Calcule les prédictions fixes selon la séquence pour les deux
    conditions.
    """
    intercept = float(
        result.fe_params[
            "Intercept"
        ]
    )

    condition_parameter = (
        get_fixed_parameter(
            result,
            "[t.standard]",
        )
    )

    condition_effect = float(
        result.fe_params[
            condition_parameter
        ]
    )

    sequence_parameter = (
        get_fixed_parameter(
            result,
            "sequence_c10",
        )
    )

    sequence_effect = float(
        result.fe_params[
            sequence_parameter
        ]
    )

    sequences = np.arange(
        1,
        65,
        dtype=float,
    )

    sequence_c10 = (
        sequences
        - sequence_mean
    ) / SEQUENCE_SCALE

    rows = []

    for condition in [
        "Neutral",
        "Standard",
    ]:
        condition_offset = (
            condition_effect
            if condition == "Standard"
            else 0.0
        )

        predictions = (
            intercept
            + condition_offset
            + sequence_effect
            * sequence_c10
        )

        for sequence, centered, prediction in zip(
            sequences,
            sequence_c10,
            predictions,
        ):
            rows.append({
                "condition":
                    condition,

                "sequence":
                    int(sequence),

                "sequence_c10":
                    float(centered),

                "predicted_confidence":
                    float(prediction),
            })

    return pd.DataFrame(
        rows
    )


def create_observation_predictions(
    dataframe,
    result,
):
    """
    Ajoute les prédictions issues uniquement des effets fixes.

    Ces prédictions n'incluent pas les effets individuels participant
    et item. Elles représentent la tendance populationnelle du modèle.
    """
    predictions = dataframe.copy()

    fixed_prediction = np.asarray(
        result.model.exog
        @ np.asarray(
            result.fe_params,
            dtype=float,
        ),
        dtype=float,
    )

    predictions[
        "fixed_prediction"
    ] = fixed_prediction

    predictions[
        "fixed_residual"
    ] = (
        predictions[
            "confidence"
        ]
        - predictions[
            "fixed_prediction"
        ]
    )

    predictions[
        "absolute_fixed_error"
    ] = np.abs(
        predictions[
            "fixed_residual"
        ]
    )

    predictions[
        "squared_fixed_error"
    ] = np.square(
        predictions[
            "fixed_residual"
        ]
    )

    output_columns = [
        "subject_id",
        "sequence",
        "item_id",
        "condition",
        "confidence",
        "sequence_c10",
        "fixed_prediction",
        "fixed_residual",
        "absolute_fixed_error",
        "squared_fixed_error",
    ]

    predictions[
        output_columns
    ].to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    return predictions


# ======================================================================
# GRAPHIQUES
# ======================================================================

def plot_condition_effect(
    adjusted_means,
):
    """
    Trace les moyennes ajustées par condition.
    """
    figure, axis = plt.subplots(
        figsize=(9, 7)
    )

    colors = {
        "Neutral": "#3b82f6",
        "Standard": "#f97316",
    }

    x_positions = np.arange(
        len(adjusted_means)
    )

    y_values = adjusted_means[
        "adjusted_mean_confidence"
    ].to_numpy()

    lower_errors = (
        y_values
        - adjusted_means[
            "ci_95_lower"
        ].to_numpy()
    )

    upper_errors = (
        adjusted_means[
            "ci_95_upper"
        ].to_numpy()
        - y_values
    )

    bars = axis.bar(
        x_positions,
        y_values,
        color=[
            colors[
                condition
            ]
            for condition in adjusted_means[
                "condition"
            ]
        ],
        alpha=0.82,
        edgecolor="white",
        yerr=np.vstack([
            lower_errors,
            upper_errors,
        ]),
        capsize=6,
    )

    axis.set_xticks(
        x_positions
    )

    axis.set_xticklabels(
        adjusted_means[
            "condition"
        ]
    )

    axis.set_ylabel(
        "Confiance moyenne ajustée"
    )

    axis.set_title(
        "Confiance ajustée selon la condition",
        fontsize=15,
        fontweight="bold",
    )

    axis.set_ylim(
        0,
        100,
    )

    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    for bar, value in zip(
        bars,
        y_values,
    ):
        axis.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height()
            + 1.0,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    figure.tight_layout()

    save_figure(
        figure,
        CONDITION_EFFECT_FILE,
    )


def plot_sequence_effect(
    sequence_predictions,
):
    """
    Trace la confiance prédite selon l'ordre de l'essai.
    """
    figure, axis = plt.subplots(
        figsize=(11, 7)
    )

    colors = {
        "Neutral": "#3b82f6",
        "Standard": "#f97316",
    }

    for condition in [
        "Neutral",
        "Standard",
    ]:
        condition_data = (
            sequence_predictions.loc[
                sequence_predictions[
                    "condition"
                ] == condition
            ]
        )

        axis.plot(
            condition_data[
                "sequence"
            ],
            condition_data[
                "predicted_confidence"
            ],
            color=colors[
                condition
            ],
            linewidth=2.5,
            label=condition,
        )

    axis.set_xlabel(
        "Position de l'essai"
    )

    axis.set_ylabel(
        "Confiance prédite"
    )

    axis.set_title(
        "Évolution prédite de la confiance au cours de l'expérience",
        fontsize=15,
        fontweight="bold",
    )

    axis.set_xlim(
        1,
        64,
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
        title="Condition"
    )

    figure.tight_layout()

    save_figure(
        figure,
        SEQUENCE_EFFECT_FILE,
    )


def plot_residuals_vs_fitted(
    predictions,
):
    """
    Trace les résidus fixes selon les prédictions populationnelles.
    """
    figure, axis = plt.subplots(
        figsize=(10, 7)
    )

    axis.scatter(
        predictions[
            "fixed_prediction"
        ],
        predictions[
            "fixed_residual"
        ],
        color="#2563eb",
        alpha=0.22,
        s=18,
        edgecolors="none",
    )

    axis.axhline(
        0,
        color="#111827",
        linestyle="--",
        linewidth=1.5,
    )

    axis.set_xlabel(
        "Confiance prédite par les effets fixes"
    )

    axis.set_ylabel(
        "Résidu : observé − prédit"
    )

    axis.set_title(
        "Modèle de contrôle : résidus et prédictions fixes",
        fontsize=14,
        fontweight="bold",
    )

    axis.grid(
        True,
        linestyle="--",
        alpha=0.35,
    )

    figure.tight_layout()

    save_figure(
        figure,
        RESIDUALS_VS_FITTED_FILE,
    )


def plot_residual_distribution(
    predictions,
):
    """
    Trace la distribution des résidus fixes.
    """
    residuals = (
        predictions[
            "fixed_residual"
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    figure, axis = plt.subplots(
        figsize=(10, 7)
    )

    sns.histplot(
        residuals,
        bins=45,
        kde=True,
        color="#8b5cf6",
        alpha=0.65,
        ax=axis,
    )

    axis.axvline(
        0,
        color="#111827",
        linestyle="--",
        linewidth=1.5,
    )

    axis.set_xlabel(
        "Résidu"
    )

    axis.set_ylabel(
        "Nombre d'observations"
    )

    axis.set_title(
        "Distribution des résidus du modèle de contrôle",
        fontsize=14,
        fontweight="bold",
    )

    figure.tight_layout()

    save_figure(
        figure,
        RESIDUAL_DISTRIBUTION_FILE,
    )


def plot_qqplot(
    predictions,
):
    """
    Trace le diagramme Q-Q des résidus fixes.
    """
    residuals = (
        predictions[
            "fixed_residual"
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
        .to_numpy(
            dtype=float
        )
    )

    figure, axis = plt.subplots(
        figsize=(8, 8)
    )

    stats.probplot(
        residuals,
        dist="norm",
        plot=axis,
    )

    axis.set_title(
        "Diagramme Q-Q des résidus du modèle de contrôle",
        fontsize=14,
        fontweight="bold",
    )

    axis.grid(
        True,
        linestyle="--",
        alpha=0.3,
    )

    figure.tight_layout()

    save_figure(
        figure,
        QQPLOT_FILE,
    )


def plot_variance_comparison(
    variance_comparison,
):
    """
    Compare graphiquement les variances du modèle nul et du modèle
    de contrôle.
    """
    plot_data = variance_comparison.copy()

    x_positions = np.arange(
        len(plot_data)
    )

    width = 0.36

    figure, axis = plt.subplots(
        figsize=(11, 7)
    )

    axis.bar(
        x_positions - width / 2,
        plot_data[
            "null_variance"
        ],
        width,
        label="Modèle nul",
        color="#94a3b8",
        alpha=0.82,
    )

    axis.bar(
        x_positions + width / 2,
        plot_data[
            "control_variance"
        ],
        width,
        label="Modèle de contrôle",
        color="#2563eb",
        alpha=0.82,
    )

    axis.set_xticks(
        x_positions
    )

    axis.set_xticklabels(
        plot_data[
            "component"
        ]
    )

    axis.set_ylabel(
        "Variance estimée"
    )

    axis.set_title(
        "Composantes de variance avant et après les contrôles",
        fontsize=15,
        fontweight="bold",
    )

    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    axis.legend()

    figure.tight_layout()

    save_figure(
        figure,
        VARIANCE_COMPARISON_PLOT_FILE,
    )


# ======================================================================
# INTERPRÉTATION
# ======================================================================

def print_fixed_effect_interpretation(
    fixed_effects,
):
    """
    Affiche une interprétation des effets condition et séquence.
    """
    report_section(
        "INTERPRÉTATION DES EFFETS FIXES"
    )

    condition_row = fixed_effects.loc[
        fixed_effects[
            "parameter"
        ]
        .str.lower()
        .str.contains(
            r"t\.standard",
            regex=True,
        )
    ]

    sequence_row = fixed_effects.loc[
        fixed_effects[
            "parameter"
        ] == "sequence_c10"
    ]

    if not condition_row.empty:
        condition_row = (
            condition_row.iloc[0]
        )

        estimate = float(
            condition_row[
                "estimate"
            ]
        )

        report_print(
            "Effet Standard − Neutral :",
            f"{estimate:.4f} point(s) de confiance.",
        )

        report_print(
            "Intervalle de confiance à 95 % :",
            (
                f"[{condition_row['ci_95_lower']:.4f}, "
                f"{condition_row['ci_95_upper']:.4f}]"
            ),
        )

        report_print(
            "Valeur p :",
            condition_row[
                "p_value"
            ],
        )

        if estimate > 0:
            report_print(
                "À la séquence moyenne, les participants Standard "
                "ont une confiance estimée plus élevée que les "
                "participants Neutral."
            )

        elif estimate < 0:
            report_print(
                "À la séquence moyenne, les participants Standard "
                "ont une confiance estimée plus faible que les "
                "participants Neutral."
            )

        else:
            report_print(
                "Aucune différence moyenne estimée entre Standard "
                "et Neutral."
            )

    if not sequence_row.empty:
        sequence_row = (
            sequence_row.iloc[0]
        )

        estimate = float(
            sequence_row[
                "estimate"
            ]
        )

        report_print("")

        report_print(
            "Effet de dix essais supplémentaires :",
            f"{estimate:.4f} point(s) de confiance.",
        )

        report_print(
            "Intervalle de confiance à 95 % :",
            (
                f"[{sequence_row['ci_95_lower']:.4f}, "
                f"{sequence_row['ci_95_upper']:.4f}]"
            ),
        )

        report_print(
            "Valeur p :",
            sequence_row[
                "p_value"
            ],
        )

        report_print(
            "Effet moyen par essai :",
            f"{estimate / SEQUENCE_SCALE:.6f} point(s).",
        )

        report_print(
            "Variation prédite du premier au dernier essai :",
            (
                f"{estimate * 6.3:.4f} point(s), "
                "car les essais 1 à 64 couvrent 6,3 unités "
                "de dix essais."
            ),
        )


def print_model_comparison(
    comparison_statistics,
):
    """
    Affiche l'interprétation de la comparaison ML.
    """
    report_section(
        "COMPARAISON ML AVEC LE MODÈLE NUL"
    )

    report_print(
        "Rapport de vraisemblance :",
        round(
            comparison_statistics[
                "likelihood_ratio"
            ],
            6,
        ),
    )

    report_print(
        "Différence de degrés de liberté :",
        comparison_statistics[
            "degrees_of_freedom"
        ],
    )

    report_print(
        "Valeur p du test de rapport de vraisemblance :",
        comparison_statistics[
            "p_value"
        ],
    )

    report_print(
        "ΔAIC = AIC contrôle − AIC nul :",
        round(
            comparison_statistics[
                "delta_aic_control_minus_null"
            ],
            6,
        ),
    )

    report_print(
        "ΔBIC = BIC contrôle − BIC nul :",
        round(
            comparison_statistics[
                "delta_bic_control_minus_null"
            ],
            6,
        ),
    )

    if (
        comparison_statistics[
            "delta_aic_control_minus_null"
        ] < 0
    ):
        report_print(
            "Selon l'AIC, le modèle de contrôle est préféré "
            "au modèle nul."
        )

    else:
        report_print(
            "Selon l'AIC, le modèle nul est préféré ou équivalent."
        )

    if (
        comparison_statistics[
            "delta_bic_control_minus_null"
        ] < 0
    ):
        report_print(
            "Selon le BIC, le modèle de contrôle est préféré "
            "au modèle nul."
        )

    else:
        report_print(
            "Selon le BIC, le modèle nul est préféré ou équivalent."
        )


# ======================================================================
# PROGRAMME PRINCIPAL
# ======================================================================

def main():
    print("=" * 80)
    print("MODÈLE MIXTE DE CONTRÔLE — EXPÉRIENCE E1")
    print("=" * 80)

    report_print(
        "Version Python :",
        sys.version.split()[0],
    )

    report_print(
        "Version pandas :",
        pd.__version__,
    )

    report_print(
        "Version NumPy :",
        np.__version__,
    )

    report_print(
        "Version SciPy :",
        scipy.__version__,
    )

    report_print(
        "Version statsmodels :",
        statsmodels.__version__,
    )

    try:
        # ==============================================================
        # 1. Chargement
        # ==============================================================

        (
            dataframe,
            sequence_mean,
        ) = load_analysis_data()

        # ==============================================================
        # 2. Modèle nul ML
        # ==============================================================

        null_model_ml = build_mixed_model(
            dataframe=dataframe,
            formula=NULL_FORMULA,
            model_label="MODÈLE NUL ML",
        )

        null_result_ml = fit_with_fallback(
            model=null_model_ml,
            reml=False,
            label="MODÈLE NUL ML",
        )

        save_model_summary(
            result=null_result_ml,
            output_file=
                NULL_ML_SUMMARY_FILE,
            title=(
                "MODÈLE MIXTE NUL E1 — ML"
            ),
        )

        # ==============================================================
        # 3. Modèle de contrôle ML
        # ==============================================================

        control_model_ml = build_mixed_model(
            dataframe=dataframe,
            formula=CONTROL_FORMULA,
            model_label="MODÈLE DE CONTRÔLE ML",
        )

        control_result_ml = fit_with_fallback(
            model=control_model_ml,
            reml=False,
            label="MODÈLE DE CONTRÔLE ML",
        )

        save_model_summary(
            result=control_result_ml,
            output_file=
                CONTROL_ML_SUMMARY_FILE,
            title=(
                "MODÈLE MIXTE DE CONTRÔLE E1 — ML"
            ),
        )

        # ==============================================================
        # 4. Modèle de contrôle REML
        # ==============================================================

        control_model_reml = build_mixed_model(
            dataframe=dataframe,
            formula=CONTROL_FORMULA,
            model_label="MODÈLE DE CONTRÔLE REML",
        )

        control_result_reml = (
            fit_with_fallback(
                model=control_model_reml,
                reml=True,
                label="MODÈLE DE CONTRÔLE REML",
            )
        )

        save_model_summary(
            result=control_result_reml,
            output_file=
                CONTROL_REML_SUMMARY_FILE,
            title=(
                "MODÈLE MIXTE DE CONTRÔLE E1 — REML"
            ),
        )

        # ==============================================================
        # 5. Résumé principal
        # ==============================================================

        report_section(
            "RÉSUMÉ DU MODÈLE DE CONTRÔLE REML"
        )

        report_print(
            control_result_reml
            .summary()
            .as_text()
        )

        # ==============================================================
        # 6. Effets fixes
        # ==============================================================

        fixed_effects = (
            create_fixed_effects_table(
                control_result_reml
            )
        )

        fixed_effects.to_csv(
            FIXED_EFFECTS_FILE,
            index=False,
        )

        print_fixed_effect_interpretation(
            fixed_effects
        )

        # ==============================================================
        # 7. Variances
        # ==============================================================

        null_variances = (
            extract_variances(
                null_result_ml
            )
        )

        control_variances = (
            extract_variances(
                control_result_reml
            )
        )

        control_variance_table = (
            create_variance_table(
                control_variances
            )
        )

        control_variance_table.to_csv(
            VARIANCE_COMPONENTS_FILE,
            index=False,
        )

        variance_comparison = (
            create_variance_comparison(
                null_variances=
                    null_variances,
                control_variances=
                    control_variances,
            )
        )

        variance_comparison.to_csv(
            VARIANCE_COMPARISON_FILE,
            index=False,
        )

        # ==============================================================
        # 8. Comparaison ML
        # ==============================================================

        (
            model_comparison,
            comparison_statistics,
        ) = compare_ml_models(
            null_result=null_result_ml,
            control_result=
                control_result_ml,
        )

        model_comparison.to_csv(
            MODEL_COMPARISON_FILE,
            index=False,
        )

        print_model_comparison(
            comparison_statistics
        )

        # ==============================================================
        # 9. R²
        # ==============================================================

        control_r2 = compute_mixed_model_r2(
            dataframe=dataframe,
            result=control_result_reml,
            variances=control_variances,
        )

        report_section(
            "R² DU MODÈLE DE CONTRÔLE"
        )

        report_print(
            "Variance des effets fixes :",
            round(
                control_r2[
                    "fixed_effect_variance"
                ],
                6,
            ),
        )

        report_print(
            "R² marginal — effets fixes seulement :",
            round(
                control_r2[
                    "marginal_r2"
                ],
                6,
            ),
        )

        report_print(
            "R² conditionnel — effets fixes et aléatoires :",
            round(
                control_r2[
                    "conditional_r2"
                ],
                6,
            ),
        )

        # ==============================================================
        # 10. Statistiques d'ajustement
        # ==============================================================

        fit_statistics = pd.DataFrame([
            {
                "model":
                    "Null ML",

                "formula":
                    NULL_FORMULA,

                "estimation":
                    "ML",

                "converged":
                    bool(
                        null_result_ml.converged
                    ),

                "log_likelihood":
                    float(
                        null_result_ml.llf
                    ),

                "aic":
                    float(
                        null_result_ml.aic
                    ),

                "bic":
                    float(
                        null_result_ml.bic
                    ),

                "residual_variance":
                    float(
                        null_result_ml.scale
                    ),
            },
            {
                "model":
                    "Control ML",

                "formula":
                    CONTROL_FORMULA,

                "estimation":
                    "ML",

                "converged":
                    bool(
                        control_result_ml.converged
                    ),

                "log_likelihood":
                    float(
                        control_result_ml.llf
                    ),

                "aic":
                    float(
                        control_result_ml.aic
                    ),

                "bic":
                    float(
                        control_result_ml.bic
                    ),

                "residual_variance":
                    float(
                        control_result_ml.scale
                    ),
            },
            {
                "model":
                    "Control REML",

                "formula":
                    CONTROL_FORMULA,

                "estimation":
                    "REML",

                "converged":
                    bool(
                        control_result_reml.converged
                    ),

                "log_likelihood":
                    float(
                        control_result_reml.llf
                    ),

                "aic":
                    safe_float(
                        control_result_reml.aic
                    ),

                "bic":
                    safe_float(
                        control_result_reml.bic
                    ),

                "residual_variance":
                    float(
                        control_result_reml.scale
                    ),
            },
        ])

        fit_statistics.to_csv(
            FIT_STATISTICS_FILE,
            index=False,
        )

        # ==============================================================
        # 11. Prédictions
        # ==============================================================

        adjusted_condition_means = (
            create_condition_adjusted_means(
                control_result_reml
            )
        )

        adjusted_condition_means.to_csv(
            CONDITION_MEANS_FILE,
            index=False,
        )

        sequence_predictions = (
            create_sequence_predictions(
                result=control_result_reml,
                sequence_mean=
                    sequence_mean,
            )
        )

        sequence_predictions.to_csv(
            SEQUENCE_PREDICTIONS_FILE,
            index=False,
        )

        observation_predictions = (
            create_observation_predictions(
                dataframe=dataframe,
                result=control_result_reml,
            )
        )

        # ==============================================================
        # 12. Graphiques
        # ==============================================================

        plot_condition_effect(
            adjusted_condition_means
        )

        plot_sequence_effect(
            sequence_predictions
        )

        plot_residuals_vs_fitted(
            observation_predictions
        )

        plot_residual_distribution(
            observation_predictions
        )

        plot_qqplot(
            observation_predictions
        )

        plot_variance_comparison(
            variance_comparison
        )

        # ==============================================================
        # 13. JSON
        # ==============================================================

        condition_parameter = (
            get_fixed_parameter(
                control_result_reml,
                "[t.standard]",
            )
        )

        sequence_parameter = (
            get_fixed_parameter(
                control_result_reml,
                "sequence_c10",
            )
        )

        results_json = {
            "input_file":
                INPUT_FILE,

            "n_observations":
                int(
                    len(dataframe)
                ),

            "n_subjects":
                int(
                    dataframe[
                        "subject_id"
                    ].nunique()
                ),

            "n_items":
                int(
                    dataframe[
                        "item_id"
                    ].nunique()
                ),

            "reference_condition":
                REFERENCE_CONDITION,

            "sequence_mean":
                sequence_mean,

            "sequence_scale":
                SEQUENCE_SCALE,

            "null_model": {
                "formula":
                    NULL_FORMULA,

                "ml_log_likelihood":
                    safe_float(
                        null_result_ml.llf
                    ),

                "ml_aic":
                    safe_float(
                        null_result_ml.aic
                    ),

                "ml_bic":
                    safe_float(
                        null_result_ml.bic
                    ),

                **{
                    key: safe_float(value)
                    for key, value
                    in null_variances.items()
                },
            },

            "control_model": {
                "formula":
                    CONTROL_FORMULA,

                "ml_log_likelihood":
                    safe_float(
                        control_result_ml.llf
                    ),

                "ml_aic":
                    safe_float(
                        control_result_ml.aic
                    ),

                "ml_bic":
                    safe_float(
                        control_result_ml.bic
                    ),

                "reml_log_likelihood":
                    safe_float(
                        control_result_reml.llf
                    ),

                "intercept":
                    safe_float(
                        control_result_reml
                        .fe_params[
                            "Intercept"
                        ]
                    ),

                "standard_vs_neutral":
                    safe_float(
                        control_result_reml
                        .fe_params[
                            condition_parameter
                        ]
                    ),

                "sequence_effect_per_10_trials":
                    safe_float(
                        control_result_reml
                        .fe_params[
                            sequence_parameter
                        ]
                    ),

                "sequence_effect_per_trial":
                    safe_float(
                        control_result_reml
                        .fe_params[
                            sequence_parameter
                        ]
                        / SEQUENCE_SCALE
                    ),

                **{
                    key: safe_float(value)
                    for key, value
                    in control_variances.items()
                },

                **{
                    key: safe_float(value)
                    for key, value
                    in control_r2.items()
                },
            },

            "comparison": {
                key: safe_float(value)
                for key, value
                in comparison_statistics.items()
            },
        }

        with open(
            JSON_RESULTS_FILE,
            "w",
            encoding="utf-8",
        ) as output_file:
            json.dump(
                results_json,
                output_file,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )

        # ==============================================================
        # 14. Fichiers produits
        # ==============================================================

        report_section(
            "FICHIERS PRODUITS"
        )

        output_files = [
            NULL_ML_SUMMARY_FILE,
            CONTROL_ML_SUMMARY_FILE,
            CONTROL_REML_SUMMARY_FILE,
            MODEL_COMPARISON_FILE,
            FIXED_EFFECTS_FILE,
            VARIANCE_COMPONENTS_FILE,
            VARIANCE_COMPARISON_FILE,
            FIT_STATISTICS_FILE,
            PREDICTIONS_FILE,
            CONDITION_MEANS_FILE,
            SEQUENCE_PREDICTIONS_FILE,
            RESIDUALS_VS_FITTED_FILE,
            RESIDUAL_DISTRIBUTION_FILE,
            QQPLOT_FILE,
            CONDITION_EFFECT_FILE,
            SEQUENCE_EFFECT_FILE,
            VARIANCE_COMPARISON_PLOT_FILE,
            JSON_RESULTS_FILE,
        ]

        for output_file in output_files:
            if os.path.isfile(
                output_file
            ):
                report_print(
                    output_file
                )

        save_report()

        print("")
        print("=" * 80)
        print("MODÈLE DE CONTRÔLE TERMINÉ")
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
        report_section(
            "ERREUR"
        )

        report_print(
            type(error).__name__,
            ":",
            error,
        )

        save_report()

        raise


if __name__ == "__main__":
    main()
