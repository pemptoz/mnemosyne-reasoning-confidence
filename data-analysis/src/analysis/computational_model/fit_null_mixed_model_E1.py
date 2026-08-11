"""
fit_null_mixed_model_E1.py

Ajuste le modèle linéaire mixte nul pour la confiance dans E1.

Entrée :
    dataset_analysis_E1.csv

Structure du modèle :

    confidence_ij = beta_0 + u_i + v_j + epsilon_ij

avec :

    beta_0 :
        moyenne générale de confiance ;

    u_i :
        interception aléatoire du participant i ;

    v_j :
        interception aléatoire de l'item j ;

    epsilon_ij :
        variation résiduelle au niveau de l'essai.

Les effets aléatoires participant et item sont croisés :

    - chaque participant répond à plusieurs items ;
    - chaque item est présenté à plusieurs participants.

Deux ajustements peuvent être réalisés :

    1. REML :
        estimation principale des composantes de variance ;

    2. ML :
        modèle de référence pour les futures comparaisons de modèles
        contenant des effets fixes différents.

Fichiers produits dans null_mixed_model_E1/ :

    null_model_REML_summary.txt
    null_model_ML_summary.txt
    null_model_variance_components.csv
    null_model_fit_statistics.csv
    null_model_fixed_effects.csv
    null_model_predictions.csv
    null_model_subject_effects.csv
    null_model_item_effects.csv

    null_model_residuals_vs_fitted.png
    null_model_residual_distribution.png
    null_model_qqplot.png
    null_model_variance_decomposition.png
    null_model_subject_effects.png
    null_model_item_effects.png

    null_model_report.txt
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
    "null_mixed_model_E1",
)

os.makedirs(
    OUTPUT_DIRECTORY,
    exist_ok=True,
)


# ----------------------------------------------------------------------
# Fichiers texte
# ----------------------------------------------------------------------

REML_SUMMARY_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "null_model_REML_summary.txt",
)

ML_SUMMARY_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "null_model_ML_summary.txt",
)

REPORT_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "null_model_report.txt",
)

JSON_RESULTS_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "null_model_results.json",
)


# ----------------------------------------------------------------------
# Fichiers CSV
# ----------------------------------------------------------------------

VARIANCE_COMPONENTS_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "null_model_variance_components.csv",
)

FIT_STATISTICS_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "null_model_fit_statistics.csv",
)

FIXED_EFFECTS_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "null_model_fixed_effects.csv",
)

PREDICTIONS_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "null_model_predictions.csv",
)

SUBJECT_EFFECTS_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "null_model_subject_effects.csv",
)

ITEM_EFFECTS_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "null_model_item_effects.csv",
)


# ----------------------------------------------------------------------
# Fichiers graphiques
# ----------------------------------------------------------------------

RESIDUALS_VS_FITTED_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "null_model_residuals_vs_fitted.png",
)

RESIDUAL_DISTRIBUTION_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "null_model_residual_distribution.png",
)

QQPLOT_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "null_model_qqplot.png",
)

VARIANCE_DECOMPOSITION_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "null_model_variance_decomposition.png",
)

SUBJECT_EFFECTS_PLOT_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "null_model_subject_effects.png",
)

ITEM_EFFECTS_PLOT_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "null_model_item_effects.png",
)


# ----------------------------------------------------------------------
# Options d'estimation
# ----------------------------------------------------------------------

# REML est utilisé pour l'estimation principale des composantes de
# variance du modèle nul.
FIT_REML_MODEL = True

# Le modèle ML servira de référence pour comparer ultérieurement des
# modèles ayant des effets fixes différents.
FIT_ML_MODEL = True

# Utilise des matrices creuses pour les variables indicatrices des
# participants et des items.
USE_SPARSE_MATRICES = True

# Nombre maximal d'itérations de l'optimiseur.
MAX_ITERATIONS = 2000

# Affichage des étapes d'optimisation de statsmodels.
OPTIMIZER_DISPLAY = False

# Méthodes essayées successivement si la première ne converge pas.
OPTIMIZATION_METHODS = [
    "lbfgs",
    "bfgs",
    "cg",
    "powell",
]

# Résolution des figures.
DPI = 300

sns.set_theme(
    style="whitegrid",
    context="notebook",
)


# ======================================================================
# RAPPORT
# ======================================================================

REPORT_LINES = []


def report_print(*values):
    """
    Affiche une ligne dans le terminal et l'ajoute au rapport.
    """
    text = " ".join(
        str(value)
        for value in values
    )

    print(text)
    REPORT_LINES.append(text)


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
    Enregistre le rapport principal.
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
# OUTILS
# ======================================================================

def normalize_subject_id(value):
    """
    Normalise l'identifiant participant.
    """
    if pd.isna(value):
        return pd.NA

    normalized = str(value).strip()

    if not normalized:
        return pd.NA

    try:
        numeric = float(normalized)

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


def safe_float(value):
    """
    Convertit une valeur numérique en float JSON-compatible.
    """
    try:
        numeric = float(value)

    except (
        TypeError,
        ValueError,
    ):
        return None

    if not np.isfinite(numeric):
        return None

    return numeric


def save_figure(
    figure,
    output_file,
):
    """
    Enregistre une figure.
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


def standard_deviation_from_variance(variance):
    """
    Calcule l'écart-type correspondant à une variance.
    """
    if (
        pd.isna(variance)
        or variance < 0
    ):
        return np.nan

    return float(
        np.sqrt(variance)
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

    report_print(
        "Nombre de colonnes :",
        len(dataframe.columns),
    )

    required_columns = {
        "subject_id",
        "item_id",
        "confidence",
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
        .apply(normalize_subject_id)
        .astype("string")
    )

    dataframe["item_id"] = (
        dataframe["item_id"]
        .apply(normalize_subject_id)
        .astype("string")
    )

    dataframe["confidence"] = pd.to_numeric(
        dataframe["confidence"],
        errors="coerce",
    )

    if "analysis_complete" in dataframe.columns:
        complete_value = (
            dataframe[
                "analysis_complete"
            ]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        dataframe["analysis_complete"] = (
            complete_value.isin([
                "true",
                "1",
                "1.0",
                "yes",
            ])
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

    before_drop = len(
        dataframe
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
                "item_id",
                "confidence",
            ]
        )
        .copy()
    )

    report_print(
        "Lignes supprimées pour donnée essentielle manquante :",
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
            "Certaines valeurs de confiance sont hors "
            "de l'intervalle [0, 100]."
        )

    if len(dataframe) < 2:
        raise ValueError(
            "Le fichier ne contient pas suffisamment "
            "d'observations."
        )

    if (
        dataframe[
            "subject_id"
        ].nunique()
        < 2
    ):
        raise ValueError(
            "Le modèle requiert au moins deux participants."
        )

    if (
        dataframe[
            "item_id"
        ].nunique()
        < 2
    ):
        raise ValueError(
            "Le modèle requiert au moins deux items."
        )

    # Statsmodels requiert un argument groups. Pour représenter des
    # effets croisés, toutes les observations sont placées dans un
    # groupe artificiel unique, et les participants/items sont définis
    # comme composantes de variance.
    dataframe["_global_group"] = "all_observations"

    dataframe = (
        dataframe
        .sort_values(
            by=[
                "subject_id",
                "item_id",
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
        "Moyenne de confiance :",
        round(
            dataframe[
                "confidence"
            ].mean(),
            6,
        ),
    )

    report_print(
        "Médiane de confiance :",
        round(
            dataframe[
                "confidence"
            ].median(),
            6,
        ),
    )

    report_print(
        "Écart-type de confiance :",
        round(
            dataframe[
                "confidence"
            ].std(),
            6,
        ),
    )

    return dataframe


# ======================================================================
# CONSTRUCTION DU MODÈLE
# ======================================================================

def build_null_model(dataframe):
    """
    Construit le modèle nul croisé.

    Les effets aléatoires participant et item sont définis comme
    composantes de variance indépendantes.
    """
    report_section(
        "CONSTRUCTION DU MODÈLE NUL"
    )

    variance_component_formulas = {
        "item": (
            "0 + C(item_id)"
        ),
        "subject": (
            "0 + C(subject_id)"
        ),
    }

    report_print(
        "Formule fixe : confidence ~ 1"
    )

    report_print(
        "Composante aléatoire participant :",
        variance_component_formulas[
            "subject"
        ],
    )

    report_print(
        "Composante aléatoire item :",
        variance_component_formulas[
            "item"
        ],
    )

    model = smf.mixedlm(
        formula="confidence ~ 1",
        data=dataframe,
        groups=dataframe[
            "_global_group"
        ],
        re_formula="0",
        vc_formula=
            variance_component_formulas,
        use_sparse=USE_SPARSE_MATRICES,
    )

    return model


# ======================================================================
# AJUSTEMENT DU MODÈLE
# ======================================================================

def fit_with_fallback(
    model,
    reml,
    model_label,
):
    """
    Essaie plusieurs optimiseurs jusqu'à obtenir un ajustement
    exploitable.
    """
    report_section(
        f"AJUSTEMENT DU MODÈLE — {model_label}"
    )

    last_error = None
    last_result = None

    for method in OPTIMIZATION_METHODS:
        report_print(
            "Tentative avec l'optimiseur :",
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

            report_print(
                "Le modèle n'a pas convergé avec",
                method,
            )

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
            "ATTENTION : aucun optimiseur n'a signalé "
            "une convergence complète."
        )

        report_print(
            "Le dernier résultat est retourné pour diagnostic."
        )

        return last_result

    raise RuntimeError(
        "Impossible d'ajuster le modèle. "
        f"Dernière erreur : {last_error!r}"
    )


# ======================================================================
# EXTRACTION DES COMPOSANTES DE VARIANCE
# ======================================================================

def get_variance_component_names(
    result,
):
    """
    Récupère l'ordre des composantes de variance utilisé par
    statsmodels.
    """
    try:
        names = list(
            result.model.exog_vc.names
        )

    except (
        AttributeError,
        TypeError,
    ):
        names = []

    if not names:
        names = [
            f"variance_component_{index}"
            for index in range(
                len(result.vcomp)
            )
        ]

    return names


def extract_variance_components(result):
    """
    Extrait les variances participant, item et résiduelle.

    Calcule également :

        ICC participant
        ICC item
        ICC total

    où :

        ICC participant =
            variance participant / variance totale

        ICC item =
            variance item / variance totale
    """
    component_names = (
        get_variance_component_names(
            result
        )
    )

    component_values = np.asarray(
        result.vcomp,
        dtype=float,
    )

    component_map = {
        str(name): float(value)
        for name, value in zip(
            component_names,
            component_values,
        )
    }

    subject_variance = np.nan
    item_variance = np.nan

    for name, value in component_map.items():
        normalized_name = (
            str(name)
            .strip()
            .lower()
        )

        if "subject" in normalized_name:
            subject_variance = value

        elif "item" in normalized_name:
            item_variance = value

    if (
        pd.isna(subject_variance)
        or pd.isna(item_variance)
    ):
        report_print(
            "Ordre des composantes retourné par statsmodels :",
            component_map,
        )

        raise RuntimeError(
            "Impossible d'identifier automatiquement les variances "
            "participant et item."
        )

    residual_variance = float(
        result.scale
    )

    total_variance = (
        subject_variance
        + item_variance
        + residual_variance
    )

    if total_variance <= 0:
        raise RuntimeError(
            "La variance totale estimée n'est pas positive."
        )

    subject_icc = (
        subject_variance
        / total_variance
    )

    item_icc = (
        item_variance
        / total_variance
    )

    residual_proportion = (
        residual_variance
        / total_variance
    )

    total_cluster_icc = (
        subject_variance
        + item_variance
    ) / total_variance

    variance_table = pd.DataFrame([
        {
            "component":
                "Participant",

            "variance":
                subject_variance,

            "standard_deviation":
                standard_deviation_from_variance(
                    subject_variance
                ),

            "proportion_total_variance":
                subject_icc,
        },
        {
            "component":
                "Item",

            "variance":
                item_variance,

            "standard_deviation":
                standard_deviation_from_variance(
                    item_variance
                ),

            "proportion_total_variance":
                item_icc,
        },
        {
            "component":
                "Residual",

            "variance":
                residual_variance,

            "standard_deviation":
                standard_deviation_from_variance(
                    residual_variance
                ),

            "proportion_total_variance":
                residual_proportion,
        },
        {
            "component":
                "Total",

            "variance":
                total_variance,

            "standard_deviation":
                standard_deviation_from_variance(
                    total_variance
                ),

            "proportion_total_variance":
                1.0,
        },
    ])

    statistics = {
        "subject_variance":
            subject_variance,

        "item_variance":
            item_variance,

        "residual_variance":
            residual_variance,

        "total_variance":
            total_variance,

        "subject_icc":
            subject_icc,

        "item_icc":
            item_icc,

        "residual_proportion":
            residual_proportion,

        "total_cluster_icc":
            total_cluster_icc,
    }

    return (
        variance_table,
        statistics,
    )


# ======================================================================
# EFFETS ESTIMÉS PAR PARTICIPANT ET ITEM
# ======================================================================

def estimate_group_intercepts(
    dataframe,
    grand_mean,
    variance_component,
    residual_variance,
    group_column,
    output_column,
):
    """
    Produit une estimation descriptive rétrécie de l'interception
    propre à chaque participant ou item.

    Pour un groupe g :

        effet brut =
            moyenne du groupe - moyenne générale

        poids de rétrécissement =
            variance_g /
            (
                variance_g
                + variance résiduelle / n_g
            )

        effet rétréci =
            poids × effet brut

    Ces valeurs sont utilisées uniquement pour les graphiques et la
    description. Les composantes de variance viennent du modèle mixte.
    """
    group_summary = (
        dataframe
        .groupby(
            group_column,
            as_index=False,
        )
        .agg(
            number_of_observations=(
                "confidence",
                "size",
            ),
            observed_mean_confidence=(
                "confidence",
                "mean",
            ),
        )
    )

    group_summary[
        "raw_effect"
    ] = (
        group_summary[
            "observed_mean_confidence"
        ]
        - grand_mean
    )

    n_values = group_summary[
        "number_of_observations"
    ].astype(float)

    denominator = (
        variance_component
        + residual_variance
        / n_values
    )

    if variance_component <= 0:
        shrinkage_weight = np.zeros(
            len(group_summary),
            dtype=float,
        )

    else:
        shrinkage_weight = (
            variance_component
            / denominator
        )

    group_summary[
        "shrinkage_weight"
    ] = shrinkage_weight

    group_summary[
        output_column
    ] = (
        group_summary[
            "raw_effect"
        ]
        * group_summary[
            "shrinkage_weight"
        ]
    )

    group_summary[
        "predicted_group_mean"
    ] = (
        grand_mean
        + group_summary[
            output_column
        ]
    )

    return group_summary


# ======================================================================
# PRÉDICTIONS ET RÉSIDUS
# ======================================================================

def create_predictions(
    dataframe,
    result,
    variance_statistics,
):
    """
    Construit les valeurs prédites et les résidus conditionnels.

    Comme les deux effets sont croisés et définis via vc_formula,
    le script calcule des estimations rétrécies par participant et
    item, puis les additionne à l'interception générale.
    """
    report_section(
        "PRÉDICTIONS DU MODÈLE NUL"
    )

    intercept = float(
        result.fe_params[
            "Intercept"
        ]
    )

    subject_effects = estimate_group_intercepts(
        dataframe=dataframe,
        grand_mean=intercept,
        variance_component=
            variance_statistics[
                "subject_variance"
            ],
        residual_variance=
            variance_statistics[
                "residual_variance"
            ],
        group_column="subject_id",
        output_column="subject_random_effect",
    )

    item_effects = estimate_group_intercepts(
        dataframe=dataframe,
        grand_mean=intercept,
        variance_component=
            variance_statistics[
                "item_variance"
            ],
        residual_variance=
            variance_statistics[
                "residual_variance"
            ],
        group_column="item_id",
        output_column="item_random_effect",
    )

    predictions = dataframe.copy()

    predictions = predictions.merge(
        subject_effects[[
            "subject_id",
            "subject_random_effect",
            "predicted_group_mean",
        ]].rename(
            columns={
                "predicted_group_mean":
                    "predicted_subject_mean",
            }
        ),
        on="subject_id",
        how="left",
        validate="many_to_one",
    )

    predictions = predictions.merge(
        item_effects[[
            "item_id",
            "item_random_effect",
            "predicted_group_mean",
        ]].rename(
            columns={
                "predicted_group_mean":
                    "predicted_item_mean",
            }
        ),
        on="item_id",
        how="left",
        validate="many_to_one",
    )

    predictions[
        "fixed_prediction"
    ] = intercept

    predictions[
        "conditional_prediction"
    ] = (
        intercept
        + predictions[
            "subject_random_effect"
        ]
        + predictions[
            "item_random_effect"
        ]
    )

    predictions[
        "marginal_residual"
    ] = (
        predictions[
            "confidence"
        ]
        - predictions[
            "fixed_prediction"
        ]
    )

    predictions[
        "conditional_residual"
    ] = (
        predictions[
            "confidence"
        ]
        - predictions[
            "conditional_prediction"
        ]
    )

    predictions[
        "squared_conditional_error"
    ] = np.square(
        predictions[
            "conditional_residual"
        ]
    )

    predictions[
        "absolute_conditional_error"
    ] = np.abs(
        predictions[
            "conditional_residual"
        ]
    )

    rmse = float(
        np.sqrt(
            predictions[
                "squared_conditional_error"
            ].mean()
        )
    )

    mae = float(
        predictions[
            "absolute_conditional_error"
        ].mean()
    )

    report_print(
        "Interception générale :",
        round(
            intercept,
            6,
        ),
    )

    report_print(
        "RMSE conditionnel descriptif :",
        round(
            rmse,
            6,
        ),
    )

    report_print(
        "MAE conditionnelle descriptive :",
        round(
            mae,
            6,
        ),
    )

    prediction_columns = [
        "subject_id",
        "sequence",
        "item_id",
        "confidence",
        "fixed_prediction",
        "subject_random_effect",
        "item_random_effect",
        "conditional_prediction",
        "marginal_residual",
        "conditional_residual",
        "squared_conditional_error",
        "absolute_conditional_error",
    ]

    available_columns = [
        column
        for column in prediction_columns
        if column in predictions.columns
    ]

    predictions[
        available_columns
    ].to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    subject_effects.to_csv(
        SUBJECT_EFFECTS_FILE,
        index=False,
    )

    item_effects.to_csv(
        ITEM_EFFECTS_FILE,
        index=False,
    )

    return (
        predictions,
        subject_effects,
        item_effects,
        rmse,
        mae,
    )


# ======================================================================
# SAUVEGARDE DES RÉSULTATS
# ======================================================================

def save_model_summary(
    result,
    output_file,
    label,
):
    """
    Sauvegarde le résumé statsmodels.
    """
    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as output:
        output.write(
            f"{label}\n"
        )

        output.write(
            "=" * 80
        )

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
    Crée le tableau des effets fixes.
    """
    fixed_parameters = result.fe_params

    standard_errors = result.bse_fe

    confidence_intervals = (
        result.conf_int()
    )

    rows = []

    for parameter_name, estimate in fixed_parameters.items():
        if parameter_name in confidence_intervals.index:
            lower_bound = confidence_intervals.loc[
                parameter_name,
                0,
            ]

            upper_bound = confidence_intervals.loc[
                parameter_name,
                1,
            ]

        else:
            lower_bound = np.nan
            upper_bound = np.nan

        rows.append({
            "parameter":
                parameter_name,

            "estimate":
                float(estimate),

            "standard_error":
                float(
                    standard_errors[
                        parameter_name
                    ]
                ),

            "ci_95_lower":
                float(lower_bound),

            "ci_95_upper":
                float(upper_bound),
        })

    return pd.DataFrame(
        rows
    )


def create_fit_statistics_table(
    reml_result,
    ml_result,
    number_of_observations,
    number_of_subjects,
    number_of_items,
):
    """
    Construit le tableau des statistiques d'ajustement.
    """
    rows = []

    for label, result in [
        (
            "REML",
            reml_result,
        ),
        (
            "ML",
            ml_result,
        ),
    ]:
        if result is None:
            continue

        rows.append({
            "estimation":
                label,

            "converged":
                bool(
                    getattr(
                        result,
                        "converged",
                        False,
                    )
                ),

            "n_observations":
                int(
                    number_of_observations
                ),

            "n_subjects":
                int(
                    number_of_subjects
                ),

            "n_items":
                int(
                    number_of_items
                ),

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

            "residual_variance":
                safe_float(
                    result.scale
                ),
        })

    return pd.DataFrame(
        rows
    )


# ======================================================================
# GRAPHIQUES
# ======================================================================

def plot_residuals_vs_fitted(
    predictions,
):
    """
    Trace les résidus conditionnels en fonction des prédictions.
    """
    figure, axis = plt.subplots(
        figsize=(10, 7)
    )

    axis.scatter(
        predictions[
            "conditional_prediction"
        ],
        predictions[
            "conditional_residual"
        ],
        s=20,
        alpha=0.25,
        color="#2563eb",
        edgecolors="none",
    )

    axis.axhline(
        0,
        color="#111827",
        linestyle="--",
        linewidth=1.5,
    )

    axis.set_title(
        "Modèle nul : résidus conditionnels et valeurs prédites",
        fontsize=14,
        fontweight="bold",
    )

    axis.set_xlabel(
        "Confiance prédite"
    )

    axis.set_ylabel(
        "Résidu : confiance observée − confiance prédite"
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
    Trace la distribution des résidus conditionnels.
    """
    residuals = (
        predictions[
            "conditional_residual"
        ]
        .dropna()
    )

    figure, axis = plt.subplots(
        figsize=(10, 7)
    )

    sns.histplot(
        residuals,
        bins=40,
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

    axis.set_title(
        "Distribution des résidus conditionnels",
        fontsize=14,
        fontweight="bold",
    )

    axis.set_xlabel(
        "Résidu"
    )

    axis.set_ylabel(
        "Nombre d'observations"
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
    Produit un diagramme quantile-quantile des résidus.
    """
    residuals = (
        predictions[
            "conditional_residual"
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
        "Diagramme Q-Q des résidus conditionnels",
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


def plot_variance_decomposition(
    variance_table,
):
    """
    Affiche la proportion de variance attribuée à chaque niveau.
    """
    plot_data = variance_table.loc[
        variance_table[
            "component"
        ] != "Total"
    ].copy()

    colors = {
        "Participant": "#2563eb",
        "Item": "#f59e0b",
        "Residual": "#6b7280",
    }

    figure, axis = plt.subplots(
        figsize=(10, 7)
    )

    bars = axis.bar(
        plot_data[
            "component"
        ],
        plot_data[
            "proportion_total_variance"
        ]
        * 100,
        color=[
            colors[
                component
            ]
            for component in plot_data[
                "component"
            ]
        ],
        alpha=0.82,
        edgecolor="white",
    )

    for bar, proportion in zip(
        bars,
        plot_data[
            "proportion_total_variance"
        ],
    ):
        axis.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height()
            + 1,
            f"{proportion * 100:.2f} %",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    axis.set_title(
        "Décomposition de la variance de la confiance",
        fontsize=15,
        fontweight="bold",
    )

    axis.set_ylabel(
        "Proportion de la variance totale (%)"
    )

    axis.set_ylim(
        0,
        max(
            100,
            (
                plot_data[
                    "proportion_total_variance"
                ].max()
                * 100
                + 10
            ),
        ),
    )

    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.35,
    )

    figure.tight_layout()

    save_figure(
        figure,
        VARIANCE_DECOMPOSITION_FILE,
    )


def plot_ordered_effects(
    effects,
    effect_column,
    label_column,
    title,
    output_file,
    color,
):
    """
    Affiche les effets estimés, triés du plus faible au plus fort.
    """
    plot_data = (
        effects[[
            label_column,
            effect_column,
        ]]
        .dropna()
        .sort_values(
            by=effect_column
        )
        .reset_index(
            drop=True
        )
    )

    plot_data[
        "rank"
    ] = np.arange(
        1,
        len(plot_data) + 1,
    )

    figure, axis = plt.subplots(
        figsize=(12, 7)
    )

    axis.scatter(
        plot_data[
            "rank"
        ],
        plot_data[
            effect_column
        ],
        s=35,
        alpha=0.72,
        color=color,
        edgecolors="white",
        linewidths=0.4,
    )

    axis.axhline(
        0,
        color="#111827",
        linestyle="--",
        linewidth=1.4,
    )

    axis.set_title(
        title,
        fontsize=14,
        fontweight="bold",
    )

    axis.set_xlabel(
        "Rang"
    )

    axis.set_ylabel(
        "Écart estimé à la moyenne générale"
    )

    axis.grid(
        True,
        linestyle="--",
        alpha=0.35,
    )

    figure.tight_layout()

    save_figure(
        figure,
        output_file,
    )


# ======================================================================
# INTERPRÉTATION
# ======================================================================

def print_variance_interpretation(
    variance_statistics,
):
    """
    Affiche une interprétation pédagogique des composantes de variance.
    """
    report_section(
        "INTERPRÉTATION DES COMPOSANTES DE VARIANCE"
    )

    subject_percentage = (
        variance_statistics[
            "subject_icc"
        ]
        * 100
    )

    item_percentage = (
        variance_statistics[
            "item_icc"
        ]
        * 100
    )

    residual_percentage = (
        variance_statistics[
            "residual_proportion"
        ]
        * 100
    )

    total_cluster_percentage = (
        variance_statistics[
            "total_cluster_icc"
        ]
        * 100
    )

    report_print(
        "Variance entre participants :",
        round(
            variance_statistics[
                "subject_variance"
            ],
            6,
        ),
    )

    report_print(
        "Part de variance attribuée aux participants :",
        f"{subject_percentage:.3f} %",
    )

    report_print(
        "Variance entre items :",
        round(
            variance_statistics[
                "item_variance"
            ],
            6,
        ),
    )

    report_print(
        "Part de variance attribuée aux items :",
        f"{item_percentage:.3f} %",
    )

    report_print(
        "Variance résiduelle :",
        round(
            variance_statistics[
                "residual_variance"
            ],
            6,
        ),
    )

    report_print(
        "Part de variance résiduelle :",
        f"{residual_percentage:.3f} %",
    )

    report_print(
        "Part totale structurée par participant ou item :",
        f"{total_cluster_percentage:.3f} %",
    )

    report_print("")

    report_print(
        "ICC participant :",
        round(
            variance_statistics[
                "subject_icc"
            ],
            6,
        ),
    )

    report_print(
        "Interprétation : deux observations provenant du même "
        "participant partagent une ressemblance attribuable au "
        "style général de confiance de cette personne."
    )

    report_print("")

    report_print(
        "ICC item :",
        round(
            variance_statistics[
                "item_icc"
            ],
            6,
        ),
    )

    report_print(
        "Interprétation : deux observations portant sur le même "
        "item partagent une ressemblance attribuable aux "
        "caractéristiques de cet item."
    )

    if subject_percentage > item_percentage:
        report_print("")

        report_print(
            "La variabilité entre participants est supérieure "
            "à la variabilité entre items."
        )

        report_print(
            "Cela suggère que les différences individuelles "
            "d'utilisation de l'échelle de confiance jouent un rôle "
            "plus important que les différences moyennes entre items."
        )

    elif item_percentage > subject_percentage:
        report_print("")

        report_print(
            "La variabilité entre items est supérieure "
            "à la variabilité entre participants."
        )

        report_print(
            "Cela suggère que les propriétés des syllogismes "
            "structurent fortement la confiance."
        )

    else:
        report_print("")

        report_print(
            "Les variabilités participant et item ont des "
            "amplitudes proches."
        )


# ======================================================================
# PROGRAMME PRINCIPAL
# ======================================================================

def main():
    print("=" * 80)
    print("MODÈLE LINÉAIRE MIXTE NUL — EXPÉRIENCE E1")
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
        # 1. Données
        # ==============================================================

        dataframe = load_analysis_data()

        # ==============================================================
        # 2. Modèle
        # ==============================================================

        null_model = build_null_model(
            dataframe
        )

        reml_result = None
        ml_result = None

        # ==============================================================
        # 3. Ajustement REML
        # ==============================================================

        if FIT_REML_MODEL:
            reml_result = fit_with_fallback(
                model=null_model,
                reml=True,
                model_label="REML",
            )

            save_model_summary(
                result=reml_result,
                output_file=
                    REML_SUMMARY_FILE,
                label=(
                    "MODÈLE MIXTE NUL E1 — REML"
                ),
            )

        # ==============================================================
        # 4. Ajustement ML
        # ==============================================================

        if FIT_ML_MODEL:
            # Reconstruit le modèle pour éviter de réutiliser un état
            # interne modifié par le premier ajustement.
            null_model_ml = build_null_model(
                dataframe
            )

            ml_result = fit_with_fallback(
                model=null_model_ml,
                reml=False,
                model_label="ML",
            )

            save_model_summary(
                result=ml_result,
                output_file=
                    ML_SUMMARY_FILE,
                label=(
                    "MODÈLE MIXTE NUL E1 — ML"
                ),
            )

        # Le résultat REML est privilégié pour la décomposition de
        # variance. Si REML n'a pas été demandé, utilise ML.
        primary_result = (
            reml_result
            if reml_result is not None
            else ml_result
        )

        if primary_result is None:
            raise RuntimeError(
                "Aucun modèle n'a été ajusté."
            )

        report_section(
            "RÉSUMÉ DU MODÈLE PRINCIPAL"
        )

        report_print(
            primary_result
            .summary()
            .as_text()
        )

        # ==============================================================
        # 5. Effet fixe
        # ==============================================================

        fixed_effects = (
            create_fixed_effects_table(
                primary_result
            )
        )

        fixed_effects.to_csv(
            FIXED_EFFECTS_FILE,
            index=False,
        )

        # ==============================================================
        # 6. Variances et ICC
        # ==============================================================

        (
            variance_table,
            variance_statistics,
        ) = extract_variance_components(
            primary_result
        )

        variance_table.to_csv(
            VARIANCE_COMPONENTS_FILE,
            index=False,
        )

        print_variance_interpretation(
            variance_statistics
        )

        # ==============================================================
        # 7. Prédictions
        # ==============================================================

        (
            predictions,
            subject_effects,
            item_effects,
            conditional_rmse,
            conditional_mae,
        ) = create_predictions(
            dataframe=dataframe,
            result=primary_result,
            variance_statistics=
                variance_statistics,
        )

        # ==============================================================
        # 8. Statistiques d'ajustement
        # ==============================================================

        fit_statistics = (
            create_fit_statistics_table(
                reml_result=reml_result,
                ml_result=ml_result,
                number_of_observations=
                    len(dataframe),
                number_of_subjects=
                    dataframe[
                        "subject_id"
                    ].nunique(),
                number_of_items=
                    dataframe[
                        "item_id"
                    ].nunique(),
            )
        )

        fit_statistics.to_csv(
            FIT_STATISTICS_FILE,
            index=False,
        )

        # ==============================================================
        # 9. Graphiques
        # ==============================================================

        plot_residuals_vs_fitted(
            predictions
        )

        plot_residual_distribution(
            predictions
        )

        plot_qqplot(
            predictions
        )

        plot_variance_decomposition(
            variance_table
        )

        plot_ordered_effects(
            effects=subject_effects,
            effect_column=
                "subject_random_effect",
            label_column="subject_id",
            title=(
                "Effets participant estimés sur la confiance"
            ),
            output_file=
                SUBJECT_EFFECTS_PLOT_FILE,
            color="#2563eb",
        )

        plot_ordered_effects(
            effects=item_effects,
            effect_column=
                "item_random_effect",
            label_column="item_id",
            title=(
                "Effets item estimés sur la confiance"
            ),
            output_file=
                ITEM_EFFECTS_PLOT_FILE,
            color="#f59e0b",
        )

        # ==============================================================
        # 10. Résultat JSON
        # ==============================================================

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

            "estimation_primary":
                (
                    "REML"
                    if reml_result
                    is not None
                    else "ML"
                ),

            "converged":
                bool(
                    getattr(
                        primary_result,
                        "converged",
                        False,
                    )
                ),

            "intercept":
                safe_float(
                    primary_result.fe_params[
                        "Intercept"
                    ]
                ),

            "subject_variance":
                safe_float(
                    variance_statistics[
                        "subject_variance"
                    ]
                ),

            "item_variance":
                safe_float(
                    variance_statistics[
                        "item_variance"
                    ]
                ),

            "residual_variance":
                safe_float(
                    variance_statistics[
                        "residual_variance"
                    ]
                ),

            "total_variance":
                safe_float(
                    variance_statistics[
                        "total_variance"
                    ]
                ),

            "subject_icc":
                safe_float(
                    variance_statistics[
                        "subject_icc"
                    ]
                ),

            "item_icc":
                safe_float(
                    variance_statistics[
                        "item_icc"
                    ]
                ),

            "residual_proportion":
                safe_float(
                    variance_statistics[
                        "residual_proportion"
                    ]
                ),

            "total_cluster_icc":
                safe_float(
                    variance_statistics[
                        "total_cluster_icc"
                    ]
                ),

            "conditional_rmse_descriptive":
                safe_float(
                    conditional_rmse
                ),

            "conditional_mae_descriptive":
                safe_float(
                    conditional_mae
                ),

            "reml_log_likelihood":
                (
                    safe_float(
                        reml_result.llf
                    )
                    if reml_result
                    is not None
                    else None
                ),

            "ml_log_likelihood":
                (
                    safe_float(
                        ml_result.llf
                    )
                    if ml_result
                    is not None
                    else None
                ),

            "ml_aic":
                (
                    safe_float(
                        ml_result.aic
                    )
                    if ml_result
                    is not None
                    else None
                ),

            "ml_bic":
                (
                    safe_float(
                        ml_result.bic
                    )
                    if ml_result
                    is not None
                    else None
                ),
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
        # 11. Résumé final
        # ==============================================================

        report_section(
            "FICHIERS PRODUITS"
        )

        output_files = [
            REML_SUMMARY_FILE,
            ML_SUMMARY_FILE,
            VARIANCE_COMPONENTS_FILE,
            FIT_STATISTICS_FILE,
            FIXED_EFFECTS_FILE,
            PREDICTIONS_FILE,
            SUBJECT_EFFECTS_FILE,
            ITEM_EFFECTS_FILE,
            RESIDUALS_VS_FITTED_FILE,
            RESIDUAL_DISTRIBUTION_FILE,
            QQPLOT_FILE,
            VARIANCE_DECOMPOSITION_FILE,
            SUBJECT_EFFECTS_PLOT_FILE,
            ITEM_EFFECTS_PLOT_FILE,
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
        print("MODÈLE NUL TERMINÉ")
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