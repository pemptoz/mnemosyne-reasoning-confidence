#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fit_gam_parsimonious_E1_n20.py

Ajuste un GAM parcimonieux pour la confiance dans l'expérience E1.

Variable dépendante
-------------------
confidence

Termes catégoriels
------------------
condition
validity_binary

Termes non linéaires
--------------------
sequence_c10
item_entropy_z

Termes linéaires
----------------
subject_accuracy_z
subject_mean_models_z
models_within_subject_z

Facteurs pénalisés
------------------
subject_code
item_code

Attention
---------
pyGAM ne fournit pas directement de véritables effets aléatoires
croisés. Les participants et les items sont donc représentés par
des facteurs catégoriels pénalisés.

Le script produit uniquement :

    gam_parsimonious_summary.txt
    gam_parsimonious_smooth_effects.png
    gam_parsimonious_smooth_effects_values.csv
"""

from contextlib import redirect_stdout
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygam
from pygam import LinearGAM, f, l, s

from prepare_gam_data_E1_n20 import (
    FEATURE_INDEX,
    build_model_matrices,
    load_data,
    prepare_data,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "analysis"
    / "GAM"
    / "gam_parsimonious_E1_n20"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------------------------------
# Fichiers produits
# --------------------------------------------------------------------------

SUMMARY_FILE = (
    OUTPUT_DIR
    / "gam_parsimonious_summary.txt"
)

SMOOTH_EFFECTS_FIGURE = (
    OUTPUT_DIR
    / "gam_parsimonious_smooth_effects.png"
)

SMOOTH_EFFECTS_VALUES_FILE = (
    OUTPUT_DIR
    / "gam_parsimonious_smooth_effects_values.csv"
)

PARAMETRIC_EFFECTS_FILE = (
    OUTPUT_DIR
    / "gam_parsimonious_parametric_effects.csv"
)


# ============================================================================
# PARAMÈTRES DU MODÈLE
# ============================================================================

# Nombre de fonctions de base pour les deux splines.
N_SPLINES = 10

# Ordre des splines :
# 3 correspond à des splines cubiques.
SPLINE_ORDER = 3

# Pénalisation des splines de sequence et item_entropy.
#
# Cette valeur reste provisoire. Elle sera évaluée dans une étape
# ultérieure par comparaison de plusieurs valeurs.
SPLINE_LAMBDA = 10.0

# Les effets linéaires ne sont pas pénalisés.
LINEAR_LAMBDA = 0.0

# La condition et la validité sont des facteurs expérimentaux.
# Une pénalisation presque nulle est utilisée afin de ne pas réduire
# artificiellement leurs différences estimées.
EXPERIMENTAL_FACTOR_LAMBDA = 1e-6

# Les facteurs participant et item sont pénalisés afin de réduire
# leurs coefficients vers zéro.
GROUP_FACTOR_LAMBDA = 10.0

# Paramètres de l'algorithme d'ajustement.
MAX_ITERATIONS = 1000
TOLERANCE = 1e-4

# Nombre de points utilisés dans les graphiques.
PLOT_GRID_SIZE = 200

# Nombre de points conservés dans l'export CSV.
EXPORT_GRID_SIZE = 25

# Résolution du graphique.
FIGURE_DPI = 300


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
# CONSTRUCTION DU MODÈLE
# ============================================================================

def build_parsimonious_gam():
    """
    Construit le GAM parcimonieux.

    Ordre des variables dans X :

        0 : condition_code
        1 : sequence_c10
        2 : subject_accuracy_z
        3 : item_entropy_z
        4 : subject_mean_models_z
        5 : models_within_subject_z
        6 : validity_binary
        7 : subject_code
        8 : item_code

    Spécification :

        f(condition)
        + s(sequence)
        + l(subject_accuracy)
        + s(item_entropy)
        + l(subject_mean_models)
        + l(models_within_subject)
        + f(validity)
        + f(subject)
        + f(item)
    """
    condition_index = FEATURE_INDEX[
        "condition_code"
    ]

    sequence_index = FEATURE_INDEX[
        "sequence_c10"
    ]

    accuracy_index = FEATURE_INDEX[
        "subject_accuracy_z"
    ]

    entropy_index = FEATURE_INDEX[
        "item_entropy_z"
    ]

    mean_models_index = FEATURE_INDEX[
        "subject_mean_models_z"
    ]

    within_models_index = FEATURE_INDEX[
        "models_within_subject_z"
    ]

    validity_index = FEATURE_INDEX[
        "validity_binary"
    ]

    subject_index = FEATURE_INDEX[
        "subject_code"
    ]

    item_index = FEATURE_INDEX[
        "item_code"
    ]

    terms = (
        # Condition expérimentale :
        # Neutral ou Standard.
        f(
            condition_index,
            lam=EXPERIMENTAL_FACTOR_LAMBDA,
        )

        # Évolution potentiellement non linéaire de la confiance
        # au cours des 64 essais.
        + s(
            sequence_index,
            n_splines=N_SPLINES,
            spline_order=SPLINE_ORDER,
            lam=SPLINE_LAMBDA,
        )

        # Association linéaire avec la précision moyenne
        # du participant.
        + l(
            accuracy_index,
            lam=LINEAR_LAMBDA,
        )

        # Association potentiellement non linéaire avec
        # l'entropie de l'item.
        + s(
            entropy_index,
            n_splines=N_SPLINES,
            spline_order=SPLINE_ORDER,
            lam=SPLINE_LAMBDA,
        )

        # Effet interindividuel linéaire du nombre moyen
        # de modèles mentaux.
        + l(
            mean_models_index,
            lam=LINEAR_LAMBDA,
        )

        # Effet intra-individuel linéaire du nombre
        # de modèles mentaux.
        + l(
            within_models_index,
            lam=LINEAR_LAMBDA,
        )

        # Validité logique :
        # Invalid ou Valid.
        + f(
            validity_index,
            lam=EXPERIMENTAL_FACTOR_LAMBDA,
        )

        # Facteur participant pénalisé.
        + f(
            subject_index,
            lam=GROUP_FACTOR_LAMBDA,
        )

        # Facteur item pénalisé.
        + f(
            item_index,
            lam=GROUP_FACTOR_LAMBDA,
        )
    )

    gam = LinearGAM(
        terms=terms,
        max_iter=MAX_ITERATIONS,
        tol=TOLERANCE,
        fit_intercept=True,
        verbose=False,
    )

    return gam


# ============================================================================
# DESCRIPTION DES TERMES
# ============================================================================

def print_model_specification(gam):
    """
    Affiche la structure du modèle avant son ajustement.
    """
    section(
        "SPÉCIFICATION DU GAM PARCIMONIEUX"
    )

    print(
        "Variable dépendante : confidence"
    )

    print("")
    print("Termes catégoriels :")
    print("  f(condition_code)")
    print("  f(validity_binary)")

    print("")
    print("Termes lisses :")
    print("  s(sequence_c10)")
    print("  s(item_entropy_z)")

    print("")
    print("Termes linéaires :")
    print("  l(subject_accuracy_z)")
    print("  l(subject_mean_models_z)")
    print("  l(models_within_subject_z)")

    print("")
    print("Facteurs pénalisés :")
    print("  f(subject_code)")
    print("  f(item_code)")

    print("")
    print(
        "Nombre de splines par terme lisse :",
        N_SPLINES,
    )

    print(
        "Ordre des splines :",
        SPLINE_ORDER,
    )

    print(
        "Pénalisation des splines :",
        SPLINE_LAMBDA,
    )

    print(
        "Pénalisation des termes linéaires :",
        LINEAR_LAMBDA,
    )

    print(
        "Pénalisation des facteurs expérimentaux :",
        EXPERIMENTAL_FACTOR_LAMBDA,
    )

    print(
        "Pénalisation des facteurs participant et item :",
        GROUP_FACTOR_LAMBDA,
    )

    print("")
    print("Représentation pyGAM des termes :")
    print(gam.terms)


# ============================================================================
# CONTRÔLES DU MODÈLE AJUSTÉ
# ============================================================================

def calculate_model_checks(
    gam,
    X,
    y,
):
    """
    Calcule des contrôles descriptifs sur les données d'ajustement.

    Les métriques calculées ici ne constituent pas une évaluation
    hors échantillon.
    """
    section(
        "CONTRÔLES DU MODÈLE AJUSTÉ"
    )

    predictions = np.asarray(
        gam.predict(X),
        dtype=float,
    )

    residuals = (
        y - predictions
    )

    squared_errors = (
        residuals ** 2
    )

    absolute_errors = np.abs(
        residuals
    )

    rmse = float(
        np.sqrt(
            squared_errors.mean()
        )
    )

    mae = float(
        absolute_errors.mean()
    )

    residual_mean = float(
        residuals.mean()
    )

    residual_standard_deviation = float(
        residuals.std(ddof=1)
    )

    predictions_are_finite = bool(
        np.isfinite(predictions).all()
    )

    predictions_below_zero = int(
        (
            predictions < 0
        ).sum()
    )

    predictions_above_hundred = int(
        (
            predictions > 100
        ).sum()
    )

    print(
        "Nombre d'observations :",
        len(y),
    )

    print(
        "Nombre brut de coefficients :",
        len(gam.coef_),
    )

    print(
        "Toutes les prédictions sont finies :",
        predictions_are_finite,
    )

    print(
        "Prédiction minimale :",
        round(
            float(predictions.min()),
            6,
        ),
    )

    print(
        "Prédiction maximale :",
        round(
            float(predictions.max()),
            6,
        ),
    )

    print(
        "Prédictions inférieures à 0 :",
        predictions_below_zero,
    )

    print(
        "Prédictions supérieures à 100 :",
        predictions_above_hundred,
    )

    print(
        "Moyenne des résidus :",
        round(
            residual_mean,
            8,
        ),
    )

    print(
        "Écart-type des résidus :",
        round(
            residual_standard_deviation,
            6,
        ),
    )

    print(
        "RMSE sur les données d'ajustement :",
        round(
            rmse,
            6,
        ),
    )

    print(
        "MAE sur les données d'ajustement :",
        round(
            mae,
            6,
        ),
    )

    print("")
    print(
        "ATTENTION : le RMSE et le MAE sont calculés sur les "
        "données utilisées pour ajuster le modèle."
    )

    print(
        "Ils ne mesurent pas encore la capacité du modèle à "
        "généraliser à de nouvelles observations."
    )

    return {
        "predictions":
            predictions,

        "residuals":
            residuals,

        "rmse_training":
            rmse,

        "mae_training":
            mae,

        "residual_mean":
            residual_mean,

        "residual_standard_deviation":
            residual_standard_deviation,

        "minimum_prediction":
            float(
                predictions.min()
            ),

        "maximum_prediction":
            float(
                predictions.max()
            ),

        "predictions_below_zero":
            predictions_below_zero,

        "predictions_above_hundred":
            predictions_above_hundred,
    }


# ============================================================================
# IDENTIFICATION DES TERMES
# ============================================================================

def get_smooth_term_definitions():
    """
    Retourne les informations nécessaires pour identifier les deux
    termes lisses.

    L'indice correspond à la position du terme dans gam.terms,
    et non directement à la position de la variable dans X.
    """
    return [
        {
            # Terme 0 : f(condition)
            # Terme 1 : s(sequence)
            "term_index":
                1,

            "predictor":
                "sequence_c10",

            "title":
                "Effet non linéaire de la séquence",

            "x_label":
                (
                    "Position de l'essai, "
                    "centrée et divisée par 10"
                ),

            "support_unit":
                "observation",
        },
        {
            # Terme 2 : l(subject_accuracy)
            # Terme 3 : s(item_entropy)
            "term_index":
                3,

            "predictor":
                "item_entropy_z",

            "title":
                "Effet non linéaire de l'entropie de l'item",

            "x_label":
                "Entropie de l'item, standardisée",

            "support_unit":
                "item",
        },
    ]


# ============================================================================
# VISUALISATION DES EFFETS LISSES
# ============================================================================

def plot_smooth_effects(
    gam,
    prepared_data,
):
    """
    Trace les effets partiels de :

        sequence_c10 ;
        item_entropy_z.

    L'axe vertical représente la contribution partielle du terme à
    la confiance prédite, et non la confiance totale prédite.
    """
    section(
        "VISUALISATION DES EFFETS LISSES"
    )

    smooth_terms = (
        get_smooth_term_definitions()
    )

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(15, 6),
    )

    for axis, term_information in zip(
        axes,
        smooth_terms,
    ):
        term_index = term_information[
            "term_index"
        ]

        predictor = term_information[
            "predictor"
        ]

        grid = gam.generate_X_grid(
            term=term_index,
            n=PLOT_GRID_SIZE,
        )

        term = gam.terms[
            term_index
        ]

        feature_index = term.feature

        x_values = np.asarray(
            grid[
                :,
                feature_index,
            ],
            dtype=float,
        )

        (
            partial_effect,
            confidence_interval,
        ) = gam.partial_dependence(
            term=term_index,
            X=grid,
            width=0.95,
        )

        partial_effect = np.asarray(
            partial_effect,
            dtype=float,
        )

        confidence_interval = np.asarray(
            confidence_interval,
            dtype=float,
        )

        # Courbe de l'effet partiel.
        axis.plot(
            x_values,
            partial_effect,
            color="#2563eb",
            linewidth=2.5,
            label="Effet estimé",
        )

        # Intervalle approximatif à 95 %.
        axis.fill_between(
            x_values,
            confidence_interval[:, 0],
            confidence_interval[:, 1],
            color="#93c5fd",
            alpha=0.35,
            label="Intervalle approximatif à 95 %",
        )

        # Ligne correspondant à une contribution partielle nulle.
        axis.axhline(
            0,
            color="#111827",
            linestyle="--",
            linewidth=1.2,
            alpha=0.75,
        )

        # Distribution des valeurs observées.
        observed_values = (
            prepared_data[
                predictor
            ]
            .dropna()
            .to_numpy(dtype=float)
        )

        y_minimum = float(
            min(
                partial_effect.min(),
                confidence_interval[:, 0].min(),
            )
        )

        y_maximum = float(
            max(
                partial_effect.max(),
                confidence_interval[:, 1].max(),
            )
        )

        y_range = (
            y_maximum - y_minimum
        )

        rug_position = (
            y_minimum
            - 0.03 * y_range
        )

        axis.plot(
            observed_values,
            np.full(
                len(observed_values),
                rug_position,
            ),
            "|",
            color="#4b5563",
            alpha=0.07,
            markersize=7,
        )

        axis.set_title(
            term_information[
                "title"
            ],
            fontsize=13,
            fontweight="bold",
        )

        axis.set_xlabel(
            term_information[
                "x_label"
            ]
        )

        axis.set_ylabel(
            "Contribution partielle à la confiance"
        )

        axis.grid(
            True,
            linestyle="--",
            alpha=0.3,
        )

    handles, labels = (
        axes[0]
        .get_legend_handles_labels()
    )

    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
    )

    figure.suptitle(
        "GAM parcimonieux — Effets lisses estimés",
        fontsize=16,
        fontweight="bold",
    )

    figure.tight_layout(
        rect=[
            0,
            0.08,
            1,
            0.95,
        ]
    )

    figure.savefig(
        SMOOTH_EFFECTS_FIGURE,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        "Figure enregistrée :",
        SMOOTH_EFFECTS_FIGURE,
    )


# ============================================================================
# SUPPORT LOCAL DES VARIABLES
# ============================================================================

def get_observed_support_values(
    prepared_data,
    predictor,
    support_unit,
):
    """
    Retourne les valeurs servant à calculer le support local.

    Pour sequence_c10 :
        une valeur par observation.

    Pour item_entropy_z :
        une valeur par item distinct.
    """
    if support_unit == "item":
        support_data = (
            prepared_data[[
                "item_id",
                predictor,
            ]]
            .dropna()
            .drop_duplicates(
                subset=[
                    "item_id"
                ]
            )
        )

        return support_data[
            predictor
        ].to_numpy(dtype=float)

    return (
        prepared_data[
            predictor
        ]
        .dropna()
        .to_numpy(dtype=float)
    )


# ============================================================================
# EXPORT NUMÉRIQUE DES EFFETS LISSES
# ============================================================================

def export_smooth_effects(
    gam,
    prepared_data,
):
    """
    Exporte les valeurs des deux effets lisses sur une grille de
    25 points.

    Le support local est exprimé :

        - en observations pour sequence_c10 ;
        - en items distincts pour item_entropy_z.
    """
    section(
        "EXPORT NUMÉRIQUE DES EFFETS LISSES"
    )

    smooth_terms = (
        get_smooth_term_definitions()
    )

    output_rows = []

    for term_information in smooth_terms:
        term_index = term_information[
            "term_index"
        ]

        predictor = term_information[
            "predictor"
        ]

        support_unit = term_information[
            "support_unit"
        ]

        grid = gam.generate_X_grid(
            term=term_index,
            n=EXPORT_GRID_SIZE,
        )

        term = gam.terms[
            term_index
        ]

        feature_index = term.feature

        x_values = np.asarray(
            grid[
                :,
                feature_index,
            ],
            dtype=float,
        )

        (
            partial_effect,
            confidence_interval,
        ) = gam.partial_dependence(
            term=term_index,
            X=grid,
            width=0.95,
        )

        partial_effect = np.asarray(
            partial_effect,
            dtype=float,
        )

        confidence_interval = np.asarray(
            confidence_interval,
            dtype=float,
        )

        observed_values = (
            get_observed_support_values(
                prepared_data=prepared_data,
                predictor=predictor,
                support_unit=support_unit,
            )
        )

        if len(x_values) > 1:
            grid_step = float(
                np.median(
                    np.diff(
                        x_values
                    )
                )
            )

        else:
            grid_step = np.nan

        if (
            np.isfinite(grid_step)
            and not np.isclose(
                grid_step,
                0.0,
            )
        ):
            local_half_width = (
                abs(grid_step) / 2.0
            )

        else:
            local_half_width = 0.1

        for grid_index, x_value in enumerate(
            x_values
        ):
            local_support_count = int(
                np.sum(
                    np.abs(
                        observed_values
                        - x_value
                    )
                    <= local_half_width
                )
            )

            lower_bound = float(
                confidence_interval[
                    grid_index,
                    0,
                ]
            )

            upper_bound = float(
                confidence_interval[
                    grid_index,
                    1,
                ]
            )

            output_rows.append({
                "predictor":
                    predictor,

                "grid_index":
                    grid_index,

                "x":
                    float(x_value),

                "partial_effect":
                    float(
                        partial_effect[
                            grid_index
                        ]
                    ),

                "ci_95_lower":
                    lower_bound,

                "ci_95_upper":
                    upper_bound,

                "ci_95_width":
                    (
                        upper_bound
                        - lower_bound
                    ),

                "local_support_count":
                    local_support_count,

                "support_unit":
                    support_unit,
            })

    effects_table = pd.DataFrame(
        output_rows
    )

    effects_table.to_csv(
        SMOOTH_EFFECTS_VALUES_FILE,
        index=False,
    )

    print(
        "Tableau enregistré :",
        SMOOTH_EFFECTS_VALUES_FILE,
    )

    return effects_table

# ============================================================================
# EXPORT DES EFFETS PARAMÉTRIQUES
# ============================================================================

def export_parametric_effects(
    gam,
    X,
):
    """
    Calcule les effets ajustés des termes non lisses du modèle.

    Les effets sont obtenus en comparant deux prédictions qui ne
    diffèrent que par la variable étudiée.

    Contrastes calculés
    -------------------
    condition :
        Standard moins Neutral.

    validity_binary :
        Valid moins Invalid.

    subject_accuracy_z :
        passage de la moyenne, z=0, à un écart-type au-dessus,
        z=1.

    subject_mean_models_z :
        passage de la moyenne, z=0, à un écart-type au-dessus,
        z=1.

    models_within_subject_z :
        passage de la moyenne personnelle, z=0, à un écart-type
        au-dessus, z=1.

    Les résultats sont exprimés en points de confiance.
    """
    section(
        "EXPORT DES EFFETS PARAMÉTRIQUES"
    )

    # ------------------------------------------------------------------
    # 1. Construction d'une observation de référence
    # ------------------------------------------------------------------

    reference_row = np.zeros(
        X.shape[1],
        dtype=float,
    )

    # Les prédicteurs standardisés sont fixés à zéro, c'est-à-dire
    # à leur moyenne.
    reference_row[
        FEATURE_INDEX[
            "sequence_c10"
        ]
    ] = 0.0

    reference_row[
        FEATURE_INDEX[
            "subject_accuracy_z"
        ]
    ] = 0.0

    reference_row[
        FEATURE_INDEX[
            "item_entropy_z"
        ]
    ] = 0.0

    reference_row[
        FEATURE_INDEX[
            "subject_mean_models_z"
        ]
    ] = 0.0

    reference_row[
        FEATURE_INDEX[
            "models_within_subject_z"
        ]
    ] = 0.0

    # Neutral est la catégorie de référence.
    reference_row[
        FEATURE_INDEX[
            "condition_code"
        ]
    ] = 0.0

    # Invalid est la catégorie de référence.
    reference_row[
        FEATURE_INDEX[
            "validity_binary"
        ]
    ] = 0.0

    # Les codes participant et item doivent correspondre à des
    # catégories existantes. Le code 0 existe nécessairement.
    #
    # Comme les deux observations d'un contraste utilisent les mêmes
    # participant et item, leurs contributions s'annulent dans la
    # différence.
    reference_row[
        FEATURE_INDEX[
            "subject_code"
        ]
    ] = 0.0

    reference_row[
        FEATURE_INDEX[
            "item_code"
        ]
    ] = 0.0

    # ------------------------------------------------------------------
    # 2. Définition des contrastes
    # ------------------------------------------------------------------

    contrast_definitions = [
        {
            "effect":
                "condition_standard_vs_neutral",

            "variable":
                "condition_code",

            "term_type":
                "categorical",

            "reference_value":
                0.0,

            "comparison_value":
                1.0,

            "reference_label":
                "Neutral",

            "comparison_label":
                "Standard",

            "interpretation":
                (
                    "Différence ajustée de confiance : "
                    "Standard moins Neutral."
                ),
        },
        {
            "effect":
                "valid_vs_invalid",

            "variable":
                "validity_binary",

            "term_type":
                "categorical",

            "reference_value":
                0.0,

            "comparison_value":
                1.0,

            "reference_label":
                "Invalid",

            "comparison_label":
                "Valid",

            "interpretation":
                (
                    "Différence ajustée de confiance : "
                    "Valid moins Invalid."
                ),
        },
        {
            "effect":
                "subject_accuracy_plus_1_sd",

            "variable":
                "subject_accuracy_z",

            "term_type":
                "linear",

            "reference_value":
                0.0,

            "comparison_value":
                1.0,

            "reference_label":
                "Mean",

            "comparison_label":
                "Mean_plus_1_SD",

            "interpretation":
                (
                    "Variation ajustée de confiance pour une "
                    "augmentation d'un écart-type de la précision "
                    "moyenne du participant."
                ),
        },
        {
            "effect":
                "subject_mean_models_plus_1_sd",

            "variable":
                "subject_mean_models_z",

            "term_type":
                "linear",

            "reference_value":
                0.0,

            "comparison_value":
                1.0,

            "reference_label":
                "Mean",

            "comparison_label":
                "Mean_plus_1_SD",

            "interpretation":
                (
                    "Variation ajustée de confiance pour une "
                    "augmentation d'un écart-type du nombre moyen "
                    "de modèles du participant."
                ),
        },
        {
            "effect":
                "models_within_subject_plus_1_sd",

            "variable":
                "models_within_subject_z",

            "term_type":
                "linear",

            "reference_value":
                0.0,

            "comparison_value":
                1.0,

            "reference_label":
                "Personal_mean",

            "comparison_label":
                "Personal_mean_plus_1_SD",

            "interpretation":
                (
                    "Variation ajustée de confiance lorsque le "
                    "nombre de modèles dépasse la moyenne personnelle "
                    "d'un écart-type."
                ),
        },
    ]

    # ------------------------------------------------------------------
    # 3. Calcul des différences de prédiction
    # ------------------------------------------------------------------

    output_rows = []

    for definition in contrast_definitions:
        feature_index = FEATURE_INDEX[
            definition[
                "variable"
            ]
        ]

        reference_scenario = (
            reference_row.copy()
        )

        comparison_scenario = (
            reference_row.copy()
        )

        reference_scenario[
            feature_index
        ] = definition[
            "reference_value"
        ]

        comparison_scenario[
            feature_index
        ] = definition[
            "comparison_value"
        ]

        scenarios = np.vstack([
            reference_scenario,
            comparison_scenario,
        ])

        predictions = np.asarray(
            gam.predict(
                scenarios
            ),
            dtype=float,
        )

        reference_prediction = float(
            predictions[0]
        )

        comparison_prediction = float(
            predictions[1]
        )

        estimated_effect = (
            comparison_prediction
            - reference_prediction
        )

        output_rows.append({
            "effect":
                definition[
                    "effect"
                ],

            "variable":
                definition[
                    "variable"
                ],

            "term_type":
                definition[
                    "term_type"
                ],

            "reference_label":
                definition[
                    "reference_label"
                ],

            "comparison_label":
                definition[
                    "comparison_label"
                ],

            "reference_value":
                definition[
                    "reference_value"
                ],

            "comparison_value":
                definition[
                    "comparison_value"
                ],

            "reference_prediction":
                reference_prediction,

            "comparison_prediction":
                comparison_prediction,

            "estimated_effect_points":
                estimated_effect,

            "direction":
                (
                    "positive"
                    if estimated_effect > 0
                    else (
                        "negative"
                        if estimated_effect < 0
                        else "null"
                    )
                ),

            "interpretation":
                definition[
                    "interpretation"
                ],
        })

    effects_table = pd.DataFrame(
        output_rows
    )

    effects_table.to_csv(
        PARAMETRIC_EFFECTS_FILE,
        index=False,
    )

    print(
        effects_table[[
            "effect",
            "estimated_effect_points",
            "direction",
        ]]
        .to_string(index=False)
    )

    print("")
    print(
        "Tableau enregistré :",
        PARAMETRIC_EFFECTS_FILE,
    )

    return effects_table


# ============================================================================
# SAUVEGARDE DU RÉSUMÉ
# ============================================================================

def save_model_summary(
    gam,
    model_checks,
):
    """
    Sauvegarde le résumé pyGAM et les principaux contrôles descriptifs.
    """
    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as output_file:
        with redirect_stdout(
            output_file
        ):
            print(
                "GAM PARCIMONIEUX DE CONFIANCE — EXPÉRIENCE E1"
            )

            print(
                "=" * 80
            )

            print("")
            print("SPÉCIFICATION")
            print("-" * 80)

            print(
                "Variable dépendante : confidence"
            )

            print(
                "Termes catégoriels : "
                "condition, validity_binary"
            )

            print(
                "Termes lisses : "
                "sequence_c10, item_entropy_z"
            )

            print(
                "Termes linéaires : "
                "subject_accuracy_z, "
                "subject_mean_models_z, "
                "models_within_subject_z"
            )

            print(
                "Facteurs pénalisés : "
                "subject_code, item_code"
            )

            print("")
            print(
                "ATTENTION : les facteurs participant et item "
                "sont des facteurs pénalisés."
            )

            print(
                "Ils constituent une approximation et non de "
                "véritables effets aléatoires croisés."
            )

            print("")
            print("PARAMÈTRES")
            print("-" * 80)

            print(
                "N_SPLINES :",
                N_SPLINES,
            )

            print(
                "SPLINE_ORDER :",
                SPLINE_ORDER,
            )

            print(
                "SPLINE_LAMBDA :",
                SPLINE_LAMBDA,
            )

            print(
                "LINEAR_LAMBDA :",
                LINEAR_LAMBDA,
            )

            print(
                "EXPERIMENTAL_FACTOR_LAMBDA :",
                EXPERIMENTAL_FACTOR_LAMBDA,
            )

            print(
                "GROUP_FACTOR_LAMBDA :",
                GROUP_FACTOR_LAMBDA,
            )

            print("")
            print("RÉSUMÉ PYGAM")
            print("-" * 80)
            print("")

            gam.summary()

            print("")
            print("CONTRÔLES DES PRÉDICTIONS")
            print("-" * 80)

            print(
                "RMSE d'ajustement :",
                model_checks[
                    "rmse_training"
                ],
            )

            print(
                "MAE d'ajustement :",
                model_checks[
                    "mae_training"
                ],
            )

            print(
                "Moyenne des résidus :",
                model_checks[
                    "residual_mean"
                ],
            )

            print(
                "Écart-type des résidus :",
                model_checks[
                    "residual_standard_deviation"
                ],
            )

            print(
                "Prédiction minimale :",
                model_checks[
                    "minimum_prediction"
                ],
            )

            print(
                "Prédiction maximale :",
                model_checks[
                    "maximum_prediction"
                ],
            )

            print(
                "Prédictions sous 0 :",
                model_checks[
                    "predictions_below_zero"
                ],
            )

            print(
                "Prédictions au-dessus de 100 :",
                model_checks[
                    "predictions_above_hundred"
                ],
            )

            print("")
            print(
                "Le RMSE et le MAE sont calculés sur les données "
                "utilisées pour l'ajustement."
            )

            print(
                "Ils ne constituent pas une évaluation "
                "hors échantillon."
            )

    print(
        "Résumé enregistré :",
        SUMMARY_FILE,
    )


# ============================================================================
# PROGRAMME PRINCIPAL
# ============================================================================

def main():
    section(
        "GAM PARCIMONIEUX DE CONFIANCE — EXPÉRIENCE E1"
    )

    print(
        "Version pyGAM :",
        pygam.__version__,
    )

    # ------------------------------------------------------------------
    # 1. Chargement
    # ------------------------------------------------------------------

    raw_data = load_data()

    # ------------------------------------------------------------------
    # 2. Préparation
    # ------------------------------------------------------------------

    (
        prepared_data,
        preparation_information,
    ) = prepare_data(
        raw_data
    )

    # preparation_information est conservé pour maintenir la même
    # interface que le script de préparation, même s'il n'est pas
    # directement utilisé ici.
    _ = preparation_information

    # ------------------------------------------------------------------
    # 3. Construction de X et y
    # ------------------------------------------------------------------

    X, y = build_model_matrices(
        prepared_data
    )

    # ------------------------------------------------------------------
    # 4. Construction du GAM
    # ------------------------------------------------------------------

    gam = build_parsimonious_gam()

    print_model_specification(
        gam
    )

    # ------------------------------------------------------------------
    # 5. Ajustement
    # ------------------------------------------------------------------

    section(
        "AJUSTEMENT DU GAM"
    )

    gam.fit(
        X,
        y,
    )

    print(
        "Ajustement terminé."
    )

    # ------------------------------------------------------------------
    # 6. Résumé dans le terminal
    # ------------------------------------------------------------------

    section(
        "RÉSUMÉ DU GAM"
    )

    gam.summary()

    # ------------------------------------------------------------------
    # 7. Contrôles
    # ------------------------------------------------------------------

    model_checks = (
        calculate_model_checks(
            gam=gam,
            X=X,
            y=y,
        )
    )

    # ------------------------------------------------------------------
    # 8. Graphique des deux effets lisses
    # ------------------------------------------------------------------

    plot_smooth_effects(
        gam=gam,
        prepared_data=
            prepared_data,
    )

    # ------------------------------------------------------------------
    # 9. Export numérique
    # ------------------------------------------------------------------

    export_smooth_effects(
        gam=gam,
        prepared_data=
            prepared_data,
    )

    # ------------------------------------------------------------------
    # 10. Effets paramétriques
    # ------------------------------------------------------------------

    parametric_effects = (
        export_parametric_effects(
            gam=gam,
            X=X,
        )
    )

    _ = parametric_effects


    # ------------------------------------------------------------------
    # 11. Résumé texte
    # ------------------------------------------------------------------

    save_model_summary(
        gam=gam,
        model_checks=
            model_checks,
    )

    # ------------------------------------------------------------------
    # 12. Fin
    # ------------------------------------------------------------------

    section(
        "TERMINÉ"
    )

    print(
        "Le GAM parcimonieux a été ajusté."
    )

    print("")
    print("Fichiers produits :")

    print(
        SUMMARY_FILE
    )

    print(
        SMOOTH_EFFECTS_FIGURE
    )

    print(
        SMOOTH_EFFECTS_VALUES_FILE
    )

    print(
        PARAMETRIC_EFFECTS_FILE
    )



if __name__ == "__main__":
    main()
