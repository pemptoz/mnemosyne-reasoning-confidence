#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygam
from pygam import LinearGAM, f, s
 

"""
fit_gam_model_E1_n20.py

Ajuste un premier modèle additif généralisé pour la confiance
dans l'expérience E1.

Variable dépendante :
    confidence

Termes catégoriels :
    condition
    validity_binary

Termes non linéaires :
    sequence_c10
    subject_accuracy_z
    item_entropy_z
    subject_mean_models_z
    models_within_subject_z

Facteurs pénalisés :
    subject_id
    item_id

Ce premier script utilise des paramètres de lissage fixés.
Aucune recherche d'hyperparamètres n'est encore réalisée.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pygam
from pygam import LinearGAM, f, s

from prepare_gam_data_E1_n20 import (
    load_data,
    prepare_data,
    build_model_matrices,
    FEATURE_INDEX,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = SCRIPT_DIR.parents[3]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "analysis"
    / "computational-model"
    / "gam"
    / "gam_model_E1_n20"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "gam_initial_summary.txt"
)

SMOOTH_EFFECTS_FIGURE = (
    OUTPUT_DIR
    / "gam_smooth_effects.png"
)

SMOOTH_EFFECTS_VALUES_FILE = (
    OUTPUT_DIR
    / "gam_smooth_effects_values.csv"
)


# ============================================================================
# PARAMÈTRES DU PREMIER GAM
# ============================================================================

# Nombre de fonctions de base utilisées pour chaque terme continu.
N_SPLINES = 10

# Paramètre de pénalisation initial des termes spline.
SPLINE_LAMBDA = 10.0

# Pénalisation des facteurs participant et item.
GROUP_FACTOR_LAMBDA = 10.0

# Condition et validité sont des facteurs expérimentaux.
# Pour ce premier ajustement, nous utilisons une pénalisation très faible
# afin d'éviter de réduire artificiellement leurs différences estimées.
EXPERIMENTAL_FACTOR_LAMBDA = 1e-6

MAX_ITERATIONS = 1000

TOLERANCE = 1e-4


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
# VISUALISATION DES EFFETS LISSES
# ============================================================================

def plot_smooth_effects(
    gam,
    prepared_data,
):
    """
    Trace les effets partiels des cinq prédicteurs continus.

    Pour chaque terme, la figure représente :

        - la contribution partielle estimée par le GAM ;
        - l'intervalle approximatif à 95 % ;
        - la distribution des observations sur l'axe horizontal.

    Une seule figure contenant cinq sous-graphiques est produite.
    """
    section(
        "VISUALISATION DES EFFETS LISSES"
    )

    smooth_terms = [
        {
            "term_index": 1,
            "feature_column": "sequence_c10",
            "title": "Effet de la séquence",
            "x_label": (
                "Position de l'essai, centrée et divisée par 10"
            ),
        },
        {
            "term_index": 2,
            "feature_column": "subject_accuracy_z",
            "title": "Effet de la précision du participant",
            "x_label": (
                "Précision du participant, standardisée"
            ),
        },
        {
            "term_index": 3,
            "feature_column": "item_entropy_z",
            "title": "Effet de l'entropie de l'item",
            "x_label": (
                "Entropie de l'item, standardisée"
            ),
        },
        {
            "term_index": 4,
            "feature_column": "subject_mean_models_z",
            "title": "Effet du nombre moyen de modèles",
            "x_label": (
                "Nombre moyen de modèles du participant, standardisé"
            ),
        },
        {
            "term_index": 5,
            "feature_column": "models_within_subject_z",
            "title": "Effet intra-individuel du nombre de modèles",
            "x_label": (
                "Écart à la moyenne personnelle, standardisé"
            ),
        },
    ]

    figure, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(14, 15),
    )

    axes = axes.flatten()

    for axis, term_information in zip(
        axes,
        smooth_terms,
    ):
        term_index = term_information[
            "term_index"
        ]

        feature_column = term_information[
            "feature_column"
        ]

        # --------------------------------------------------------------
        # 1. Construction d'une grille régulière
        # --------------------------------------------------------------

        grid = gam.generate_X_grid(
            term=term_index,
            n=200,
        )

        term = gam.terms[
            term_index
        ]

        feature_index = term.feature

        x_values = grid[
            :,
            feature_index,
        ]

        # --------------------------------------------------------------
        # 2. Contribution partielle et intervalle à 95 %
        # --------------------------------------------------------------

        (
            partial_effect,
            confidence_interval,
        ) = gam.partial_dependence(
            term=term_index,
            X=grid,
            width=0.95,
        )

        # --------------------------------------------------------------
        # 3. Courbe estimée
        # --------------------------------------------------------------

        axis.plot(
            x_values,
            partial_effect,
            color="#2563eb",
            linewidth=2.5,
            label="Effet estimé",
        )

        # --------------------------------------------------------------
        # 4. Intervalle approximatif à 95 %
        # --------------------------------------------------------------

        axis.fill_between(
            x_values,
            confidence_interval[:, 0],
            confidence_interval[:, 1],
            color="#93c5fd",
            alpha=0.35,
            label="Intervalle à 95 %",
        )

        # --------------------------------------------------------------
        # 5. Ligne de référence à zéro
        # --------------------------------------------------------------

        axis.axhline(
            0,
            color="#111827",
            linestyle="--",
            linewidth=1.2,
            alpha=0.75,
        )

        # --------------------------------------------------------------
        # 6. Distribution des observations
        # --------------------------------------------------------------

        observed_values = (
            prepared_data[
                feature_column
            ]
            .dropna()
            .to_numpy(dtype=float)
        )

        rug_y = np.full(
            len(observed_values),
            axis.get_ylim()[0],
        )

        axis.plot(
            observed_values,
            rug_y,
            "|",
            color="#4b5563",
            alpha=0.08,
            markersize=8,
        )

        # --------------------------------------------------------------
        # 7. Mise en forme
        # --------------------------------------------------------------

        axis.set_title(
            term_information["title"],
            fontsize=13,
            fontweight="bold",
        )

        axis.set_xlabel(
            term_information["x_label"]
        )

        axis.set_ylabel(
            "Contribution partielle à la confiance"
        )

        axis.grid(
            True,
            linestyle="--",
            alpha=0.3,
        )

    # Le sixième sous-graphique n'est pas utilisé.
    axes[-1].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()

    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
    )

    figure.suptitle(
        "GAM de la confiance — Effets lisses estimés",
        fontsize=17,
        fontweight="bold",
        y=0.995,
    )

    figure.tight_layout(
        rect=[
            0,
            0.04,
            1,
            0.98,
        ]
    )

    figure.savefig(
        SMOOTH_EFFECTS_FIGURE,
        dpi=300,
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
# CONSTRUCTION DU GAM
# ============================================================================

def build_initial_gam():
    """
    Construit le premier GAM.

    Les indices correspondent à l'ordre des colonnes défini dans
    prepare_gam_data_E1_n20.py.
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
        f(
            condition_index,
            lam=EXPERIMENTAL_FACTOR_LAMBDA,
        )
        + s(
            sequence_index,
            n_splines=N_SPLINES,
            spline_order=3,
            lam=SPLINE_LAMBDA,
        )
        + s(
            accuracy_index,
            n_splines=N_SPLINES,
            spline_order=3,
            lam=SPLINE_LAMBDA,
        )
        + s(
            entropy_index,
            n_splines=N_SPLINES,
            spline_order=3,
            lam=SPLINE_LAMBDA,
        )
        + s(
            mean_models_index,
            n_splines=N_SPLINES,
            spline_order=3,
            lam=SPLINE_LAMBDA,
        )
        + s(
            within_models_index,
            n_splines=N_SPLINES,
            spline_order=3,
            lam=SPLINE_LAMBDA,
        )
        + f(
            validity_index,
            lam=EXPERIMENTAL_FACTOR_LAMBDA,
        )
        + f(
            subject_index,
            lam=GROUP_FACTOR_LAMBDA,
        )
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
# CONTRÔLES DU MODÈLE AJUSTÉ
# ============================================================================

def print_model_checks(gam, X, y):
    """
    Affiche quelques contrôles élémentaires après l'ajustement.
    """
    section(
        "CONTRÔLES DU MODÈLE AJUSTÉ"
    )

    predictions = gam.predict(X)

    residuals = (
        y - predictions
    )

    rmse = float(
        np.sqrt(
            np.mean(
                residuals ** 2
            )
        )
    )

    mae = float(
        np.mean(
            np.abs(residuals)
        )
    )

    print(
        "Nombre d'observations :",
        len(y),
    )

    print(
        "Nombre de coefficients estimés :",
        len(gam.coef_),
    )

    print(
        "Toutes les prédictions sont-elles finies ?",
        bool(
            np.isfinite(predictions).all()
        ),
    )

    print(
        "Confiance prédite minimale :",
        round(
            float(predictions.min()),
            4,
        ),
    )

    print(
        "Confiance prédite maximale :",
        round(
            float(predictions.max()),
            4,
        ),
    )

    print(
        "Moyenne des résidus :",
        round(
            float(residuals.mean()),
            6,
        ),
    )

    print(
        "RMSE sur les données d'ajustement :",
        round(
            rmse,
            4,
        ),
    )

    print(
        "MAE sur les données d'ajustement :",
        round(
            mae,
            4,
        ),
    )

    print("")
    print(
        "ATTENTION : RMSE et MAE sont calculées sur les mêmes "
        "données que celles utilisées pour ajuster le modèle."
    )

    print(
        "Elles ne constituent pas encore une évaluation "
        "hors échantillon."
    )

    return {
        "rmse_training":
            rmse,

        "mae_training":
            mae,

        "minimum_prediction":
            float(predictions.min()),

        "maximum_prediction":
            float(predictions.max()),

        "mean_residual":
            float(residuals.mean()),
    }


# ============================================================================
# SAUVEGARDE DU RÉSUMÉ
# ============================================================================

def save_summary(gam):
    """
    Sauvegarde le résumé textuel du GAM.

    pyGAM écrit directement son résumé dans la sortie standard.
    Nous redirigeons donc temporairement cette sortie vers un fichier.
    """
    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as output_file:
        original_stdout = sys.stdout

        try:
            sys.stdout = output_file

            print(
                "PREMIER GAM DE CONFIANCE — EXPÉRIENCE E1"
            )

            print(
                "=" * 80
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

            gam.summary()

        finally:
            sys.stdout = original_stdout

    print(
        "Résumé enregistré :",
        SUMMARY_FILE,
    )

# ============================================================================
# EXPORT NUMÉRIQUE DES EFFETS LISSES
# ============================================================================

def export_smooth_effects(
    gam,
    prepared_data,
):
    """
    Exporte les effets partiels des cinq prédicteurs continus.

    Le fichier contient, pour chaque prédicteur :

        - la valeur du prédicteur ;
        - la contribution partielle estimée ;
        - les bornes de l'intervalle approximatif à 95 % ;
        - une estimation de la densité locale des observations.

    Les valeurs de x sont calculées sur une grille régulière.
    """
    section(
        "EXPORT NUMÉRIQUE DES EFFETS LISSES"
    )

    smooth_terms = [
        {
            "term_index": 1,
            "predictor": "sequence_c10",
        },
        {
            "term_index": 2,
            "predictor": "subject_accuracy_z",
        },
        {
            "term_index": 3,
            "predictor": "item_entropy_z",
        },
        {
            "term_index": 4,
            "predictor": "subject_mean_models_z",
        },
        {
            "term_index": 5,
            "predictor": "models_within_subject_z",
        },
    ]

    output_rows = []

    for term_information in smooth_terms:
        term_index = term_information[
            "term_index"
        ]

        predictor = term_information[
            "predictor"
        ]

        # Une grille de 25 points suffit pour une lecture textuelle.
        grid = gam.generate_X_grid(
            term=term_index,
            n=25,
        )

        term = gam.terms[
            term_index
        ]

        feature_index = term.feature

        x_values = grid[
            :,
            feature_index,
        ]

        (
            partial_effect,
            confidence_interval,
        ) = gam.partial_dependence(
            term=term_index,
            X=grid,
            width=0.95,
        )

        observed_values = (
            prepared_data[
                predictor
            ]
            .dropna()
            .to_numpy(dtype=float)
        )

        # Distance entre deux points successifs de la grille.
        if len(x_values) > 1:
            grid_step = float(
                np.median(
                    np.diff(x_values)
                )
            )
        else:
            grid_step = np.nan

        # Compte approximativement le nombre d'observations proches
        # de chaque point de la grille.
        local_half_width = (
            abs(grid_step) / 2
            if np.isfinite(grid_step)
            and grid_step != 0
            else 0.1
        )

        for index, x_value in enumerate(x_values):
            local_count = int(
                np.sum(
                    np.abs(
                        observed_values
                        - x_value
                    )
                    <= local_half_width
                )
            )

            output_rows.append({
                "predictor":
                    predictor,

                "grid_index":
                    index,

                "x":
                    float(x_value),

                "partial_effect":
                    float(
                        partial_effect[index]
                    ),

                "ci_95_lower":
                    float(
                        confidence_interval[
                            index,
                            0,
                        ]
                    ),

                "ci_95_upper":
                    float(
                        confidence_interval[
                            index,
                            1,
                        ]
                    ),

                "ci_95_width":
                    float(
                        confidence_interval[
                            index,
                            1,
                        ]
                        - confidence_interval[
                            index,
                            0,
                        ]
                    ),

                "local_observation_count":
                    local_count,
            })

    effects_table = pd.DataFrame(
        output_rows
    )

    effects_table.to_csv(
        SMOOTH_EFFECTS_VALUES_FILE,
        index=False,
    )

    print(
        "Valeurs des effets enregistrées :",
        SMOOTH_EFFECTS_VALUES_FILE,
    )

    return effects_table


# ============================================================================
# PROGRAMME PRINCIPAL
# ============================================================================

def main():
    section(
        "PREMIER GAM DE CONFIANCE — EXPÉRIENCE E1"
    )

    print(
        "Version pyGAM :",
        pygam.__version__,
    )

    # ------------------------------------------------------------------
    # 1. Préparation
    # ------------------------------------------------------------------

    raw_data = load_data()

    (
        prepared_data,
        preparation_information,
    ) = prepare_data(
        raw_data
    )

    X, y = build_model_matrices(
        prepared_data
    )

    # ------------------------------------------------------------------
    # 2. Construction du modèle
    # ------------------------------------------------------------------

    section(
        "CONSTRUCTION DU GAM"
    )

    gam = build_initial_gam()

    print(
        "Nombre de splines par prédicteur continu :",
        N_SPLINES,
    )

    print(
        "Pénalisation des splines :",
        SPLINE_LAMBDA,
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
    print(
        "Termes du modèle :"
    )

    print(
        gam.terms
    )

    # ------------------------------------------------------------------
    # 3. Ajustement
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
    # 4. Résumé
    # ------------------------------------------------------------------

    section(
        "RÉSUMÉ DU GAM"
    )

    gam.summary()

    # ------------------------------------------------------------------
    # 5. Contrôles
    # ------------------------------------------------------------------

    print_model_checks(
        gam=gam,
        X=X,
        y=y,
    )

    # ------------------------------------------------------------------
    # 6 Effets lisses
    # ------------------------------------------------------------------

    plot_smooth_effects(
        gam=gam,
        prepared_data=prepared_data,
    )

    smooth_effects_table = export_smooth_effects(
        gam=gam,
        prepared_data=prepared_data,
    )

    # ------------------------------------------------------------------
    # 7. Sauvegarde minimale
    # ------------------------------------------------------------------

    save_summary(
        gam
    )

    section(
        "TERMINÉ"
    )

    print(
        "Le premier GAM a été ajusté."
    )

    print(
        "Fichiers produits :"
    )

    print(
        SUMMARY_FILE
    )

    print(
        SMOOTH_EFFECTS_FIGURE
    )
    print(
        SMOOTH_EFFECTS_VALUES_FILE
    )


if __name__ == "__main__":
    main()
