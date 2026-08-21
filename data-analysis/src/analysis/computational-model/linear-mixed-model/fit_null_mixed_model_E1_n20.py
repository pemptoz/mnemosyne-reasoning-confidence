"""
fit_null_mixed_model_E1.py

Ajuste le modèle linéaire mixte nul de la confiance pour
l'expérience E1.

Fichier d'entrée
----------------
dataset_analysis_E1_n20.csv

Modèle
------
confidence_ij = beta_0 + u_i + v_j + epsilon_ij

avec :

    beta_0 :
        niveau moyen général de confiance ;

    u_i :
        intercept aléatoire du participant i ;

    v_j :
        intercept aléatoire de l'item j ;

    epsilon_ij :
        variation résiduelle au niveau de l'essai.

Les effets participant et item sont croisés.

Deux ajustements sont réalisés :

    REML :
        utilisé pour l'estimation principale des composantes
        de variance ;

    ML :
        conservé comme référence pour les futures comparaisons
        avec des modèles possédant davantage d'effets fixes.

Fichiers produits
-----------------
null_mixed_model_E1/null_model_REML_summary.txt
null_mixed_model_E1/null_model_variance_components.csv
null_mixed_model_E1/null_model_fit_statistics.csv
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# ======================================================================
# CONFIGURATION
# ======================================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        SCRIPT_DIR,
        "..",
        "..",
        "..",
        "..",
    )
)

INPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "tables",
    "computational-model",
    "dataset_analysis_E1_n20.csv",
)

OUTPUT_DIRECTORY = os.path.join(
    PROJECT_ROOT,
    "results",
    "analysis",
    "computational-model",
    "linear-mixed-model",
    "null_mixed_model_E1_n20",
)

os.makedirs(
    OUTPUT_DIRECTORY,
    exist_ok=True,
)


# ----------------------------------------------------------------------
# Fichiers produits
# ----------------------------------------------------------------------

REML_SUMMARY_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "null_model_REML_summary.txt",
)

VARIANCE_COMPONENTS_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "null_model_variance_components.csv",
)

FIT_STATISTICS_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "null_model_fit_statistics.csv",
)



# ----------------------------------------------------------------------
# Options d'estimation
# ----------------------------------------------------------------------

# Les matrices creuses réduisent la quantité de mémoire nécessaire
# pour représenter les indicatrices des participants et des items.
USE_SPARSE_MATRICES = True

# Nombre maximal d'itérations accordé à chaque optimiseur.
MAX_ITERATIONS = 2000

# Les optimiseurs sont essayés dans cet ordre.
OPTIMIZATION_METHODS = [
    "lbfgs",
    "bfgs",
    "cg",
    "powell",
]

# Si False, statsmodels n'affiche pas le détail de chaque itération.
OPTIMIZER_DISPLAY = False


# ======================================================================
# AFFICHAGE
# ======================================================================

def print_section(title):
    """Affiche un titre de section dans le terminal."""
    separator = "=" * 80

    print("")
    print(separator)
    print(title)
    print(separator)


# ======================================================================
# OUTILS GÉNÉRAUX
# ======================================================================

def normalize_identifier(value):
    """
    Normalise un identifiant de participant ou d'item.

    Exemples :
        63873   -> "63873"
        63873.0 -> "63873"
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


def safe_float(value):
    """
    Convertit une valeur numérique en float exploitable.

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


def standard_deviation_from_variance(variance):
    """Calcule l'écart-type correspondant à une variance."""
    if pd.isna(variance) or variance < 0:
        return np.nan

    return float(
        np.sqrt(variance)
    )


# ======================================================================
# CHARGEMENT DES DONNÉES
# ======================================================================

def load_analysis_data():
    """
    Charge et valide le dataset analytique minimal.
    """
    print_section(
        "CHARGEMENT DES DONNÉES"
    )

    if not os.path.isfile(INPUT_FILE):
        raise FileNotFoundError(
            "Fichier analytique introuvable : "
            f"{INPUT_FILE}"
        )

    dataframe = pd.read_csv(
        INPUT_FILE
    )

    print(
        "Fichier :",
        INPUT_FILE,
    )

    print(
        "Nombre de lignes brutes :",
        len(dataframe),
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
            "Colonnes absentes du dataset analytique : "
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

    # Si la colonne existe, seules les lignes déclarées complètes
    # sont utilisées.
    if "analysis_complete" in dataframe.columns:
        complete_values = (
            dataframe["analysis_complete"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        dataframe["analysis_complete"] = (
            complete_values.isin({
                "true",
                "1",
                "1.0",
                "yes",
            })
        )

        before_filter = len(dataframe)

        dataframe = dataframe.loc[
            dataframe["analysis_complete"]
        ].copy()

        print(
            "Lignes retirées car analysis_complete=False :",
            before_filter - len(dataframe),
        )

    before_drop = len(dataframe)

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

    print(
        "Lignes supprimées pour donnée essentielle manquante :",
        before_drop - len(dataframe),
    )

    invalid_confidence = (
        (dataframe["confidence"] < 0)
        | (dataframe["confidence"] > 100)
    )

    if invalid_confidence.any():
        raise ValueError(
            "Certaines valeurs de confiance sont hors "
            "de l'intervalle [0, 100]."
        )

    if len(dataframe) < 2:
        raise ValueError(
            "Le dataset ne contient pas suffisamment "
            "d'observations."
        )

    number_of_subjects = (
        dataframe["subject_id"].nunique()
    )

    number_of_items = (
        dataframe["item_id"].nunique()
    )

    if number_of_subjects < 2:
        raise ValueError(
            "Le modèle nécessite au moins deux participants."
        )

    if number_of_items < 2:
        raise ValueError(
            "Le modèle nécessite au moins deux items."
        )

    # Statsmodels exige une variable de regroupement principale.
    # Pour représenter des effets croisés, toutes les observations
    # sont placées dans un groupe artificiel unique.
    #
    # Les effets participant et item sont ensuite définis dans
    # vc_formula comme composantes de variance.
    dataframe["_global_group"] = "all_observations"

    dataframe = (
        dataframe
        .sort_values(
            by=[
                "subject_id",
                "item_id",
            ]
        )
        .reset_index(drop=True)
    )

    print(
        "Nombre de lignes utilisées :",
        len(dataframe),
    )

    print(
        "Nombre de participants :",
        number_of_subjects,
    )

    print(
        "Nombre d'items :",
        number_of_items,
    )

    print(
        "Confiance moyenne :",
        round(
            dataframe["confidence"].mean(),
            6,
        ),
    )

    return dataframe


# ======================================================================
# CONSTRUCTION DU MODÈLE
# ======================================================================

def build_null_model(dataframe):
    """
    Construit le modèle mixte nul avec des intercepts aléatoires
    croisés pour les participants et les items.
    """
    variance_component_formulas = {
        "item":
            "0 + C(item_id)",

        "subject":
            "0 + C(subject_id)",
    }

    model = smf.mixedlm(
        formula="confidence ~ 1",
        data=dataframe,
        groups=dataframe["_global_group"],
        re_formula="0",
        vc_formula=variance_component_formulas,
        use_sparse=USE_SPARSE_MATRICES,
    )

    return model


# ======================================================================
# AJUSTEMENT DU MODÈLE
# ======================================================================

def fit_with_fallback(
    model,
    reml,
    estimation_label,
):
    """
    Ajuste un modèle en essayant successivement plusieurs
    optimiseurs.

    Le premier ajustement convergé est conservé.
    """
    print_section(
        f"AJUSTEMENT DU MODÈLE — {estimation_label}"
    )

    last_error = None
    last_result = None

    for method in OPTIMIZATION_METHODS:
        print(
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
                    method,
                )

                return result, method

        except Exception as error:
            last_error = error

            print(
                "Échec avec",
                method,
                ":",
                repr(error),
            )

    if last_result is not None:
        raise RuntimeError(
            "Un résultat a été obtenu, mais aucun optimiseur "
            "n'a signalé une convergence complète."
        )

    raise RuntimeError(
        "Impossible d'ajuster le modèle. "
        f"Dernière erreur : {last_error!r}"
    )


# ======================================================================
# COMPOSANTES DE VARIANCE
# ======================================================================

def get_variance_component_names(result):
    """
    Récupère les noms des composantes de variance dans l'ordre
    utilisé par statsmodels.
    """
    try:
        names = list(
            result.model.exog_vc.names
        )

    except (AttributeError, TypeError):
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

    Les proportions de variance sont également les ICC correspondant
    aux effets participant et item dans ce modèle nul croisé.
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
        str(name).strip().lower():
            float(value)

        for name, value in zip(
            component_names,
            component_values,
        )
    }

    subject_variance = np.nan
    item_variance = np.nan

    for name, value in component_map.items():
        if "subject" in name:
            subject_variance = value

        elif "item" in name:
            item_variance = value

    if (
        pd.isna(subject_variance)
        or pd.isna(item_variance)
    ):
        raise RuntimeError(
            "Impossible d'identifier automatiquement les "
            "variances participant et item. "
            f"Composantes trouvées : {component_map}"
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

    subject_proportion = (
        subject_variance
        / total_variance
    )

    item_proportion = (
        item_variance
        / total_variance
    )

    residual_proportion = (
        residual_variance
        / total_variance
    )

    clustered_proportion = (
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
                subject_proportion,
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
                item_proportion,
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

    additional_statistics = {
        "subject_icc":
            subject_proportion,

        "item_icc":
            item_proportion,

        "clustered_proportion":
            clustered_proportion,
    }

    return (
        variance_table,
        additional_statistics,
    )


# ======================================================================
# STATISTIQUES D'AJUSTEMENT
# ======================================================================

def count_estimated_parameters(result):
    """
    Retourne le nombre de paramètres utilisé pour le calcul
    des critères AIC et BIC.

    Dans ce modèle :
        - un effet fixe : l'interception ;
        - deux composantes de variance ;
        - la variance résiduelle.
    """
    try:
        return int(
            result.df_modelwc + 1
        )

    except (
        AttributeError,
        TypeError,
        ValueError,
    ):
        return np.nan


def create_fit_statistics_table(
    dataframe,
    reml_result,
    reml_optimizer,
    ml_result,
    ml_optimizer,
):
    """
    Construit un tableau synthétique des ajustements REML et ML.

    Les AIC et BIC ne sont conservés que pour ML, car les critères
    REML ne doivent pas être utilisés pour comparer des modèles ayant
    des structures d'effets fixes différentes.
    """
    number_of_observations = len(
        dataframe
    )

    number_of_subjects = (
        dataframe["subject_id"].nunique()
    )

    number_of_items = (
        dataframe["item_id"].nunique()
    )

    rows = []

    for (
        estimation,
        result,
        optimizer,
    ) in [
        (
            "REML",
            reml_result,
            reml_optimizer,
        ),
        (
            "ML",
            ml_result,
            ml_optimizer,
        ),
    ]:
        rows.append({
            "estimation":
                estimation,

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

            "intercept":
                safe_float(
                    result.fe_params[
                        "Intercept"
                    ]
                ),

            "intercept_standard_error":
                safe_float(
                    result.bse_fe[
                        "Intercept"
                    ]
                ),

            "log_likelihood":
                safe_float(
                    result.llf
                ),

            # Les AIC et BIC REML ne sont pas utilisés ici.
            "aic":
                (
                    safe_float(result.aic)
                    if estimation == "ML"
                    else np.nan
                ),

            "bic":
                (
                    safe_float(result.bic)
                    if estimation == "ML"
                    else np.nan
                ),

            "number_of_estimated_parameters":
                count_estimated_parameters(
                    result
                ),

            "residual_variance":
                safe_float(
                    result.scale
                ),
        })

    return pd.DataFrame(rows)


# ======================================================================
# SAUVEGARDE DU RÉSUMÉ REML
# ======================================================================

def save_reml_summary(
    result,
    variance_table,
    additional_statistics,
):
    """
    Sauvegarde le résumé statsmodels REML et une interprétation
    concise des composantes de variance.
    """
    participant_row = variance_table.loc[
        variance_table["component"]
        == "Participant"
    ].iloc[0]

    item_row = variance_table.loc[
        variance_table["component"]
        == "Item"
    ].iloc[0]

    residual_row = variance_table.loc[
        variance_table["component"]
        == "Residual"
    ].iloc[0]

    with open(
        REML_SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as output_file:
        output_file.write(
            "MODÈLE LINÉAIRE MIXTE NUL E1 — REML\n"
        )

        output_file.write(
            "=" * 80
        )

        output_file.write("\n\n")

        output_file.write(
            result.summary().as_text()
        )

        output_file.write("\n\n")

        output_file.write(
            "DÉCOMPOSITION DE LA VARIANCE\n"
        )

        output_file.write(
            "=" * 80
        )

        output_file.write("\n\n")

        output_file.write(
            "Participant\n"
        )

        output_file.write(
            f"  Variance : "
            f"{participant_row['variance']:.6f}\n"
        )

        output_file.write(
            f"  Écart-type : "
            f"{participant_row['standard_deviation']:.6f}\n"
        )

        output_file.write(
            f"  Proportion : "
            f"{participant_row['proportion_total_variance']:.6f}\n"
        )

        output_file.write("\n")

        output_file.write(
            "Item\n"
        )

        output_file.write(
            f"  Variance : "
            f"{item_row['variance']:.6f}\n"
        )

        output_file.write(
            f"  Écart-type : "
            f"{item_row['standard_deviation']:.6f}\n"
        )

        output_file.write(
            f"  Proportion : "
            f"{item_row['proportion_total_variance']:.6f}\n"
        )

        output_file.write("\n")

        output_file.write(
            "Résiduel\n"
        )

        output_file.write(
            f"  Variance : "
            f"{residual_row['variance']:.6f}\n"
        )

        output_file.write(
            f"  Écart-type : "
            f"{residual_row['standard_deviation']:.6f}\n"
        )

        output_file.write(
            f"  Proportion : "
            f"{residual_row['proportion_total_variance']:.6f}\n"
        )

        output_file.write("\n")

        output_file.write(
            f"ICC participant : "
            f"{additional_statistics['subject_icc']:.6f}\n"
        )

        output_file.write(
            f"ICC item : "
            f"{additional_statistics['item_icc']:.6f}\n"
        )

        output_file.write(
            f"Proportion totale participant + item : "
            f"{additional_statistics['clustered_proportion']:.6f}\n"
        )


# ======================================================================
# NETTOYAGE DES ANCIENNES SORTIES
# ======================================================================

def remove_obsolete_output_files():
    """
    Supprime les anciennes sorties qui ne sont plus produites par
    la version simplifiée du script.
    """
    obsolete_filenames = [
        "null_model_ML_summary.txt",
        "null_model_fixed_effects.csv",
        "null_model_predictions.csv",
        "null_model_subject_effects.csv",
        "null_model_item_effects.csv",
        "null_model_residuals_vs_fitted.png",
        "null_model_residual_distribution.png",
        "null_model_qqplot.png",
        "null_model_variance_decomposition.png",
        "null_model_subject_effects.png",
        "null_model_item_effects.png",
        "null_model_results.json",
        "null_model_report.txt",
    ]

    for filename in obsolete_filenames:
        path = os.path.join(
            OUTPUT_DIRECTORY,
            filename,
        )

        if os.path.isfile(path):
            os.remove(path)

            print(
                "Ancienne sortie supprimée :",
                path,
            )


# ======================================================================
# PROGRAMME PRINCIPAL
# ======================================================================

def main():
    print("=" * 80)
    print("MODÈLE LINÉAIRE MIXTE NUL — EXPÉRIENCE E1")
    print("=" * 80)

    remove_obsolete_output_files()

    try:
        # ==============================================================
        # 1. Chargement des données
        # ==============================================================

        dataframe = load_analysis_data()

        # ==============================================================
        # 2. Ajustement REML
        # ==============================================================

        reml_model = build_null_model(
            dataframe
        )

        (
            reml_result,
            reml_optimizer,
        ) = fit_with_fallback(
            model=reml_model,
            reml=True,
            estimation_label="REML",
        )

        # ==============================================================
        # 3. Ajustement ML
        # ==============================================================

        # Le modèle est reconstruit pour éviter de réutiliser un objet
        # potentiellement modifié par le premier ajustement.
        ml_model = build_null_model(
            dataframe
        )

        (
            ml_result,
            ml_optimizer,
        ) = fit_with_fallback(
            model=ml_model,
            reml=False,
            estimation_label="ML",
        )

        # ==============================================================
        # 4. Composantes de variance REML
        # ==============================================================

        (
            variance_table,
            additional_statistics,
        ) = extract_variance_components(
            reml_result
        )

        variance_table.to_csv(
            VARIANCE_COMPONENTS_FILE,
            index=False,
        )

        # ==============================================================
        # 5. Statistiques d'ajustement REML et ML
        # ==============================================================

        fit_statistics = (
            create_fit_statistics_table(
                dataframe=dataframe,
                reml_result=reml_result,
                reml_optimizer=
                    reml_optimizer,
                ml_result=ml_result,
                ml_optimizer=
                    ml_optimizer,
            )
        )

        fit_statistics.to_csv(
            FIT_STATISTICS_FILE,
            index=False,
        )

        # ==============================================================
        # 6. Résumé REML
        # ==============================================================

        save_reml_summary(
            result=reml_result,
            variance_table=
                variance_table,
            additional_statistics=
                additional_statistics,
        )

        # ==============================================================
        # 7. Affichage des résultats principaux
        # ==============================================================

        print_section(
            "RÉSULTATS PRINCIPAUX — REML"
        )

        print(
            reml_result
            .summary()
            .as_text()
        )

        print_section(
            "COMPOSANTES DE VARIANCE"
        )

        print(
            variance_table
            .to_string(
                index=False
            )
        )

        print("")

        print(
            "ICC participant :",
            round(
                additional_statistics[
                    "subject_icc"
                ],
                6,
            ),
        )

        print(
            "ICC item :",
            round(
                additional_statistics[
                    "item_icc"
                ],
                6,
            ),
        )

        print(
            "Proportion participant + item :",
            round(
                additional_statistics[
                    "clustered_proportion"
                ],
                6,
            ),
        )

        print_section(
            "STATISTIQUES D'AJUSTEMENT"
        )

        print(
            fit_statistics
            .to_string(
                index=False
            )
        )

        print_section(
            "FICHIERS PRODUITS"
        )

        print(
            REML_SUMMARY_FILE
        )

        print(
            VARIANCE_COMPONENTS_FILE
        )

        print(
            FIT_STATISTICS_FILE
        )

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
        print_section(
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
