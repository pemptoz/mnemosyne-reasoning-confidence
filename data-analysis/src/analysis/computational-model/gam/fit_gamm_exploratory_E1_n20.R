#!/usr/bin/env Rscript

# ============================================================================
# fit_gamm_exploratory_E1_n20.R
#
# Ajuste un GAMM exploratoire pour la confiance dans l'expérience E1.
#
# Effets fixes catégoriels :
#     condition
#     validity_binary
#
# Effets lisses :
#     sequence_c10
#     subject_accuracy_z
#     item_entropy_z
#     subject_mean_models_z
#     models_within_subject_z
#
# Effets aléatoires croisés :
#     intercept participant
#     intercept item
#
# Estimation :
#     REML
#
# Sorties :
#     gamm_exploratory_summary.txt
#     gamm_exploratory_variance_components.csv
# ============================================================================


# ============================================================================
# DÉPENDANCES
# ============================================================================

suppressPackageStartupMessages({
    library(mgcv)
})


# ============================================================================
# OUTILS DE CHEMIN
# ============================================================================



get_script_path <- function() {
    arguments <- commandArgs(
        trailingOnly = FALSE
    )

    file_argument <- grep(
        "^--file=",
        arguments,
        value = TRUE
    )

    if (length(file_argument) != 1) {
        stop(
            "Impossible de déterminer le chemin du script."
        )
    }

    normalizePath(
        sub(
            "^--file=",
            "",
            file_argument
        ),
        mustWork = TRUE
    )
}


script_file <- get_script_path()

script_dir <- dirname(
    script_file
)

# Le script se trouve dans :
# src/analysis/GAM/
#
# Il faut donc remonter de trois niveaux pour retrouver la racine.
project_root <- normalizePath(
    file.path(
        script_dir,
        "..",
        "..",
        "..",
        ".."
    ),
    mustWork = TRUE
)


# ============================================================================
# FICHIERS
# ============================================================================

data_file <- file.path(
    project_root,
    "results",
    "tables",
    "computational-model",
    "dataset_analysis_E1_n20.csv"
)

output_dir <- file.path(
    project_root,
    "results",
    "analysis",
    "computational-model",
    "gam",
    "gamm_exploratory_E1_n20"
)

dir.create(
    output_dir,
    recursive = TRUE,
    showWarnings = FALSE
)

summary_file <- file.path(
    output_dir,
    "gamm_exploratory_summary.txt"
)

variance_file <- file.path(
    output_dir,
    "gamm_exploratory_variance_components.csv"
)

smooth_plot_file <- file.path(
    output_dir,
    "gamm_exploratory_smooth_effects.png"
)

k_check_file <- file.path(
    output_dir,
    "gamm_exploratory_k_check.csv"
)

smooth_values_file <- file.path(
    output_dir,
    "gamm_exploratory_smooth_effects.csv"
)



# ============================================================================
# AFFICHAGE
# ============================================================================

section <- function(title) {
    cat("\n")
    cat(
        paste0(
            rep("=", 80),
            collapse = ""
        )
    )
    cat("\n")
    cat(title)
    cat("\n")
    cat(
        paste0(
            rep("=", 80),
            collapse = ""
        )
    )
    cat("\n")
}


# ============================================================================
# STANDARDISATION
# ============================================================================

standardize_variable <- function(values, variable_name) {
    numeric_values <- suppressWarnings(
        as.numeric(values)
    )

    variable_mean <- mean(
        numeric_values,
        na.rm = TRUE
    )

    variable_sd <- sd(
        numeric_values,
        na.rm = TRUE
    )

    if (
        !is.finite(variable_sd)
        || variable_sd <= 0
    ) {
        stop(
            paste0(
                "Impossible de standardiser ",
                variable_name,
                " : écart-type = ",
                variable_sd
            )
        )
    }

    standardized <- (
        numeric_values
        - variable_mean
    ) / variable_sd

    list(
        values = standardized,
        mean = variable_mean,
        standard_deviation = variable_sd
    )
}


# ============================================================================
# CHARGEMENT
# ============================================================================

section(
    "CHARGEMENT DES DONNÉES"
)

if (!file.exists(data_file)) {
    stop(
        paste(
            "Fichier introuvable :",
            data_file
        )
    )
}

data <- read.csv(
    data_file,
    stringsAsFactors = FALSE,
    check.names = FALSE
)

cat(
    "Fichier :",
    data_file,
    "\n"
)

cat(
    "Nombre de lignes brutes :",
    nrow(data),
    "\n"
)

cat(
    "Nombre de colonnes :",
    ncol(data),
    "\n"
)


# ============================================================================
# VÉRIFICATION DES COLONNES
# ============================================================================

required_columns <- c(
    "confidence",
    "condition",
    "sequence",
    "subject_id",
    "item_id",
    "subject_accuracy",
    "item_entropy",
    "subject_mean_models",
    "models_within_subject",
    "validity_binary"
)

missing_columns <- setdiff(
    required_columns,
    names(data)
)

if (length(missing_columns) > 0) {
    stop(
        paste(
            "Colonnes absentes :",
            paste(
                missing_columns,
                collapse = ", "
            )
        )
    )
}


# ============================================================================
# FILTRAGE DES LIGNES COMPLÈTES
# ============================================================================

section(
    "PRÉPARATION DES DONNÉES"
)

if ("analysis_complete" %in% names(data)) {
    complete_values <- tolower(
        trimws(
            as.character(
                data$analysis_complete
            )
        )
    )

    complete_mask <- complete_values %in% c(
        "true",
        "1",
        "1.0",
        "yes",
        "oui"
    )

    before_filter <- nrow(data)

    data <- data[
        complete_mask,
        ,
        drop = FALSE
    ]

    cat(
        "Lignes retirées car analysis_complete=False :",
        before_filter - nrow(data),
        "\n"
    )
}


# ============================================================================
# CONVERSIONS
# ============================================================================

numeric_columns <- c(
    "confidence",
    "sequence",
    "subject_accuracy",
    "item_entropy",
    "subject_mean_models",
    "models_within_subject",
    "validity_binary"
)

for (column in numeric_columns) {
    data[[column]] <- suppressWarnings(
        as.numeric(
            data[[column]]
        )
    )
}

data$condition <- trimws(
    as.character(
        data$condition
    )
)

data$subject_id <- factor(
    trimws(
        as.character(
            data$subject_id
        )
    )
)

data$item_id <- factor(
    trimws(
        as.character(
            data$item_id
        )
    )
)


# ============================================================================
# SUPPRESSION DES VALEURS MANQUANTES
# ============================================================================

before_complete_cases <- nrow(data)

complete_case_mask <- complete.cases(
    data[
        required_columns
    ]
)

data <- data[
    complete_case_mask,
    ,
    drop = FALSE
]

cat(
    "Lignes supprimées pour données essentielles manquantes :",
    before_complete_cases - nrow(data),
    "\n"
)


# ============================================================================
# VALIDATIONS
# ============================================================================

if (
    any(
        data$confidence < 0
        | data$confidence > 100
    )
) {
    stop(
        "Certaines valeurs de confidence sont hors de [0, 100]."
    )
}

observed_conditions <- sort(
    unique(
        data$condition
    )
)

expected_conditions <- c(
    "Neutral",
    "Standard"
)

if (
    !identical(
        observed_conditions,
        expected_conditions
    )
) {
    stop(
        paste(
            "Conditions observées inattendues :",
            paste(
                observed_conditions,
                collapse = ", "
            )
        )
    )
}

validity_values <- sort(
    unique(
        data$validity_binary
    )
)

if (
    !identical(
        validity_values,
        c(0, 1)
    )
) {
    stop(
        paste(
            "Valeurs de validity_binary inattendues :",
            paste(
                validity_values,
                collapse = ", "
            )
        )
    )
}


# ============================================================================
# FACTEURS EXPÉRIMENTAUX
# ============================================================================

data$condition <- factor(
    data$condition,
    levels = c(
        "Neutral",
        "Standard"
    )
)

data$validity <- factor(
    data$validity_binary,
    levels = c(
        0,
        1
    ),
    labels = c(
        "Invalid",
        "Valid"
    )
)


# ============================================================================
# CENTRAGE DE LA SÉQUENCE
# ============================================================================

sequence_mean <- mean(
    data$sequence
)

data$sequence_c10 <- (
    data$sequence
    - sequence_mean
) / 10


# ============================================================================
# STANDARDISATION DES PRÉDICTEURS
# ============================================================================

standardized_variables <- c(
    "subject_accuracy",
    "item_entropy",
    "subject_mean_models",
    "models_within_subject"
)

standardization_rows <- list()

for (
    variable_name
    in standardized_variables
) {
    result <- standardize_variable(
        data[[variable_name]],
        variable_name
    )

    standardized_name <- paste0(
        variable_name,
        "_z"
    )

    data[[standardized_name]] <- (
        result$values
    )

    standardization_rows[[
        length(standardization_rows) + 1
    ]] <- data.frame(
        variable = variable_name,
        standardized_variable = standardized_name,
        mean = result$mean,
        standard_deviation =
            result$standard_deviation,
        stringsAsFactors = FALSE
    )
}

standardization_table <- do.call(
    rbind,
    standardization_rows
)


# ============================================================================
# CONTRÔLES DES DONNÉES PRÉPARÉES
# ============================================================================

cat(
    "Nombre de lignes utilisées :",
    nrow(data),
    "\n"
)

cat(
    "Nombre de participants :",
    nlevels(data$subject_id),
    "\n"
)

cat(
    "Nombre d'items :",
    nlevels(data$item_id),
    "\n"
)

cat(
    "Moyenne de la séquence :",
    sequence_mean,
    "\n"
)

cat("\n")
cat("Paramètres de standardisation :\n")

print(
    standardization_table,
    row.names = FALSE
)


# ============================================================================
# FORMULE DU GAMM
# ============================================================================

gamm_formula <- (
    confidence
    ~ condition
    + validity
    + s(
        sequence_c10,
        k = 10,
        bs = "tp"
    )
    + s(
        subject_accuracy_z,
        k = 6,
        bs = "tp"
    )
    + s(
        item_entropy_z,
        k = 8,
        bs = "tp"
    )
    + s(
        subject_mean_models_z,
        k = 5,
        bs = "tp"
    )
    + s(
        models_within_subject_z,
        k = 5,
        bs = "tp"
    )
    + s(
        subject_id,
        bs = "re"
    )
    + s(
        item_id,
        bs = "re"
    )
)


# ============================================================================
# AJUSTEMENT
# ============================================================================

section(
    "AJUSTEMENT DU GAMM EXPLORATOIRE"
)

cat(
    "Formule :\n"
)

print(
    gamm_formula
)

cat(
    "\nMéthode d'estimation : REML\n"
)

gamm_result <- gam(
    formula = gamm_formula,
    data = data,
    family = gaussian(
        link = "identity"
    ),
    method = "REML",
    select = FALSE
)

cat(
    "Ajustement terminé.\n"
)


# ============================================================================
# EXPORT NUMÉRIQUE DES EFFETS LISSES
# ============================================================================

smooth_definitions <- list(
    list(
        term = "s(sequence_c10)",
        variable = "sequence_c10"
    ),
    list(
        term = "s(subject_accuracy_z)",
        variable = "subject_accuracy_z"
    ),
    list(
        term = "s(item_entropy_z)",
        variable = "item_entropy_z"
    ),
    list(
        term = "s(subject_mean_models_z)",
        variable = "subject_mean_models_z"
    ),
    list(
        term = "s(models_within_subject_z)",
        variable = "models_within_subject_z"
    )
)

reference_data <- data[
    rep(1, 100),
    ,
    drop = FALSE
]

reference_data$condition <- factor(
    "Neutral",
    levels = levels(data$condition)
)

reference_data$validity <- factor(
    "Invalid",
    levels = levels(data$validity)
)

reference_data$sequence_c10 <- 0
reference_data$subject_accuracy_z <- 0
reference_data$item_entropy_z <- 0
reference_data$subject_mean_models_z <- 0
reference_data$models_within_subject_z <- 0

smooth_rows <- list()

for (definition in smooth_definitions) {
    variable <- definition$variable
    term_name <- definition$term

    grid_values <- seq(
        min(data[[variable]], na.rm = TRUE),
        max(data[[variable]], na.rm = TRUE),
        length.out = 100
    )

    new_data <- reference_data
    new_data[[variable]] <- grid_values

    prediction <- predict(
        gamm_result,
        newdata = new_data,
        type = "terms",
        terms = term_name,
        se.fit = TRUE,
        unconditional = TRUE
    )

    effect <- as.numeric(
        prediction$fit[, 1]
    )

    standard_error <- as.numeric(
        prediction$se.fit[, 1]
    )

    smooth_rows[[
        length(smooth_rows) + 1
    ]] <- data.frame(
        predictor = variable,
        x = grid_values,
        partial_effect = effect,
        standard_error = standard_error,
        ci_95_lower = effect - 1.96 * standard_error,
        ci_95_upper = effect + 1.96 * standard_error
    )
}

smooth_values <- do.call(
    rbind,
    smooth_rows
)

write.csv(
    smooth_values,
    smooth_values_file,
    row.names = FALSE
)

cat(
    "Valeurs des courbes enregistrées :",
    smooth_values_file,
    "\n"
)



# ============================================================================
# RÉSUMÉ DANS LE TERMINAL
# ============================================================================

section(
    "RÉSUMÉ DU GAMM"
)

print(
    summary(
        gamm_result
    )
)

# ============================================================================
# GRAPHIQUES DES EFFETS LISSES
# ============================================================================

section(
    "GRAPHIQUES DES EFFETS LISSES"
)

png(
    filename = smooth_plot_file,
    width = 2400,
    height = 1800,
    res = 200
)

par(
    mfrow = c(3, 2),
    mar = c(5, 5, 4, 2)
)

# Les termes 1 à 5 correspondent aux cinq prédicteurs continus.
for (term_index in 1:5) {
    plot(
        gamm_result,
        select = term_index,
        shade = TRUE,
        shade.col = "lightblue",
        seWithMean = TRUE,
        residuals = FALSE,
        rug = TRUE,
        pages = 0,
        scale = 0
    )

    abline(
        h = 0,
        lty = 2,
        col = "gray40"
    )
}

plot.new()

dev.off()

cat(
    "Graphique des effets lisses enregistré :",
    smooth_plot_file,
    "\n"
)


# ============================================================================
# CONTRÔLE NUMÉRIQUE DE k
# ============================================================================

section(
    "CONTRÔLE DES DIMENSIONS DE BASE"
)

set.seed(12345)

k_check_result <- k.check(
    gamm_result,
    subsample = 5000,
    n.rep = 400
)

print(
    k_check_result
)

k_check_table <- data.frame(
    smooth = rownames(
        k_check_result
    ),
    k_check_result,
    row.names = NULL,
    check.names = FALSE
)

write.csv(
    k_check_table,
    k_check_file,
    row.names = FALSE
)

cat(
    "Contrôle de k enregistré :",
    k_check_file,
    "\n"
)


# ============================================================================
# SAUVEGARDE DU RÉSUMÉ
# ============================================================================

sink(
    summary_file
)

cat(
    "GAMM EXPLORATOIRE DE CONFIANCE — EXPÉRIENCE E1\n"
)

cat(
    paste0(
        rep("=", 80),
        collapse = ""
    )
)

cat("\n\n")

cat(
    "Fichier de données :\n",
    data_file,
    "\n\n"
)

cat(
    "Nombre d'observations : ",
    nrow(data),
    "\n",
    sep = ""
)

cat(
    "Nombre de participants : ",
    nlevels(data$subject_id),
    "\n",
    sep = ""
)

cat(
    "Nombre d'items : ",
    nlevels(data$item_id),
    "\n\n",
    sep = ""
)

cat(
    "Méthode d'estimation : REML\n"
)

cat(
    "Famille : gaussienne\n"
)

cat(
    "Lien : identité\n\n"
)

cat(
    "FORMULE\n"
)

cat(
    paste0(
        rep("-", 80),
        collapse = ""
    )
)

cat("\n")

print(
    gamm_formula
)

cat("\n")
cat("RÉSUMÉ DU MODÈLE\n")

cat(
    paste0(
        rep("-", 80),
        collapse = ""
    )
)

cat("\n")

print(
    summary(
        gamm_result
    )
)

cat("\n")
cat("COMPOSANTES DE VARIANCE\n")

cat(
    paste0(
        rep("-", 80),
        collapse = ""
    )
)

cat("\n")

print(
    variance_components
)

cat("\n")
cat("PARAMÈTRES DE STANDARDISATION\n")

cat(
    paste0(
        rep("-", 80),
        collapse = ""
    )
)

cat("\n")

print(
    standardization_table,
    row.names = FALSE
)

cat(
    smooth_plot_file,
    "\n"
)

cat(
    k_check_file,
    "\n"
)



sink()


# ============================================================================
# FIN
# ============================================================================

section(
    "TERMINÉ"
)

cat(
    "Premier GAMM exploratoire ajusté.\n"
)

cat(
    "Fichiers produits :\n"
)

cat(
    summary_file,
    "\n"
)

cat(
    variance_file,
    "\n"
)
