import copy
import hashlib
import itertools
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import ccobra
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent

# .../data-analysis/src/models
MODELS_DIR = SCRIPT_DIR.parent

# .../data-analysis
PROJECT_ROOT = MODELS_DIR.parent.parent

# Permet d'importer :
# .../data-analysis/src/models/mreasoner/
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(MODELS_DIR),
    )

import mreasoner

def resolve_project_path(path_value):
        """
        Résout un chemin par rapport à la racine du projet.

        Les chemins absolus sont conservés.
        Les chemins relatifs sont interprétés depuis PROJECT_ROOT.
        """
        path = Path(
            os.path.expanduser(
                str(path_value)
            )
        )

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        return str(
            path.resolve()
        )


class CCobraMReasoner(ccobra.CCobraModel):
    """
    Version individualisée de mReasoner pour le domaine conditional.

    Pour chaque participant, le modèle recherche la meilleure combinaison
    des quatre paramètres :

        epsilon
        lambda
        omega
        sigma

    Les résultats de mReasoner sont mis en cache afin d'éviter de refaire
    les mêmes appels Lisp pour chaque participant.

    Règle de conversion en réponse binaire :

        uniquement NVC / sortie vide -> No
        au moins une conclusion valide -> Yes

    Exemple :

        ["NVC"]          -> No
        ["Aac"]          -> Yes
        ["Aac", "NVC"]   -> Yes
    """

    PARAMETER_NAMES = (
        "epsilon",
        "lambda",
        "omega",
        "sigma",
    )

    CACHE_VERSION = 2

    # ==================================================================
    # Initialisation
    # ==================================================================

    def __init__(
        self,
        name="mReasoner",
        fit_its=2,
        n_samples=5,
        cache_file="mreasoner_conditional_cache_E1.npz",
        log_file="full_log_E1.json",
        random_tie_break=False,
        random_seed=42,
        **kwargs,
    ):
        super().__init__(
            name,
            ["conditional"],
            ["single-choice"],
        )

        # --------------------------------------------------------------
        # Chemins du projet
        # --------------------------------------------------------------
        self.ccl_dir = PROJECT_ROOT / ".ccl"
        self.mreasoner_source_dir = PROJECT_ROOT / ".mreasoner"

        # --------------------------------------------------------------
        # Configuration
        # --------------------------------------------------------------
        self.fit_its = int(fit_its)
        self.n_samples = int(n_samples)

        if self.fit_its < 2:
            raise ValueError(
                "fit_its doit être supérieur ou égal à 2."
            )

        if self.n_samples < 1:
            raise ValueError(
                "n_samples doit être supérieur ou égal à 1."
            )

        self.cache_file = resolve_project_path(
            cache_file
        )

        self.log_file = resolve_project_path(
            log_file
        )

        self.random_tie_break = bool(
            random_tie_break
        )

        self.random_seed = int(
            random_seed
        )

        self.rng = np.random.default_rng(
            self.random_seed
        )

        # --------------------------------------------------------------
        # Grille des paramètres
        # --------------------------------------------------------------
        self.param_values = (
            self._create_parameter_values()
        )

        self.param_matrix = (
            self._create_parameter_matrix()
        )

        self.n_param_combinations = (
            self.param_matrix.shape[0]
        )

        # --------------------------------------------------------------
        # Historiques
        # --------------------------------------------------------------
        self.history = {}
        self.pre_train_history = {}

        # --------------------------------------------------------------
        # Paramètres actuellement sélectionnés
        # --------------------------------------------------------------
        self.best_score = None

        self.best_indices = np.array(
            [],
            dtype=np.int64,
        )

        self.best_param_dicts = []

        self.selected_index = (
            self._default_parameter_index()
        )

        self.params = self._params_from_index(
            self.selected_index
        )

        # --------------------------------------------------------------
        # Cache
        # --------------------------------------------------------------
        self.task_keys = []
        self.task_premises = {}
        self.task_index = {}

        self.prediction_cache = np.zeros(
            (
                self.n_param_combinations,
                0,
            ),
            dtype=self._cache_dtype(),
        )

        self._load_cache()

        # --------------------------------------------------------------
        # État CCOBRA
        # --------------------------------------------------------------
        self.current_subject_id = None
        self.start_time = None
        self._log_written = False

        # Le processus Lisp est démarré uniquement si nécessaire.
        self.mr = None

        print(
            "DEBUG: grille mReasoner initialisée : "
            f"{self.n_param_combinations} configurations, "
            f"{len(self.task_keys)} tâche(s) en cache, "
            f"{self.n_samples} échantillon(s) par configuration."
        )

    # ==================================================================
    # Grille des paramètres
    # ==================================================================

    def _create_parameter_values(self):
        """
        Construit les valeurs testées pour chacun des quatre paramètres.
        """
        fallback_bounds = {
            "epsilon": (0.0, 1.0),
            "lambda": (0.1, 8.0),
            "omega": (0.0, 1.0),
            "sigma": (0.0, 1.0),
        }

        bounds = getattr(
            mreasoner,
            "PARAM_BOUNDS",
            None,
        )

        values = {}

        for parameter_index, parameter_name in enumerate(
            self.PARAMETER_NAMES
        ):
            if bounds is None:
                lower_bound, upper_bound = (
                    fallback_bounds[parameter_name]
                )

            elif isinstance(bounds, dict):
                lower_bound, upper_bound = (
                    bounds[parameter_name]
                )

            else:
                lower_bound, upper_bound = (
                    bounds[parameter_index]
                )

            values[parameter_name] = np.linspace(
                float(lower_bound),
                float(upper_bound),
                self.fit_its,
                dtype=np.float64,
            )

        return values

    def _create_parameter_matrix(self):
        """
        Produit une matrice de forme :

            nombre_configurations × 4

        Chaque ligne correspond à une combinaison de paramètres.
        """
        combinations = itertools.product(
            self.param_values["epsilon"],
            self.param_values["lambda"],
            self.param_values["omega"],
            self.param_values["sigma"],
        )

        return np.asarray(
            list(combinations),
            dtype=np.float64,
        )

    def _default_parameter_index(self):
        """
        Retourne l'index de la configuration la plus proche des
        paramètres par défaut de mReasoner.
        """
        default_params = getattr(
            mreasoner,
            "DEFAULT_PARAMS",
            {
                "epsilon": 1.0,
                "lambda": 4.0,
                "omega": 1.0,
                "sigma": 1.0,
            },
        )

        default_vector = np.asarray([
            float(default_params[name])
            for name in self.PARAMETER_NAMES
        ], dtype=np.float64)

        distances = np.sum(
            np.square(
                self.param_matrix - default_vector
            ),
            axis=1,
        )

        return int(
            np.argmin(distances)
        )

    def _params_from_index(
        self,
        configuration_index,
    ):
        """
        Convertit une ligne de la grille en dictionnaire.
        """
        row = self.param_matrix[
            int(configuration_index)
        ]

        return {
            parameter_name: float(row[index])
            for index, parameter_name
            in enumerate(self.PARAMETER_NAMES)
        }

    # ==================================================================
    # Processus Lisp
    # ==================================================================

    def _initialize_mreasoner(self):
        """Initialise mReasoner uniquement lorsque cela est nécessaire."""
        if self.mr is not None:
            return

        print("DEBUG: [ccobra] Initialisation de mReasoner...")

        clozure = mreasoner.ClozureCL(
            ccl_dir=str(self.ccl_dir)
        )

        ccl_path = clozure.exec_path()

        if ccl_path is None:
            raise FileNotFoundError(
                f"Exécutable Clozure CL introuvable dans {self.ccl_dir}"
            )

        mreasoner_dir = mreasoner.source_path(
            mreas_path=str(self.mreasoner_source_dir)
        )

        self.mr = mreasoner.MReasoner(
            ccl_path,
            mreasoner_dir,
        )

        print(
            "DEBUG: [ccobra] mReasoner initialisé : "
            f"CCL={ccl_path}, sources={mreasoner_dir}"
        )


    def _terminate_mreasoner(self):
        """
        Ferme proprement le processus mReasoner.
        """
        mr = getattr(
            self,
            "mr",
            None,
        )

        if mr is None:
            return

        try:
            terminate = getattr(
                mr,
                "terminate",
                None,
            )

            if callable(terminate):
                terminate()

        except Exception as error:
            print(
                "WARNING: impossible de fermer mReasoner : "
                f"{error!r}"
            )

        finally:
            self.mr = None

    def __getstate__(self):
        """
        Empêche CCOBRA de copier le processus Lisp.
        """
        state = self.__dict__.copy()

        state["mr"] = None
        state.pop("rng", None)

        return state

    def __setstate__(self, state):
        """
        Restaure une copie individuelle du modèle.

        Chaque copie recharge le cache depuis le disque.
        """
        self.__dict__.update(state)

        self.mr = None

        self.rng = np.random.default_rng(
            self.random_seed
        )

        self.task_keys = []
        self.task_premises = {}
        self.task_index = {}

        self.prediction_cache = np.zeros(
            (
                self.n_param_combinations,
                0,
            ),
            dtype=self._cache_dtype(),
        )

        self._load_cache()

        print(
            "DEBUG: copie CCOBRA restaurée : "
            f"{len(self.task_keys)} tâche(s) chargée(s)."
        )

    # ==================================================================
    # Normalisation des tâches
    # ==================================================================

    @staticmethod
    def _normalize_premises(item):
        """
        Transforme item.task en tuple de deux chaînes.
        """
        raw_task = item.task

        if isinstance(raw_task, np.ndarray):
            raw_task = raw_task.tolist()

        if isinstance(raw_task, (list, tuple)):
            premises = []

            for premise in raw_task:
                if isinstance(premise, np.ndarray):
                    premise = premise.tolist()

                # Retire les niveaux d'imbrication unitaires.
                while (
                    isinstance(premise, (list, tuple))
                    and len(premise) == 1
                ):
                    premise = premise[0]

                if isinstance(premise, (list, tuple)):
                    premise = " ".join(
                        str(part)
                        for part in premise
                    )

                premise = str(premise).strip()

                if premise:
                    premises.append(premise)

        else:
            premises = [
                part.strip()
                for part in str(raw_task).split("/")
                if part.strip()
            ]



        if len(premises) != 2:
            raise ValueError(
                "MReasoner.query() attend exactement deux prémisses, "
                f"mais {len(premises)} ont été trouvées : "
                f"{premises!r}"
            )

        return tuple(premises)

    @staticmethod
    def _task_key(premises):
        """
        Produit une clé stable pour une paire de prémisses.
        """
        serialized = json.dumps(
            list(premises),
            ensure_ascii=False,
            separators=(",", ":"),
        )

        return hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest()

    # ==================================================================
    # Normalisation des réponses humaines
    # ==================================================================

    @staticmethod
    def _normalize_truth(truth):
        """
        Convertit la réponse du participant en :

            1 = Yes
            0 = No

        Gère les formes imbriquées comme [['Yes']] et [['No']].
        """
        while isinstance(
            truth,
            (list, tuple, np.ndarray),
        ):
            if np.size(truth) != 1:
                raise ValueError(
                    "Réponse humaine ambiguë : "
                    f"{truth!r}"
                )

            if isinstance(truth, np.ndarray):
                truth = truth.reshape(-1)[0]
            else:
                truth = truth[0]

        if isinstance(truth, np.generic):
            truth = truth.item()

        if isinstance(truth, (bool, np.bool_)):
            return int(truth)

        if isinstance(
            truth,
            (int, float, np.integer, np.floating),
        ):
            if truth == 1:
                return 1

            if truth == 0:
                return 0

        normalized = str(
            truth
        ).strip().lower()

        if normalized in {
            "yes",
            "y",
            "true",
            "1",
            "oui",
        }:
            return 1

        if normalized in {
            "no",
            "n",
            "false",
            "0",
            "non",
        }:
            return 0

        raise ValueError(
            f"Réponse non reconnue : {truth!r}. "
            f"Valeur normalisée : {normalized!r}."
        )

    # ==================================================================
    # Conversion des conclusions de mReasoner
    # ==================================================================

    @staticmethod
    def _normalize_prediction_token(value):
        """
        Nettoie une conclusion produite par mReasoner.
        """
        normalized = str(value).strip()

        changed = True

        while changed and len(normalized) >= 2:
            changed = False

            for left, right in (
                ('"', '"'),
                ("'", "'"),
                ("[", "]"),
                ("(", ")"),
                ("{", "}"),
            ):
                if (
                    normalized.startswith(left)
                    and normalized.endswith(right)
                ):
                    normalized = (
                        normalized[1:-1].strip()
                    )

                    changed = True
                    break

        return normalized

    @classmethod
    def _flatten_predictions(
        cls,
        predictions,
    ):
        """
        Transforme récursivement la sortie en liste plate.
        """
        if predictions is None:
            return []

        if isinstance(predictions, np.ndarray):
            predictions = predictions.tolist()

        if isinstance(predictions, np.generic):
            return [
                predictions.item()
            ]

        if isinstance(predictions, dict):
            flattened = []

            for value in predictions.values():
                flattened.extend(
                    cls._flatten_predictions(value)
                )

            return flattened

        if isinstance(
            predictions,
            (list, tuple, set),
        ):
            flattened = []

            for value in predictions:
                flattened.extend(
                    cls._flatten_predictions(value)
                )

            return flattened

        return [predictions]

    @classmethod
    def _valid_conclusions(
        cls,
        predictions,
    ):
        """
        Retourne uniquement les conclusions différentes de NVC.

        Exemples :

            ["NVC"]          -> []
            ["Aac"]          -> ["AAC"]
            ["Aac", "NVC"]   -> ["AAC"]
        """
        flattened = cls._flatten_predictions(
            predictions
        )

        nvc_values = {
            "",
            "NVC",
            "NO VALID CONCLUSION",
            "NO-VALID-CONCLUSION",
            "NO_VALID_CONCLUSION",
            "NOTHING FOLLOWS",
            "NOTHING-FOLLOWS",
            "NOTHING_FOLLOWS",
            "NONE",
            "NIL",
            "NULL",
            "()",
        }

        valid = []

        for value in flattened:
            normalized = (
                cls._normalize_prediction_token(
                    value
                ).upper()
            )

            if normalized not in nvc_values:
                valid.append(normalized)

        return valid

    @classmethod
    def _is_nvc_prediction(
        cls,
        predictions,
    ):
        """
        Retourne True s'il n'existe aucune conclusion valide.
        """
        return (
            len(cls._valid_conclusions(predictions))
            == 0
        )

    @classmethod
    def _prediction_to_binary(
        cls,
        predictions,
    ):
        """
        Convertit une sortie complète de mReasoner en une seule réponse.

            [] ou ["NVC"]      -> 0 / No
            ["Aac"]            -> 1 / Yes
            ["Aac", "NVC"]     -> 1 / Yes

        Une sortie mixte n'est donc pas simultanément Yes et No.
        NVC est ignoré si une conclusion valide existe.
        """
        valid_conclusions = (
            cls._valid_conclusions(predictions)
        )

        return int(
            len(valid_conclusions) > 0
        )

    # ==================================================================
    # Gestion du cache
    # ==================================================================

    def _cache_dtype(self):
        """
        Choisit un type entier adapté à n_samples.
        """
        if self.n_samples <= np.iinfo(
            np.uint8
        ).max:
            return np.uint8

        if self.n_samples <= np.iinfo(
            np.uint16
        ).max:
            return np.uint16

        return np.uint32

    def _grid_signature(self):
        """
        Produit une signature du cache.

        CACHE_VERSION empêche le chargement de l'ancien cache qui
        traitait incorrectement ["NVC"] comme Yes.
        """
        payload = {
            "cache_version": self.CACHE_VERSION,
            "prediction_rule":
                "yes_if_at_least_one_non_nvc_conclusion",
            "fit_its": self.fit_its,
            "n_samples": self.n_samples,
            "parameters": {
                name: [
                    float(value)
                    for value
                    in self.param_values[name]
                ]
                for name in self.PARAMETER_NAMES
            },
        }

        serialized = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        )

        return hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest()

    def _reset_memory_cache(self):
        """
        Vide uniquement le cache en mémoire.
        """
        self.task_keys = []
        self.task_premises = {}
        self.task_index = {}

        self.prediction_cache = np.zeros(
            (
                self.n_param_combinations,
                0,
            ),
            dtype=self._cache_dtype(),
        )

    def _reload_cache_from_disk(self):
        """
        Recharge la dernière version du cache.
        """
        if not os.path.isfile(self.cache_file):
            return

        self._reset_memory_cache()
        self._load_cache()

    def _load_cache(self):
        """
        Charge le cache s'il correspond à la configuration actuelle.
        """
        if not os.path.isfile(self.cache_file):
            return

        try:
            with np.load(
                self.cache_file,
                allow_pickle=False,
            ) as cache_data:
                signature = str(
                    cache_data["signature"].item()
                )

                if signature != self._grid_signature():
                    print(
                        "WARNING: cache incompatible ignoré."
                    )
                    return

                task_keys = (
                    cache_data["task_keys"]
                    .astype(str)
                    .tolist()
                )

                premises_json = (
                    cache_data["premises_json"]
                    .astype(str)
                    .tolist()
                )

                predictions = (
                    cache_data["predictions"]
                )

                expected_shape = (
                    self.n_param_combinations,
                    len(task_keys),
                )

                if predictions.shape != expected_shape:
                    print(
                        "WARNING: dimensions du cache "
                        f"incorrectes : {predictions.shape}, "
                        f"attendu : {expected_shape}."
                    )
                    return

                self.task_keys = task_keys

                self.task_premises = {
                    key: tuple(
                        json.loads(serialized_premises)
                    )
                    for key, serialized_premises
                    in zip(task_keys, premises_json)
                }

                self.task_index = {
                    key: index
                    for index, key
                    in enumerate(task_keys)
                }

                self.prediction_cache = (
                    predictions.astype(
                        self._cache_dtype(),
                        copy=False,
                    )
                )

                print(
                    "DEBUG: cache chargé depuis "
                    f"{self.cache_file} : "
                    f"{len(self.task_keys)} tâche(s)."
                )

        except Exception as error:
            print(
                "WARNING: impossible de charger le cache : "
                f"{error!r}"
            )

    def _save_cache(self):
        """
        Sauvegarde atomiquement le cache.
        """
        cache_directory = (
            os.path.dirname(self.cache_file)
            or "."
        )

        os.makedirs(
            cache_directory,
            exist_ok=True,
        )

        premises_json = np.asarray(
            [
                json.dumps(
                    list(self.task_premises[key]),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                for key in self.task_keys
            ],
            dtype=np.str_,
        )

        temporary_file = None

        try:
            file_descriptor, temporary_file = (
                tempfile.mkstemp(
                    prefix=".mreasoner-cache-",
                    suffix=".npz",
                    dir=cache_directory,
                )
            )

            os.close(file_descriptor)

            np.savez_compressed(
                temporary_file,
                signature=np.asarray(
                    self._grid_signature()
                ),
                task_keys=np.asarray(
                    self.task_keys,
                    dtype=np.str_,
                ),
                premises_json=premises_json,
                predictions=self.prediction_cache,
            )

            os.replace(
                temporary_file,
                self.cache_file,
            )

            temporary_file = None

            print(
                "DEBUG: cache sauvegardé dans "
                f"{self.cache_file}."
            )

        finally:
            if (
                temporary_file
                and os.path.exists(temporary_file)
            ):
                os.remove(temporary_file)

    def _print_cache_diagnostic(self):
        """
        Affiche un résumé du contenu du cache.
        """
        if self.prediction_cache.shape[1] == 0:
            return

        unique_profiles = np.unique(
            self.prediction_cache,
            axis=0,
        )

        print(
            "DEBUG: profils de prédiction distincts : "
            f"{len(unique_profiles)} / "
            f"{self.n_param_combinations}"
        )

        for task_column in range(
            self.prediction_cache.shape[1]
        ):
            values, counts = np.unique(
                self.prediction_cache[
                    :,
                    task_column,
                ],
                return_counts=True,
            )

            distribution = {
                int(value): int(count)
                for value, count
                in zip(values, counts)
            }

            task_key = self.task_keys[
                task_column
            ]

            print(
                f"DEBUG: tâche {task_column} "
                f"{self.task_premises[task_key]!r} : "
                f"{distribution}"
            )

        if len(unique_profiles) == 1:
            print(
                "WARNING: toutes les configurations "
                "produisent le même profil."
            )

    def _ensure_tasks_cached(self, tasks):
        """
        Calcule uniquement les tâches absentes du cache.
        """
        normalized_tasks = [
            tuple(premises)
            for premises in tasks
        ]

        missing_tasks = {
            self._task_key(premises): premises
            for premises in normalized_tasks
            if self._task_key(premises)
            not in self.task_index
        }

        if not missing_tasks:
            return

        # Une autre copie CCOBRA a peut-être rempli le cache.
        self._reload_cache_from_disk()

        missing_tasks = {
            self._task_key(premises): premises
            for premises in normalized_tasks
            if self._task_key(premises)
            not in self.task_index
        }

        if not missing_tasks:
            print(
                "DEBUG: tâches récupérées depuis "
                "le cache disque."
            )
            return

        print(
            f"DEBUG: {len(missing_tasks)} nouvelle(s) "
            "tâche(s) à calculer."
        )

        self._initialize_mreasoner()

        old_task_count = len(
            self.task_keys
        )

        new_items = list(
            missing_tasks.items()
        )

        total_task_count = (
            old_task_count + len(new_items)
        )

        expanded_cache = np.zeros(
            (
                self.n_param_combinations,
                total_task_count,
            ),
            dtype=self._cache_dtype(),
        )

        if old_task_count:
            expanded_cache[
                :,
                :old_task_count,
            ] = self.prediction_cache

        total_queries = (
            self.n_param_combinations
            * len(new_items)
            * self.n_samples
        )

        print(
            "DEBUG: génération du cache : "
            f"{self.n_param_combinations} configurations × "
            f"{len(new_items)} tâche(s) × "
            f"{self.n_samples} échantillon(s) = "
            f"{total_queries} requêtes Lisp."
        )

        progress_step = max(
            1,
            self.n_param_combinations // 100,
        )

        start_time = time.time()

        for configuration_index, parameter_row in enumerate(
            self.param_matrix
        ):
            param_dict = {
                parameter_name: float(
                    parameter_row[index]
                )
                for index, parameter_name
                in enumerate(self.PARAMETER_NAMES)
            }

            for new_task_offset, (
                _,
                premises,
            ) in enumerate(new_items):
                yes_count = 0

                for _ in range(self.n_samples):
                    predictions = self.mr.query(
                        list(premises),
                        param_dict=param_dict,
                    )

                    # Une sortie entière correspond à une seule
                    # décision Yes ou No.
                    yes_count += (
                        self._prediction_to_binary(
                            predictions
                        )
                    )

                cache_column = (
                    old_task_count
                    + new_task_offset
                )

                expanded_cache[
                    configuration_index,
                    cache_column,
                ] = yes_count

            if (
                configuration_index % progress_step == 0
                or configuration_index
                == self.n_param_combinations - 1
            ):
                percentage = (
                    100.0
                    * (configuration_index + 1)
                    / self.n_param_combinations
                )

                elapsed = (
                    time.time() - start_time
                )

                print(
                    f"\rDEBUG: cache {percentage:6.2f}% "
                    f"({elapsed:.1f}s)",
                    end="",
                    flush=True,
                )

        print()

        self.prediction_cache = expanded_cache

        for key, premises in new_items:
            self.task_index[key] = len(
                self.task_keys
            )

            self.task_keys.append(key)
            self.task_premises[key] = tuple(
                premises
            )

        self._print_cache_diagnostic()
        self._save_cache()

    # ==================================================================
    # Historique des observations
    # ==================================================================

    def _add_observation(
        self,
        history,
        item,
        truth,
    ):
        premises = self._normalize_premises(
            item
        )

        key = self._task_key(
            premises
        )

        encoded_truth = self._normalize_truth(
            truth
        )

        if key not in history:
            history[key] = {
                "premises": premises,
                "yes": 0,
                "no": 0,
            }

        if encoded_truth == 1:
            history[key]["yes"] += 1
        else:
            history[key]["no"] += 1

    def _merged_history(self):
        """
        Fusionne le pré-entraînement général et individuel.
        """
        merged = copy.deepcopy(
            self.pre_train_history
        )

        for key, participant_data in self.history.items():
            merged[key] = copy.deepcopy(
                participant_data
            )

        return merged

    # ==================================================================
    # Recherche des meilleurs paramètres
    # ==================================================================

    def fit(self):
        """
        Calcule le score de toutes les configurations avec NumPy.
        """
        train_history = (
            self._merged_history()
        )

        if not train_history:
            self.best_score = None

            self.best_indices = np.asarray(
                [self.selected_index],
                dtype=np.int64,
            )

            self.best_param_dicts = [
                self._params_from_index(
                    self.selected_index
                )
            ]

            return

        all_premises = [
            task_data["premises"]
            for task_data
            in train_history.values()
        ]

        self._ensure_tasks_cached(
            all_premises
        )

        task_columns = np.fromiter(
            (
                self.task_index[key]
                for key in train_history
            ),
            dtype=np.int64,
            count=len(train_history),
        )

        yes_probabilities = (
            self.prediction_cache[
                :,
                task_columns,
            ].astype(
                np.float64,
                copy=False,
            )
            / float(self.n_samples)
        )

        yes_counts = np.fromiter(
            (
                task_data["yes"]
                for task_data
                in train_history.values()
            ),
            dtype=np.float64,
            count=len(train_history),
        )

        no_counts = np.fromiter(
            (
                task_data["no"]
                for task_data
                in train_history.values()
            ),
            dtype=np.float64,
            count=len(train_history),
        )

        scores = (
            yes_probabilities @ yes_counts
            + (
                1.0 - yes_probabilities
            ) @ no_counts
        )

        best_score = float(
            np.max(scores)
        )

        best_indices = np.flatnonzero(
            np.isclose(
                scores,
                best_score,
                rtol=1e-12,
                atol=1e-12,
            )
        )

        if best_indices.size == 0:
            raise RuntimeError(
                "Aucune configuration optimale trouvée."
            )

        if self.random_tie_break:
            selected_index = int(
                self.rng.choice(best_indices)
            )
        else:
            best_rows = self.param_matrix[
                best_indices
            ]

            centroid = np.mean(
                best_rows,
                axis=0,
            )

            distances = np.sum(
                np.square(
                    best_rows - centroid
                ),
                axis=1,
            )

            selected_index = int(
                best_indices[
                    np.argmin(distances)
                ]
            )



        self.best_score = best_score
        self.best_indices = best_indices
        self.selected_index = selected_index

        self.params = self._params_from_index(
            selected_index
        )

        self.best_param_dicts = [
            self._params_from_index(index)
            for index in best_indices
        ]

    # ==================================================================
    # Cycle CCOBRA
    # ==================================================================

    def pre_train(
        self,
        dataset,
        **kwargs,
    ):
        """
        Pré-entraînement collectif éventuel.
        """
        if not dataset:
            return

        self.pre_train_history = {}
        tasks_to_cache = []

        for participant_data in dataset:
            for trial_data in participant_data:
                item = trial_data["item"]
                response = trial_data["response"]

                self._add_observation(
                    self.pre_train_history,
                    item,
                    response,
                )

                tasks_to_cache.append(
                    self._normalize_premises(item)
                )

        unique_tasks = list(
            dict.fromkeys(tasks_to_cache)
        )

        self._ensure_tasks_cached(
            unique_tasks
        )

        self.fit()

    def pre_train_person(
        self,
        dataset,
        **kwargs,
    ):
        """
        Ajoute les données d'entraînement du participant.
        """
        if not dataset:
            return

        tasks_to_cache = []

        for trial_data in dataset:
            item = trial_data["item"]
            response = trial_data["response"]

            self._add_observation(
                self.history,
                item,
                response,
            )

            tasks_to_cache.append(
                self._normalize_premises(item)
            )

        unique_tasks = list(
            dict.fromkeys(tasks_to_cache)
        )

        self._ensure_tasks_cached(
            unique_tasks
        )

        self.fit()

    def start_participant(
        self,
        subj_id=None,
        **kwargs,
    ):
        """
        Démarre l'évaluation d'un participant.

        Attention : on ne vide pas self.history ici, car certaines
        versions de CCOBRA appellent pre_train_person() avant
        start_participant().
        """
        self.current_subject_id = subj_id
        self.start_time = time.time()
        self._log_written = False

        self._reload_cache_from_disk()
        self.fit()

    def predict(
        self,
        item,
        **kwargs,
    ):
        """
        Prédit Yes ou No avec la configuration actuellement sélectionnée.
        """
        premises = self._normalize_premises(
            item
        )

        key = self._task_key(
            premises
        )

        self._ensure_tasks_cached(
            [premises]
        )

        task_column = self.task_index[
            key
        ]

        yes_count = int(
            self.prediction_cache[
                self.selected_index,
                task_column,
            ]
        )

        yes_probability = (
            yes_count / float(self.n_samples)
        )

        if yes_probability > 0.5:
            return "Yes"

        if yes_probability < 0.5:
            return "No"

        if self.random_tie_break:
            return str(
                self.rng.choice(
                    ["Yes", "No"]
                )
            )

        # Choix déterministe en cas d'égalité.
        return "Yes"

    def adapt(
        self,
        item,
        truth,
        **kwargs,
    ):
        """
        Adapte les paramètres après chaque nouvelle réponse.
        """
        self._add_observation(
            self.history,
            item,
            truth,
        )

        self.fit()

    def end_participant(
        self,
        subj_id=None,
        model_log=None,
        **kwargs,
    ):
        """
        Enregistre les paramètres finaux du participant.
        """
        if subj_id is None:
            subj_id = (
                self.current_subject_id
            )

        if subj_id is None:
            subj_id = "unknown"

        self.fit()

        participant_log = {
            parameter_name: float(
                self.params[parameter_name]
            )
            for parameter_name
            in self.PARAMETER_NAMES
        }

        participant_log["best_params"] = [
            {
                parameter_name: float(
                    param_dict[parameter_name]
                )
                for parameter_name
                in self.PARAMETER_NAMES
            }
            for param_dict
            in self.best_param_dicts
        ]

        if model_log is not None:
            model_log.update(
                participant_log
            )

        self._write_full_log(
            subject_id=str(subj_id),
            participant_log=participant_log,
        )

        elapsed = (
            time.time() - self.start_time
            if self.start_time is not None
            else 0.0
        )

        print(
            "End Participant "
            f"({elapsed:.2f}s, "
            f"{self.fit_its} valeurs/paramètre) "
            f"id={subj_id} "
            f"params={self.params} "
            f"score={self.best_score} "
            f"ex_aequo={len(self.best_param_dicts)}"
        )

        self._terminate_mreasoner()

    # ==================================================================
    # Journal JSON
    # ==================================================================

    def _write_full_log(
        self,
        subject_id,
        participant_log,
    ):
        """
        Écrit le fichier :

            {
                "mReasoner": {
                    "participant": {
                        "epsilon": ...,
                        "lambda": ...,
                        "omega": ...,
                        "sigma": ...,
                        "best_params": [...]
                    }
                }
            }
        """
        if self._log_written:
            return

        log_directory = (
            os.path.dirname(self.log_file)
            or "."
        )

        os.makedirs(
            log_directory,
            exist_ok=True,
        )

        existing_log = {}

        if os.path.isfile(self.log_file):
            try:
                with open(
                    self.log_file,
                    "r",
                    encoding="utf-8",
                ) as input_file:
                    existing_log = json.load(
                        input_file
                    )

            except (
                json.JSONDecodeError,
                OSError,
                TypeError,
            ) as error:
                print(
                    "WARNING: ancien journal illisible : "
                    f"{error!r}"
                )

                existing_log = {}

        if not isinstance(existing_log, dict):
            existing_log = {}

        model_section = existing_log.setdefault(
            self.name,
            {},
        )

        if not isinstance(model_section, dict):
            model_section = {}
            existing_log[self.name] = model_section

        model_section[
            str(subject_id)
        ] = participant_log

        temporary_file = None

        try:
            file_descriptor, temporary_file = (
                tempfile.mkstemp(
                    prefix=".full-log-",
                    suffix=".json",
                    dir=log_directory,
                    text=True,
                )
            )

            with os.fdopen(
                file_descriptor,
                "w",
                encoding="utf-8",
            ) as output_file:
                json.dump(
                    existing_log,
                    output_file,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )

                output_file.flush()
                os.fsync(
                    output_file.fileno()
                )

            os.replace(
                temporary_file,
                self.log_file,
            )

            temporary_file = None
            self._log_written = True

            print(
                "DEBUG: paramètres sauvegardés dans "
                f"{self.log_file}."
            )

        finally:
            if (
                temporary_file
                and os.path.exists(temporary_file)
            ):
                os.remove(temporary_file)
