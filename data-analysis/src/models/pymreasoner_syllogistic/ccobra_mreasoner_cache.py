""" CCOBRA model wrapper for mReasoner (Optimisé avec Vectorisation NumPy).
"""

import sys
import os
import logging
import time
from pathlib import Path

import ccobra
import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
MODELS_DIR = CURRENT_DIR.parent

# Permet de trouver pymreasoner_syllogistic/create_cache.py
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# Permet de trouver models/mreasoner/
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

import mreasoner
import create_cache


# ======================================================================
# CHEMINS
# ======================================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

BASE_DIR = os.path.abspath(
    os.path.join(
        SCRIPT_DIR,
        "..",
        "..",
        "..",
    )
)

DEFAULT_CACHE_FILE = os.path.join(
    BASE_DIR,
    "data",
    "cache",
    "pymreasoner_syllogistic",
    "2020-09-09-cache-11-10.npy",
)


logger = logging.getLogger(__name__)

class CCobraMReasoner(ccobra.CCobraModel):
    """ mReasoner CCOBRA model implementation.
    """

    def __init__(self, name='mReasoner', n_samples=2, fit_its=5, cache_file=None):
        super(CCobraMReasoner, self).__init__(name, ['syllogistic'], ['single-choice'])

        self.n_samples = n_samples
        self.fit_its = fit_its

        self.n_pre_train_dudes = 0
        self.pre_train_data = np.zeros((64, 9))
        self.history = np.zeros((64, 9))    

        # Cache par défaut stocké dans data/cache/pymreasoner_syllogistic/.
        if cache_file is None:
            cache_file = DEFAULT_CACHE_FILE

        # Un chemin relatif fourni dans un benchmark est interprété
        # relativement à la racine du dépôt.
        elif not os.path.isabs(cache_file):
            cache_file = os.path.join(
                BASE_DIR,
                cache_file,
            )

        cache_file = os.path.abspath(
            cache_file
        )


        self.prediction_cache = None
        if cache_file:
            self.prediction_cache = np.load(cache_file)
            if self.prediction_cache.shape[0] != fit_its:
                logger.warning('WARNING: fit_its mismatch between model and cache.')
            self.fit_its = self.prediction_cache.shape[0]
        else:
            self.prediction_cache = create_cache.generate_cache(self.fit_its, self.n_samples)

        self.params = {}

        for param_idx, param in enumerate(['epsilon', 'lambda', 'omega', 'sigma']):
            default_value = mreasoner.DEFAULT_PARAMS[param]
            conf_values = np.linspace(*mreasoner.PARAM_BOUNDS[param_idx], self.fit_its)
            diffs = np.abs(conf_values - default_value)
            closest_idxs = np.arange(len(conf_values))[diffs == diffs.min()]
            closest_idx = np.random.choice(closest_idxs)
            self.params[param] = (closest_idx, conf_values[closest_idx])

        self.best_param_dicts = []
        self.start_time = None

    def end_participant(self, subj_id, model_log, **kwargs):
        print('End Participant ({:.2f}s, {} its) id={} params={}'.format(
            time.time() - self.start_time,
            self.fit_its,
            subj_id,
            str([(x, y[1]) for x, y in self.params.items()]).replace(' ', ''),
        ))
        sys.stdout.flush()

        model_log.update({x: y[1] for x, y in self.params.items()})
        model_log['best_params'] = [{y: z[1] for y, z in x.items()} for x in self.best_param_dicts]

    def start_participant(self, **kwargs):
        self.start_time = time.time()

    def pre_train(self, dataset):
        if self.fit_its == 0 or self.evaluation_type == 'coverage':
            return

        self.n_pre_train_dudes = len(dataset)
        self.pre_train_data = np.zeros((64, 9))
        for subj_data in dataset:
            for task_data in subj_data:
                item = task_data['item']
                enc_task = ccobra.syllogistic.encode_task(item.task)
                enc_resp = ccobra.syllogistic.encode_response(task_data['response'], item.task)

                task_idx = ccobra.syllogistic.SYLLOGISMS.index(enc_task)
                resp_idx = ccobra.syllogistic.RESPONSES.index(enc_resp)
                self.pre_train_data[task_idx, resp_idx] += 1

        div_mask = (self.pre_train_data.sum(axis=1) != 0)
        self.pre_train_data[div_mask] /= self.pre_train_data[div_mask].sum(axis=1, keepdims=True)
        self.fit()

    def pre_train_person(self, dataset, **kwargs):
        print('Person training...')
        if self.fit_its == 0:
            return

        for task_data in dataset:
            item = task_data['item']
            enc_task = ccobra.syllogistic.encode_task(item.task)
            enc_resp = ccobra.syllogistic.encode_response(task_data['response'], item.task)

            task_idx = ccobra.syllogistic.SYLLOGISMS.index(enc_task)
            resp_idx = ccobra.syllogistic.RESPONSES.index(enc_resp)
            self.history[task_idx, resp_idx] += 1

        self.fit()

    def fit(self):
        history_copy = self.history.copy()
        div_mask = (history_copy.sum(axis=1) != 0)
        history_copy[div_mask] /= history_copy[div_mask].sum(axis=1, keepdims=True)

        train_data = self.pre_train_data.copy()
        train_data[div_mask] = history_copy[div_mask]

        # ==========================================
        # OPTIMISATION ULTRA-RAPIDE (VECTORISATION)
        # Remplace les 4 boucles for très lentes.
        # ==========================================
        
        # 1. On trouve les valeurs max pour chaque syllogisme (sur le dernier axe)
        max_preds = self.prediction_cache.max(axis=-1, keepdims=True)
        
        # 2. On crée le masque booléen des meilleures prédictions
        pred_mask = (self.prediction_cache == max_preds)
        
        # 3. On calcule le score pour les 14641 grilles d'un seul coup (magie de NumPy)
        scores = np.sum(np.mean(train_data * pred_mask, axis=-1), axis=-1)
        
        # 4. On trouve le score maximum
        best_score = scores.max()
        
        # 5. On récupère toutes les coordonnées (idx_epsilon, idx_lambda, etc.) où ce score apparait
        best_indices = np.argwhere(np.isclose(scores, best_score))
        
        # 6. On reconstruit le dictionnaire de paramètres pour rester compatible avec CCOBRA
        epsilons = np.linspace(*mreasoner.PARAM_BOUNDS[0], self.fit_its)
        lambdas = np.linspace(*mreasoner.PARAM_BOUNDS[1], self.fit_its)
        omegas = np.linspace(*mreasoner.PARAM_BOUNDS[2], self.fit_its)
        sigmas = np.linspace(*mreasoner.PARAM_BOUNDS[3], self.fit_its)
        
        best_param_dicts = []
        for idxs in best_indices:
            idx_epsilon, idx_lambda, idx_omega, idx_sigma = idxs
            best_param_dicts.append({
                'epsilon': (idx_epsilon, epsilons[idx_epsilon]),
                'lambda': (idx_lambda, lambdas[idx_lambda]),
                'omega': (idx_omega, omegas[idx_omega]),
                'sigma': (idx_sigma, sigmas[idx_sigma])
            })

        # Choix aléatoire parmi les meilleurs
        self.params = best_param_dicts[int(np.random.randint(0, len(best_param_dicts)))]
        self.best_param_dicts = best_param_dicts

    def predict(self, item, **kwargs):
        syllog = ccobra.syllogistic.Syllogism(item)

        pred_mat = self.prediction_cache[
            self.params['epsilon'][0], self.params['lambda'][0], self.params['omega'][0], self.params['sigma'][0]]
        syl_idx = ccobra.syllogistic.SYLLOGISMS.index(syllog.encoded_task)
        preds = pred_mat[syl_idx]

        cand_choices = np.array(ccobra.syllogistic.RESPONSES)[preds == preds.max()]
        return syllog.decode_response(np.random.choice(cand_choices))

    def adapt(self, item, truth, **kwargs):
        enc_task = ccobra.syllogistic.encode_task(item.task)
        enc_resp = ccobra.syllogistic.encode_response(truth, item.task)

        task_idx = ccobra.syllogistic.SYLLOGISMS.index(enc_task)
        resp_idx = ccobra.syllogistic.RESPONSES.index(enc_resp)
        self.history[task_idx, resp_idx] += 1

        self.fit()