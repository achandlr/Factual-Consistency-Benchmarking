'''

# TODO: These are flyingsquid model repo:

https://github.com/HazyResearch/flyingsquid?tab=readme-ov-file

there are 3 different ways to install. maybe I should try cl

1. git clone https://github.com/HazyResearch/flyingsquid.git

cd flyingsquid

conda env create -f environment.yml
conda activate flyingsquid

2. then try  lternatively, you can install the dependencies yourself:

Pgmpy
PyTorch (only necessary for the PyTorch integration)
And then install the actual package:

pip install flyingsquid



sample usage


from flyingsquid.label_model import LabelModel
import numpy as np

L_train = np.load('...')

m = L_train.shape[1]
label_model = LabelModel(m)
label_model.fit(L_train)

preds = label_model.predict(L_train)
'''

# Source: https://github.com/JieyuZ2/wrench/blob/main/wrench/labelmodel/flyingsquid.py

import logging
from typing import Any, List, Optional, Union

import numpy as np
# from flyingsquid.label_model import LabelModel

# from ..basemodel import BaseLabelModel
# from ..dataset import BaseDataset
# from ..dataset.utils import check_weak_labels

logger = logging.getLogger(__name__)

ABSTAIN = -1

import copy
import json
import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Union, Callable

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
from snorkel.labeling import LFAnalysis
from tqdm.auto import tqdm

from snorkel.utils import probs_to_preds


logger = logging.getLogger(__name__)


import random
from collections import Counter
from typing import Dict, Optional, Union


from functools import partial
from typing import List

import numpy as np
import seqeval.metrics as seq_metric
import sklearn.metrics as cls_metric
# from seqeval.scheme import IOB2
from snorkel.utils import probs_to_preds



def brier_score_loss(y_true: np.ndarray, y_proba: np.ndarray, ):
    r = len(np.unique(y_true))
    return np.mean((np.eye(r)[y_true] - y_proba) ** 2)


def accuracy_score_(y_true: np.ndarray, y_proba: np.ndarray, **kwargs):
    y_pred = probs_to_preds(y_proba, **kwargs)
    return cls_metric.accuracy_score(y_true, y_pred)


def f1_score_(y_true: np.ndarray, y_proba: np.ndarray, average: str, **kwargs):
    if average == 'binary' and len(np.unique(y_true)) > 2:
        return 0.0
    y_pred = probs_to_preds(y_proba, **kwargs)
    return cls_metric.f1_score(y_true, y_pred, average=average)


def recall_score_(y_true: np.ndarray, y_proba: np.ndarray, average: str, **kwargs):
    if average == 'binary' and len(np.unique(y_true)) > 2:
        return 0.0
    y_pred = probs_to_preds(y_proba, **kwargs)
    return cls_metric.recall_score(y_true, y_pred, average=average)


def precision_score_(y_true: np.ndarray, y_proba: np.ndarray, average: str, **kwargs):
    if average == 'binary' and len(np.unique(y_true)) > 2:
        return 0.0
    y_pred = probs_to_preds(y_proba, **kwargs)
    return cls_metric.precision_score(y_true, y_pred, average=average)


def auc_score_(y_true: np.ndarray, y_proba: np.ndarray, **kwargs):
    if len(np.unique(y_true)) > 2:
        return 0.0
    return cls_metric.roc_auc_score(y_true, y_proba[:, 1], **kwargs)


def ap_score_(y_true: np.ndarray, y_proba: np.ndarray, **kwargs):
    if len(np.unique(y_true)) > 2:
        return 0.0
    return cls_metric.average_precision_score(y_true, y_proba[:, 1], pos_label=1, **kwargs)


def f1_score_seq(y_true: List[List], y_pred: List[List], id2label: dict, strict=True):
    y_true = [[id2label[x] for x in y] for y in y_true]
    y_pred = [[id2label[x] for x in y] for y in y_pred]
    if strict:
        return seq_metric.f1_score(y_true, y_pred, mode='strict', scheme=IOB2)
    else:
        return seq_metric.f1_score(y_true, y_pred)


def precision_seq(y_true: List[List], y_pred: List[List], id2label: dict, strict=True):
    y_true = [[id2label[x] for x in y] for y in y_true]
    y_pred = [[id2label[x] for x in y] for y in y_pred]
    if strict:
        return seq_metric.precision_score(y_true, y_pred, mode='strict', scheme=IOB2)
    else:
        return seq_metric.precision_score(y_true, y_pred)


def recall_seq(y_true: List[List], y_pred: List[List], id2label: dict, strict=True):
    y_true = [[id2label[x] for x in y] for y in y_true]
    y_pred = [[id2label[x] for x in y] for y in y_pred]
    if strict:
        return seq_metric.recall_score(y_true, y_pred, mode='strict', scheme=IOB2)
    else:
        return seq_metric.recall_score(y_true, y_pred)


METRIC = {
    'acc'               : accuracy_score_,
    'auc'               : auc_score_,
    'ap'                : ap_score_,
    'f1_binary'         : partial(f1_score_, average='binary'),
    'f1_micro'          : partial(f1_score_, average='micro'),
    'f1_macro'          : partial(f1_score_, average='macro'),
    'f1_weighted'       : partial(f1_score_, average='weighted'),
    'recall_binary'     : partial(recall_score_, average='binary'),
    'recall_micro'      : partial(recall_score_, average='micro'),
    'recall_macro'      : partial(recall_score_, average='macro'),
    'recall_weighted'   : partial(recall_score_, average='weighted'),
    'precision_binary'  : partial(precision_score_, average='binary'),
    'precision_micro'   : partial(precision_score_, average='micro'),
    'precision_macro'   : partial(precision_score_, average='macro'),
    'precision_weighted': partial(precision_score_, average='weighted'),
    'logloss'           : cls_metric.log_loss,
    'brier'             : brier_score_loss,
}

SEQ_METRIC = {
    'f1_seq'       : partial(f1_score_seq),
    'precision_seq': partial(precision_seq),
    'recall_seq'   : partial(recall_seq),
}


def metric_to_direction(metric: str) -> str:
    if metric in ['acc', 'f1_binary', 'f1_micro', 'f1_macro', 'f1_weighted', 'auc']:
        return 'maximize'
    if metric in ['logloss', 'brier']:
        return 'minimize'
    if metric in SEQ_METRIC:
        return 'maximize'
    raise NotImplementedError(f'cannot automatically decide the direction for {metric}!')

def array_to_marginals(y, cardinality=None):
    class_counts = Counter(y)
    if cardinality is None:
        sorted_counts = np.array([v for k, v in sorted(class_counts.items())])
    else:
        sorted_counts = np.zeros(len(cardinality))
        for i, c in enumerate(cardinality):
            sorted_counts[i] = class_counts.get(c, 0)
    marginal = sorted_counts / sum(sorted_counts)
    return marginal


def calc_cmi_matrix(y, L):
    n, m = L.shape
    lf_cardinality = [sorted(np.unique(L[:, i])) for i in range(m)]

    n_class = len(np.unique(y))
    c_idx_l = [y == c for c in range(n_class)]
    c_cnt_l = [np.sum(c_idx) for c_idx in c_idx_l]
    class_marginal = [c_cnt / n for c_cnt in c_cnt_l]

    cond_probs = np.zeros((n_class, m, max(map(len, lf_cardinality))))
    for c, c_idx in enumerate(c_idx_l):
        for i in range(m):
            card_i = lf_cardinality[i]
            cond_probs[c, i][:len(card_i)] = array_to_marginals(L[:, i][c_idx], card_i)

    cmi_matrix = -np.ones((m, m)) * np.inf
    for i in range(m):
        L_i = L[:, i]
        card_i = lf_cardinality[i]
        for j in range(i + 1, m):
            L_j = L[:, j]
            card_j = lf_cardinality[j]

            cmi_ij = 0.0
            for c, (c_idx, n_c) in enumerate(zip(c_idx_l, c_cnt_l)):
                cmi = 0.0
                for ci_idx, ci in enumerate(card_i):
                    for cj_idx, cj in enumerate(card_j):
                        p = np.sum(np.logical_and(L_i[c_idx] == ci, L_j[c_idx] == cj)) / n_c
                        if p > 0:
                            cur = p * np.log(p / (cond_probs[c, i, ci_idx] * cond_probs[c, j, cj_idx]))
                            cmi += cur

                cmi_ij += class_marginal[c] * cmi
            cmi_matrix[i, j] = cmi_matrix[j, i] = cmi_ij

    return cmi_matrix


def cluster_based_accuracy_variance(Y, L, cluster_labels):
    correct = Y == L
    acc_l = []
    cluster_idx = np.unique(cluster_labels)
    for cluster in cluster_idx:
        cluster_correct = correct[cluster_labels == cluster]
        cluster_acc = np.sum(cluster_correct) / len(cluster_correct)
        acc_l.append(cluster_acc)
    return np.var(acc_l)





class BaseModel(ABC):
    """Abstract model class."""
    hyperparas: Dict

    @abstractmethod
    def __init__(self, **kwargs: Any):
        pass

    def _update_hyperparas(self, **kwargs: Any):
        for k, v in self.hyperparas.items():
            if k in kwargs: self.hyperparas[k] = kwargs[k]

    @abstractmethod
    def fit(self, dataset_train, y_train=None, dataset_valid=None, y_valid=None,
            verbose: Optional[bool] = False, *args: Any, **kwargs: Any):
        """Abstract method for fitting training data.

        Parameters
        ----------
        """
        pass

    @abstractmethod
    def predict(self, dataset, **kwargs: Any):
        pass

    @abstractmethod
    def test(self, dataset, metric_fn: Union[Callable, str], y_true=None, **kwargs):
        pass

    def save(self, destination: str) -> None:
        """Save label model.
        Parameters
        ----------
        destination
            Filename for saving model
        Example
        -------
        >>> model.save('./saved_model.pkl')  # doctest: +SKIP
        """
        f = open(destination, "wb")
        pickle.dump(self.__dict__, f)
        f.close()

    def load(self, source: str) -> None:
        """Load existing label model.
        Parameters
        ----------
        source
            Filename to load model from
        Example
        -------
        Load parameters saved in ``saved_model``
        >>> model.load('./saved_model.pkl')  # doctest: +SKIP
        """
        f = open(source, "rb")
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)


class BaseDataset(ABC):
    """Abstract data class."""

    def __init__(self,
                 path: str = None,
                 split: Optional[str] = None,
                 feature_cache_name: Optional[str] = None,
                 **kwargs: Any) -> None:
        self.ids: List = []
        self.labels: List = []
        self.examples: List = []
        self.weak_labels: List[List] = []
        self.features = None
        self.id2label = None

        self.split = split
        self.path = path

        if path is not None and split is not None:
            self.load(path=path, split=split)
            self.load_features(feature_cache_name)
            self.n_class = len(self.id2label)
            self.n_lf = len(self.weak_labels[0])

    def __len__(self):
        return len(self.ids)

    def load(self, path: str, split: str):
        """Method for loading data given the split.

        Parameters
        ----------
        split
            A str with values in {"train", "valid", "test", None}. If None, then do not load any data.
        Returns
        -------
        self
        """

        assert split in ["train", "valid", "test"], 'Parameter "split" must be in ["train", "valid", "test", None]'

        path = Path(path)

        self.split = split
        self.path = path

        data_path = path / f'{split}.json'
        logger.info(f'loading data from {data_path}')
        data = json.load(open(data_path, 'r'))
        for i, item in tqdm(data.items()):
            self.ids.append(i)
            self.labels.append(item['label'])
            self.weak_labels.append(item['weak_labels'])
            self.examples.append(item['data'])

        label_path = self.path / f'label.json'
        self.id2label = {int(k): v for k, v in json.load(open(label_path, 'r')).items()}

        return self

    def load_labeled_ids_and_lf_exemplars(self, path: str):

        path = Path(path)

        assert self.split == 'train', 'labeled data can only be loaded by train'
        logger.info(f'loading labeled ids and lf exemplars from {path}')
        data = json.load(open(path, 'r'))
        labeled_ids = data.get('labeled_ids', [])
        lf_exemplar_ids = data.get('lf_exemplar_ids', [])

        # map to real data idx in self
        labeled_ids = [self.ids.index(i) for i in labeled_ids]
        lf_exemplar_ids = [self.ids.index(i) for i in lf_exemplar_ids]

        return labeled_ids, lf_exemplar_ids

    def load_features(self, cache_name: Optional[str] = None):
        """Method for loading data feature given the split and cache_name.

        Parameters
        ----------
        cache_name
            A str used to locate the feature file.
        Returns
        -------
        features
            np.ndarray
        """
        if cache_name is None:
            self.features = None
            return None

        path = self.path / f'{self.split}_{cache_name}.pkl'
        logger.info(f'loading features from {path}')
        features = pickle.load(open(path, 'rb'))
        self.features = features
        return features

    def save_features(self, cache_name: Optional[str] = None):
        if cache_name is None:
            return None
        path = self.path / f'{self.split}_{cache_name}.pkl'
        logger.info(f'saving features into {path}')
        pickle.dump(self.features, open(path, 'wb'), protocol=4)
        return path

    def extract_feature(self,
                        extract_fn: Union[str, Callable],
                        return_extractor: bool,
                        cache_name: str = None,
                        force: bool = False,
                        normalize=False,
                        **kwargs: Any):
        if cache_name is not None:
            path = self.path / f'{self.split}_{cache_name}.pkl'
            if path.exists() and (not force):
                self.load_features(cache_name=cache_name)
                return

        if isinstance(extract_fn, Callable):
            self.features = extract_fn(self.examples)
        else:
            extractor = self.extract_feature_(extract_fn=extract_fn, return_extractor=return_extractor, **kwargs)
            if normalize:
                features = self.features
                scaler = preprocessing.StandardScaler().fit(features)
                self.features = scaler.transform(features)
                extract_fn = lambda x: scaler.transform(extractor(x))
            else:
                extract_fn = extractor

        if cache_name is not None:
            self.save_features(cache_name=cache_name)

        if return_extractor:
            return extract_fn

    @abstractmethod
    def extract_feature_(self, extract_fn: str, return_extractor: bool, **kwargs: Any):
        """Abstract method for extracting features given the mode.

        Parameters
        ----------
        """
        pass

    def create_subset(self, idx: List[int]):
        dataset = self.__class__()
        for i in idx:
            dataset.ids.append(self.ids[i])
            dataset.labels.append(self.labels[i])
            dataset.examples.append(self.examples[i])
            dataset.weak_labels.append(self.weak_labels[i])

        if self.features is not None:
            dataset.features = self.features[idx]

        dataset.id2label = copy.deepcopy(self.id2label)
        dataset.split = self.split
        dataset.path = self.path
        dataset.n_class = self.n_class
        dataset.n_lf = self.n_lf

        return dataset

    def create_split(self, idx: List[int]):
        chosen = self.create_subset(idx)
        remain = self.create_subset([i for i in range(len(self)) if i not in idx])
        return chosen, remain

    def sample(self, alpha: Union[int, float], return_dataset=True):
        if isinstance(alpha, float):
            alpha = int(len(self) * alpha)
        idx = np.random.choice(len(self), alpha, replace=False)
        if return_dataset:
            return self.create_subset(idx)
        else:
            return list(idx)

    def get_covered_subset(self):
        idx = [i for i in range(len(self)) if np.any(np.array(self.weak_labels[i]) != -1)]
        return self.create_subset(idx)

    def get_conflict_labeled_subset(self):
        idx = [i for i in range(len(self)) if len({l for l in set(self.weak_labels[i]) if l != -1}) > 1]
        return self.create_subset(idx)

    def get_agreed_labeled_subset(self):
        idx = [i for i in range(len(self)) if len({l for l in set(self.weak_labels[i]) if l != -1}) == 1]
        return self.create_subset(idx)

    def lf_summary(self):
        L = np.array(self.weak_labels)
        Y = np.array(self.labels)
        lf_summary = LFAnalysis(L=L).lf_summary(Y=Y)
        return lf_summary

    def summary(self, n_clusters=10, features=None, return_lf_summary=False):
        summary_d = {}
        L = np.array(self.weak_labels)
        Y = np.array(self.labels)

        summary_d['n_class'] = self.n_class
        summary_d['n_data'], summary_d['n_lfs'] = L.shape
        summary_d['n_uncovered_data'] = np.sum(np.all(L == -1, axis=1))
        uncovered_rate = summary_d['n_uncovered_data'] / summary_d['n_data']
        summary_d['overall_coverage'] = (1 - uncovered_rate)

        lf_summary = LFAnalysis(L=L).lf_summary(Y=Y)
        summary_d['lf_avr_acc'] = lf_summary['Emp. Acc.'].mean()
        summary_d['lf_var_acc'] = lf_summary['Emp. Acc.'].var()
        summary_d['lf_avr_propensity'] = lf_summary['Coverage'].mean()
        summary_d['lf_var_propensity'] = lf_summary['Coverage'].var()
        summary_d['lf_avr_overlap'] = lf_summary['Overlaps'].mean()
        summary_d['lf_var_overlap'] = lf_summary['Overlaps'].var()
        summary_d['lf_avr_conflict'] = lf_summary['Conflicts'].mean()
        summary_d['lf_var_conflict'] = lf_summary['Conflicts'].var()

        # calc cmi
        # from ..utils import calc_cmi_matrix, cluster_based_accuracy_variance
        cmi_matrix = calc_cmi_matrix(Y, L)
        lf_cmi = np.ma.masked_invalid(cmi_matrix).mean(1).data
        summary_d['correlation'] = lf_cmi.mean()
        lf_summary['correlation'] = pd.Series(lf_cmi)

        # calc data dependency
        if hasattr(self, 'features') and features is None:
            features = self.features
        if features is not None:
            kmeans = KMeans(n_clusters=n_clusters).fit(features)
            cluster_labels = kmeans.labels_
            acc_var = np.array([cluster_based_accuracy_variance(Y, L[:, i], cluster_labels) for i in range(self.n_lf)])
            summary_d['data-dependency'] = acc_var.mean()
            lf_summary['data-dependency'] = pd.Series(acc_var)

        if return_lf_summary:
            return summary_d, lf_summary
        else:
            return summary_d

class BaseClassModel(BaseModel, ABC):

    @abstractmethod
    def fit(self, dataset_train: Union[BaseDataset, np.ndarray], y_train: Optional[np.ndarray] = None,
            dataset_valid: Optional[Union[BaseDataset, np.ndarray]] = None, y_valid: Optional[np.ndarray] = None,
            verbose: Optional[bool] = False, *args: Any, **kwargs: Any):
        """Abstract method for fitting training data.

        Parameters
        ----------
        """
        pass

    @abstractmethod
    def predict_proba(self, dataset: Union[BaseDataset, np.ndarray], **kwargs: Any) -> np.ndarray:
        """Abstract method for outputting probabilistic predictions on given dataset.

        Parameters
        ----------
        """
        pass

    def predict(self, dataset: Union[BaseDataset, np.ndarray], **kwargs: Any) -> np.ndarray:
        """Method for predicting on given dataset.

        Parameters
        ----------
        """
        proba = self.predict_proba(dataset, **kwargs)
        return probs_to_preds(probs=proba)

    def test(self, dataset: Union[BaseDataset, np.ndarray], metric_fn: Union[Callable, str], y_true: Optional[np.ndarray] = None, **kwargs):
        if isinstance(metric_fn, str):
            metric_fn = METRIC[metric_fn]
        if y_true is None:
            y_true = np.array(dataset.labels)
        probas = self.predict_proba(dataset, **kwargs)
        return metric_fn(y_true, probas)




class BaseLabelModel(BaseClassModel):
    """Abstract label model class."""

    @abstractmethod
    def fit(self,
            dataset_train: Union[BaseDataset, np.ndarray],
            dataset_valid: Optional[Union[BaseDataset, np.ndarray]] = None,
            y_valid: Optional[np.ndarray] = None,
            n_class: Optional[int] = None,
            verbose: Optional[bool] = False,
            *args: Any,
            **kwargs: Any):
        pass

    @staticmethod
    def _init_balance(L: np.ndarray,
                      dataset_valid: Optional[Union[BaseDataset, np.ndarray]] = None,
                      y_valid: Optional[np.ndarray] = None,
                      n_class: Optional[int] = None):
        if y_valid is not None:
            y = y_valid
        elif dataset_valid is not None:
            y = np.array(dataset_valid.labels)
        else:
            y = np.arange(L.max() + 1)
        class_counts = Counter(y)

        if isinstance(dataset_valid, BaseDataset):
            if n_class is not None:
                assert n_class == dataset_valid.n_class
            else:
                n_class = dataset_valid.n_class

        if n_class is None:
            sorted_counts = np.array([v for k, v in sorted(class_counts.items())])
        else:
            sorted_counts = np.zeros(n_class)
            for c, cnt in class_counts.items():
                sorted_counts[c] = cnt
        balance = (sorted_counts + 1) / sum(sorted_counts)

        return balance



class FlyingSquid(BaseLabelModel):
    def __init__(self, **kwargs: Any):
        super().__init__()
        self.hyperparas = {}
        self.model = None
        self.n_class = None

    def fit(self,
            dataset_train: Union[BaseDataset, np.ndarray],
            dataset_valid: Optional[Union[BaseDataset, np.ndarray]] = None,
            y_valid: Optional[np.ndarray] = None,
            n_class: Optional[int] = None,
            balance: Optional[np.ndarray] = None,
            dependency_graph: Optional[List] = [],
            verbose: Optional[bool] = False,
            **kwargs: Any):

        self._update_hyperparas(**kwargs)
        if isinstance(dataset_train, BaseDataset):
            if n_class is not None:
                assert n_class == dataset_train.n_class
            else:
                n_class = dataset_train.n_class
        if n_class is not None and balance is not None:
            assert len(balance) == n_class

        L = check_weak_labels(dataset_train)
        if balance is None:
            balance = self._init_balance(L, dataset_valid, y_valid, n_class)
        n_class = len(balance)
        self.n_class = n_class

        n, m = L.shape
        if n_class > 2:
            model = []
            for i in range(n_class):
                label_model = LabelModel(m=m, lambda_edges=dependency_graph)
                L_i = np.copy(L)
                target_mask = L_i == i
                abstain_mask = L_i == ABSTAIN
                other_mask = (~target_mask) & (~abstain_mask)
                L_i[target_mask] = 1
                L_i[abstain_mask] = 0
                L_i[other_mask] = -1
                label_model.fit(L_train=L_i, class_balance=np.array([1 - balance[i], balance[i]]), verbose=verbose, **kwargs)
                model.append(label_model)
        else:
            model = LabelModel(m=m, lambda_edges=dependency_graph)
            L_i = np.copy(L)
            abstain_mask = L_i == -1
            negative_mask = L_i == 0
            L_i[abstain_mask] = 0
            L_i[negative_mask] = -1
            model.fit(L_train=L_i, class_balance=balance, verbose=verbose, **kwargs)

        self.model = model

    def predict_proba(self, dataset: Union[BaseDataset, np.ndarray], **kwargs: Any) -> np.ndarray:
        L = check_weak_labels(dataset)
        if self.n_class > 2:
            probas = np.zeros((len(L), self.n_class))
            for i in range(self.n_class):
                L_i = np.copy(L)
                target_mask = L_i == i
                abstain_mask = L_i == ABSTAIN
                other_mask = (~target_mask) & (~abstain_mask)
                L_i[target_mask] = 1
                L_i[abstain_mask] = 0
                L_i[other_mask] = -1
                probas[:, i] = self.model[i].predict_proba(L_matrix=L_i)[:, 1]
            probas = np.nan_to_num(probas, nan=-np.inf)  # handle NaN
            probas = np.exp(probas) / np.sum(np.exp(probas), axis=1, keepdims=True)
        else:
            L_i = np.copy(L)
            abstain_mask = L_i == -1
            negative_mask = L_i == 0
            L_i[abstain_mask] = 0
            L_i[negative_mask] = -1
            probas = self.model.predict_proba(L_matrix=L_i)
        return probas