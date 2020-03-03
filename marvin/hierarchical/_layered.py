# This file is part of "marvin" which is released under GPL.
#
# See file LICENCE or go to http://www.gnu.org/licenses/ for full license
# details.
#
# Copyright (c) 2019 Gabriele N. Tornetta <phoenix1987@gmail.com>.
# All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from functools import reduce
import numpy as np
from operator import mul

from scipy.stats import entropy

from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.utils.metaestimators import _BaseComposition

from marvin.metrics import entropy_score


class LayeredHierarchicalClassifier(_BaseComposition, ClassifierMixin):
    def __init__(self, layers, hierarchy):
        # ---- Preconditions ----
        if len(layers) == 1:
            raise RuntimeError(
                "A layered hierarchical model with a single layer "
                "is equivalent to the layer itself, so please use that directly."
            )
        if len(layers) != len(hierarchy) + 1:
            raise ValueError(
                "There should be a mapping for each pair of adjacent layers "
                f"(was given {len(layers)} layers and {len(hierarchy)} maps)."
            )

        self.layers = layers
        self.hierarchy = hierarchy
        self._maps = None

        self._validate_layers()
        self._validate_hierarchy()

    def _validate_layers(self):
        names, estimators = zip(*self.layers)

        # validate names
        self._validate_names(names)

        for estimator in estimators:
            if not (hasattr(estimator, "fit") and hasattr(estimator, "predict_proba")):
                raise TypeError("Each layer must be implement fit and predict_proba")

    def _validate_hierarchy(self):
        if len(self.hierarchy) == 1:
            return

        for i in range(len(self.hierarchy) - 1):
            if set(self.hierarchy[i].keys()) != set(
                [v for _, v in self.hierarchy[i + 1].items()]
            ):
                raise ValueError(
                    f"Incompatible hierarchy at position {i}. "
                    f"Its domain must coincide with the image of the map at position {i+1}."
                )

    def _compute_maps(self):
        """Turn the hierarchy maps into maps defined on the leaves to any intermediate
        layer."""
        if self._maps:
            return

        # Recursive relation:
        # - mu*_n = id
        # - mu*_{n-1} = mu_{n-1}
        # - mu*_k = mu_k mu*_{k+1}  for k = 1, ..., n-1
        self._maps = [self.hierarchy[-1]]

        for map in self.hierarchy[-2::-1]:
            self._maps.append({c: map[v] for c, v in self._maps[-1].items()})

        self._maps = self._maps[::-1]

    def _path(self, c):
        path = [map[c] for map in self._maps]
        path.append(c)
        return path

    def fit(self, data, targets):
        self._compute_maps()
        try:
            for (_, layer), map in zip(self.layers[:-1], self._maps):
                layer.fit(data, [map[t] for t in targets])
        except KeyError as e:
            raise RuntimeError(
                "The map at the base of the hierarchy is not supported on all the"
                "classes seen from the given targets."
            ) from e

        self.layers[-1][1].fit(data, targets)
        self.classes_ = self.layers[-1][1].classes_
        self._indices = [
            {c: i for i, c in enumerate(layer.classes_)} for _, layer in self.layers
        ]
        self._layer_map = dict(self.layers)

    def predict(self, data):
        return self.classes_[np.argmax(self._predict_proba(data), axis=1)]

    def _predict_proba(self, data):
        return np.array(
            [
                np.array(
                    [
                        reduce(
                            mul,
                            (
                                h[index[map[c]]]
                                for h, index, map in zip(
                                    slices[:-1], self._indices[:-1], self._maps
                                )
                            ),
                            slices[-1][i],
                        )
                        for i, c in enumerate(self.classes_)
                    ]
                )
                for slices in zip(
                    *(layer.predict_proba(data) for _, layer in self.layers)
                )
            ]
        )

    def predict_proba(self, data):
        predictions = self._predict_proba(data)
        return predictions / predictions.sum(axis=1)[:, np.newaxis]

    def get_params(self, deep=True):
        return self._get_params("layers", deep=deep)

    def set_params(self, **kwargs):
        self._set_params("layers", **kwargs)
        return self

    def score(self, X, y, sample_weight=None):
        return accuracy_score(self.predict(X), y, sample_weight=sample_weight)

    def h_score(self, X, y):
        """Hierarchical score.

        This is a hierarchical generalisation of micro-precision and
        micro-recall which, in the multi-class setting, coincide with the
        accuracy score.
        """
        y_true_paths = np.array([self._path(c) for c in y])
        y_pred_paths = np.array([self._path(c) for c in self.predict(X)])

        return np.mean(y_true_paths == y_pred_paths)

    def entropy_score(self, X, y, alpha=0.5):
        return entropy_score(y, self.predict_proba(X), self.classes_, alpha)
