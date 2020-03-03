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

from sklearn.base import ClassifierMixin, clone
from sklearn.metrics import accuracy_score
from sklearn.utils.metaestimators import _BaseComposition


def _flatten(l):
    for i in l:
        if isinstance(i, list):
            yield from _flatten(i)
        else:
            yield i


class ParentNodeHierarchicalClassifier(_BaseComposition, ClassifierMixin):
    def __init__(self, estimator, hierarchy):
        # ---- Preconditions ----
        if not hierarchy:
            raise ValueError("Empty hierarchy.")

        if len(hierarchy) == 1:
            raise ValueError(
                "A hierarchy of height 1 is equivalent to the given estimator. "
                "Please train that directly instead of using this model."
            )

        self.estimator = estimator
        self.hierarchy = hierarchy
        self._leaves_map = None
        self._parent_map = None
        self._root = None

        build_estimator = (
            estimator if callable(estimator) else lambda c: clone(estimator)
        )
        self._estimators = {c: build_estimator(c) for c in hierarchy}
        self._estimator_list = list(self._estimators.items())

        self._validate_estimators()
        self._validate_hierarchy()

    def _validate_estimators(self):
        names, estimators = zip(*self._estimator_list)

        # validate names
        self._validate_names(names)

        for _, estimator in self._estimator_list:
            if not (hasattr(estimator, "fit") and hasattr(estimator, "predict_proba")):
                raise TypeError(
                    "Each estimator in the hierarchy must be implement fit and predict_proba"
                )

    def _validate_hierarchy(self):

        # Find the root node
        child_nodes = {n for _, children in self.hierarchy.items() for n in children}
        parent_nodes = set(self.hierarchy.keys())

        try:
            self._root, = parent_nodes - child_nodes
        except ValueError:
            raise ValueError(
                "The given hierarchy is not in the shape of a rooted tree."
            )
        self._leaves = child_nodes - parent_nodes

        # TODO: check fo cycles

    def _compute_leaves_map(self):
        leaves = {}

        def collect_children(n):
            if n in leaves:
                return leaves[n]

            if n in self.hierarchy:
                leaves[n] = [collect_children(m) for m in self.hierarchy[n]]
                return leaves[n]

            return n

        collect_children(self._root)

        self._leaves_map = {p: list(_flatten(l)) for p, l in leaves.items()}

    def _path(self, c):
        path = [c]
        while path[0] != self._root:
            parent = self._parent_map[path[0]]
            if parent in path:
                # TODO: Try to detect cycles earlier
                raise RuntimeError("Cycle detected in the given hierarchy.")
            path.insert(0, parent)
        return path

    def _compute_parent_map(self):
        self._parent_map = {}
        for p, children in self.hierarchy.items():
            for c in children:
                self._parent_map[c] = p

    def fit(self, data, targets):
        self._compute_leaves_map()
        self._compute_parent_map()

        # Fit each parent model
        for p, estimator in self._estimators.items():
            classes = set(self.hierarchy[p])
            where = np.asarray(np.isin(targets, self._leaves_map[p])).nonzero()
            estimator.fit(
                np.array(data)[where],
                np.array(
                    [
                        a
                        for e in [
                            set(self._path(c)) & classes
                            for c in np.array(targets)[where]
                        ]
                        for a in e
                    ]
                ),
            )

        def collect_classes(n):
            estimator = self._estimators.get(n, None)
            if estimator:
                return [collect_classes(c) for c in estimator.classes_]
            return n

        self.classes_ = np.array(list(_flatten(collect_classes(self._root))))

    def predict(self, data):
        return self.classes_[np.argmax(self.predict_proba(data), axis=1)]

    def predict_proba(self, data):
        probas = {
            c: estimator.predict_proba(data) for c, estimator in self._estimator_list
        }

        def proba(c):
            path = self._path(c)
            return reduce(
                mul,
                [
                    probas[path[i]][
                        :,
                        np.where(self._estimators[path[i]].classes_ == path[i + 1])[0][
                            0
                        ],
                    ]
                    for i in range(len(path) - 1)
                ],
            )

        return np.column_stack([proba(c) for c in self.classes_])

    def get_params(self, deep=True):
        return self._get_params("_estimator_list", deep=deep)

    def set_params(self, **kwargs):
        self._set_params("_estimator_list", **kwargs)
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
