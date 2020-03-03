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

from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from marvin.hierarchical import ParentNodeHierarchicalClassifier
from marvin.metrics import entropy_score


def test_parent_node_hierarchical_classifier():
    data, targets = load_digits(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        data, targets, test_size=0.2, random_state=42
    )

    hierarchy = {
        "root": ["r", "b"],
        "b": ["m", "s"],
        "r": [0, 3, 6, 8, 9],
        "s": [1, 4, 7],
        "m": [2, 5],
    }

    def build_estimator(c):
        if c == "root":
            return SVC(gamma=0.001, probability=True)

        if c in ["a", "b"]:
            return LR(solver="lbfgs", multi_class="multinomial", max_iter=5000)

        if c in ["r", "s", "m"]:
            return SVC(gamma=0.001, probability=True)

        raise ValueError(f"Unknown class: {c}")

    model = ParentNodeHierarchicalClassifier(build_estimator, hierarchy)
    model.fit(X_train, y_train)

    assert 0.95 <= model.score(X_test, y_test) <= model.h_score(X_test, y_test)

    Y_pred = model.predict_proba(X_test)
    assert 0.95 <= entropy_score(y_test, Y_pred, model.classes_)
