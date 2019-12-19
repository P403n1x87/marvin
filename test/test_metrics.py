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

import numpy as np
import pytest
from marvin.metrics import *


def test_cm_purity():
    assert cm_purity(np.array([[1, 0], [0, 1]])) == 1.0
    assert cm_purity(np.array([[0, 1], [1, 0]])) == 0.0
    assert cm_purity(np.array([[1, 0], [1, 0]])) == 1 - np.sqrt(0.5)


def test_mean_entropy():
    assert mean_entropy([[1 / 2, 1 / 2], [1, 0], [0, 1]]) == pytest.approx(
        np.mean([1, 0, 0])
    )


def test_entropy_score():
    assert (
        entropy_score(
            y_true=["a", "b"],
            Y_pred=np.array([[1, 0], [0.51, 0.49]]),
            classes=np.array(["a", "b"]),
            alpha=1,
        )
        > 0.95
    )


def test_probabilistic_confusion_matrix():
    classes = np.array(["a", "b"])
    y_true = np.array(["a", "a", "b", "b"])
    Y_pred = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.8, 0.2]])

    assert np.array_equal(
        probabilistic_confusion_matrix(y_true, Y_pred, classes),
        np.array([[0.5, 0.5], [0.8, 0.2]]),
    )


def test_mean_purity():
    assert (
        mean_purity(
            y_true=["a", "b"],
            Y_pred=np.array([[1, 0], [1, 0]]),
            classes=np.array(["a", "b"]),
        )
        == 0.5
    )
