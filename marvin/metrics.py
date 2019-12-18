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

from scipy.stats import entropy


def confusion_entropy():
    # See http://gent.uab.cat/rosario_delgado/sites/gent.uab.cat.rosario_delgado/files/cen_revised.pdf
    pass


def entropy_score(y_true, Y_pred, classes, alpha=0.5):
    """Compute the entropy score of the predicted probabilities.

    The entropy score is a way to combine both accuracy and entropic
    information into a single metric. Accuracy measures the fraction of
    features that have been predicted correctly, regardless of how confident
    the prediction was. Entropy, on the other hand, gives a measure of
    how certain the model was in making a certain prediction. Features that
    are predicted correctly with good confidence should be valued more than
    features predicted with low confidence. Similarly, wrong predictions
    where a model shows a certain degree of confusion shouldn't be penalised
    too harshly with a score of 0.

    A parameter, alpha, can be used to control how much of the entropic
    information should be used to correct the accuracy score. A value of
    0 yields the ordinary accuracy score, whereas with a value of 1 one
    would get a score close to 0 for each feature that is correctly classified
    albeit with poor certainty. The default is set to .5.

    Args:
        y_true (array): The 1D array of the ground truth for each sample.

        Y_pred (array): The 2D array of predicted probabilities for each
            sample.

        classes (array9: The 1D array of the classes (normally the
            ``classes_`` attribute of a scikit-learn estimator.

        alpha (float): A parameter between 0 and 1 weighting the entropy
            contribution to the final score. With ``alpha=0`` the result
            coicides with the ordinary accuracy score.

    Returns:
        (float) the entropy-corrected accuracy score.
    """
    y_pred = classes[np.argmax(Y_pred, axis=1)]

    h = np.apply_along_axis(lambda x: entropy(x, base=len(x)), axis=1, arr=Y_pred,)

    return np.mean(np.where(y_true == y_pred, 1 - alpha * h, alpha * h))


def mean_entropy(Y):
    """Compute the mean entropy.

    The result is a normalised mean entropy of all the rows of ``Y``.
    """
    return np.mean(
        np.apply_along_axis(lambda x: entropy(x, base=len(x)), axis=1, arr=Y,)
    )
