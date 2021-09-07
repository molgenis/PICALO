"""
File:         force_normaliser.py
Created:      2021/03/02
Last Changed: 2020/07/28
Author:       M.Vochteloo

Copyright (C) 2020 M.Vochteloo

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

A copy of the GNU General Public License can be found in the LICENSE file in the
root directory of this source tree. If not, see <https://www.gnu.org/licenses/>.
"""

# Standard imports.

# Third party imports.
import pandas as pd
import numpy as np
from scipy.special import ndtri

# Local application imports.


class ForceNormaliser:
    def __init__(self, dataset_m, samples, log):
        self.dataset_m = dataset_m
        self.samples = samples
        self.log = log

    def process(self, data):
        # Make 1d arrays 2d.
        squeeze = False
        if data.ndim == 1:
            squeeze = True
            data = data[:, np.newaxis]

        # Check which axis to use.
        axis = None
        if data.shape[0] == len(self.samples):
            # normalise per column
            axis = 0
        elif data.shape[1] == len(self.samples):
            # normalise per row
            axis = 1
        else:
            self.log.error("Matrix and sample shape do not match.")
            exit()

        # Force normalise.
        normal = np.empty_like(data, dtype=np.float64)
        for cohort_index in range(self.dataset_m.shape[1]):
            mask = self.dataset_m[:, cohort_index].astype(bool)
            if np.sum(mask) > 0:
                if axis == 1:
                    normal[:, mask] = self.force_normalise(data[:, mask], axis=axis)
                elif axis == 0:
                    normal[mask, :] = self.force_normalise(data[mask, :], axis=axis)

        if squeeze:
            normal = np.squeeze(normal)

        return normal

    @staticmethod
    def force_normalise(x, axis):
        return ndtri((pd.DataFrame(x).rank(axis=axis, ascending=True) - 0.5) / x.shape[axis])
