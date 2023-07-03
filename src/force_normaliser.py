"""
File:         force_normaliser.py
Created:      2021/03/02
Last Changed: 2021/09/23
Author:       M.Vochteloo

Copyright (C) 2020 University Medical Center Groningen.

A copy of the BSD 3-Clause "New" or "Revised" License can be found in the
LICENSE file in the root directory of this source tree.
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
        if data.shape[0] == np.size(self.samples):
            # normalise per column
            axis = 0
        elif data.shape[1] == np.size(self.samples):
            # normalise per row
            axis = 1
        else:
            self.log.error("Matrix and sample shape do not match.")
            exit()

        # Force normalise.
        normal = np.empty(data.shape, dtype=np.float64)
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
