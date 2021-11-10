#!/usr/bin/env python3

"""
File:         plot_force_normalise.py
Created:      2021/02/0/04
Last Changed:
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
from __future__ import print_function
from pathlib import Path
import os

# Third party imports.
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Local application imports.

# Metadata
__program__ = "Plot Force Normalise"
__author__ = "Martijn Vochteloo"
__maintainer__ = "Martijn Vochteloo"
__email__ = "m.vochteloo@rug.nl"
__license__ = "GPLv3"
__version__ = 1.0
__description__ = "{} is a program developed and maintained by {}. " \
                  "This program is licensed under the {} license and is " \
                  "provided 'as-is' without any warranty or indemnification " \
                  "of any kind.".format(__program__,
                                        __author__,
                                        __license__)


"""
Syntax:
./plot_force_normalise.py -h
"""


class main():
    def __init__(self):
        # Set variables.
        self.outdir = os.path.join(str(Path(__file__).parent.parent), 'plot')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

    def start(self):
        print("generate data")
        s = pd.Series(self.generate())
        # s.sort_values(inplace=True)
        print(s)

        # print("Rank")
        # rank = s.rank(ascending=True)

        print("Force normalise")
        normal_s = pd.Series(stats.norm.ppf((s.rank(ascending=True) - 0.5) / s.size), index=s.index)
        print(normal_s)

        print("Combine")
        df = pd.concat([s, normal_s], axis=1)
        df.columns = ["x", "norm"]
        df.reset_index(drop=False, inplace=True)
        df_m = df.melt(id_vars=["index"])
        print(df_m)

        print("Plot")
        self.plot_distribution(df["x"], "before")
        self.plot_distribution(df["norm"], "after")

        sns.set(rc={'figure.figsize': (10, 7.5)})
        sns.set_style("ticks")
        fig, ax = plt.subplots()
        sns.despine(fig=fig, ax=ax)

        sns.lineplot(data=df_m,
                     x="variable",
                     y="value",
                     units="index",
                     estimator=None,
                     ax=ax)

        fig.savefig(os.path.join(self.outdir, "simulated_force_normal2.png"))
        plt.close()

    @staticmethod
    def generate(median=0.2, err=0.1, outlier_err=0.2, size=80, outlier_size=10):
        errs = err * np.random.rand(size) * np.random.choice((-1, 1), size)
        data = median + errs

        lower_errs = outlier_err * np.random.rand(outlier_size)
        lower_outliers = median - err - lower_errs

        upper_errs = outlier_err * np.random.rand(outlier_size)
        upper_outliers = median + err + upper_errs

        data = np.concatenate((data, lower_outliers, upper_outliers))
        np.random.shuffle(data)

        data[data < 0] = 0

        return data

    def plot_distribution(self, data, name):
        sns.set_style("darkgrid", {"axes.facecolor": ".9"})
        fig, ax = plt.subplots()
        g = sns.distplot(data)
        g.set_title(name)
        g.set_ylabel('frequency')
        g.set_xlabel('cell fraction')
        fig.savefig(os.path.join(self.outdir, "{}_distribution.png".format(name)))


if __name__ == '__main__':
    m = main()
    m.start()
