#!/usr/bin/env python3

"""
File:         picalo_graphical_overview.py
Created:      2022/02/02
Last Changed: 2022/07/20
Author:       M.Vochteloo

Copyright (C) 2020 University Medical Center Groningen.

A copy of the BSD 3-Clause "New" or "Revised" License can be found in the
LICENSE file in the root directory of this source tree.
"""

# Standard imports.
from __future__ import print_function
import random
import os

# Third party imports.
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib
from matplotlib import pyplot as plt

# Local application imports.

# Metadata
__program__ = "PICALO Graphical Plot"
__author__ = "Martijn Vochteloo"
__maintainer__ = "Martijn Vochteloo"
__email__ = "m.vochteloo@rug.nl"
__license__ = "BSD (3-Clause)"
__version__ = 1.0
__description__ = "{} is a program developed and maintained by {}. " \
                  "This program is licensed under the {} license and is " \
                  "provided 'as-is' without any warranty or indemnification " \
                  "of any kind.".format(__program__,
                                        __author__,
                                        __license__)


class main():
    def __init__(self):
        self.n_points = 500
        self.color_map = {0.0: "#E69F00",
                          1.0: "#0072B2",
                          2.0: "#D55E00"}
        self.palette = {"ieQTL1": "#E69F00",
                        "ieQTL2": "#0072B2",
                        "sum": "#009E73"}
        self.example_name = "Jane Doe"
        self.example_index = 168

        # Base model parameters.
        self.context_sd = 1

        # Model 1 parameters.
        self.maf1 = 0.42
        self.error_sd1 = 0.4
        self.betas1 = np.array([-2.5, 2.1, 0.5, 0.6])

        # Model 2 parameters.
        self.maf2 = 0.34
        self.error_sd2 = 0.6
        self.betas2 = np.array([-2.5, 3, 0.6, 1])

        # Set ll function x evaluation points.
        self.x1, self.x3 = -4, 4

        # Set the right pdf font for exporting.
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42

        # Create the output directory
        self.outdir = os.path.join(str(os.path.dirname(os.path.abspath(__file__))), 'picalo_graphical_overview')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

    def start(self):
        X = self.generate_base_model()

        # Generate ieQTL models.
        ieqtl1, betas1 = self.generate_ieqtl_model(maf=self.maf1,
                                                   base_matrix=X,
                                                   error_sd=self.error_sd1,
                                                   betas=self.betas1,
                                                   base_seed=1,
                                                   )
        ieqtl2, betas2 = self.generate_ieqtl_model(maf=self.maf2,
                                                   base_matrix=X,
                                                   error_sd=self.error_sd2,
                                                   betas=self.betas2,
                                                   base_seed=3,
                                                   )

        # Plot before.
        xlim1, ylim1 = self.plot_ieqtl(X=ieqtl1,
                                       title="ieQTL1",
                                       xlabel="eQTL context",
                                       ylabel="gene expression",
                                       filename="ieQTL1_before"
                                       )
        xlim2, ylim2 = self.plot_ieqtl(X=ieqtl2,
                                       title="ieQTL2",
                                       xlabel="eQTL context",
                                       ylabel="gene expression",
                                       filename="ieQTL2_before"
                                       )

        # Optimize ieQTLs.
        coef_a1, coef_b1 = self.optimize_ieqtl(X=ieqtl1, betas=betas1)
        coef_a2, coef_b2 = self.optimize_ieqtl(X=ieqtl2, betas=betas2)

        coef_a = coef_a1 + coef_a2
        coef_b = coef_b1 + coef_b2

        opt_context = self.calc_vertex_xpos(a=coef_a, b=coef_b)

        # Plot the optimization curve.
        self.plot_log_likelihood(context=X.loc[self.example_name, "context"],
                                 a1=coef_a1.loc[self.example_name],
                                 b1=coef_b1.loc[self.example_name],
                                 xlim1=xlim1,
                                 a2=coef_a2.loc[self.example_name],
                                 b2=coef_b2.loc[self.example_name],
                                 xlim2=xlim2,
                                 palette=self.palette,
                                 title=self.example_name,
                                 xlabel="context",
                                 ylabel="log likelihood",
                                 filename="log_likelihood_{}".format(self.example_name),
                                 )

        # Change context values.
        ieqtl1_opt = ieqtl1.copy()
        ieqtl1_opt["context"] = opt_context
        ieqtl1_opt["interaction"] = ieqtl1_opt["context"] * ieqtl1_opt["genotype"]
        ieqtl1_opt["y_hat"] = self.predict(X=ieqtl1_opt[["intercept", "genotype", "context", "interaction"]], betas=betas1)

        ieqtl2_opt = ieqtl2.copy()
        ieqtl2_opt["context"] = opt_context
        ieqtl2_opt["interaction"] = ieqtl2_opt["context"] * ieqtl2_opt["genotype"]
        ieqtl2_opt["y_hat"] = self.predict(X=ieqtl2_opt[["intercept", "genotype", "context", "interaction"]], betas=betas2)

        _, _ = self.plot_ieqtl(X=ieqtl1_opt,
                               xlim=xlim1,
                               ylim=ylim1,
                               title="ieQTL1 - optimized",
                               xlabel="eQTL context - optimized",
                               ylabel="gene expression",
                               filename="ieQTL1_after"
                               )
        _, _ = self.plot_ieqtl(X=ieqtl2_opt,
                               xlim=xlim2,
                               ylim=ylim2,
                               title="ieQTL2 - optimized",
                               xlabel="eQTL context - optimized",
                               ylabel="gene expression",
                               filename="ieQTL2_after"
                               )

    def generate_base_model(self):
        X = pd.DataFrame(np.nan,
                         columns=["expression", "intercept", "genotype", "context", "interaction", "y_hat", "residuals"],
                         index=["sample{}".format(i) for i in range(self.n_points)])

        X.index = [x if i != self.example_index else self.example_name for i, x in enumerate(X.index)]

        X["intercept"] = 1
        np.random.seed(0)
        X["context"] = np.random.normal(0, self.context_sd, self.n_points)

        return X

    def generate_ieqtl_model(self, maf, base_matrix, error_sd, betas, base_seed=0):
        X = base_matrix.copy()

        # Generate genotype array.
        genotype = np.array([0] * round((1 - maf) ** 2. * self.n_points) +
                            [1] * round(2 * (1 - maf) * maf * self.n_points) +
                            [2] * round(maf ** 2 * self.n_points)
                            )
        random.seed(base_seed)
        random.shuffle(genotype)
        X["genotype"] = genotype

        # Calculate interaction term.
        X["interaction"] = X["context"] * X["genotype"]

        # Calculate expression.
        np.random.seed(base_seed + 1)
        X["expression"] = X.loc[:, ["intercept", "genotype", "context", "interaction"]].dot(betas) + np.random.normal(0, error_sd, self.n_points)

        # Model the ieQTL.
        betas = self.fit(X=X[["intercept", "genotype", "context", "interaction"]], y=X["expression"])
        X["y_hat"] = self.predict(X=X[["intercept", "genotype", "context", "interaction"]], betas=betas)
        X["residuals"] = X["expression"] - X["y_hat"]

        return X, betas

    @staticmethod
    def fit(X, y):
        return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    @staticmethod
    def predict(X, betas):
        return np.dot(X, betas)

    # def optimize_ieqtl(self, X, betas):
    #     # Initialize the evaluation matrix.
    #     eval_df = X.copy()
    #
    #     # Calculate the residuals squared.
    #     eval_df["residuals_squared"] = X["residuals"] * X["residuals"]
    #     rss = eval_df["residuals_squared"].sum()
    #
    #     # Evaluate the log likelihood function for each sample on position
    #     # x1 and x3.
    #     y_values_df = pd.DataFrame(np.nan, index=eval_df.index, columns=[self.x1, self.x3])
    #     for eval_pos in [self.x1, self.x3]:
    #         # Replace the context with the evaluation position and recalculate
    #         # the interaction term.
    #         eval_df.loc[:, "context"] = eval_pos
    #         eval_df["interaction"] = eval_df["context"] * eval_df["genotype"]
    #
    #         # Calculate the y_hat of the model for the original betas.
    #         eval_df["adj_y_hat"] = self.predict(X=eval_df[["intercept", "genotype", "context", "interaction"]], betas=betas)
    #
    #         # Calculate the residuals squared per sample for this model.
    #         eval_df["adj_residuals"] = eval_df["expression"] - eval_df["adj_y_hat"]
    #         eval_df["adj_residuals_squared"] = eval_df["adj_residuals"] * eval_df["adj_residuals"]
    #
    #         # Save the adjusted residuals squared.
    #         y_values_df.loc[:, eval_pos] = rss - eval_df["residuals_squared"] + eval_df["adj_residuals_squared"]
    #
    #     # Determine the coefficients.
    #     coef_a, coef_b = self.calc_parabola_vertex(x1=self.x1,
    #                                                x2=X["context"],
    #                                                x3=self.x3,
    #                                                y1=y_values_df[self.x1],
    #                                                y2=rss,
    #                                                y3=y_values_df[self.x3])
    #
    #     return coef_a, coef_b

    def optimize_ieqtl(self, X, betas):
        # Initialize the evaluation matrix.
        eval_df = X.copy()

        # Calculate the log likelihood.
        eval_df["log_likelihood"] = np.log(stats.norm.pdf(X["residuals"], 0, 1))
        ll = eval_df["log_likelihood"].sum()

        # Evaluate the log likelihood function for each sample on position
        # x1 and x3.
        y_values_df = pd.DataFrame(np.nan, index=eval_df.index, columns=[self.x1, self.x3])
        for eval_pos in [self.x1, self.x3]:
            # Replace the context with the evaluation position and recalculate
            # the interaction term.
            eval_df.loc[:, "context"] = eval_pos
            eval_df["interaction"] = eval_df["context"] * eval_df["genotype"]

            # Calculate the y_hat of the model for the original betas.
            eval_df["adj_y_hat"] = self.predict(X=eval_df[["intercept", "genotype", "context", "interaction"]], betas=betas)

            # Calculate the residuals squared per sample for this model.
            eval_df["adj_residuals"] = eval_df["expression"] - eval_df["adj_y_hat"]
            eval_df["adj_log_likelihood"] = np.log(stats.norm.pdf(eval_df["adj_residuals"], 0, 1))

            # Save the adjusted residuals squared.
            y_values_df.loc[:, eval_pos] = ll - eval_df["log_likelihood"] + eval_df["adj_log_likelihood"]

        # Determine the coefficients.
        coef_a, coef_b = self.calc_parabola_vertex(x1=self.x1,
                                                   x2=X["context"],
                                                   x3=self.x3,
                                                   y1=y_values_df[self.x1],
                                                   y2=ll,
                                                   y3=y_values_df[self.x3])

        return coef_a, coef_b

    @staticmethod
    def calc_parabola_vertex(x1, x2, x3, y1, y2, y3):
        """
        Function to return the coefficient representation of the (simplified)
        log likelihood function: a x^2 + b x + c. The c term is not returned
        since this does not influence the vertex x position.

        :param x1: x position of point 1
        :param x2: x position of point 2
        :param x3: x position of point 3
        :param y1: y position of point 1
        :param y2: y position of point 2
        :param y2: y position of point 3

        Adapted and modifed to get the unknowns for defining a parabola:
        http://stackoverflow.com/questions/717762/how-to-calculate-the-vertex-of-a-parabola-given-three-points
        """
        denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
        a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
        b = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / denom
        # c = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom

        return a, b

    @staticmethod
    def calc_vertex_xpos(a, b):
        return -b / (2 * a)

    def plot_ieqtl(self, X, xlim=None, ylim=None, xlabel="", ylabel="",
                   title="", filename=None, annot=None):
        # Calculate R2.
        pearsonr = self.calc_pearsonr_vector(x=X["expression"], y=X["y_hat"])
        r_squared = pearsonr * pearsonr

        sns.set(rc={'figure.figsize': (12, 9)})
        sns.set_style("ticks")
        fig, ax = plt.subplots()
        sns.despine(fig=fig, ax=ax)

        for i, group_id in enumerate([2.0, 1.0, 0.0]):
            subset = X.loc[X["genotype"] == group_id, :].copy()

            if len(subset.index) > 1:
                if self.example_name not in subset.index or 1 == 1:
                    sns.regplot(x="context",
                                y="expression",
                                data=subset,
                                scatter_kws={'facecolors': self.color_map[group_id],
                                             's': 50,
                                             'linewidth': 0,
                                             'alpha': 0.8},
                                line_kws={"color": self.color_map[group_id],
                                          "alpha": 1},
                                ax=ax
                                )
                else:
                    sns.regplot(x="context",
                                y="expression",
                                data=subset.loc[[x for x in subset.index if x != self.example_name], :],
                                scatter_kws={'facecolors': "#000000",
                                             's': 50,
                                             'linewidth': 0,
                                             'alpha': 0.8},
                                line_kws={"color": self.color_map[group_id],
                                          "alpha": 1},
                                ax=ax
                                )
                    sns.scatterplot(x="context",
                                    y="expression",
                                    data=subset.loc[[self.example_name], :],
                                    color=self.color_map[group_id],
                                    s=200,
                                    ax=ax
                                    )

        # Add the text.
        ax.annotate(
            'R\u00b2 = {:.2f}'.format(r_squared),
            xy=(0.05, 0.9),
            xycoords=ax.transAxes,
            color="#000000",
            alpha=1,
            fontsize=30,
            fontweight='bold')
        if annot is not None:
            # Add the text.
            ax.annotate(
                'betas = {}'.format(annot),
                xy=(0.05, 0.80),
                xycoords=ax.transAxes,
                color="#000000",
                alpha=1,
                fontsize=20,
                fontweight='bold')

        ax.set_title(title,
                     fontsize=30,
                     fontweight='bold')
        ax.set_ylabel(ylabel,
                      fontsize=22,
                      fontweight='bold')
        ax.set_xlabel(xlabel,
                      fontsize=22,
                      fontweight='bold')

        if xlim is None:
            current_xlim = ax.get_xlim()
            ax.set_xlim((current_xlim[0] - 0.5, current_xlim[1] + 0.5))
        else:
            ax.set_xlim(xlim)

        if ylim is None:
            current_ylim = ax.get_ylim()
            ax.set_ylim((current_ylim[0] - 0.5, current_ylim[1] + 0.5))
        else:
            ax.set_ylim(ylim)

        plt.show()
        fig.savefig(os.path.join(self.outdir, "{}.pdf".format(filename)))

        return ax.get_xlim(), ax.get_ylim()

    @staticmethod
    def calc_pearsonr_vector(x, y):
        x_dev = x - np.mean(x)
        y_dev = y - np.mean(y)
        dev_sum = np.sum(x_dev * y_dev)
        x_rss = np.sum(x_dev * x_dev)
        y_rss = np.sum(y_dev * y_dev)
        return dev_sum / np.sqrt(x_rss * y_rss)

    def plot_log_likelihood(self, context, a1, b1, xlim1, a2, b2, xlim2, stepsize=100, xlabel="", ylabel="", title="", palette=None, filename=None):
        # Generate data.
        line_data = []
        point_data = []
        for (a, b, xlim, label) in [(a1, b1, xlim1, "ieQTL1"),
                                    (a2, b2, xlim2, "ieQTL2"),
                                    (a1 + a2, b1 + b2, (min(xlim1[0], xlim2[0]), max(xlim1[1], xlim2[1])), "sum")]:
            for x in np.linspace(xlim[0], xlim[1], stepsize):
                line_data.append([label, x, a*x**2 + b*x + 0])

            vertex_xpos = self.calc_vertex_xpos(a=a, b=b)
            point_data.append([label, vertex_xpos, a*vertex_xpos**2 + b*vertex_xpos + 0])
        line_df = pd.DataFrame(line_data, columns=["hue", "x", "y"])
        point_df = pd.DataFrame(point_data, columns=["hue", "x", "y"])

        sns.set(rc={'figure.figsize': (12, 9)})
        sns.set_style("ticks")
        fig, ax = plt.subplots()
        sns.despine(fig=fig, ax=ax)

        sns.lineplot(data=line_df,
                     x="x",
                     y="y",
                     hue="hue",
                     palette=palette,
                     estimator=None,
                     legend=None,
                     ax=ax)
        sns.scatterplot(data=point_df,
                        x="x",
                        y="y",
                        hue="hue",
                        s=200,
                        palette=palette,
                        legend=None,
                        ax=ax)

        for _, row in point_df.iterrows():
            color = "#000000"
            if palette is not None:
                color = palette[row["hue"]]
            ax.axvline(row["x"], ls='--', color=color, zorder=-1)
        ax.axvline(context, ls='-', color="#000000", zorder=-1)

        i = 0
        for i, (a, b, label) in enumerate([(a1, b1, "ieQTL1"),
                                           (a2, b2, "ieQTL2"),
                                           (a1 + a2, b1 + b2, "sum")]
                                          ):

            color = "#000000"
            if palette is not None:
                color = palette[label]

            # Add the text.
            ax.annotate(
                '{}: {:.2f}x\u00b2 + {:.2f}x'.format(label, a, b),
                xy=(0.67, 0.05 + (i * 0.05)),
                xycoords=ax.transAxes,
                color=color,
                alpha=1,
                fontsize=20,
                fontweight='bold')

        # Add the text.
        ax.annotate(
            'Initial guess: {:.2f}'.format(context),
            xy=(0.67, 0.05 + ((i+1) * 0.05)),
            xycoords=ax.transAxes,
            color="#000000",
            alpha=1,
            fontsize=20,
            fontweight='bold')

        ax.set_title(title,
                     fontsize=30,
                     fontweight='bold')
        ax.set_ylabel(ylabel,
                      fontsize=22,
                      fontweight='bold')
        ax.set_xlabel(xlabel,
                      fontsize=22,
                      fontweight='bold')

        plt.show()
        fig.savefig(os.path.join(self.outdir, "{}.pdf".format(filename)))


if __name__ == "__main__":
    m = main()
    m.start()
