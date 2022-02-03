#!/usr/bin/env python3

"""
File:         manim_scene.py
Created:      2022/02/02
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
import random
import math

# Third party imports.
import numpy as np
import pandas as pd
from scipy import stats
from manimlib import *

# Local application imports.

# Metadata
__program__ = "Manim Scene"
__author__ = "Martijn Vochteloo"
__maintainer__ = "Martijn Vochteloo"
__email__ = "m.vochteloo@st.hanze.nl"
__license__ = "GPLv3"
__version__ = 1.0
__description__ = "{} is a program developed and maintained by {}. " \
                  "This program is licensed under the {} license and is " \
                  "provided 'as-is' without any warranty or indemnification " \
                  "of any kind.".format(__program__,
                                        __author__,
                                        __license__)


class GraphicalAbstract(Scene):
    def construct(self):
        example_index = 168
        example_name = "Jane Doe"
        n_points = 500
        context_sd = 1

        X = self.generate_base_model(example_index=example_index,
                                     example_name=example_name,
                                     n_points=n_points,
                                     context_sd=context_sd
                                     )

        # Generate ieQTL model.
        ieqtl, betas = self.generate_ieqtl_model(maf=0.41,
                                                 base_matrix=X,
                                                 error_sd=0.4,
                                                 betas=np.array([-2.5, 2.1, 0.5, 0.6]),
                                                 base_seed=1,
                                                 )
        regression_lines = self.calc_regression_lines(ieqtl=ieqtl)

        example_genotype = ieqtl.loc[example_name, "genotype"]

        x_range = (math.floor(ieqtl["context"].min()) - 1, math.ceil(ieqtl["context"].max()) + 1, 1)
        y_range = (math.floor(ieqtl["expression"].min()), math.ceil(ieqtl["expression"].max()), 1)

        axes = Axes(
            x_range=x_range,
            y_range=y_range,
            axis_config={
                "include_tip": False,
                "numbers_to_exclude": [0],
                "stroke_width": 1,
            }
        )
        axes.add_coordinate_labels(
            font_size=20,
            num_decimal_places=0,
        )
        axis_labels = axes.get_axis_labels(x_label_tex='context',
                                           y_label_tex='expression').set_color(GREY)

        self.play(
            DrawBorderThenFill(axes),
            Write(axis_labels)
        )

        genotype_groups = [0, 1, 2]
        example_dot = None
        example_text = None
        dots = {x: [] for x in genotype_groups}
        for i, row in ieqtl.iterrows():
            color = BLACK
            if row["genotype"] == 0:
                color = GREEN
            elif row["genotype"] == 1:
                color = BLUE
            elif row["genotype"] == 2:
                color = RED

            dot = Dot(color=color, opacity=0.5)
            dot.move_to(axes.c2p(row["context"], row["expression"]))

            if i == example_name:
                example_dot = dot

                example_text = Text(example_name).scale(0.3)
                example_text.move_to(axes.c2p(row["context"], row["expression"] + 0.5))
            else:
                dots[row["genotype"]].append(dot)

        lines = {}
        for genotype_group, (start, end) in regression_lines.items():
            color = BLACK
            if genotype_group == 0:
                color = GREEN
            elif genotype_group == 1:
                color = BLUE
            elif genotype_group == 2:
                color = RED

            start_dot = Dot()
            start_dot.move_to(axes.c2p(start[0], start[1]))

            end_dot = Dot()
            end_dot.move_to(axes.c2p(end[0], end[1]))

            lines[genotype_group] = Line(start_dot, end_dot, color=color)

        self.play(
            FadeIn(VGroup(*dots[0])),
            FadeIn(VGroup(*dots[1])),
            FadeIn(VGroup(*dots[2])),
            FadeIn(lines[0]),
            FadeIn(lines[1]),
            FadeIn(lines[2]),
            FadeIn(example_dot)
        )
        self.wait(2)

        self.play(
            FadeOut(VGroup(*dots[0])),
            FadeOut(VGroup(*dots[1])),
            FadeOut(VGroup(*dots[2]))
        )
        del dots
        self.play(
            FadeIn(example_text)
        )
        self.wait(2)

        ########################################################################

        # Optimize.
        coef_a, coef_b = self.optimize_ieqtl(X=ieqtl, betas=betas)
        opt_context = self.calc_vertex_xpos(a=coef_a, b=coef_b)

        ieqtl_opt = ieqtl.copy()
        ieqtl_opt["context"] = opt_context
        ieqtl_opt["interaction"] = ieqtl_opt["context"] * ieqtl_opt["genotype"]
        ieqtl_opt["y_hat"] = self.predict(X=ieqtl_opt[["intercept", "genotype", "context", "interaction"]], betas=betas)

        example_optimal = ieqtl_opt.loc[example_name, "context"]

        ########################################################################

        x_tracker = ValueTracker(ieqtl.loc[example_name, "context"])
        f_always(
            example_dot.move_to,
            lambda: axes.c2p(x_tracker.get_value(), ieqtl.loc[example_name, "expression"])
        )
        f_always(
            example_text.move_to,
            lambda: axes.c2p(x_tracker.get_value(), ieqtl.loc[example_name, "expression"] + 0.5)
        )

        self.play(x_tracker.animate.set_value(x_range[0]),
                  run_time=2)
        self.play(x_tracker.animate.set_value(x_range[1]),
                  run_time=3)
        self.play(x_tracker.animate.set_value(example_optimal),
                  run_time=2)
        self.wait(2)

        self.play(
            FadeOut(example_dot),
            FadeOut(example_text)
        )

        dots = []
        animations = []
        for i in range(ieqtl.shape[0]):
            ieqltl_row = ieqtl.iloc[i, :]
            ieqtl_opt_row = ieqtl_opt.iloc[i, :]

            color = BLACK
            if ieqltl_row["genotype"] == 0:
                color = GREEN
            elif ieqltl_row["genotype"] == 1:
                color = BLUE
            elif ieqltl_row["genotype"] == 2:
                color = RED

            dot = Dot(color=color, opacity=0.5)
            dot.move_to(axes.c2p(ieqltl_row["context"], ieqltl_row["expression"]))
            dots.append(dot)

            animation = ApplyMethod(dot.move_to, axes.c2p(ieqtl_opt_row["context"], ieqtl_opt_row["expression"]))
            animations.append(animation)

        self.play(
            FadeIn(VGroup(*dots))
        )
        self.play(*animations)

    @staticmethod
    def generate_base_model(example_index=0, example_name="Jane Doe",
                            n_points=100, context_sd=1):
        X = pd.DataFrame(np.nan,
                         columns=["expression", "intercept", "genotype", "context", "interaction", "y_hat", "residuals"],
                         index=["sample{}".format(i) for i in range(n_points)])

        X.index = [x if i != example_index else example_name for i, x in enumerate(X.index)]

        X["intercept"] = 1
        np.random.seed(0)
        X["context"] = np.random.normal(0, context_sd, n_points)

        return X

    def generate_ieqtl_model(self, maf, base_matrix, error_sd, betas, base_seed=0):
        X = base_matrix.copy()

        # Generate genotype array.
        genotype = np.array([0] * round((1 - maf) ** 2. * X.shape[0]) +
                            [1] * round(2 * (1 - maf) * maf * X.shape[0]) +
                            [2] * round(maf ** 2 * X.shape[0])
                            )
        random.seed(base_seed)
        random.shuffle(genotype)
        X["genotype"] = genotype

        # Calculate interaction term.
        X["interaction"] = X["context"] * X["genotype"]

        # Calculate expression.
        np.random.seed(base_seed + 1)
        X["expression"] = X.loc[:, ["intercept", "genotype", "context", "interaction"]].dot(betas) + np.random.normal(0, error_sd, X.shape[0])

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

    def calc_regression_lines(self, ieqtl):
        regression_lines = {}
        for genotype_group in ieqtl["genotype"].unique():
            subset = ieqtl.loc[ieqtl["genotype"] == genotype_group, :].copy()
            betas = self.fit(X=subset[["intercept", "context"]], y=subset["expression"])
            subset["y_hat"] = self.predict(X=subset[["intercept", "context"]], betas=betas)
            subset.sort_values("context", inplace=True)
            regression_lines[genotype_group] = [np.array([subset.iloc[0, :]["context"], subset.iloc[0, :]["y_hat"], 0]),
                                                np.array([subset.iloc[-1, :]["context"], subset.iloc[-1, :]["y_hat"], 0])
                                                ]
        return regression_lines

    def optimize_ieqtl(self, X, betas, x1=-4, x3=4):
        # Initialize the evaluation matrix.
        eval_df = X.copy()

        # Calculate the log likelihood.
        eval_df["log_likelihood"] = np.log(stats.norm.pdf(X["residuals"], 0, 1))
        ll = eval_df["log_likelihood"].sum()

        # Evaluate the log likelihood function for each sample on position
        # x1 and x3.
        y_values_df = pd.DataFrame(np.nan, index=eval_df.index, columns=[x1, x3])
        for eval_pos in [x1, x3]:
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
        coef_a, coef_b = self.calc_parabola_vertex(x1=x1,
                                                   x2=X["context"],
                                                   x3=x3,
                                                   y1=y_values_df[x1],
                                                   y2=ll,
                                                   y3=y_values_df[x3])

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


if __name__ == "__main__":
    m = main()
    m.start()
