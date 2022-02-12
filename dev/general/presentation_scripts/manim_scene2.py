#!/usr/bin/env python3

"""
File:         manim_scene2.py
Created:      2022/02/10
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
import os

# Third party imports.
import numpy as np
import pandas as pd
from scipy import stats
from manimlib import *

# Local application imports.

# Metadata
__program__ = "Manim Scene 2"
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


class GraphicalAbstractPart3(Scene):
    def construct(self):
        # Defining palette.
        colors = {
            "LL": GREEN_C,
            "RS": ORANGE,
            "LLS_660Q": RED,
            "NTR_AFFY": BLUE_D,
            "LLS_OmniExpr": GREY,
            "CODAM": GREEN_A,
            "PAN": PURPLE,
            "NTR_GONL": BLUE_A,
            "GONL": YELLOW_C,
        }

        # Loading data.
        iterations_df = pd.read_csv(os.path.join("/Users/mvochteloo/Downloads/PICALO_animations/PIC1/iteration.txt.gz"), sep="\t", header=0, index_col=0)
        std_df = pd.read_csv(os.path.join("/Users/mvochteloo/Downloads/PICALO_animations/sample_to_dataset.txt.gz"), sep="\t", header=0, index_col=None)
        info_df = pd.read_csv(os.path.join("/Users/mvochteloo/Downloads/PICALO_animations/PIC1/info.txt.gz"), sep="\t", header=0, index_col=0)

        iterations_df = iterations_df.iloc[:17, :]
        info_df = info_df.iloc[:16, :]

        std_df["hue"] = std_df["dataset"].map(colors)
        if list(iterations_df.columns) != list(std_df["sample"]):
            print("Error, samples do not match.")
            exit()
        colors = list(std_df["hue"])

        n_iterations = iterations_df.shape[0]
        line_xstep = 5
        line_xlim = (0, math.ceil(n_iterations / line_xstep) * line_xstep, line_xstep)

        max_ieqtls = info_df["N"].max()
        line_ystep = 1000
        line_ylim = (0, math.ceil(max_ieqtls / line_ystep) * line_ystep, line_ystep)

        n_samples = iterations_df.shape[1]
        n_samples = 2932
        scatter_xstep = 250
        scatter_xlim = (0, math.ceil(n_samples / scatter_xstep) * scatter_xstep, scatter_xstep)

        min_value = iterations_df.min(axis=1).min()
        max_value = iterations_df.max(axis=1).max()
        scatter_ystep = 1
        scatter_ylim = (math.floor(min_value / scatter_ystep) * scatter_ystep, math.ceil(max_value / scatter_ystep) * scatter_ystep, scatter_ystep)

        print(line_xlim, line_ylim)
        print(scatter_xlim, scatter_ylim)

        ########################################################################

        # Construct the axis for the scatterplot.
        line_axes = Axes(
            x_range=line_xlim,
            y_range=line_ylim,
            axis_config={
                "include_tip": False,
                "stroke_width": 1,
            }
        ).set_color(GREY)
        line_axes.add_coordinate_labels(
            font_size=20,
            num_decimal_places=0,
        ).set_color(GREY)
        line_axes_labels = line_axes.get_axis_labels(
            x_label_tex='iteration',
            y_label_tex='ieQTLs'
        ).set_color(GREY)
        VGroup(line_axes, line_axes_labels).scale(0.5).shift(UP * 1.75)

        # Construct the axis for the scatterplot.
        scatter_axes = Axes(
            x_range=scatter_xlim,
            y_range=scatter_ylim,
            axis_config={
                "include_tip": False,
                "stroke_width": 1,
            }
        ).set_color(GREY)
        scatter_axes_labels = scatter_axes.get_axis_labels(
            x_label_tex='',
            y_label_tex='value'
        ).set_color(GREY)
        VGroup(scatter_axes, scatter_axes_labels).scale(0.5).shift(DOWN * 2)

        # Draw the axes.
        self.play(
            DrawBorderThenFill(line_axes),
            DrawBorderThenFill(scatter_axes),
            Write(line_axes_labels),
            Write(scatter_axes_labels),
            run_time=2
        )

        # Initialize start positions.
        n_ieqtls_dot = Dot(point=line_axes.c2p(0, 0), radius=0.08, color=WHITE, fill_opacity=1)

        sample_dots = []
        for i, (scatter_ypos, color) in enumerate(zip(iterations_df.iloc[0, :], colors)):
            sample_dot = Dot(point=scatter_axes.c2p(i, scatter_ypos), radius=0.02, color=color, fill_opacity=0.8)
            sample_dots.append(sample_dot)

        # Add the text.
        iteration = "start"
        iteration_text = always_redraw(lambda: Text("{}".format(iteration), color=WHITE).scale(0.8).shift(UP * 3.5))

        n_ieqtls = np.nan
        n_ieqtls_text = always_redraw(
            lambda: Text("N = {:,}".format(n_ieqtls), color=GREY_B).scale(0.4).move_to(
                line_axes.c2p(line_xlim[1] * 0.9, line_ylim[1] * 0.8)
            )
        )

        pearson_r = np.nan
        pearson_r_text = always_redraw(
            lambda: Text("r = {:.4f}".format(pearson_r), color=GREY_B).scale(0.4).move_to(
                scatter_axes.c2p(scatter_xlim[1] * 0.9, scatter_ylim[1] * 0.8)
            )
        )

        self.play(
            FadeIn(n_ieqtls_dot),
            FadeIn(VGroup(*sample_dots)),
            FadeIn(iteration_text)
        )
        self.wait(3)

        # Do the iterations.
        run_time = 4
        run_time_change = 0.925
        prev_index = None
        prev_label = None
        for row_index, (iteration_label, row) in enumerate(iterations_df.iterrows()):
            if iteration_label == "start":
                continue
            print(row_index, iteration_label)

            # Calculate the run time.
            run_time = run_time * run_time_change

            # Load the animations.
            animations = []
            for i, (dot, scatter_ypos) in enumerate(zip(sample_dots, row)):
                animation = ApplyMethod(dot.move_to, scatter_axes.c2p(i, scatter_ypos))
                animations.append(animation)

            # Construct the new ieQTLs line.
            line = None
            if prev_index is not None and prev_label is not None:
                line = Line(start=line_axes.c2p(prev_index, info_df.loc[prev_label, "N"]),
                            end=line_axes.c2p(row_index, info_df.loc[iteration_label, "N"]),
                            stroke_width=5,
                            color=WHITE)

            # Play the animations.
            new_n_ieqtls = info_df.loc[iteration_label, "N"]
            if line is not None:
                self.play(
                    *animations,
                    ApplyMethod(n_ieqtls_dot.move_to, line_axes.c2p(row_index, new_n_ieqtls)),
                    ShowCreation(line),
                    run_time=run_time)
            else:
                self.play(
                    *animations,
                    ApplyMethod(n_ieqtls_dot.move_to,
                                line_axes.c2p(row_index, new_n_ieqtls)),
                    run_time=run_time)
            self.wait()

            # Update the text.
            pearson_r = info_df.loc[iteration_label, "Pearson r"]
            n_ieqtls = int(info_df.loc[iteration_label, "N"])
            iteration = iteration_label

            # Save for line.
            prev_index = row_index
            prev_label = iteration_label

            if row_index > 0:
                self.add(n_ieqtls_text, pearson_r_text)
