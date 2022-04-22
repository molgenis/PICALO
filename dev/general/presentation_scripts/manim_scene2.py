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
__email__ = "m.vochteloo@rug.nl"
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


class GraphicalAbstractPart4(Scene):
    def construct(self):
        func = lambda x: -x * np.sin(x)
        # func = lambda x: x ** 2 - 8 * x
        # func = lambda x: np.sin(x) + np.sin((10.0 / 3.0) * x)
        x_range = (0, 10, 2)
        radius = 0.15
        y_values = np.array([func(x) for x in np.arange(x_range[0], x_range[1], 0.01)])
        y_step = 2
        y_range = ((math.floor(np.min(y_values) / y_step) * y_step, (math.ceil(np.max(y_values) / y_step) + 2) * y_step, y_step))

        # Construct the axis for the scatterplot.
        axes = Axes(
            x_range=x_range,
            y_range=y_range,
            axis_config={
                "include_tip": False,
                "stroke_width": 1,
            }
        ).set_color(GREY)
        axes.add_coordinate_labels(
            font_size=20,
            num_decimal_places=0,
        ).set_color(GREY)

        graph = axes.get_graph(func, color=WHITE)

        self.play(
            DrawBorderThenFill(axes),
            run_time=1
        )
        self.play(
            ShowCreation(graph),
            run_time=2
        )
        self.wait()

        local_text = Text("local minima").scale(0.6).move_to(axes.c2p(2.1, func(2.1) - 1.5))
        global_text = Text("global minima").scale(0.6).move_to(axes.c2p(8, func(8) - 1.5))
        self.play(
            FadeIn(VGroup(local_text, global_text)),
            run_time=1
        )
        self.wait()

        # Drop ball 1.
        start_x = 7
        start_y = y_range[1]
        dot1 = Dot(point=axes.c2p(start_x, start_y), radius=radius, color=WHITE)
        data = calculate_drop(x=start_x, y=start_y, func=func, x_range=x_range)
        print(data)
        run_time = (data.iloc[-1, :]["time"] / data.shape[0]) * 50
        print(run_time, data.shape[0] * run_time)
        for i, row in data.iterrows():
            self.play(dot1.animate.move_to(axes.c2p(row["x"], row["y"])),
                      run_time=run_time)
        self.wait(3)

        # Drop ball 2.
        start_x = 4
        dot2 = Dot(point=axes.c2p(start_x, start_y), radius=radius, color=WHITE)
        data = calculate_drop(x=start_x, y=start_y, func=func, x_range=x_range)
        print(data)
        run_time = (data.iloc[-1, :]["time"] / data.shape[0]) * 50
        print(run_time, data.shape[0] * run_time)
        for i, row in data.iterrows():
            self.play(dot2.animate.move_to(axes.c2p(row["x"], row["y"])),
                      run_time=run_time)
        self.wait()

        # Remove.
        self.play(
            FadeOut(VGroup(dot1, dot2)),
            run_time=1
        )

        combined_data = []
        max_time = 0
        max_rows = 0
        for start_x in np.arange(1, 10, 1):
            dot = Dot(point=axes.c2p(start_x, start_y), radius=radius, color=WHITE)
            data = calculate_drop(x=start_x, y=start_y, func=func, x_range=x_range)

            if data.iloc[-1, :]["time"] > max_time:
                max_time = data.iloc[-1, :]["time"]

            if data.shape[0] > max_rows:
                max_rows = data.shape[0]

            combined_data.append((dot, data))

        run_time = (max_time / max_rows) * 50
        for i in range(max_rows):
            animations = []
            for (dot, data) in combined_data:
                if i < data.shape[0]:
                    row = data.iloc[i, :]
                    animations.append(ApplyMethod(dot.move_to, axes.c2p(row["x"], row["y"])))
            if len(animations) > 0:
                self.play(*animations,
                          run_time=run_time)
        self.wait()


def calculate_drop(x, y, func, x_range):
    """
    x: starting value of X
    func: the function

    https://www.mathworks.com/matlabcentral/answers/477104-how-would-you-plot-a-graph-which-a-ball-then-rolls-down-say-a-y-x-2-graph
    """
    # Start vectors.
    dx = 0.1  # step used to compute numerical derivatives
    dt = 0.00005  # integration time step
    grav = 9.806  # acceleration due to gravity
    drop_speed = 2500
    roll_speed = 1000
    speed = drop_speed  # initial speed
    G = [0, -grav]  # gravity vector
    max_steps = 1000

    # initial energy state (per unit mass)
    Ep = grav * y  # potential energy
    Ek = 0.5 * speed ** 2  # kinetic energy
    Etot = Ep + Ek  # total system energy

    # initialize saved data table
    data = np.empty((max_steps + 1, 7), dtype=np.float64)
    data[0, :] = np.array([0, x, y, speed, Ep, Ek, Etot])

    # simulate the falling.
    cnt = 1
    stop = False
    for i in range(max_steps):
        time = cnt * dt

        if y > func(x):
            # update speed and y pos.
            speed = speed + grav * dt
            y = y - (speed * dt)

            # update energy states
            Ep = grav * y
            Ek = 0.5 * speed ** 2
            Etot = Ep + Ek

            # set speed to zero once we land.
            if y <= func(x):
                y = func(x)
                if func(x + dx) > func(x):
                    speed = roll_speed * -1
                else:
                    speed = roll_speed

            # save data
            data[cnt, :] = np.array([time, x, y, speed, Ep, Ek, Etot])

        else:
            stop = True

            dy = (func(x + dx / 2) - func(x - dx / 2)) / dx  # first derivative
            deltax = dx  # step change in X value
            deltay = dy * dx  # corresponding change in Y value
            mag = np.sqrt(deltax ** 2 + deltay ** 2)  # magnitude of step change

            # compute the unit tangent vector
            Tx = deltax / mag
            Ty = deltay / mag
            T = [Tx, Ty]  # unit tangent vector

            # compute accelerations
            At = np.dot(G, T)  # acceration in the tangential direction

            # update states (numerical integration)
            speed = speed + At * dt
            delta = speed * dt  # dstance traveled along curve
            x = x + delta * Tx  # updated X position
            y = func(x)

            # update energy states
            Ep = grav * y
            Ek = 0.5 * speed ** 2
            Etot = Ep + Ek

            # Check if we go uphill.
            if func(x) <= data[cnt-1, 2]:
                stop = False

            # check if we exit the x_range.
            if x <= x_range[0]:
                x = x_range[0]
                stop = True
                print("left border reached")
            if x >= x_range[1]:
                x = x_range[1]
                stop = True
                print("right border reached")

            # save data
            data[cnt, :] = np.array([time, x, y, speed, Ep, Ek, Etot])

            if stop:
                print("going uphill")

        cnt += 1

        if stop:
            break

    data_df = pd.DataFrame(data, columns=["time", "x", "y", "speed", "ep", "ek", "etot"])
    return data_df.iloc[:cnt, :]

