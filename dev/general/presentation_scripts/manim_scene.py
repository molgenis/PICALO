#!/usr/bin/env python3

"""
File:         manim_scene.py
Created:      2022/02/02
Last Changed: 2022/07/20
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
__program__ = "Manim Scene"
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


class GraphicalAbstractPart1(Scene):
    def construct(self):
        scale_wait = 0.9

        example_index = 168
        example_name = "Jane Doe"
        n_points = 500
        context_sd = 1

        base_model = generate_base_model(example_index=example_index,
                                         example_name=example_name,
                                         n_points=n_points,
                                         context_sd=context_sd
                                         )

        # Generate ieQTL model.
        ieqtl1, betas1 = generate_ieqtl_model(maf=0.41,
                                              base_matrix=base_model,
                                              error_sd=0.4,
                                              betas=np.array([-2.5, 2.1, 0.5, 0.6]),
                                              base_seed=1,
                                              )
        regression_lines1 = calc_regression_lines(ieqtl=ieqtl1)

        # Determine the range of the plot axes.
        context_range1 = (math.floor(ieqtl1["context"].min()) - 1, math.ceil(ieqtl1["context"].max()) + 1, 1)
        scatter_y_range1 = (math.floor(ieqtl1["expression"].min()), math.ceil(ieqtl1["expression"].max()), 2)

        # Construct the axis for the scatterplot.
        scatter_axes1 = Axes(
            x_range=context_range1,
            y_range=scatter_y_range1,
            axis_config={
                "include_tip": False,
                "numbers_to_exclude": [0],
                "stroke_width": 1,
            }
        ).set_color(GREY)
        scatter_axes1.add_coordinate_labels(
            font_size=20,
            num_decimal_places=0,
        ).set_color(GREY)
        scatter_axis_labels1 = scatter_axes1.get_axis_labels(
            x_label_tex='context',
            y_label_tex='expression'
        ).set_color(GREY)

        # Draw the axes.
        self.play(
            DrawBorderThenFill(scatter_axes1),
            Write(scatter_axis_labels1)
        )

        # Create a dot for each sample on x-axis = context and y-axis =
        # expression.
        example_color1 = None
        example_dot_scatter_x1 = ieqtl1.loc[example_name, "context"]
        example_dot_scatter_y1 = ieqtl1.loc[example_name, "expression"]
        example_dot_scatter1 = None
        example_dot_scatter_radius1 = 0.08
        example_dot_scatter_fill1 = 0.5
        example_text_scatter1 = None
        example_text_scatter_scale1 = 0.6
        context_value = ValueTracker(example_dot_scatter_x1)
        scatter_dots1 = []
        for i, row in ieqtl1.iterrows():
            color = BLACK
            if row["genotype"] == 0:
                color = GREEN
            elif row["genotype"] == 1:
                color = BLUE
            elif row["genotype"] == 2:
                color = RED

            if i == example_name:
                example_color1 = color
                example_dot_scatter1 = always_redraw(
                    lambda: Dot(radius=example_dot_scatter_radius1,
                                color=example_color1,
                                fill_opacity=example_dot_scatter_fill1).move_to(
                        scatter_axes1.c2p(context_value.get_value(), example_dot_scatter_y1)
                    )
                )

                example_text_scatter1 = always_redraw(
                    lambda: Text(example_name).scale(example_text_scatter_scale1).move_to(
                        scatter_axes1.c2p(context_value.get_value(), example_dot_scatter_y1 + 0.75)
                    )
                )
            else:
                dot = Dot(color=color, fill_opacity=0.5)
                dot.move_to(scatter_axes1.c2p(row["context"], row["expression"]))
                scatter_dots1.append(dot)

        # Construct the regression lines for each genotype group.
        scatter_lines1 = []
        for genotype_group, (start, end) in regression_lines1.items():
            color = BLACK
            if genotype_group == 0:
                color = GREEN
            elif genotype_group == 1:
                color = BLUE
            elif genotype_group == 2:
                color = RED

            start_dot = Dot().move_to(scatter_axes1.c2p(start[0], start[1]))
            end_dot = Dot().move_to(scatter_axes1.c2p(end[0], end[1]))

            scatter_lines1.append(Line(start_dot, end_dot, color=color))

        # Show the interaction eQTL regression plot.
        self.play(
            FadeIn(VGroup(*scatter_dots1)),
            FadeIn(VGroup(*scatter_lines1)),
            FadeIn(example_dot_scatter1)
        )
        self.wait(3 * scale_wait)

        # Remove the individual dots except the example dot.
        self.play(
            FadeOut(VGroup(*scatter_dots1)),
            FadeIn(example_text_scatter1)
        )
        self.wait(2 * scale_wait)
        example_dot_scatter_fill1 = 1

        # Zoom out and add a second graph for the log likelihood.
        self.play(
            VGroup(scatter_axes1,
                   scatter_axis_labels1,
                   *scatter_lines1,
                   example_dot_scatter1,
                   example_text_scatter1).animate.scale(0.5).shift(LEFT * 3.5),
            run_time=2*scale_wait
        )
        example_dot_scatter_radius1 = example_dot_scatter_radius1 * 0.5
        example_text_scatter_scale1 = example_text_scatter_scale1 * 0.5
        self.wait(2 * scale_wait)

        # Add the x-axis movement line.
        start_dot = Dot().move_to(scatter_axes1.c2p(context_range1[0], example_dot_scatter_y1))
        end_dot = Dot().move_to(scatter_axes1.c2p(context_range1[1], example_dot_scatter_y1))
        example_dot_scatter_hline1 = DashedLine(start_dot, end_dot, stroke_width=3, color=WHITE)

        ########################################################################

        # Calculate the optimum for this ieQTL.
        coef_a1, coef_b1, coef_c1 = optimize_ieqtl(X=ieqtl1, betas=betas1)
        opt_context1 = calc_vertex_xpos(a=coef_a1, b=coef_b1)

        ieqtl_opt1 = ieqtl1.copy()
        ieqtl_opt1["context"] = opt_context1
        ieqtl_opt1["interaction"] = ieqtl_opt1["context"] * ieqtl_opt1["genotype"]
        ieqtl_opt1["y_hat"] = predict(X=ieqtl_opt1[["intercept", "genotype", "context", "interaction"]], betas=betas1)

        print("{}x^2 + {}x + {}".format(coef_a1[example_name],
                                        coef_b1[example_name],
                                        coef_c1[example_name]))

        example_ll_function = lambda x: coef_a1[example_name] * x ** 2 + coef_b1[example_name] * x + coef_c1[example_name]

        ########################################################################

        # Determine log likelihood values.
        example_ll_values = []
        for x in np.arange(context_range1[0], context_range1[1], 0.01):
            example_ll_values.append(example_ll_function(x))
        example_ll_values = np.array(example_ll_values)

        # Determine the lowest value and set that to 0.
        example_ll_values_min = math.floor(np.min(example_ll_values))

        # Determine the range of the plot axes.
        graph_y_range1 = (0, math.ceil(np.max(example_ll_values)) + (example_ll_values_min * -1), 5)

        graph_axes1 = Axes(
            x_range=context_range1,
            y_range=graph_y_range1,
            axis_config={
                "include_tip": False,
                "stroke_width": 1,
            }
        ).set_color(GREY)
        graph_axes1.add_coordinate_labels(
            font_size=20,
            num_decimal_places=0,
        ).set_color(GREY)
        graph_axes_labels1 = graph_axes1.get_axis_labels(
            x_label_tex='context',
            y_label_tex='log(likelihood)'
        ).set_color(GREY)

        example_x_start1 = ieqtl1.loc[example_name, "context"]
        example_x_optimal1 = ieqtl_opt1.loc[example_name, "context"]

        # Construct the log likelihood parabola.
        example_parabola1_left = graph_axes1.get_graph(lambda x: example_ll_function(x) + (example_ll_values_min * -1), x_range=(example_x_start1, -4))
        example_parabola1_left.reverse_points()
        example_parabola1_left.set_stroke(example_color1)
        example_parabola1 = graph_axes1.get_graph(lambda x: example_ll_function(x) + (example_ll_values_min * -1))
        example_parabola1.set_stroke(example_color1)

        example_dot_graph_radius1 = 0.08
        example_dot_graph1 = always_redraw(
            lambda: Dot(radius=example_dot_graph_radius1, color=example_color1).move_to(
                graph_axes1.i2gp(context_value.get_value(),
                                 example_parabola1)
            )
        )
        value_text = always_redraw(
            lambda: Text("[{:.2f}, {:.2f}]".format(context_value.get_value(),
                                                   coef_a1[example_name] * context_value.get_value() ** 2 + coef_b1[example_name] * context_value.get_value() + coef_c1[example_name] + (example_ll_values_min * -1))).scale(0.3).set_color(GREY).move_to(graph_axes1.c2p(context_range1[0] + 0.5,
                                                                                                                                                                                                                                                            coef_a1[example_name] * example_x_optimal1 ** 2 + coef_b1[example_name] * example_x_optimal1 + coef_c1[example_name] + (example_ll_values_min * -1))
                             )
        )

        example_vline_graph1 = always_redraw(
            lambda: graph_axes1.get_v_line(example_dot_graph1.get_bottom())
        )

        VGroup(graph_axes1,
               graph_axes_labels1,
               example_dot_graph1,
               example_vline_graph1,
               example_parabola1_left,
               example_parabola1).scale(0.5).shift(RIGHT * 3.5)
        example_dot_graph_radius1 = example_dot_graph_radius1 * 0.5

        # Draw the axes.
        self.play(
            DrawBorderThenFill(graph_axes1),
            Write(graph_axes_labels1)
        )
        self.add(example_dot_scatter_hline1,
                 example_dot_graph1,
                 example_vline_graph1,
                 value_text)
        self.wait(1 * scale_wait)

        # Move the example dot along the x-axis.
        dps = 1
        self.play(context_value.animate.set_value(context_range1[0]),
                  ShowCreation(example_parabola1_left),
                  run_time=(abs(example_dot_scatter_x1 - context_range1[0]) / dps) * scale_wait)
        self.play(context_value.animate.set_value(context_range1[1]),
                  ShowCreation(example_parabola1),
                  run_time=(abs(context_range1[0] - context_range1[1]) / dps) * scale_wait)
        self.play(context_value.animate.set_value(example_x_optimal1),
                  FadeOut(example_parabola1_left),
                  run_time=3 * scale_wait)
        self.wait(2 * scale_wait)

        self.play(
            FadeOut(VGroup(graph_axes1,
                           graph_axes_labels1,
                           example_dot_scatter1,
                           example_text_scatter1,
                           example_dot_scatter_hline1,
                           example_dot_graph1,
                           example_vline_graph1,
                           example_parabola1_left,
                           example_parabola1,
                           value_text)),
            run_time=2 * scale_wait
        )
        self.wait(1 * scale_wait)

        self.play(
            VGroup(scatter_axes1,
                   scatter_axis_labels1,
                   *scatter_lines1).animate.scale(2).shift(RIGHT * 3.5),
            run_time=2 * scale_wait
        )
        self.wait(2 * scale_wait)

        scatter_dots1 = []
        animations = []
        for i in range(ieqtl1.shape[0]):
            ieqltl_row = ieqtl1.iloc[i, :]
            ieqtl_opt_row = ieqtl_opt1.iloc[i, :]

            color = BLACK
            if ieqltl_row["genotype"] == 0:
                color = GREEN
            elif ieqltl_row["genotype"] == 1:
                color = BLUE
            elif ieqltl_row["genotype"] == 2:
                color = RED

            dot = Dot(color=color, fill_opacity=0.5)
            dot.move_to(scatter_axes1.c2p(ieqltl_row["context"], ieqltl_row["expression"]))
            scatter_dots1.append(dot)

            animation = ApplyMethod(dot.move_to, scatter_axes1.c2p(ieqtl_opt_row["context"], ieqtl_opt_row["expression"]))
            animations.append(animation)

        self.play(
            FadeIn(VGroup(*scatter_dots1))
        )
        self.wait(1 * scale_wait)
        self.play(*animations, run_time=6 * scale_wait)
        self.wait(2 * scale_wait)


class GraphicalAbstractPart2(Scene):
    def construct(self):
        scale_wait = 0.6

        example_index = 168
        example_name = "Jane Doe"
        n_points = 500
        context_sd = 1

        base_model = generate_base_model(example_index=example_index,
                                         example_name=example_name,
                                         n_points=n_points,
                                         context_sd=context_sd
                                         )

        # Generate ieQTL models.
        ieqtl1, betas1 = generate_ieqtl_model(maf=0.42,
                                              base_matrix=base_model,
                                              error_sd=0.4,
                                              betas=np.array([-2.5, 2.1, 0.5, 0.6]),
                                              base_seed=1
                                              )
        regression_lines1 = calc_regression_lines(ieqtl=ieqtl1)

        ieqtl2, betas2 = generate_ieqtl_model(maf=0.34,
                                              base_matrix=base_model,
                                              error_sd=0.6,
                                              betas=np.array([-2.5, 3, 0.6, 1]),
                                              base_seed=3
                                              )
        regression_lines2 = calc_regression_lines(ieqtl=ieqtl2)

        # Determine the range of the plot axes.
        context_range1 = (math.floor(min(ieqtl1["context"].min(), ieqtl2["context"].min())) - 1, math.ceil(max(ieqtl1["context"].max(), ieqtl1["context"].max())) + 1, 1)
        scatter_y_range1 = (math.floor(min(ieqtl1["expression"].min(), ieqtl2["expression"].min())), math.ceil(max(ieqtl1["expression"].max(), ieqtl2["expression"].max())), 2)

        example_dot_x = base_model.loc[example_name, "context"]
        context_value1 = ValueTracker(example_dot_x)
        context_value2 = ValueTracker(example_dot_x)

        ########################################################################

        # Construct the axis for the scatterplot.
        scatter_axes1 = Axes(
            x_range=context_range1,
            y_range=scatter_y_range1,
            axis_config={
                "include_tip": False,
                "numbers_to_exclude": [0],
                "stroke_width": 1,
            }
        ).set_color(GREY)
        scatter_axes1.add_coordinate_labels(
            font_size=20,
            num_decimal_places=0,
        ).set_color(GREY)
        scatter_axis_labels1 = scatter_axes1.get_axis_labels(
            x_label_tex='context',
            y_label_tex='expression'
        ).set_color(GREY)

        # Create a dot for each sample on x-axis = context and y-axis =
        # expression.
        example_color1 = None
        example_dot_scatter_y1 = ieqtl1.loc[example_name, "expression"]
        example_dot_scatter1 = None
        example_dot_scatter_radius1 = 0.08
        example_dot_scatter_fill1 = 0.5
        example_text_scatter1 = None
        example_text_scatter_scale1 = 0.6
        scatter_dots1 = []
        for i, row in ieqtl1.iterrows():
            color = BLACK
            if row["genotype"] == 0:
                color = GREEN
            elif row["genotype"] == 1:
                color = BLUE
            elif row["genotype"] == 2:
                color = RED

            if i == example_name:
                example_color1 = color
                example_dot_scatter1 = always_redraw(
                    lambda: Dot(radius=example_dot_scatter_radius1,
                                color=example_color1,
                                fill_opacity=example_dot_scatter_fill1).move_to(
                        scatter_axes1.c2p(context_value1.get_value(), example_dot_scatter_y1)
                    )
                )

                example_text_scatter1 = always_redraw(
                    lambda: Text(example_name).scale(example_text_scatter_scale1).move_to(
                        scatter_axes1.c2p(context_value1.get_value(), example_dot_scatter_y1 + 0.75)
                    )
                )
            else:
                dot = Dot(color=color, fill_opacity=0.5)
                dot.move_to(scatter_axes1.c2p(row["context"], row["expression"]))
                scatter_dots1.append(dot)

        # Construct the regression lines for each genotype group.
        scatter_lines1 = []
        for genotype_group, (start, end) in regression_lines1.items():
            color = BLACK
            if genotype_group == 0:
                color = GREEN
            elif genotype_group == 1:
                color = BLUE
            elif genotype_group == 2:
                color = RED

            start_dot = Dot().move_to(scatter_axes1.c2p(start[0], start[1]))
            end_dot = Dot().move_to(scatter_axes1.c2p(end[0], end[1]))
            scatter_lines1.append(Line(start_dot, end_dot, color=color))

        VGroup(scatter_axes1,
               scatter_axis_labels1,
               *scatter_dots1,
               *scatter_lines1,
               example_dot_scatter1).scale(0.5).shift(LEFT * 3.5 + UP * 2)
        example_dot_scatter_radius1 = example_dot_scatter_radius1 * 0.5
        example_text_scatter_scale1 = example_text_scatter_scale1 * 0.5

        ########################################################################

        # Construct the axis for the scatterplot.
        scatter_axes2 = Axes(
            x_range=context_range1,
            y_range=scatter_y_range1,
            axis_config={
                "include_tip": False,
                "numbers_to_exclude": [0],
                "stroke_width": 1,
            }
        ).set_color(GREY)
        scatter_axes2.add_coordinate_labels(
            font_size=20,
            num_decimal_places=0,
        ).set_color(GREY)
        scatter_axis_labels2 = scatter_axes2.get_axis_labels(
            x_label_tex='context',
            y_label_tex='expression'
        ).set_color(GREY)

        # Create a dot for each sample on x-axis = context and y-axis =
        # expression.
        example_color2 = None
        example_dot_scatter_y2 = ieqtl2.loc[example_name, "expression"]
        example_dot_scatter2 = None
        example_dot_scatter_radius2 = 0.08
        example_dot_scatter_fill2 = 0.5
        example_text_scatter2 = None
        example_text_scatter_scale2 = 0.6
        scatter_dots2 = []
        for i, row in ieqtl2.iterrows():
            color = BLACK
            if row["genotype"] == 0:
                color = GREEN
            elif row["genotype"] == 1:
                color = BLUE
            elif row["genotype"] == 2:
                color = RED

            if i == example_name:
                example_color2 = color
                example_dot_scatter2 = always_redraw(
                    lambda: Dot(radius=example_dot_scatter_radius2,
                                color=example_color2,
                                fill_opacity=example_dot_scatter_fill2).move_to(
                        scatter_axes2.c2p(context_value2.get_value(), example_dot_scatter_y2)
                    )
                )

                example_text_scatter2 = always_redraw(
                    lambda: Text(example_name).scale(example_text_scatter_scale2).move_to(
                        scatter_axes2.c2p(context_value2.get_value(), example_dot_scatter_y2 + 0.75)
                    )
                )
            else:
                dot = Dot(color=color, fill_opacity=0.5)
                dot.move_to(scatter_axes2.c2p(row["context"], row["expression"]))
                scatter_dots2.append(dot)

        # Construct the regression lines for each genotype group.
        scatter_lines2 = []
        for genotype_group, (start, end) in regression_lines2.items():
            color = BLACK
            if genotype_group == 0:
                color = GREEN
            elif genotype_group == 1:
                color = BLUE
            elif genotype_group == 2:
                color = RED

            start_dot = Dot().move_to(scatter_axes2.c2p(start[0], start[1]))
            end_dot = Dot().move_to(scatter_axes2.c2p(end[0], end[1]))
            scatter_lines2.append(Line(start_dot, end_dot, color=color))

        VGroup(scatter_axes2,
               scatter_axis_labels2,
               *scatter_dots2,
               *scatter_lines2,
               example_dot_scatter2).scale(0.5).shift(LEFT * 3.5 + DOWN * 2)
        example_dot_scatter_radius2 = example_dot_scatter_radius2 * 0.5
        example_text_scatter_scale2 = example_text_scatter_scale2 * 0.5

        ########################################################################

        # Calculate the optimum for this ieQTL.
        coef_a1, coef_b1, coef_c1 = optimize_ieqtl(X=ieqtl1, betas=betas1)
        coef_a2, coef_b2, coef_c2 = optimize_ieqtl(X=ieqtl2, betas=betas2)
        opt_context1 = calc_vertex_xpos(a=coef_a1, b=coef_b1)
        opt_context2 = calc_vertex_xpos(a=coef_a2, b=coef_b2)
        opt_context = calc_vertex_xpos(a=coef_a1 + coef_a2, b=coef_b1 + coef_b2)

        ieqtl_opt1 = ieqtl1.copy()
        ieqtl_opt1["context_solo"] = opt_context1
        ieqtl_opt1["context"] = opt_context
        ieqtl_opt1["interaction"] = ieqtl_opt1["context"] * ieqtl_opt1["genotype"]
        ieqtl_opt1["y_hat"] = predict(X=ieqtl_opt1[["intercept", "genotype", "context", "interaction"]], betas=betas1)

        ieqtl_opt2 = ieqtl2.copy()
        ieqtl_opt2["context_solo"] = opt_context2
        ieqtl_opt2["context"] = opt_context
        ieqtl_opt2["interaction"] = ieqtl_opt2["context"] * ieqtl_opt2["genotype"]
        ieqtl_opt2["y_hat"] = predict(X=ieqtl_opt2[["intercept", "genotype", "context", "interaction"]], betas=betas2)

        example_ll_function1 = lambda x: coef_a1[example_name] * x ** 2 + \
                                         coef_b1[example_name] * x

        example_ll_function2 = lambda x: coef_a2[example_name] * x ** 2 + \
                                         coef_b2[example_name] * x

        example_ll_function3 = lambda x: (coef_a1[example_name] + coef_a2[example_name]) * x ** 2 + \
                                         (coef_b1[example_name] + coef_b2[example_name]) * x

        ########################################################################

        # Determine log likelihood values.
        example_ll_values = []
        for x in np.arange(context_range1[0], context_range1[1], 0.01):
            example_ll_values.append(example_ll_function1(x))
            example_ll_values.append(example_ll_function2(x))
            example_ll_values.append(example_ll_function3(x))
        example_ll_values = np.array(example_ll_values)

        # Determine the lowest value and set that to 0.
        example_ll_values_min = math.floor(np.min(example_ll_values))

        # Determine the range of the plot axes.
        graph_y_range1 = (0, math.ceil(np.max(example_ll_values)) + (example_ll_values_min * -1), 5)

        graph_axes = Axes(
            x_range=context_range1,
            y_range=graph_y_range1,
            axis_config={
                "include_tip": False,
                "stroke_width": 1,
            }
        ).set_color(GREY)
        graph_axes.add_coordinate_labels(
            font_size=20,
            num_decimal_places=0,
        ).set_color(GREY)
        graph_axes_labels = graph_axes.get_axis_labels(
            x_label_tex='context',
            y_label_tex='log(likelihood)'
        ).set_color(GREY)

        VGroup(graph_axes,
               graph_axes_labels).scale(0.5).shift(RIGHT * 3.5)

        ########################################################################

        # Draw the axes.
        self.play(
            DrawBorderThenFill(scatter_axes1),
            DrawBorderThenFill(scatter_axes2),
            DrawBorderThenFill(graph_axes),
            Write(scatter_axis_labels1),
            Write(scatter_axis_labels2),
            Write(graph_axes_labels)
        )

        # Show the interaction eQTL regression plot.
        self.play(
            FadeIn(VGroup(*scatter_dots1)),
            FadeIn(VGroup(*scatter_lines1)),
            FadeIn(example_dot_scatter1),
            FadeIn(VGroup(*scatter_dots2)),
            FadeIn(VGroup(*scatter_lines2)),
            FadeIn(example_dot_scatter2),
            run_time=2 * scale_wait
        )
        self.wait(3 * scale_wait)

        ########################################################################

        example_x_start1 = ieqtl1.loc[example_name, "context"]
        example_x_optimal1 = ieqtl_opt1.loc[example_name, "context_solo"]

        example_x_start2 = ieqtl2.loc[example_name, "context"]
        example_x_optimal2 = ieqtl_opt2.loc[example_name, "context_solo"]

        example_x_optimal3 = ieqtl_opt1.loc[example_name, "context"]

        # Construct the log likelihood parabola.
        example_parabola1_left = graph_axes.get_graph(lambda x: example_ll_function1(x) + (example_ll_values_min * -1), x_range=(example_x_start1, -4))
        example_parabola1_left.reverse_points()
        example_parabola1_left.set_stroke(example_color1)

        example_parabola1 = graph_axes.get_graph(lambda x: example_ll_function1(x) + (example_ll_values_min * -1))
        example_parabola1.set_stroke(example_color1)

        example_parabola2_left = graph_axes.get_graph(lambda x: example_ll_function2(x) + (example_ll_values_min * -1), x_range=(example_x_start2, -4))
        example_parabola2_left.reverse_points()
        example_parabola2_left.set_stroke(example_color2)

        example_parabola2 = graph_axes.get_graph(lambda x: example_ll_function2(x) + (example_ll_values_min * -1))
        example_parabola2.set_stroke(example_color2)

        example_parabola3 = graph_axes.get_graph(lambda x: example_ll_function3(x) + (example_ll_values_min * -1))
        example_parabola3.set_stroke(WHITE)

        example_dot_graph_radius1 = 0.04
        example_dot_graph1 = always_redraw(
            lambda: Dot(radius=example_dot_graph_radius1,
                        color=example_color1).move_to(
                graph_axes.i2gp(context_value1.get_value(),
                                example_parabola1)
            )
        )
        value_text1 = always_redraw(
            lambda: Text("[{:.2f}, {:.2f}]".format(context_value1.get_value(),
                                                   example_ll_function1(context_value1.get_value()) + (example_ll_values_min * -1))).scale(
                0.3).set_color(example_color1).move_to(
                graph_axes.c2p(context_range1[0] + 0.5,
                               example_ll_function1(example_x_optimal1) + (example_ll_values_min * -1))
                )
        )
        example_vline_graph1 = always_redraw(
            lambda: graph_axes.get_v_line(example_dot_graph1.get_bottom())
        )

        example_dot_graph_radius2 = 0.04
        example_dot_graph2 = always_redraw(
            lambda: Dot(radius=example_dot_graph_radius2,
                        color=example_color2).move_to(
                graph_axes.i2gp(context_value2.get_value(),
                                example_parabola2)
            )
        )
        value_text2 = always_redraw(
            lambda: Text("[{:.2f}, {:.2f}]".format(context_value2.get_value(),
                                                   example_ll_function2(context_value2.get_value()) + (example_ll_values_min * -1))).scale(
                0.3).set_color(example_color2).move_to(
                graph_axes.c2p(context_range1[0] + 0.5,
                               example_ll_function1(example_x_optimal1) + (example_ll_values_min * -1) - 2)
                )
        )
        example_vline_graph2 = always_redraw(
            lambda: graph_axes.get_v_line(example_dot_graph2.get_bottom())
        )

        example_dot_graph_radius3 = 0.04
        example_dot_graph3 = Dot(radius=example_dot_graph_radius3, color=WHITE).move_to(graph_axes.i2gp(example_x_optimal3, example_parabola3))

        value_text3 = Text("[{:.2f}, {:.2f}]".format(context_value2.get_value(), example_ll_function3(example_x_optimal3) + (example_ll_values_min * -1))).scale(0.3).set_color(WHITE).move_to(graph_axes.c2p(context_range1[0] + 0.5, example_ll_function1(example_x_optimal3) + (example_ll_values_min * -1) - 4))

        example_vline_graph3 = graph_axes.get_v_line(example_dot_graph3.get_bottom())

        self.play(
            FadeOut(VGroup(*scatter_dots1)),
            FadeOut(VGroup(*scatter_dots2))
        )
        example_dot_scatter_fill1 = 1
        example_dot_scatter_fill2 = 1
        self.add(example_text_scatter1,
                 example_text_scatter2,
                 example_dot_graph1,
                 example_dot_graph2,
                 example_vline_graph1,
                 example_vline_graph2,
                 value_text1,
                 value_text2)
        self.wait(2 * scale_wait)

        # Move the example dot along the x-axis.
        dps = 1
        self.play(context_value1.animate.set_value(context_range1[0]),
                  context_value2.animate.set_value(context_range1[0]),
                  ShowCreation(example_parabola1_left),
                  ShowCreation(example_parabola2_left),
                  run_time=(abs(example_dot_x - context_range1[0]) / dps) * scale_wait)
        self.play(context_value1.animate.set_value(context_range1[1]),
                  context_value2.animate.set_value(context_range1[1]),
                  ShowCreation(example_parabola1),
                  ShowCreation(example_parabola2),
                  run_time=(abs(context_range1[0] - context_range1[1]) / dps) * scale_wait)
        self.play(context_value1.animate.set_value(example_x_optimal1),
                  context_value2.animate.set_value(example_x_optimal2),
                  FadeOut(example_parabola1_left),
                  FadeOut(example_parabola2_left),
                  run_time=3 * scale_wait)
        self.wait(4 * scale_wait)

        ########################################################################

        graph_objects = VGroup(
            graph_axes,
            graph_axes_labels,
            example_parabola1,
            example_parabola2,
            example_dot_graph1,
            example_dot_graph2,
            example_vline_graph1,
            example_vline_graph2,
            value_text1,
            value_text2
        )
        self.play(graph_objects.animate.shift(DOWN * 1.5))

        to_isolate = ["A_1", "B_1", "A_2", "B_2", "(", ")"]
        lines = VGroup(
            Tex("A_1", "x^2", "+", "B_1", "x"),
            Tex("A_2", "x^2", "+", "B_2", "x"),
            Tex("(", "A_1", "x^2", "+", "B_1", "x", ")", "+", "(", "A_2", "x^2", "+", "B_2", "x", ")", isolate=to_isolate),
            Tex("(", "A_1", "+", "A_2", ")", "x^2", "+", "(", "B_1", "+", "B_2", ")", "x", isolate=to_isolate),
        )
        lines.arrange(DOWN, buff=MED_LARGE_BUFF).scale(0.6).shift(RIGHT * 3.5 + UP * 2)
        for line in lines:
            line.set_color_by_tex_to_color_map({
                "A": BLUE,
                "B": TEAL,
                "x": WHITE
            })

        self.play(
            FadeIn(lines[0]),
            FadeIn(lines[1]),
            run_time=2 * scale_wait
        )
        self.wait(2 * scale_wait)

        self.play(
            TransformMatchingTex(lines[0].copy(), lines[2]),
            TransformMatchingTex(lines[1].copy(), lines[2]),
            run_time=2 * scale_wait
        )
        self.wait(2 * scale_wait)

        self.play(
            TransformMatchingTex(lines[2].copy(), lines[3]),
            run_time=2 * scale_wait
        )
        self.wait(4 * scale_wait)

        self.play(
            FadeOut(lines),
            graph_objects.animate.shift(UP * 1.5),
            run_time=2 * scale_wait
        )

        ########################################################################

        self.play(ShowCreation(example_parabola3),
                  run_time=2 * scale_wait)
        self.wait(1 * scale_wait)

        self.play(FadeIn(example_dot_graph3),
                  FadeIn(value_text3),
                  FadeIn(example_vline_graph3),
                  run_time=2 * scale_wait)
        self.wait(2 * scale_wait)

        self.play(context_value1.animate.set_value(example_x_optimal3),
                  context_value2.animate.set_value(example_x_optimal3),
                  FadeOut(example_parabola1_left),
                  FadeOut(example_parabola2_left),
                  run_time=4 * scale_wait)
        self.wait(1 * scale_wait)

        self.play(FadeOut(VGroup(example_dot_graph1,
                                 example_dot_graph2,
                                 example_dot_graph3,
                                 example_vline_graph1,
                                 example_vline_graph2,
                                 example_vline_graph3,
                                 value_text1,
                                 value_text2,
                                 value_text3,
                                 example_parabola1,
                                 example_parabola2,
                                 example_parabola3,
                                 example_dot_scatter1,
                                 example_dot_scatter2,
                                 example_text_scatter1,
                                 example_text_scatter2,
                                 graph_axes,
                                 graph_axes_labels)),
                  run_time=2 * scale_wait)
        self.wait(1 * scale_wait)

        ########################################################################

        scatter_dots1 = []
        animations1 = []
        for i in range(ieqtl1.shape[0]):
            ieqltl_row = ieqtl1.iloc[i, :]
            ieqtl_opt_row = ieqtl_opt1.iloc[i, :]

            color = BLACK
            if ieqltl_row["genotype"] == 0:
                color = GREEN
            elif ieqltl_row["genotype"] == 1:
                color = BLUE
            elif ieqltl_row["genotype"] == 2:
                color = RED

            dot = Dot(color=color, fill_opacity=0.5).scale(0.5)
            dot.move_to(scatter_axes1.c2p(ieqltl_row["context"], ieqltl_row["expression"]))
            scatter_dots1.append(dot)

            animation = ApplyMethod(dot.move_to, scatter_axes1.c2p(ieqtl_opt_row["context"], ieqtl_opt_row["expression"]))
            animations1.append(animation)

        scatter_dots2 = []
        animations2 = []
        for i in range(ieqtl2.shape[0]):
            ieqltl_row = ieqtl2.iloc[i, :]
            ieqtl_opt_row = ieqtl_opt2.iloc[i, :]

            color = BLACK
            if ieqltl_row["genotype"] == 0:
                color = GREEN
            elif ieqltl_row["genotype"] == 1:
                color = BLUE
            elif ieqltl_row["genotype"] == 2:
                color = RED

            dot = Dot(color=color, fill_opacity=0.5).scale(0.5)
            dot.move_to(scatter_axes2.c2p(ieqltl_row["context"], ieqltl_row["expression"]))
            scatter_dots2.append(dot)

            animation = ApplyMethod(dot.move_to, scatter_axes2.c2p(ieqtl_opt_row["context"], ieqtl_opt_row["expression"]))
            animations2.append(animation)

        self.play(
            FadeIn(VGroup(*scatter_dots1)),
            FadeIn(VGroup(*scatter_dots2))
        )
        self.wait(1 * scale_wait)
        self.play(*animations1, *animations2, run_time=6 * scale_wait)
        self.wait(2 * scale_wait)


def generate_base_model(example_index=0, example_name="Jane Doe", n_points=100,
                        context_sd=1):
    X = pd.DataFrame(np.nan,
                     columns=["expression", "intercept", "genotype", "context",
                              "interaction", "y_hat", "residuals"],
                     index=["sample{}".format(i) for i in range(n_points)])

    X.index = [x if i != example_index else example_name for i, x in
               enumerate(X.index)]

    X["intercept"] = 1
    np.random.seed(0)
    X["context"] = np.random.normal(0, context_sd, n_points)

    return X


def generate_ieqtl_model(maf, base_matrix, error_sd, betas, base_seed=0):
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
    X["expression"] = X.loc[:,["intercept", "genotype", "context", "interaction"]].dot(
        betas) + np.random.normal(0, error_sd, X.shape[0])

    # Model the ieQTL.
    betas = fit(X=X[["intercept", "genotype", "context", "interaction"]], y=X["expression"])
    X["y_hat"] = predict(X=X[["intercept", "genotype", "context", "interaction"]], betas=betas)
    X["residuals"] = X["expression"] - X["y_hat"]

    return X, betas


def fit(X, y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)


def predict(X, betas):
    return np.dot(X, betas)


def calc_regression_lines(ieqtl):
    regression_lines = {}
    for genotype_group in ieqtl["genotype"].unique():
        subset = ieqtl.loc[ieqtl["genotype"] == genotype_group, :].copy()
        betas = fit(X=subset[["intercept", "context"]], y=subset["expression"])
        subset["y_hat"] = predict(X=subset[["intercept", "context"]], betas=betas)
        subset.sort_values("context", inplace=True)
        regression_lines[genotype_group] = [np.array(
            [subset.iloc[0, :]["context"], subset.iloc[0, :]["y_hat"], 0]),
                                            np.array(
                                                [subset.iloc[-1, :]["context"],
                                                 subset.iloc[-1, :]["y_hat"],
                                                 0])
                                            ]
    return regression_lines


def optimize_ieqtl(X, betas, x1=-4, x3=4):
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
        eval_df["adj_y_hat"] = predict(X=eval_df[["intercept", "genotype", "context", "interaction"]], betas=betas)

        # Calculate the residuals squared per sample for this model.
        eval_df["adj_residuals"] = eval_df["expression"] - eval_df["adj_y_hat"]
        eval_df["adj_log_likelihood"] = np.log(stats.norm.pdf(eval_df["adj_residuals"], 0, 1))

        # Save the adjusted residuals squared.
        y_values_df.loc[:, eval_pos] = ll - eval_df["log_likelihood"] + eval_df["adj_log_likelihood"]

    # Determine the coefficients.
    coef_a, coef_b, coef_c = calc_parabola_vertex(
        x1=x1,
        x2=X["context"],
        x3=x3,
        y1=y_values_df[x1],
        y2=ll,
        y3=y_values_df[x3]
    )

    return coef_a, coef_b, coef_c


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
    c = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom

    return a, b, c


def calc_vertex_xpos(a, b):
    return -b / (2 * a)