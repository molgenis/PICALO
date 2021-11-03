"""
File:         ieqtl.py
Created:      2021/04/08
Last Changed: 2021/11/01
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
import numpy as np
from scipy import stats

# Local application imports.
from src.statistics import inverse, fit, predict, calc_residuals, calc_rss, fit_and_predict, calc_std, calc_p_value, calc_regression_log_likelihood


class IeQTL:
    def __init__(self, snp, gene, cov, genotype, covariate, expression):
        # Safe arguments.
        self.snp = snp
        self.gene = gene
        self.cov = cov

        # Initialize flags.
        self.is_computed = False
        self.is_analyzed = False

        # Save data.
        self.mask = genotype != -1
        self.n = np.sum(self.mask)
        self.X = self.construct_model_matrix(genotype=genotype,
                                             covariate=covariate)
        # self.covariate = covariate[self.mask]
        self.y = expression[self.mask]
        del genotype, covariate, expression

        # Initialize empty properties variables.
        self.betas = np.empty(4, dtype=np.float64)
        self.residuals = np.empty(self.n, dtype=np.float64)
        self.rss = None
        self.std = None
        self.p_value = None

        # Set ll function x evaluation points.
        self.x1, self.x3 = -4, 4

        # Calculate the optimisation function and determine the optimum.
        self.coef_a = np.empty(self.n, dtype=np.float64)
        self.coef_b = np.empty(self.n, dtype=np.float64)

    def construct_model_matrix(self, genotype, covariate):
        X = np.empty((self.n, 4), np.float32)
        X[:, 0] = 1
        X[:, 1] = genotype[self.mask]
        X[:, 2] = covariate[self.mask]
        X[:, 3] = X[:, 1] * X[:, 2]

        return X

    def get_gene(self):
        return self.gene

    def get_snp(self):
        return self.snp

    def get_eqtl_id(self):
        return "{}:{}".format(self.gene, self.snp)

    def get_ieqtl_id(self):
        return "{}:{}:{}".format(self.gene, self.snp, self.cov)

    def compute(self):
        # First calculate the rss for the matrix - interaction term.
        rss_null = calc_rss(y=self.y, y_hat=fit_and_predict(X=self.X[:, :3], y=self.y))

        # Calculate the rss for the interaction mode. Safe the betas as well as
        # the residuals for optimization later.
        inv_m = inverse(self.X)
        self.betas = fit(X=self.X, y=self.y, inv_m=inv_m)
        y_hat = predict(X=self.X, betas=self.betas)
        self.residuals = calc_residuals(y=self.y, y_hat=y_hat)
        self.rss = np.sum(self.residuals * self.residuals)
        self.std = calc_std(rss=self.rss, n=self.n, df=self.X.shape[1], inv_m=inv_m)

        # Calculate interaction p-value.
        self.p_value = calc_p_value(rss1=rss_null, rss2=self.rss, df1=3, df2=4, n=self.n)

        # Set the flag.
        self.is_computed = True

    def set_mll_coef_representation(self):
        """
        This function evaluaties the (simplified) log likelihood function
        at 3 points. In reality we only evaluate at 2 points and use
        the initial guess as the center point.

        Then, the coefficients a and b in the function a x^2 + b x + c are
        calculated analytically and returned.
        """
        if not self.is_computed:
            self.compute()

        # Initialize the evaluation matrix.
        eval_m = np.copy(self.X)

        # Calculate the residuals squared.
        rs = self.residuals * self.residuals

        # Evaluate the log likelihood function for each sample on position
        # x1 and x3.
        y_values = []
        for eval_pos in [self.x1, self.x3]:
            # Replace the covariate with the evaluation position and recalculate
            # the interaction term.
            eval_m[:, 2] = eval_pos
            eval_m[:, 3] = eval_m[:, 1] * eval_m[:, 2]

            # Calculate the y_hat of the model for the original betas.
            adj_y_hat = predict(X=eval_m, betas=self.betas)

            # Calculate the residuals squared per sample for this model.
            adj_residuals = calc_residuals(y=self.y, y_hat=adj_y_hat)
            adj_rs = adj_residuals * adj_residuals

            # Save the adjusted residuals squared.
            y_values.append(self.rss - rs + adj_rs)

        # Determine the coefficients.
        self.coef_a, self.coef_b = self.calc_parabola_vertex(x1=self.x1,
                                                             x2=self.X[:, 2],
                                                             x3=self.x3,
                                                             y1=y_values[0],
                                                             y2=self.rss,
                                                             y3=y_values[1])

        # Set the flag.
        self.is_analyzed = True

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

    def get_mll_coef_representation(self, full_array=False):
        if not self.is_analyzed:
            self.set_mll_coef_representation()

        if full_array:
            # Make the vector complete again.
            full_a = np.zeros(np.size(self.mask), np.float64)
            full_a[self.mask] = self.coef_a

            full_b = np.zeros(np.size(self.mask), np.float64)
            full_b[self.mask] = self.coef_b

            return full_a, full_b
        else:
            return self.coef_a, self.coef_b

    def get_mask(self):
        return self.mask

    def calc_log_likelihood(self, new_cov=None):
        residuals = self.residuals.copy()
        if new_cov is not None:
            new_X = self.X.copy()
            new_X[:, 2] = new_cov[self.mask]
            new_X[:, 3] = new_X[:, 1] * new_X[:, 2]

            residuals = calc_residuals(y=self.y, y_hat=predict(X=new_X, betas=self.betas))

        return calc_regression_log_likelihood(residuals=residuals)

    def __str__(self):
        return "IeQTL(snp={}, gene={}, cov={}, is_computed={}, " \
               "is_analyzed={}, n={}, betas={}, rss={:.2f}, p_value={:.2e}, " \
               "coef_a={}, coef_b={})".format(self.snp,
                                              self.gene,
                                              self.cov,
                                              self.is_computed,
                                              self.is_analyzed,
                                              self.n,
                                              self.betas,
                                              self.rss,
                                              self.p_value,
                                              self.coef_a,
                                              self.coef_b)
