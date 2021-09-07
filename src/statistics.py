"""
File:         statistics.py
Created:      2021/04/14
Last Changed: 2021/07/13
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

# Third party imports.
from scipy.special import betainc
import numpy as np
from statsmodels.regression.linear_model import OLS


def remove_multicollinearity(X, threshold=0.95):
    dropped_indices = []
    indices = np.arange(X.shape[1])
    max_vif = np.inf
    while len(indices) > 1 and max_vif > threshold:
        vif = np.array([calc_ols_rsquared(X[:, indices], ix) for ix in range(len(indices))])
        max_vif = max(vif)

        if max_vif > threshold:
            max_index = np.where(vif == max_vif)[0][0]
            dropped_indices.append(indices[max_index])
            indices = np.delete(indices, max_index)

    dropped_indices.sort()

    return X[:, indices], dropped_indices


def calc_ols_rsquared(m, idx):
    n_vars = m.shape[1]
    mask = np.arange(n_vars) != idx
    return OLS(m[:, idx], m[:, mask]).fit().rsquared


def remove_covariates(y_m, X_m=None, X_inter_m=None, inter_m=None,
                      include_intercept=False, include_inter_as_covariate=False):
    if X_m is None and X_inter_m is None:
        return y_m
    if X_inter_m is not None and inter_m is None:
        print("Error in remove_covariates")
        exit()
    if inter_m is not None and (y_m.shape != inter_m.shape):
        print("Error in remove_covariates")
        exit()

    # Prepare X_m
    X_m_tmp = None
    if X_m is not None:
        X_m_tmp = np.copy(X_m)

        # Force 2D matrix.
        if np.ndim(X_m_tmp) == 1:
            X_m_tmp = X_m_tmp[:, np.newaxis]

    # Add the intercept.
    if include_intercept:
        intercept = np.ones((X_m_tmp.shape[0], 1))
        if X_m_tmp is not None:
            X_m_tmp = np.hstack((intercept, X_m_tmp))
        else:
            X_m_tmp = intercept

    # Prepare X_inter_m
    X_inter_m_tmp = None
    if X_inter_m is not None:
        X_inter_m_tmp = np.copy(X_inter_m)

        # Force 2D matrix.
        if np.ndim(X_inter_m_tmp) == 1:
            X_inter_m_tmp = X_inter_m_tmp[:, np.newaxis]

    # Loop over expression rows.
    y_m_corrected = np.empty_like(y_m, dtype=np.float64)
    for i in range(y_m.shape[0]):
        X = None

        # Add the covariates without interaction.
        if X_m_tmp is not None:
            X = X_m_tmp

        # Add the covariates with interaction termn.
        if X_inter_m_tmp is not None and inter_m is not None:
            inter_a = inter_m[i, :][:, np.newaxis]
            x_times_inter_m = X_inter_m_tmp * inter_a

            if X is None:
                X = x_times_inter_m
            else:
                X = np.concatenate((X, x_times_inter_m), axis=1)

            if include_inter_as_covariate:
                X = np.hstack((X, inter_a))

        X = summarize_matrix(X)
        y_m_corrected[i, :] = calc_residuals(X=X, y=y_m[i, :])

    return y_m_corrected


def summarize_matrix(m):
    m = m[:, m.std(axis=0) != 0]
    zscore = (m - m.mean(axis=0)) / m.std(axis=0)
    corr_matrix = np.dot(zscore.T, zscore) / (zscore.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eig(corr_matrix)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = np.real(eigenvalues[order])
    eigenvectors = np.real(eigenvectors[:, order])

    # Replace negative eigenvalues with 0
    eigenvalues[eigenvalues < 0] = 0

    # Find the number of eigenvalues that explain 99.99999999% of the variance.
    variance_expl = np.cumsum(eigenvalues / np.sum(eigenvalues))
    mask = np.round(variance_expl, 10) != 1

    return np.dot(zscore, eigenvectors[:, mask])


def remove_interaction_covariates(y_m, X_m, inter_m,
                                  include_covariate=False,
                                  include_intercept=False):
    y_m_corrected = np.empty_like(y_m, dtype=np.float64)

    X_m_tmp = np.copy(X_m)
    if np.ndim(X_m_tmp) == 1:
        X_m_tmp = X_m_tmp[:, np.newaxis]

    for i in range(y_m.shape[0]):
        X = X_m_tmp * inter_m[i, :][:, np.newaxis]

        if include_covariate:
            X = np.concatenate((X_m_tmp, X), axis=1)
        if include_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))

        y_m_corrected[i, :] = calc_residuals(X=X, y=y_m[i, :])

    return y_m_corrected


def remove_covariates_elementwise(y_m, X_m, a=None,
                                  include_intercept=False):
    y_m_corrected = np.empty_like(y_m, dtype=np.float64)

    for i in range(y_m.shape[0]):

        X = X_m[i, :][:, np.newaxis]
        if a is not None:
            X = np.hstack((X, a[:, np.newaxis]))

        if include_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))

        y_m_corrected[i, :] = calc_residuals(X=X, y=y_m[i, :])

    return y_m_corrected


def inverse(X):
    X_square = X.T.dot(X)
    try:
        return np.linalg.inv(X_square)
    except np.linalg.LinAlgError:
        print("Warning: using pseudo-inverse")
        return np.linalg.pinv(X_square)


def fit(X, y, inv_m=None):
    if inv_m is None:
        inv_m = inverse(X)
    return inv_m.dot(X.T).dot(y)


def predict(X, betas):
    return np.dot(X, betas)


def fit_and_predict(X, y):
    return predict(X=X, betas=fit(X=X, y=y))


def calc_residuals(X, y):
    y_hat = fit_and_predict(X=X, y=y)
    return y - y_hat


def calc_rss(y, y_hat, sum=True):
    res = y - y_hat
    res_squared = res * res
    if sum:
        return np.sum(res_squared)
    else:
        return res_squared


def calc_std(rss, n, df, inv_m):
    return np.sqrt(rss / (n - df) * np.diag(inv_m))


def calc_p_value(rss1, rss2, df1, df2, n):
    """
    last row == stats.f.sf(f_value, dfn=(df2 - df1), dfd=(n - df2)) but this
    is faster somehow.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.betainc.html

    1 - I(a,b,x) = I(b, a, 1-x)
    """
    if rss2 >= rss1:
        return 1
    dfn = df2 - df1
    dfd = n - df2
    f_value = ((rss1 - rss2) / dfn) / (rss2 / dfd)
    p_value = betainc(dfd / 2, dfn / 2, 1 - ((dfn * f_value) / ((dfn * f_value) + dfd)))
    if p_value == 0:
        p_value = 2.2250738585072014e-308
    return p_value


def calc_vertex_xpos(a, b):
    a[a == 0] = np.nan
    vertex_xpos = -b / (2 * a)
    return vertex_xpos.astype(np.float64)


def calc_pearsonr(x, y):
    x_dev = x - np.mean(x)
    y_dev = y - np.mean(y)
    dev_sum = np.sum(x_dev * y_dev)
    x_rss = np.sum(x_dev * x_dev)
    y_rss = np.sum(y_dev * y_dev)
    return dev_sum / np.sqrt(x_rss * y_rss)


def calc_eucledian_distance(x, y):
    return np.linalg.norm(x - y)
