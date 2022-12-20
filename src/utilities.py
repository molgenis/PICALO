"""
File:         utilities.py
Created:      2021/04/28
Last Changed: 2022/12/15
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
import os

# Third party imports.
import pandas as pd
import numpy as np
from statsmodels.stats import multitest

# Local application imports.
from src.objects.ieqtl import IeQTL


def load_dataframe(inpath, header, index_col, sep="\t", low_memory=True,
                   nrows=None, skiprows=None, log=None):
    df = pd.read_csv(inpath, sep=sep, header=header, index_col=index_col,
                     low_memory=low_memory, nrows=nrows, skiprows=skiprows)

    message = "\tLoaded dataframe: {} with shape: {}".format(os.path.basename(inpath), df.shape)
    if log is None:
        print(message)
    else:
        log.info(message)

    return df


def save_dataframe(df, outpath, header, index, sep="\t", log=None):
    if df is None:
        return

    compression = 'infer'
    if outpath.endswith('.gz'):
        compression = 'gzip'

    df.to_csv(outpath, sep=sep, index=index, header=header,
              compression=compression)

    message = "\tSaved dataframe: {} with shape: {}".format(os.path.basename(outpath), df.shape)
    if log is None:
        print(message)
    else:
        log.info(message)


def get_ieqtls(eqtl_m, geno_m, expr_m, context_a, cov, alpha):
    n_eqtls = eqtl_m.shape[0]

    ieqtls = []
    sample_masks = []
    results = []
    p_values = np.empty(n_eqtls, dtype=np.float64)
    for row_index in range(n_eqtls):
        snp, gene = eqtl_m[row_index, :]
        ieqtl = IeQTL(snp=snp,
                      gene=gene,
                      cov=cov,
                      genotype=geno_m[row_index, :],
                      covariate=context_a,
                      expression=expr_m[row_index, :]
                      )
        sample_mask = ieqtl.get_mask()
        sample_masks.append(sample_mask)

        ieqtl.compute()
        p_values[row_index] = ieqtl.p_value
        ieqtls.append(ieqtl)
        results.append([snp, gene, cov, ieqtl.n] + ieqtl.betas.tolist() + ieqtl.std.tolist() + [ieqtl.p_value])

    # Calculate the FDR.
    fdr_values = multitest.multipletests(p_values, method='fdr_bh')[1]

    # Calculate the number of significant hits.
    mask = fdr_values <= alpha
    n_hits = np.sum(mask)

    # Calculate the number of hits per sample.
    n_hits_per_sample = np.stack(sample_masks, axis=0)[mask, :].sum(axis=0)

    results_df = pd.DataFrame(results,
                              columns=["SNP", "gene", "covariate", "N",
                                       "beta-intercept", "beta-genotype",
                                       "beta-covariate", "beta-interaction",
                                       "std-intercept", "std-genotype",
                                       "std-covariate", "std-interaction",
                                       "p-value"])
    results_df["FDR"] = fdr_values

    return n_hits, n_hits_per_sample, [ieqtl for ieqtl, include in zip(ieqtls, mask) if include], results_df
