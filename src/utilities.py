"""
File:         utilities.py
Created:      2021/04/28
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
import os

# Third party imports.
import pandas as pd


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
