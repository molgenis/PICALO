#!/usr/bin/env python3

"""
File:         plot_picalo_with_multiple_start_pos.py
Created:      2021/05/18
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
import argparse
import os

# Third party imports.
import numpy as np
import pandas as pd
from scipy.special import betainc
from statsmodels.stats import multitest
from scipy import stats
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Local application imports.

# Metadata
__program__ = "Plot PICALO with Multiple Start Positions"
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
./plot_picalo_with_multiple_start_pos.py -h

### MetaBrain ###

### BIOS ###

./plot_picalo_with_multiple_start_pos.py \
    -i /groups/umcg-bios/tmp01/projects/PICALO/output \
    -f 2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA \
    -rx PIC2-Comp13 \
    -ry PIC3-Comp13 \
    -e png pdf

"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.indir = getattr(arguments, 'indir')
        self.filename = getattr(arguments, 'filename')
        self.rx = getattr(arguments, 'regplot_x')
        self.ry = getattr(arguments, 'regplot_y')
        self.extensions = getattr(arguments, 'extension')
        self.n_pics = 5
        self.n_comps = 25

        # Set variables.
        base_dir = str(os.path.dirname(os.path.abspath(__file__)))
        self.file_outdir = os.path.join(base_dir, 'plot_picalo_with_multiple_start_pos', self.filename)
        self.plot_outdir = os.path.join(self.file_outdir, 'plot')
        for outdir in [self.plot_outdir, self.file_outdir]:
            if not os.path.exists(outdir):
                os.makedirs(outdir)

        # Set the right pdf font for exporting.
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42

    @staticmethod
    def create_argument_parser():
        parser = argparse.ArgumentParser(prog=__program__,
                                         description=__description__)

        # Add optional arguments.
        parser.add_argument("-v",
                            "--version",
                            action="version",
                            version="{} {}".format(__program__,
                                                   __version__),
                            help="show program's version number and exit.")
        parser.add_argument("-i",
                            "--indir",
                            type=str,
                            required=True,
                            help="The path to the PICALO output directory.")
        parser.add_argument("-f",
                            "--filename",
                            type=str,
                            required=True,
                            help="The PICALO filename.")
        parser.add_argument("-rx",
                            "--regplot_x",
                            type=str,
                            default=None,
                            help="The x-axis values for the regplot.")
        parser.add_argument("-ry",
                            "--regplot_y",
                            type=str,
                            default=None,
                            help="The y-axis values for the regplot.")
        parser.add_argument("-e",
                            "--extension",
                            nargs="+",
                            type=str,
                            choices=["png", "pdf", "eps"],
                            default=["png"],
                            help="The figure file extension. "
                                 "Default: 'png'.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        # print("Loading data")
        # data = []
        # indices = []
        # for pic_index in range(1, self.n_pics + 1):
        #     for comp_index in range(1, self.n_comps + 1):
        #         index = "PIC{}-Comp{}".format(pic_index, comp_index)
        #         fpath = os.path.join(self.indir, "{}-{}AsCov".format(self.filename, index), "PIC1", "iteration.txt.gz")
        #         if not os.path.exists(fpath):
        #             continue
        #
        #         df = self.load_file(fpath, header=0, index_col=0)
        #         if df.shape[0] == 1:
        #             df.iloc[0, :] = np.nan
        #         data.append(df.iloc[-1, :].copy())
        #         indices.append(index)
        #
        #         del df
        #
        # df = pd.concat(data, axis=1).T
        # df.index = indices
        # print(df)
        #
        # print("Correlating.")
        # corr_m, pvalue_m = self.corrcoef(m=df.to_numpy())
        #
        # print("Perform BH correction")
        # fdr_df = pd.DataFrame({"pvalue": pvalue_m.flatten()})
        # mask = ~fdr_df["pvalue"].isnull()
        # fdr_df["FDR"] = np.nan
        # fdr_df.loc[mask, "FDR"] = multitest.multipletests(fdr_df.loc[mask, "pvalue"], method='fdr_bh')[1]
        # fdr_m = fdr_df["FDR"].to_numpy().reshape(pvalue_m.shape)
        # del fdr_df, mask
        #
        # print("Saving output.")
        # self.save_file(df=df,
        #                outpath=os.path.join(self.file_outdir, "{}_data.txt.gz".format(self.filename)))
        # for m, suffix in ([corr_m, "correlation_coefficient"],
        #                   [pvalue_m, "correlation_pvalue"],
        #                   [fdr_m, "correlation_FDR"]):
        #     tmp_df = pd.DataFrame(m, index=indices, columns=indices)
        #     self.save_file(df=tmp_df,
        #                    outpath=os.path.join(self.file_outdir,
        #                                         "{}_{}.txt.gz".format(self.filename, suffix)))
        #     del tmp_df
        # exit()

        # corr_df = pd.DataFrame(m, index=indices, columns=indices)
        # fdr_df = pd.DataFrame(fdr_m, index=indices, columns=indices)
        df = self.load_file(os.path.join(self.file_outdir, "{}_data.txt.gz".format(self.filename)), header=0, index_col=0)
        corr_df = self.load_file(os.path.join(self.file_outdir, "{}_correlation_coefficient.txt.gz".format(self.filename)), header=0, index_col=0)
        fdr_df = self.load_file(os.path.join(self.file_outdir, "{}_correlation_FDR.txt.gz".format(self.filename)), header=0, index_col=0)
        print(df)
        print(corr_df)
        print(fdr_df)

        corr_df = corr_df * corr_df

        print("Masking matrix.")
        color_corr_df = corr_df.copy()
        color_corr_df[fdr_df > 0.05] = np.nan

        print("Plotting heatmap")
        self.plot_heatmap(heatmap_df=color_corr_df,
                          reg_df=df.loc[[self.rx, self.ry], :].T,
                          reg_x=self.rx,
                          reg_y=self.ry)

    @staticmethod
    def load_file(inpath, header, index_col, sep="\t", low_memory=True,
                  nrows=None, skiprows=None):
        df = pd.read_csv(inpath, sep=sep, header=header, index_col=index_col,
                         low_memory=low_memory, nrows=nrows, skiprows=skiprows)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(inpath),
                                      df.shape))
        return df

    @staticmethod
    def corrcoef(m):
        """
        Pearson correlation over the columns.

        https://stackoverflow.com/questions/24432101/correlation-coefficients-and-p-values-for-all-pairs-of-rows-of-a-matrix
        """
        r = np.corrcoef(m)
        rf = r[np.triu_indices(r.shape[0], 1)]
        df = m.shape[1] - 2
        ts = rf * rf * (df / (1 - rf * rf))
        pf = betainc(0.5 * df, 0.5, df / (df + ts))
        p = np.zeros(shape=r.shape)
        p[np.triu_indices(p.shape[0], 1)] = pf
        p[np.tril_indices(p.shape[0], -1)] = p.T[
            np.tril_indices(p.shape[0], -1)]
        p[np.diag_indices(p.shape[0])] = np.ones(p.shape[0])
        return r, p

    @staticmethod
    def save_file(df, outpath, header=True, index=True, sep="\t"):
        compression = 'infer'
        if outpath.endswith('.gz'):
            compression = 'gzip'

        df.to_csv(outpath, sep=sep, index=index, header=header,
                  compression=compression)
        print("\tSaved dataframe: {} "
              "with shape: {}".format(os.path.basename(outpath),
                                      df.shape))

    def plot_heatmap(self, heatmap_df, reg_df, reg_x="x", reg_y="y"):
        sns.set_style("ticks")
        fig, axes = plt.subplots(nrows=self.n_pics,
                                 ncols=self.n_pics,
                                 sharex="none",
                                 sharey="none",
                                 figsize=(1 * heatmap_df.shape[1] + 10,
                                          1 * heatmap_df.shape[0] + 10))
        sns.set(color_codes=True)

        for row_index, row_pic_index in enumerate(range(1, self.n_pics + 1)):
            row_start_index = (row_pic_index - 1) * self.n_comps
            row_end_index = row_pic_index * self.n_comps
            for col_index, col_pic_index in enumerate(range(1, self.n_pics + 1)):
                print(row_index, col_index)
                ax = axes[row_index, col_index]
                if (row_index == 0) and (col_index == (self.n_pics - 1)):
                    sns.despine(fig=fig, ax=ax)
                    sns.regplot(x=reg_x,
                                y=reg_y,
                                data=reg_df,
                                scatter_kws={'facecolors': "#000000",
                                             'linewidth': 0,
                                             's': 60,
                                             'alpha': 0.75},
                                line_kws={"color": "#b22222",
                                          'linewidth': 5},
                                ax=ax)

                    pearson_coef, _ = stats.pearsonr(reg_df[reg_y], reg_df[reg_x])
                    ax.annotate(
                        'N = {:,}'.format(reg_df.shape[0]),
                        xy=(0.03, 0.94),
                        xycoords=ax.transAxes,
                        color="#000000",
                        fontsize=40,
                        fontweight='bold')
                    ax.annotate(
                        'r = {:.2f}'.format(pearson_coef),
                        xy=(0.03, 0.90),
                        xycoords=ax.transAxes,
                        color="#000000",
                        fontsize=40,
                        fontweight='bold')

                    ax.set_xlabel(reg_x,
                                  fontsize=40,
                                  fontweight='bold')
                    ax.set_ylabel(reg_y,
                                  fontsize=40,
                                  fontweight='bold')
                    continue
                elif row_index < col_index:
                    ax.set_axis_off()
                    continue

                col_start_index = (col_pic_index - 1) * self.n_comps
                col_end_index = col_pic_index * self.n_comps
                subset_df = heatmap_df.iloc[row_start_index:row_end_index, col_start_index:col_end_index]
                subset_df.index = [index.split("-")[1] for index in subset_df.index]
                subset_df.columns = [index.split("-")[1] for index in subset_df.columns]
                annot_df = subset_df.copy()
                annot_df = annot_df.round(2)
                annot_df.fillna("", inplace=True)

                mask = None
                if row_index == col_index:
                    mask = np.zeros((self.n_comps, self.n_comps))
                    mask[np.triu_indices_from(mask)] = True

                hm = sns.heatmap(subset_df,
                                 cmap=sns.diverging_palette(246, 24,
                                                            as_cmap=True),
                                 vmin=-1,
                                 vmax=1,
                                 center=0,
                                 square=True,
                                 annot=annot_df,
                                 xticklabels=row_index == (self.n_pics - 1),
                                 yticklabels=col_index == 0,
                                 mask=mask,
                                 fmt='',
                                 cbar=False,
                                 annot_kws={"size": 14, "color": "#000000"},
                                 ax=ax)

                if row_index == (self.n_pics - 1):
                    ax.set_xlabel("PIC{}".format(col_pic_index), fontsize=45)

                if col_index == 0:
                    ax.set_ylabel("PIC{}".format(row_pic_index), fontsize=45)

                hm.set_xticklabels(hm.get_xmajorticklabels(), fontsize=30, rotation=90)
                hm.set_yticklabels(hm.get_ymajorticklabels(), fontsize=30, rotation=0)

        for extension in self.extensions:
            fig.savefig(os.path.join(self.plot_outdir, "{}_picalo_multiple_start_pos_correlation_heatmap.{}".format(self.filename, extension)))
        plt.close()

    def print_arguments(self):
        print("Arguments:")
        print("  > Input directory: {}".format(self.indir))
        print("  > Filename: {}".format(self.filename))
        print("  > Regression plot:")
        print("  >     x-axis: {}".format(self.rx))
        print("  >     y-axis: {}".format(self.ry))
        print("  > Extensions: {}".format(self.extensions))
        print("  > File outpath {}".format(self.file_outdir))
        print("  > Plot outpath {}".format(self.plot_outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
