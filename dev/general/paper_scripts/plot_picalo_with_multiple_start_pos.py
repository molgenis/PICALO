#!/usr/bin/env python3

"""
File:         plot_picalo_with_multiple_start_pos.py
Created:      2021/05/18
Last Changed:
Author:       M.Vochteloo

Copyright (C) 2020 University Medical Center Groningen.

A copy of the BSD 3-Clause "New" or "Revised" License can be found in the
LICENSE file in the root directory of this source tree.
"""

# Standard imports.
from __future__ import print_function
import argparse
import glob
import re
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
__license__ = "BSD (3-Clause)"
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
    -e png

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
        self.n_pics = 3
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

        print("Loading data")
        iteration_data = []
        indices = []
        first_ieqtls = {}
        last_ieqtls = {}
        for pic_index in range(1, self.n_pics + 1):
            for comp_index in range(1, self.n_comps + 1):
                index = "PIC{}-Comp{}".format(pic_index, comp_index)
                iteration_fpath = os.path.join(self.indir, "{}-{}AsCov".format(self.filename, index), "PIC1", "iteration.txt.gz")
                if not os.path.exists(iteration_fpath):
                    continue

                iteration_df = self.load_file(iteration_fpath, header=0, index_col=0)
                if iteration_df.shape[0] == 1:
                    iteration_df.iloc[0, :] = np.nan
                iteration_data.append(iteration_df.iloc[-1, :].copy())
                indices.append(index)

                result_files = glob.glob(os.path.join(self.indir, "{}-{}AsCov".format(self.filename, index), "PIC1", "results_iteration*.txt.gz"))
                result_files.sort(key=self.natural_keys)

                first_result_df = self.load_file(result_files[0], header=0, index_col=None)
                first_result_df.index = first_result_df["SNP"] + "_" + first_result_df["gene"]
                first_ieqtls[index] = set(first_result_df.loc[first_result_df["FDR"] <= 0.05, :].index)

                last_result_df = self.load_file(result_files[-1], header=0, index_col=None)
                last_result_df.index = last_result_df["SNP"] + "_" + last_result_df["gene"]
                last_ieqtls[index] = set(last_result_df.loc[last_result_df["FDR"] <= 0.05, :].index)

                del iteration_df, first_result_df, last_result_df

        iteration_df = pd.concat(iteration_data, axis=1).T
        iteration_df.index = indices
        print(iteration_df)

        print("Correlating.")
        corr_m, pvalue_m = self.corrcoef(m=iteration_df.to_numpy())

        print("Overlap.")
        first_overlap_df, first_annot_df = self.overlap(ieqtls=first_ieqtls, indices=indices)
        last_overlap_df, last_annot_df = self.overlap(ieqtls=last_ieqtls, indices=indices)

        print("Perform BH correction")
        fdr_df = pd.DataFrame({"pvalue": pvalue_m.flatten()})
        mask = ~fdr_df["pvalue"].isnull()
        fdr_df["FDR"] = np.nan
        fdr_df.loc[mask, "FDR"] = multitest.multipletests(fdr_df.loc[mask, "pvalue"], method='fdr_bh')[1]
        fdr_m = fdr_df["FDR"].to_numpy().reshape(pvalue_m.shape)
        del fdr_df, mask

        corr_df = pd.DataFrame(corr_m, index=indices, columns=indices)
        pvalue_df = pd.DataFrame(pvalue_m, index=indices, columns=indices)
        fdr_df = pd.DataFrame(fdr_m, index=indices, columns=indices)

        print("Saving output.")
        for tmp_df, suffix in ([iteration_df, "iteration_data"],
                               [corr_df, "correlation_coefficient"],
                               [pvalue_df, "correlation_pvalue"],
                               [fdr_df, "correlation_FDR"],
                               [first_overlap_df, "first_overlap"],
                               [first_annot_df, "first_min_size"],
                               [last_overlap_df, "last_overlap"],
                               [last_annot_df, "last_min_size"],
                               ):
            self.save_file(df=tmp_df,
                           outpath=os.path.join(self.file_outdir,
                                                "{}_{}.txt.gz".format(self.filename, suffix)))
        # exit()

        # iteration_df = self.load_file(os.path.join(self.file_outdir, "{}_data.txt.gz".format(self.filename)), header=0, index_col=0)
        # corr_df = self.load_file(os.path.join(self.file_outdir, "{}_correlation_coefficient.txt.gz".format(self.filename)), header=0, index_col=0)
        # fdr_df = self.load_file(os.path.join(self.file_outdir, "{}_correlation_FDR.txt.gz".format(self.filename)), header=0, index_col=0)
        # first_overlap_df = self.load_file(os.path.join(self.file_outdir, "{}_first_overlap.txt.gz".format(self.filename)), header=0, index_col=0)
        # first_annot_df = self.load_file(os.path.join(self.file_outdir, "{}_first_min_size.txt.gz".format(self.filename)), header=0, index_col=0)
        # last_overlap_df = self.load_file(os.path.join(self.file_outdir, "{}_first_min_size.txt.gz".format(self.filename)), header=0, index_col=0)
        # last_annot_df = self.load_file(os.path.join(self.file_outdir, "{}_last_min_size.txt.gz".format(self.filename)), header=0, index_col=0)
        print(iteration_df)
        print(corr_df)
        print(fdr_df)
        print(first_overlap_df)
        print(first_annot_df)
        print(last_overlap_df)
        print(last_annot_df)

        corr_df = corr_df * corr_df

        print("Masking matrix.")
        color_corr_df = corr_df.copy()
        color_corr_df[fdr_df > 0.05] = np.nan

        print("Plotting heatmap")
        self.plot_heatmap(heatmap_df=color_corr_df,
                          reg_df=iteration_df.loc[[self.rx, self.ry], :].T,
                          reg_x=self.rx,
                          reg_y=self.ry,
                          data_type="correlation")
        self.plot_heatmap(heatmap_df=first_overlap_df,
                          annot_df=first_annot_df,
                          data_type="first_overlap")
        self.plot_heatmap(heatmap_df=last_overlap_df,
                          annot_df=last_annot_df,
                          data_type="last_overlap")

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
    def natural_keys(text):
        return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

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
    def overlap(ieqtls, indices):
        overlap = pd.DataFrame(np.nan, index=indices, columns=indices)
        annot = pd.DataFrame("", index=indices, columns=indices)
        for index1 in indices:
            for index2 in indices:
                n_overlap = len(ieqtls[index1].intersection(ieqtls[index2]))
                min_size = min(len(ieqtls[index1]), len(ieqtls[index2]))

                fraction = np.nan
                if n_overlap > 0 and min_size > 0:
                    fraction = n_overlap / min_size
                overlap.loc[index1, index2] = fraction
                annot.loc[index1, index2] = "{:,.2f}\n{:,}/{:,}".format(fraction, n_overlap, min_size)

        return overlap, annot

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

    def plot_heatmap(self, heatmap_df, annot_df=None, reg_df=None,
                     reg_x="x", reg_y="y", data_type=""):
        fontsize = 16
        if annot_df is None:
            fontsize = 22
            annot_df = heatmap_df.round(2)

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
                if (row_index == 0) and (col_index == (self.n_pics - 1)) and reg_df is not None:
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
                subset_annot_df = annot_df.iloc[row_start_index:row_end_index, col_start_index:col_end_index]

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
                                 annot=subset_annot_df,
                                 xticklabels=row_index == (self.n_pics - 1),
                                 yticklabels=col_index == 0,
                                 mask=mask,
                                 fmt='',
                                 cbar=False,
                                 annot_kws={"size": fontsize, "color": "#000000"},
                                 ax=ax)

                if row_index == (self.n_pics - 1):
                    ax.set_xlabel("PIC{}".format(col_pic_index), fontsize=45)

                if col_index == 0:
                    ax.set_ylabel("PIC{}".format(row_pic_index), fontsize=45)

                hm.set_xticklabels(hm.get_xmajorticklabels(), fontsize=30, rotation=90)
                hm.set_yticklabels(hm.get_ymajorticklabels(), fontsize=30, rotation=0)

        for extension in self.extensions:
            filename = "{}_picalo_multiple_start_pos_{}_heatmap.{}".format(self.filename, data_type, extension)
            fig.savefig(os.path.join(self.plot_outdir, filename))
            print("Saved '{}'.".format(filename))
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
