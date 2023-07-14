#!/usr/bin/env python3

"""
File:         plot_pca.py
Created:      2023/07/07
Last Changed: 2023/07/14
Author:       M.Vochteloo

Copyright (C) 2020 University Medical Center Groningen.

A copy of the BSD 3-Clause "New" or "Revised" License can be found in the
LICENSE file in the root directory of this source tree.
"""

# Standard imports.
from __future__ import print_function
import argparse
import json
import os

# Third party imports.
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Local application imports.

# Metadata
__program__ = "Plot PCA"
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
./plot_pca.py -h

### MetaBrain 2 covariates ###

./plot_pca.py \
    -d /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/simulate_expression/2023-07-13-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-first2ExprPCForceNormalised/simulation1/expression_table.txt.gz \
    -zscore \
    -std /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/sample_to_dataset.txt.gz \
    -p /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/data/MetaBrainColorPalette.json \
    -o 2023-07-13-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-first2ExprPCForceNormalised-simulation1
    
./plot_pca.py \
    -d /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/simulate_expression2/2023-07-13-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-first2ExprPCForceNormalised/expression_table.txt.gz \
    -zscore \
    -o 2023-07-13-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-first2ExprPCForceNormalised-V2
    
### Bios 2 covaraites ###

./plot_pca.py \
    -d /groups/umcg-bios/tmp01/projects/PICALO/simulate_expression/2023-07-13-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-first2ExprPCForceNormalised/simulation1/expression_table.txt.gz \
    -zscore \
    -std /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/sample_to_dataset.txt.gz \
    -p /groups/umcg-bios/tmp01/projects/PICALO/data/BIOSColorPalette.json \
    -o 2023-07-13-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-first2ExprPCForceNormalised-simulation1
    
./plot_pca.py \
    -d /groups/umcg-bios/tmp01/projects/PICALO/simulate_expression2/2023-07-13-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-first2ExprPCForceNormalised/expression_table.txt.gz \
    -zscore \
    -o 2023-07-13-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-first2ExprPCForceNormalised-V2
    

"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.data_path = getattr(arguments, 'data')
        self.transpose = getattr(arguments, 'transpose')
        self.zscore = getattr(arguments, 'zscore')
        self.save = getattr(arguments, 'save')
        self.n_save = getattr(arguments, 'n_save')
        self.n_plot = getattr(arguments, 'n_plot')
        self.std_path = getattr(arguments, 'sample_to_dataset')
        self.palette_path = getattr(arguments, 'palette')
        outdir = getattr(arguments, 'outdir')
        outfolder = getattr(arguments, 'outfolder')
        self.extensions = getattr(arguments, 'extensions')

        # Set variables.
        if outdir is None:
            outdir = str(os.path.dirname(os.path.abspath(__file__)))
        self.outdir = os.path.join(outdir, "plot_pca", outfolder)
        self.plot_outdir = os.path.join(outdir, "plot_pca", outfolder, "plot")
        for outdir in [self.outdir, self.plot_outdir]:
            if not os.path.exists(outdir):
                os.makedirs(outdir)

        # Loading palette.
        self.palette = None
        if self.palette_path is not None:
            with open(self.palette_path) as f:
                self.palette = json.load(f)
            f.close()

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
        parser.add_argument("-d",
                            "--data",
                            type=str,
                            required=True,
                            help="The path to the data matrix.")
        parser.add_argument("-ns",
                            "--n_save",
                            type=int,
                            required=False,
                            default=25,
                            help="The number of components to save. Default: 25.")
        parser.add_argument("-np",
                            "--n_plot",
                            type=int,
                            required=False,
                            default=5,
                            help="The number of components to plot. Default: 5.")
        parser.add_argument("-transpose",
                            action='store_true',
                            help="Transpose.")
        parser.add_argument("-zscore",
                            action='store_true',
                            help="Z-score transform.")
        parser.add_argument("-save",
                            action='store_true',
                            help="Save.")
        parser.add_argument("-std",
                            "--sample_to_dataset",
                            type=str,
                            required=False,
                            default=None,
                            help="The path to the sample-dataset link matrix.")
        parser.add_argument("-p",
                            "--palette",
                            type=str,
                            required=False,
                            default=None,
                            help="The path to a json file with the"
                                 "dataset to color combinations.")
        parser.add_argument("-od",
                            "--outdir",
                            type=str,
                            required=False,
                            default=None,
                            help="The name of the output path.")
        parser.add_argument("-of",
                            "--outfolder",
                            type=str,
                            required=False,
                            default="output",
                            help="The name of the output folder.")
        parser.add_argument("-e",
                            "--extensions",
                            type=str,
                            nargs="+",
                            default=["png"],
                            choices=["eps", "pdf", "pgf", "png", "ps", "raw", "rgba", "svg", "svgz"],
                            help="The output file format(s), default: ['png']")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("Loading data.")
        df = self.load_file(self.data_path, header=0, index_col=0)
        if self.transpose:
            df = df.T
        print(df)

        if self.zscore:
            df = (df - df.mean(axis=0)) / df.std(axis=0)
        pca = PCA(n_components=self.n_save)
        pca.fit(df)
        pca_df = pd.DataFrame(pca.components_).T
        columns = ["PC{}".format(i + 1) for i in range(self.n_save)]
        pca_df.columns = columns
        pca_df.index = df.columns
        print(pca_df)

        explained_var_df = pd.DataFrame(pca.explained_variance_ratio_ * 100,
                                        index=["PC{}".format(i + 1) for i in range(self.n_save)],
                                        columns=["ExplainedVariance"])
        print(explained_var_df)

        if self.save:
            self.save_file(df=pca_df,
                           outpath=os.path.join(self.outdir, "first{}ExpressionPCs.txt.gz".format(self.n_save)))
            self.save_file(df=explained_var_df,
                           outpath=os.path.join(self.outdir, "first{}ExpressionPCsExplainedVariance.txt.gz".format(self.n_save)))

        print("Loading color data.")
        hue = None
        palette = None
        if self.std_path is not None:
            sa_df = self.load_file(self.std_path, header=0, index_col=None)
            sa_df.set_index(sa_df.columns[0], inplace=True)
            sa_df.columns = ["hue"]
            sa_df["hue"] = sa_df["hue"].astype(str)
            pca_df = pca_df.iloc[:, :self.n_plot].merge(sa_df, left_index=True, right_index=True)

            hue = "hue"
            palette = self.palette


        # Plotting.
        self.plot(df=pca_df,
                  columns=columns[:self.n_plot],
                  explained_variance=explained_var_df,
                  hue=hue,
                  palette=palette)

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
    def save_file(df, outpath, header=True, index=True, sep="\t"):
        compression = 'infer'
        if outpath.endswith('.gz'):
            compression = 'gzip'

        df.to_csv(outpath, sep=sep, index=index, header=header,
                  compression=compression)
        print("\tSaved dataframe: {} "
              "with shape: {}".format(os.path.basename(outpath),
                                      df.shape))

    def plot(self, df, columns, explained_variance, hue, palette):
        ncols = len(columns)
        nrows = len(columns)

        fig, axes = plt.subplots(nrows=nrows,
                                 ncols=ncols,
                                 sharex='col',
                                 sharey='row',
                                 figsize=(10 * ncols, 10 * nrows))
        sns.set(color_codes=True)

        for i, y_col in enumerate(columns):
            for j, x_col in enumerate(columns):
                print(i, j)

                ax = axes[i, j]
                if i == 0 and j == (ncols - 1):
                    ax.set_axis_off()
                    if hue is not None and palette is not None:
                        groups_present = df[hue].unique().tolist()
                        handles = []
                        for key, value in palette.items():
                            if key in groups_present:
                                handles.append(mpatches.Patch(color=value, label=key))
                        ax.legend(handles=handles, loc=4, fontsize=25)

                elif i < j:
                    ax.set_axis_off()
                    continue
                elif i == j:
                    ax.set_axis_off()

                    ax.annotate("{}\n{:.2f}%".format(y_col, explained_variance.loc[y_col, "ExplainedVariance"]),
                                xy=(0.5, 0.5),
                                ha='center',
                                xycoords=ax.transAxes,
                                color="#000000",
                                fontsize=40,
                                fontweight='bold')
                else:
                    sns.despine(fig=fig, ax=ax)

                    sns.scatterplot(x=x_col,
                                    y=y_col,
                                    hue=hue,
                                    data=df,
                                    s=100,
                                    palette=palette,
                                    linewidth=0,
                                    legend=False,
                                    ax=ax)

                    ax.set_ylabel("",
                                  fontsize=20,
                                  fontweight='bold')
                    ax.set_xlabel("",
                                  fontsize=20,
                                  fontweight='bold')

        for extension in self.extensions:
            fig.savefig(os.path.join(self.plot_outdir, "pca_plot.{}".format(extension)))
        plt.close()

    def print_arguments(self):
        print("Arguments:")
        print("  > Data path: {}".format(self.data_path))
        print("  > Transpose: {}".format(self.transpose))
        print("  > Z-score: {}".format(self.zscore))
        print("  > Save: {}".format(self.save))
        print("  > N components to save: {}".format(self.n_save))
        print("  > N components to plot: {}".format(self.n_plot))
        print("  > Sample-to-dataset path: {}".format(self.std_path))
        print("  > Output directory: {}".format(self.outdir))
        print("  > Extensions: {}".format(self.extensions))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
