#!/usr/bin/env python3

"""
File:         plot_double_regplot.py
Created:      2021/05/24
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
import json
import os

# Third party imports.
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Local application imports.

# Metadata
__program__ = "Plot Double Regplot"
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
./plot_double_regplot.py -h

./plot_double_regplot.py \
    -mx /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/PICs.txt.gz \
    -metabrain_filex_transpose \
    -mxi PIC2 \
    -my /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/data/MetaBrain_CellFractionPercentages_forPlotting.txt.gz \
    -myi Glia \
    -mp /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/data/MetaBrainColorPalette.json \
    -bx /groups/umcg-bios/tmp01/projects/PICALO/output/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/PICs.txt.gz \
    -bios_filex_transpose \
    -bxi PIC2 \
    -by /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_phenotype_matrix/BIOS_CellFractionPercentages_forPlotting.txt.gz \
    -byi Myeloid_Lineage \
    -bp /groups/umcg-bios/tmp01/projects/PICALO/data/BIOSColorPalette.json \
    -o 2022-03-24-MetaBrain_and_BIOS_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_PICs_vs_MainCellFractions \
    -e pdf png
    
./plot_double_regplot.py \
    -mx /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/PICs.txt.gz \
    -metabrain_filex_transpose \
    -mxi PIC1 \
    -my /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/correlate_samples_with_avg_gene_expression/MetaBrain_CorrelationsWithAverageExpression.txt.gz \
    -myi AvgExprCorrelation \
    -mstd /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/sample_to_dataset.txt.gz \
    -mp /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/data/MetaBrainColorPalette.json \
    -bx /groups/umcg-bios/tmp01/projects/PICALO/output/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/PICs.txt.gz \
    -bios_filex_transpose \
    -bxi PIC1 \
    -by /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/correlate_samples_with_avg_gene_expression/BIOS_CorrelationsWithAverageExpression.txt.gz \
    -byi AvgExprCorrelation \
    -bstd /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs/sample_to_dataset.txt.gz \
    -bp /groups/umcg-bios/tmp01/projects/PICALO/data/BIOSColorPalette.json \
    -o 2022-03-24-MetaBrain_and_BIOS_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_PICs_vs_AvgExprCorrelation \
    -e pdf png
    
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.metax_path = getattr(arguments, 'metabrain_filex')
        self.metax_transpose = getattr(arguments, 'metabrain_filex_transpose')
        self.metax_index = getattr(arguments, 'metabrain_filex_index').replace("_", " ")

        self.metay_path = getattr(arguments, 'metabrain_filey')
        self.metay_transpose = getattr(arguments, 'metabrain_filey_transpose')
        self.metay_index = getattr(arguments, 'metabrain_filey_index').replace("_", " ")

        self.meta_std_path = getattr(arguments, 'metabrain_std')
        self.meta_palette_path = getattr(arguments, 'metabrain_palette_path')

        self.biosx_path = getattr(arguments, 'bios_filex')
        self.biosx_transpose = getattr(arguments, 'bios_filex_transpose')
        self.biosx_index = getattr(arguments, 'bios_filex_index').replace("_", " ")

        self.biosy_path = getattr(arguments, 'bios_filey')
        self.biosy_transpose = getattr(arguments, 'bios_filey_transpose')
        self.biosy_index = getattr(arguments, 'bios_filey_index').replace("_", " ")

        self.bios_std_path = getattr(arguments, 'bios_std')
        self.bios_palette_path = getattr(arguments, 'bios_palette_path')

        self.extensions = getattr(arguments, 'extensions')
        self.outfile = getattr(arguments, 'outfile')

        # Set variables.
        base_dir = str(os.path.dirname(os.path.abspath(__file__)))
        self.outdir = os.path.join(base_dir, 'plot')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        # Set the right pdf font for exporting.
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42

        # Loading palette.
        self.meta_palette = None
        if self.meta_palette_path is not None:
            with open(self.meta_palette_path) as f:
                self.meta_palette = json.load(f)
            f.close()

        self.bios_palette = None
        if self.bios_palette_path is not None:
            with open(self.bios_palette_path) as f:
                self.bios_palette = json.load(f)
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
        parser.add_argument("-mx",
                            "--metabrain_filex",
                            type=str,
                            required=True,
                            help="The path to MetaBrain x-axis file.")
        parser.add_argument("-metabrain_filex_transpose",
                            action='store_true',
                            help="Transpose the x-axis MetaBrain file.")
        parser.add_argument("-mxi",
                            "--metabrain_filex_index",
                            type=str,
                            required=True,
                            help="The index of the MetaBrain x-axis data.")

        parser.add_argument("-my",
                            "--metabrain_filey",
                            type=str,
                            required=True,
                            help="The path to MetaBrain y-axis file.")
        parser.add_argument("-metabrain_filey_transpose",
                            action='store_true',
                            help="Transpose the y-axis MetaBrain file.")
        parser.add_argument("-myi",
                            "--metabrain_filey_index",
                            type=str,
                            required=True,
                            help="The index of the MetaBrain y-axis data.")

        parser.add_argument("-mstd",
                            "--metabrain_std",
                            type=str,
                            required=False,
                            default=None,
                            help="The path to the MetaBrain sample-to-dataset"
                                 "file.")
        parser.add_argument("-mp",
                            "--metabrain_palette_path",
                            type=str,
                            required=False,
                            default=None,
                            help="The path to the MetaBrain json file with the"
                                 "dataset to color combinations.")

        parser.add_argument("-bx",
                            "--bios_filex",
                            type=str,
                            required=True,
                            help="The path to BIOS x-axis file.")
        parser.add_argument("-bios_filex_transpose",
                            action='store_true',
                            help="Transpose the x-axis BIOS file.")
        parser.add_argument("-bxi",
                            "--bios_filex_index",
                            type=str,
                            required=True,
                            help="The index of the BIOS x-axis data.")

        parser.add_argument("-by",
                            "--bios_filey",
                            type=str,
                            required=True,
                            help="The path to BIOS y-axis file.")
        parser.add_argument("-bios_filey_transpose",
                            action='store_true',
                            help="Transpose the y-axis BIOS file.")
        parser.add_argument("-byi",
                            "--bios_filey_index",
                            type=str,
                            required=True,
                            help="The index of the BIOS y-axis data.")

        parser.add_argument("-bstd",
                            "--bios_std",
                            type=str,
                            required=False,
                            default=None,
                            help="The path to the BIOS sample-to-dataset"
                                 "file.")
        parser.add_argument("-bp",
                            "--bios_palette_path",
                            type=str,
                            required=False,
                            default=None,
                            help="The path to the BIOS json file with the"
                                 "dataset to color combinations.")

        parser.add_argument("-e",
                            "--extensions",
                            type=str,
                            nargs="+",
                            default=["png"],
                            choices=["eps", "pdf", "pgf", "png", "ps", "raw", "rgba", "svg", "svgz"],
                            help="The output file format(s), default: ['png']")
        parser.add_argument("-o",
                            "--outfile",
                            type=str,
                            required=True,
                            help="The name of the output file")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("Loading data")
        meta_df = self.load_data(pathx=self.metax_path,
                                 pathy=self.metay_path,
                                 transposex=self.metax_transpose,
                                 transposey=self.metay_transpose,
                                 indexx=self.metax_index,
                                 indexy=self.metay_index,
                                 path_std=self.meta_std_path
                                 )
        bios_df = self.load_data(pathx=self.biosx_path,
                                 pathy=self.biosy_path,
                                 transposex=self.biosx_transpose,
                                 transposey=self.biosy_transpose,
                                 indexx=self.biosx_index,
                                 indexy=self.biosy_index,
                                 path_std=self.bios_std_path
                                 )

        print(meta_df)
        print(bios_df)

        print("Plotting regression plot")
        self.plot_double_regplot(bios_df=bios_df,
                                 bios_hue="dataset" if self.bios_std_path is not None else None,
                                 meta_df=meta_df,
                                 meta_hue="dataset" if self.meta_std_path is not None else None)

    def load_data(self, pathx, pathy, transposex, transposey, indexx, indexy, path_std):
        dfx = self.load_file(pathx, header=0, index_col=0)
        dfy = self.load_file(pathy, header=0, index_col=0)

        if transposex:
            dfx = dfx.T

        if transposey:
            dfy = dfy.T

        df = dfx[[indexx]].merge(dfy[[indexy]], left_index=True, right_index=True)
        df.dropna(inplace=True)

        if path_std is not None:
            std = self.load_file(path_std, header=0, index_col=0)
            df = df.merge(std, left_index=True, right_index=True)

        return df

    @staticmethod
    def load_file(inpath, header, index_col, sep="\t", low_memory=True,
                  nrows=None, skiprows=None):
        df = pd.read_csv(inpath, sep=sep, header=header, index_col=index_col,
                         low_memory=low_memory, nrows=nrows, skiprows=skiprows)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(inpath),
                                      df.shape))
        return df

    def plot_double_regplot(self, bios_df, bios_hue, meta_df, meta_hue):
        sns.set_style("ticks")
        fig, axes = plt.subplots(nrows=2,
                                 ncols=2,
                                 figsize=(24, 10),
                                 sharex="none",
                                 sharey="none",
                                 gridspec_kw={"height_ratios": [0.9, 0.1]})
        sns.set(color_codes=True)

        self.single_regplot(fig=fig,
                            plot_ax=axes[0, 0],
                            legend_ax=axes[1, 0],
                            df=bios_df,
                            x=self.biosx_index,
                            y=self.biosy_index,
                            hue=bios_hue,
                            palette=self.bios_palette,
                            title="blood")
        self.single_regplot(fig=fig,
                            plot_ax=axes[0, 1],
                            legend_ax=axes[1, 1],
                            df=meta_df,
                            x=self.metax_index,
                            y=self.metay_index,
                            hue=meta_hue,
                            palette=self.meta_palette,
                            title="brain")

        for extension in self.extensions:
            fig.savefig(os.path.join(self.outdir, "{}.{}".format(self.outfile, extension)))
        plt.close()

    @staticmethod
    def single_regplot(fig, plot_ax, legend_ax, df, x="x", y="y",
                       hue=None, palette=None, xlabel=None, ylabel=None,
                       title=""):
        if xlabel is None:
            xlabel = x
        if ylabel is None:
            ylabel = y

        sns.despine(fig=fig, ax=plot_ax)
        legend_ax.set_axis_off()

        # Set annotation.
        pearson_coef, _ = stats.pearsonr(df[y], df[x])
        annot_xpos = 0.03
        if pearson_coef < 0:
            annot_xpos = 0.8
        plot_ax.annotate(
            'total N = {:,}'.format(df.shape[0]),
            xy=(annot_xpos, 0.94),
            xycoords=plot_ax.transAxes,
            color="#000000",
            fontsize=14,
            fontweight='bold')
        plot_ax.annotate(
            'total r = {:.2f}'.format(pearson_coef),
            xy=(annot_xpos, 0.90),
            xycoords=plot_ax.transAxes,
            color="#000000",
            fontsize=14,
            fontweight='bold')

        group_column = hue
        ci = None
        alpha = 0.8
        if hue is None:
            df["hue"] = "all"
            group_column = "hue"
            ci = 95
            alpha = 0.2

        handles = []
        for i, hue_group in enumerate(df[group_column].unique()):
            subset = df.loc[df[group_column] == hue_group, :]
            if subset.shape[0] < 2:
                continue

            facecolors = "#000000"
            color = "#b22222"
            if palette is not None:
                if hue_group in palette:
                    facecolors = palette[hue_group]
                    color = facecolors
                elif y in palette:
                    facecolors = palette[y]
                    color = "#000000"

            sns.regplot(x=x, y=y, data=subset, ci=ci,
                        scatter_kws={'facecolors': facecolors,
                                     'alpha': alpha,
                                     'linewidth': 0},
                        line_kws={"color": color},
                        ax=plot_ax)

            subset_pearson_coef, _ = stats.pearsonr(subset[y], subset[x])
            handles.append([mpatches.Patch(color=color,
                                           label="{} [n={:,}; r={:.2f}]".format(hue_group, subset.shape[0], subset_pearson_coef)), subset_pearson_coef])

        if len(df[group_column].unique()) > 1:
            handles.sort(key=lambda x: -x[1])
            handles = [x[0] for x in handles]
            legend_ax.legend(handles=handles, loc=8, fontsize=8, ncol=3)

        plot_ax.set_xlabel(xlabel,
                           fontsize=14,
                           fontweight='bold')
        plot_ax.set_ylabel(ylabel,
                           fontsize=14,
                           fontweight='bold')
        plot_ax.set_title(title,
                          fontsize=18,
                          fontweight='bold')

        # Change margins.
        xlim = (df[x].min(), df[x].max())
        ylim = (df[y].min(), df[y].max())

        xmargin = (xlim[1] - xlim[0]) * 0.1
        ymargin = (ylim[1] - ylim[0]) * 0.1
        new_xlim = (xlim[0] - xmargin, xlim[1] + xmargin)
        new_ylim = (ylim[0] - ymargin, ylim[1] + ymargin)

        plot_ax.set_xlim(new_xlim[0], new_xlim[1])
        plot_ax.set_ylim(new_ylim[0], 1)

    def print_arguments(self):
        print("Arguments:")
        print("  > MetaBrain:")
        print("  >     (1) {}: {} {}".format(self.metax_index, self.metax_path, "[T]" if self.metax_transpose else ""))
        print("  >     (2) {}: {} {}".format(self.metay_index, self.metay_path, "[T]" if self.metay_transpose else ""))
        print("  >     Sample-to-dataset: {}".format(self.meta_std_path))
        print("  >     Palette: {}".format(self.meta_palette_path))
        print("  > BIOS:")
        print("  >     (1) {}: {} {}".format(self.biosx_index, self.biosx_path, "[T]" if self.biosx_transpose else ""))
        print("  >     (2) {}: {} {}".format(self.biosy_index, self.biosy_path, "[T]" if self.biosy_transpose else ""))
        print("  >     Sample-to-dataset: {}".format(self.bios_std_path))
        print("  >     Palette: {}".format(self.bios_palette_path))
        print("  > Extensions: {}".format(self.extensions))
        print("  > Outfile: {}".format(self.outfile))
        print("  > Output directory: {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
