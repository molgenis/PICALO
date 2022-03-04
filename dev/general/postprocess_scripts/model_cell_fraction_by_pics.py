#!/usr/bin/env python3

"""
File:         model_cell_fractions_by_pics.py
Created:      2022/02/25
Last Changed: 2022/03/04
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
from pathlib import Path
import argparse
import os

# Third party imports.
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Local application imports.

# Metadata
__program__ = "Model Cell Fractions by PICS"
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
./model_cell_fraction_by_pics.py -h


### BIOS ###

./model_cell_fraction_by_pics.py \
    -cf /groups/umcg-bios/tmp01/projects/PICALO/data/BIOS_cell_types_DeconCell_2019-03-08.txt.gz \
    -p /groups/umcg-bios/tmp01/projects/PICALO/output/2021-12-09-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs/PICs.txt.gz \
    -o 2021-12-09-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs-Complete
    
./model_cell_fraction_by_pics.py \
    -cf /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_phenotype_matrix/BIOS_CellFractionPercentages.txt.gz \
    -p /groups/umcg-bios/tmp01/projects/PICALO/output/2021-12-09-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs/PICs.txt.gz \
    -o 2021-12-09-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs
    
    
### METABRAIN ###

./model_cell_fraction_by_pics.py \
    -cf /groups/umcg-biogen/tmp01/output/2019-11-06-FreezeTwoDotOne/2020-10-12-deconvolution/deconvolution/matrix_preparation/2022-01-21-CortexEUR-cis-NegativeToZero-DatasetAndRAMCorrected/perform_deconvolution/deconvolution_table_complete.txt.gz \
    -p /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/2021-12-09-MetaBrain-CortexEUR-cis-NoENA-NoMDSOutlier-GT1AvgExprFilter-PrimaryeQTLs/PICs.txt.gz \
    -o 2021-12-09-MetaBrain-CortexEUR-cis-NoENA-NoMDSOutlier-GT1AvgExprFilter-PrimaryeQTLs-Complete
    
./model_cell_fraction_by_pics.py \
    -cf /groups/umcg-biogen/tmp01/output/2019-11-06-FreezeTwoDotOne/2020-10-12-deconvolution/deconvolution/matrix_preparation/2022-01-21-CortexEUR-cis-NegativeToZero-DatasetAndRAMCorrected/perform_deconvolution/deconvolution_table.txt.gz \
    -p /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/2021-12-09-MetaBrain-CortexEUR-cis-NoENA-NoMDSOutlier-GT1AvgExprFilter-PrimaryeQTLs/PICs.txt.gz \
    -o 2021-12-09-MetaBrain-CortexEUR-cis-NoENA-NoMDSOutlier-GT1AvgExprFilter-PrimaryeQTLs
    

"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.cf_path = getattr(arguments, 'cell_fractions')
        self.pics_path = getattr(arguments, 'pics')
        self.out_filename = getattr(arguments, 'outfile')

        # Set variables.
        base_dir = str(Path(__file__).parent.parent)
        self.file_outdir = os.path.join(base_dir, 'model_cell_fraction_by_pics')
        self.plot_outdir = os.path.join(self.file_outdir, 'plot')
        for outdir in [self.plot_outdir, self.file_outdir]:
            if not os.path.exists(outdir):
                os.makedirs(outdir)

        self.ct_trans_dict ={
            "Baso": "Basophil",
            "Neut": "Neutrophil",
            "Eos": "Eosinophil",
            "Granulocyte": "Granulocyte",
            "Mono": "Monocyte",
            "LUC": "LUC",
            "Lymph": "Lymphocyte",
        }

        self.palette = {
            'Adult-Ex1': '#56B4E9',
            'Adult-Ex2': '#56B4E9',
            'Adult-Ex3': '#56B4E9',
            'Adult-Ex4': '#56B4E9',
            'Adult-Ex5': '#56B4E9',
            'Adult-Ex6': '#56B4E9',
            'Adult-Ex7': '#56B4E9',
            'Adult-Ex8': '#56B4E9',
            'Adult-In1': '#2690ce',
            'Adult-In2': '#2690ce',
            'Adult-In3': '#2690ce',
            'Adult-In4': '#2690ce',
            'Adult-In5': '#2690ce',
            'Adult-In6': '#2690ce',
            'Adult-In7': '#2690ce',
            'Adult-In8': '#2690ce',
            'Adult-Microglia': '#E69F00',
            'Adult-OPC': '#1b8569',
            'Adult-Endothelial': '#CC79A7',
            'Adult-Astrocytes': '#D55E00',
            'Adult-Oligo': '#009E73',
            'Adult-OtherNeuron': '#2690ce',
            'Dev-Replicating': '#000000',
            'Dev-Quiescent': '#808080',
            "Excitatory": "#56B4E9",
            "Inhibitory": "#2690ce",
            'OtherNeuron': '#0072B2',
            "Oligodendrocyte": "#009E73",
            "EndothelialCell": "#CC79A7",
            "Microglia": "#E69F00",
            "Astrocyte": "#D55E00",
            "Basophil": "#009E73",
            "Neutrophil": "#D55E00",
            "Eosinophil": "#0072B2",
            "Granulocyte": "#808080",
            "Monocyte": "#E69F00",
            "LUC": "#F0E442",
            "Lymphocyte": "#CC79A7",
        }

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
        parser.add_argument("-cf",
                            "--cell_fractions",
                            type=str,
                            required=True,
                            help="The path to the cell fraction matrix.")
        parser.add_argument("-p",
                            "--pics",
                            type=str,
                            required=True,
                            help="The path to the PICS matrix.")
        parser.add_argument("-o",
                            "--outfile",
                            type=str,
                            default="output",
                            help="The name of the outfile. Default: output.")

        return parser.parse_args()

    def start(self):
        print("Starting program")
        self.print_arguments()

        print("Loading data")
        cf_df = self.load_file(self.cf_path)
        pics_df = self.load_file(self.pics_path)

        print("Preprocessing data")
        pics_df = pics_df.T
        if cf_df.shape[0] < cf_df.shape[1]:
            cf_df = cf_df.T

        samples = [sample for sample in pics_df.index if sample in cf_df.index]
        print("\tUsing {} samples".format(len(samples)))
        pics_df = pics_df.loc[samples, :]
        cf_df = cf_df.loc[samples, :]
        pics = pics_df.columns

        cf_df.columns = [col.split("_")[0] for col in cf_df.columns]
        cf_df.columns = [self.ct_trans_dict[col] if col in self.ct_trans_dict else col for col in cf_df.columns]

        pics_df.insert(0, "INTERCEPT", 1)

        print("Modelling")
        correlation_m = np.empty((cf_df.shape[1], len(pics)), dtype=np.float64)
        pvalue_m = np.empty((cf_df.shape[1], len(pics)), dtype=np.float64)
        ols_results_m = np.empty((cf_df.shape[1], 2 + (len(pics) * 2)), dtype=np.float64)
        index = []
        full_index = []
        for i, cell_type in enumerate(cf_df.columns):
            # Creat mask.
            mask = ~cf_df.loc[:, cell_type].isna()
            n = mask.sum()

            print("\t{} [N={:,}]".format(cell_type, n))

            # Correlations.
            for j, pic in enumerate(pics):
                coef, pvalue = stats.pearsonr(cf_df.loc[mask, cell_type], pics_df.loc[mask, pic])
                correlation_m[i, j] = coef
                pvalue_m[i, j] = pvalue

            # OLS model.
            ols = OLS(cf_df.loc[mask, cell_type], pics_df.loc[mask, :])
            results = ols.fit()

            # Save results.
            ols_results_m[i, :] = np.hstack((np.array([n, results.rsquared]), results.params[1:], results.bse[1:]))
            index.append(cell_type)
            full_index.append("{} [N={:,}]".format(cell_type, n))

        correlation_df = pd.DataFrame(correlation_m,
                                      index=index,
                                      columns=pics
                                      )
        print(correlation_df)
        pvalue_df = pd.DataFrame(pvalue_m,
                                 index=index,
                                 columns=pics
                                 )

        ols_results_df = pd.DataFrame(ols_results_m,
                                      index=index,
                                      columns=["N", "R2"] +
                                              ["{} beta".format(pic) for pic in pics] +
                                              ["{} std err".format(pic) for pic in pics]
                                      )
        for pic in pics:
            ols_results_df["{} t-value".format(pic)] = ols_results_df["{} beta".format(pic)] / ols_results_df["{} std err".format(pic)]
        print(ols_results_df)

        print("Saving file.")
        self.save_file(df=correlation_df,
                       outpath=os.path.join(self.file_outdir, "{}_correlation_df.txt.gz".format(self.out_filename)))
        self.save_file(df=ols_results_df,
                       outpath=os.path.join(self.file_outdir, "{}_ols_results_df.txt.gz".format(self.out_filename)))

        print("Visualising")
        bar_df = ols_results_df[["R2"]].copy()
        bar_df["index"] = bar_df.index
        palette = self.palette
        for ct in bar_df["index"]:
            if ct not in palette:
                palette = None
                break
        self.plot_barplot(
            df=bar_df,
            x="R2",
            y="index",
            xlabel="R\u00b2",
            ylabel="cell type",
            palette=palette,
            appendix="_R2_barplot"
        )

        correlation_df.index = full_index
        correlation_annot_df = correlation_df.copy()
        correlation_annot_df.index = full_index
        correlation_annot_df = correlation_annot_df.round(2).astype(str)
        correlation_df[pvalue_df > 0.05] = 0
        self.plot_heatmap(df=correlation_df,
                          annot_df=correlation_annot_df,
                          vmin=-1,
                          vmax=1,
                          xlabel="PICs",
                          ylabel="cell fraction %",
                          title="Pearson correlations",
                          appendix="correlation_heatmap")

        tvalue_df = ols_results_df.loc[:, [col for col in ols_results_df.columns if col.endswith(" t-value")]].copy()
        tvalue_df.index = full_index
        tvalue_df.columns = [col.replace(" t-value", "") for col in tvalue_df.columns]
        tvalue_annot_df = tvalue_df.copy()
        tvalue_annot_df = tvalue_annot_df.round(2).astype(str)
        tvalue_df[tvalue_df.abs() < 1.96] = 0
        self.plot_heatmap(df=tvalue_df,
                          annot_df=tvalue_annot_df,
                          xlabel="PICs",
                          ylabel="cell fraction %",
                          title="OLS t-values",
                          appendix="tvalue_heatmap")

    @staticmethod
    def load_file(inpath, header=0, index_col=0, sep="\t", low_memory=True,
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

    def plot_barplot(self, df, x="x", y="y", xlabel="", ylabel="", title="",
                     palette=None, appendix=""):
        sns.set_style("ticks")
        fig, ax = plt.subplots(figsize=(12, 12))

        sns.despine(fig=fig, ax=ax)

        color = None
        if palette is None:
            color = "#808080"

        g = sns.barplot(x=x,
                        y=y,
                        data=df,
                        color=color,
                        palette=palette,
                        dodge=False,
                        ax=ax)

        ax.set_title(title,
                     fontsize=22,
                     fontweight='bold')
        ax.set_ylabel(ylabel,
                      fontsize=14,
                      fontweight='bold')
        ax.set_xlabel(xlabel,
                      fontsize=14,
                      fontweight='bold')

        plt.tight_layout()
        fig.savefig(os.path.join(self.plot_outdir, "{}_{}.png".format(self.out_filename, appendix)))
        plt.close()

    def plot_heatmap(self, df, annot_df, vmin=None, vmax=None, xlabel="",
                     ylabel="", title="", appendix=""):
        sns.set_style("ticks")
        annot_df.fillna("", inplace=True)

        fig, ax = plt.subplots(figsize=(df.shape[1], df.shape[0]))
        sns.set(color_codes=True)

        sns.heatmap(df,
                    vmin=vmin,
                    vmax=vmax,
                    cmap=sns.diverging_palette(246, 24, as_cmap=True),
                    cbar=False,
                    center=0,
                    square=True,
                    annot=annot_df,
                    fmt='',
                    annot_kws={"size": 14, "color": "#000000"},
                    ax=ax)

        plt.setp(ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize=20,
                                    rotation=0))
        plt.setp(ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=20,
                                    rotation=90))

        ax.set_xlabel(xlabel, fontsize=14)
        ax.xaxis.set_label_position('top')

        ax.set_ylabel(ylabel, fontsize=14)
        ax.yaxis.set_label_position('right')

        fig.suptitle(title,
                     fontsize=22,
                     fontweight='bold')

        plt.tight_layout()
        fig.savefig(os.path.join(self.plot_outdir, "{}_{}.png".format(self.out_filename, appendix)))
        plt.close()

    def print_arguments(self):
        print("Arguments:")
        print("  > Cell fraction path: {}".format(self.cf_path))
        print("  > PICs path: {}".format(self.pics_path))
        print("  > Plot output directory {}".format(self.plot_outdir))
        print("  > File output directory {}".format(self.file_outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()