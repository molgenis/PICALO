#!/usr/bin/env python3

"""
File:         pre_process_expression_matrix.py
Created:      2021/05/25
Last Changed: 2021/07/22
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
from functools import reduce
import argparse
import time
import glob
import os

# Third party imports.
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Local application imports.

# Metadata
__program__ = "Pre-process Expression Matrix"
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
./pre_process_expression_matrix.py -h
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.data_path = getattr(arguments, 'data')
        self.tcov_path = getattr(arguments, 'technical_covariates')
        self.solo_correction = getattr(arguments, 'solo_correction')
        self.gte_path = getattr(arguments, 'gene_to_expression')
        self.gte_prefix = getattr(arguments, 'gte_prefix')
        self.exclude = getattr(arguments, 'exclude')
        outdir = getattr(arguments, 'outdir')
        outfolder = getattr(arguments, 'outfolder')

        # Set variables.
        if outdir is None:
            outdir = str(Path(__file__).parent.parent)
        self.plot_outdir = os.path.join(outdir, 'pre_process_expression_matrix', outfolder, 'plot')
        self.file_outdir = os.path.join(outdir, 'pre_process_expression_matrix', outfolder, 'data')
        for outdir in [self.plot_outdir, self.file_outdir]:
            if not os.path.exists(outdir):
                os.makedirs(outdir)

        self.file_cohort_dict = {
            "AMPAD-MAYO-V2": "MAYO",
            "CMC_HBCC_set2": "CMC HBCC",
            "GTEx": "GTEx",
            "AMPAD-ROSMAP-V2": "ROSMAP",
            "BrainGVEX-V2": "Brain GVEx",
            "TargetALS": "Target ALS",
            "AMPAD-MSBB-V2": "MSBB",
            "NABEC-H610": "NABEC",
            "LIBD_1M": "LIBD",
            "ENA": "ENA",
            "LIBD_h650": "LIBD",
            "GVEX": "GVEX",
            "NABEC-H550": "NABEC",
            "CMC_HBCC_set3": "CMC HBCC",
            "UCLA_ASD": "UCLA ASD",
            "CMC": "CMC",
            "CMC_HBCC_set1": "CMC HBCC"
        }

        self.palette = {
            "MAYO": "#9c9fa0",
            "CMC HBCC": "#0877b4",
            "GTEx": "#0fa67d",
            "ROSMAP": "#6950a1",
            "Brain GVEx": "#48b2e5",
            "Target ALS": "#d5c77a",
            "MSBB": "#5cc5bf",
            "NABEC": "#6d743a",
            "LIBD": "#e49d26",
            "ENA": "#d46727",
            "GVEX": "#000000",
            "UCLA ASD": "#f36d2a",
            "CMC": "#eae453",
            "NA": "#808080"
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
        parser.add_argument("-d",
                            "--data",
                            type=str,
                            required=True,
                            help="The path to the data matrix.")
        parser.add_argument("-t",
                            "--technical_covariates",
                            type=str,
                            required=True,
                            help="The path to the technical covariates matrix.")
        parser.add_argument("-gte",
                            "--gene_to_expression",
                            type=str,
                            required=True,
                            help="The path to the gene-expression link files.")
        parser.add_argument("-p",
                            "--gte_prefix",
                            type=str,
                            required=True,
                            help="The gene-expression link file prefix.")
        parser.add_argument("-e",
                            "--exclude",
                            nargs="*",
                            type=str,
                            default=[],
                            help="The gene-expression link files to exclude.")
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
        parser.add_argument("-sc",
                            "--solo_correction",
                            nargs="*",
                            type=str,
                            required=False,
                            default=None,
                            choices=["TechnicalCovariates", "MDS", "Datasets"],
                            help="Include a covariate correction with a"
                                 "subset of the correction matrix. Default:"
                                 " None.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        # Construct the output filename.
        filename = os.path.basename(self.data_path).replace(".gz", "").replace(".txt", "")

        # Loading samples.
        print("Loading samples.")
        gte_combined_df = None
        dataset_to_sample_dict = {}
        for infile in glob.glob(os.path.join(self.gte_path, "{}*.txt".format(self.gte_prefix))):
            file = os.path.basename(infile).replace(".txt", "").replace(self.gte_prefix, "")
            if file in self.exclude:
                continue
            gte_df = self.load_file(infile, header=None, index_col=None)
            gte_df["file"] = file
            if gte_combined_df is None:
                gte_combined_df = gte_df
            else:
                gte_combined_df = pd.concat([gte_combined_df, gte_df], axis=0, ignore_index=True)

            dataset_to_sample_dict[file] = gte_df.iloc[:, 1]
        gte_combined_df["cohort"] = gte_combined_df.iloc[:, 2].map(self.file_cohort_dict)
        sample_to_cohort = dict(zip(gte_combined_df.iloc[:, 1], gte_combined_df.iloc[:, 3]))
        samples = gte_combined_df.iloc[:, 1].values.tolist()
        print("\tN samples: {}".format(len(samples)))

        # Safe sample cohort data frame.
        sample_cohort_df = gte_combined_df.iloc[:, [1, 3]]
        sample_cohort_df.columns = ["sample", "cohort"]
        self.save_file(sample_cohort_df, outpath=os.path.join(self.file_outdir, "SampleToCohorts.txt.gz"), index=False)
        sample_dataset_df = gte_combined_df.iloc[:, [1, 2]]
        sample_dataset_df.columns = ["sample", "dataset"]
        self.save_file(sample_dataset_df, outpath=os.path.join(self.file_outdir, "SampleToDataset.txt.gz"), index=False)

        # Create cohort matrix.
        dataset_sample_counts = list(zip(*np.unique(gte_combined_df["file"], return_counts=True)))
        dataset_sample_counts.sort(key=lambda x: -x[1])
        datasets = [csc[0] for csc in dataset_sample_counts]
        print("\tDatasets: {} [N = {}]".format(", ".join(datasets), len(datasets)))

        dataset_df = pd.DataFrame(0, index=samples, columns=datasets)
        for dataset in datasets:
            dataset_df.loc[dataset_to_sample_dict[dataset], dataset] = 1
        dataset_df.index.name = "-"

        # Load data.
        print("Loading data.")
        df = self.load_file(self.data_path, header=0, index_col=0)
        print(df)

        print("Step 1: sample selection.")
        print("\tUsing {}/{} samples.".format(len(samples), df.shape[1]))
        df = df.loc[:, samples]

        print("Step 2: remove probes with zero variance.")
        mask = df.std(axis=1) != 0
        print("\tUsing {}/{} probes.".format(np.sum(mask), np.size(mask)))
        df = df.loc[mask, :]

        print("Step 3: log2 transform.")
        min_value = df.min(axis=1).min()
        if min_value <= 0:
            df = np.log2(df - min_value + 1)
        else:
            df = np.log2(df + 1)

        print("\tSaving file.")
        self.save_file(df=df, outpath=os.path.join(self.file_outdir, "{}.SampleSelection.ProbesWithZeroVarianceRemoved.Log2Transformed.txt.gz".format(filename)))

        print("Step 4: center probes.")
        df = df.subtract(df.mean(axis=1), axis=0)

        print("Step 5: sample z-transform.")
        df = (df - df.mean(axis=0)) / df.std(axis=0)

        print("\tSaving file.")
        self.save_file(df=df, outpath=os.path.join(self.file_outdir, "{}.SampleSelection.ProbesWithZeroVarianceRemoved.Log2Transformed.ProbesCentered.SamplesZTransformed.txt.gz".format(filename)))

        print("Step 6: PCA analysis.")
        self.pca(df=df, filename=filename, sample_to_cohort=sample_to_cohort,
                 file_appendix="SampleSelection.ProbesWithZeroVarianceRemoved.Log2Transformed.ProbesCentered.SamplesZTransformed",
                 plot_appendix="_1")

        print("Step 7: Construct technical covariate matrix.")
        tcov_df = self.load_file(self.tcov_path, header=0, index_col=0)
        correction_df, correction_segments = self.prepare_correction_matrix(tcov_df=tcov_df.loc[samples, :], dataset_df=dataset_df)

        print("\tSaving file.")
        self.save_file(df=correction_df, outpath=os.path.join(self.file_outdir, "correction_matrix.txt.gz"))

        print("Step 8: remove technical covariates OLS.")
        corrected_df = self.calculate_residuals(df=df, correction_df=correction_df)

        print("\tSaving file.")
        self.save_file(df=corrected_df, outpath=os.path.join(self.file_outdir, "{}.SampleSelection.ProbesWithZeroVarianceRemoved.Log2Transformed.ProbesCentered.SamplesZTransformed.CovariatesRemovedOLS.txt.gz".format(filename)))

        print("Step 9: PCA analysis.")
        self.pca(df=corrected_df, filename=filename,
                 sample_to_cohort=sample_to_cohort,
                 file_appendix="SampleSelection.ProbesWithZeroVarianceRemoved.Log2Transformed.ProbesCentered.SamplesZTransformed.CovariatesRemovedOLS",
                 plot_appendix="_2_CovariatesRemovedOLS")
        del correction_df, corrected_df

        step = 10
        for name in ["TechnicalCovariates", "MDS", "Datasets"]:
            if name in self.solo_correction:
                print("Step {}: remove {} components OLS.".format(step, name))
                correction_df = correction_segments[name]
                self.save_file(df=correction_df, outpath=os.path.join(self.file_outdir, "{}_matrix.txt.gz".format(name)))
                corrected_df = self.calculate_residuals(df=df, correction_df=correction_df)
                step += 1

                print("\tSaving file.")
                self.save_file(df=corrected_df,
                               outpath=os.path.join(self.file_outdir,
                                                    "{}.SampleSelection.ProbesWithZeroVarianceRemoved.Log2Transformed.ProbesCentered.SamplesZTransformed.{}RemovedOLS.txt.gz".format(filename, name)))

                print("Step {}: PCA analysis.".format(step))
                self.pca(df=corrected_df, filename=filename,
                         sample_to_cohort=sample_to_cohort,
                         file_appendix="SampleSelection.ProbesWithZeroVarianceRemoved.Log2Transformed.ProbesCentered.SamplesZTransformed.{}RemovedOLS".format(name),
                         plot_appendix="_2_{}RemovedOLS".format(name))
                step += 1

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

    @staticmethod
    def reverse_dict(dict):
        out_dict = {}
        seen_keys = set()
        for key, value in dict.items():
            if key in seen_keys:
                print("Key {} has muiltiple values.".format(key))
            seen_keys.add(key)

            if value in out_dict.keys():
                keys = out_dict[value]
                keys.append(key)
                out_dict[value] = keys
            else:
                out_dict[value] = [key]

        return out_dict

    def prepare_correction_matrix(self, tcov_df, dataset_df):
        df = tcov_df.copy()

        # Remove columns without variance.
        df = df.loc[:, df.std(axis=0) != 0]

        # Split the technical covariates matrix into technical covariates,
        # MDS components and cohort dummy variables.
        tech_cov_mask = np.zeros(df.shape[1], dtype=bool)
        mds_mask = np.zeros(df.shape[1], dtype=bool)
        cohort_mask = np.zeros(df.shape[1], dtype=bool)
        for i, col in enumerate(df.columns):
            if "MDS" in col:
                mds_mask[i] = True
            elif set(df[col].unique()) == {0, 1}:
                cohort_mask[i] = True
            else:
                tech_cov_mask[i] = True
        print("\tColumn in input file:")
        print("\t  > N-technical covariates: {}".format(np.sum(tech_cov_mask)))
        print("\t  > N-MDS components: {}".format(np.sum(mds_mask)))
        print("\t  > N-cohort dummy variables: {}".format(np.sum(cohort_mask)))

        # Create an intercept data frame.
        intecept_df = pd.DataFrame(1, index=df.index, columns=["INTERCEPT"])

        # filter the technical covariates on VIF.
        tech_cov_df = self.remove_multicollinearity(df.loc[:, tech_cov_mask].copy())

        # split the MDS components per cohort.
        mds_columns = df.columns[mds_mask]
        mds_df = pd.DataFrame(0, index=df.index, columns=["{}_{}".format(a, b) for a in cohort_df.columns for b in mds_columns])
        for cohort in dataset_df.columns:
            mask = dataset_df.loc[:, cohort] == 1
            for mds_col in mds_columns:
                col = "{}_{}".format(cohort, mds_col)
                mds_df.loc[mask, col] = df.loc[mask, mds_col]

        # replace cohort variables with the complete set defined by the GTE
        # file. Don't include the cohort with the most samples.
        tmp_dataset_df = dataset_df.iloc[:, 1:]

        # construct the complete technical covariates matrix. Start with an
        # intercept. Don't include the cohort with the most samples.
        correction_df = reduce(lambda left, right: pd.merge(left,
                                                            right,
                                                            left_index=True,
                                                            right_index=True),
                               [intecept_df, tech_cov_df, mds_df, tmp_dataset_df])
        correction_df.index.name = "-"

        print("\tColumn in technical covariates matrix:")
        print("\t  > N-intercept: {}".format(intecept_df.shape[1]))
        print("\t  > N-technical covariates: {}".format(tech_cov_df.shape[1]))
        print("\t  > N-MDS components: {}".format(mds_df.shape[1]))
        print("\t  > N-dataset dummy variables: {}".format(tmp_dataset_df.shape[1]))

        return correction_df, {"TechnicalCovariates": tech_cov_df, "MDS": mds_df, "Datasets": tmp_dataset_df}

    def remove_multicollinearity(self, df, threshold=0.9999):
        indices = np.arange(df.shape[1])
        max_vif = np.inf
        while len(indices) > 1 and max_vif > threshold:
            vif = np.array([self.calc_ols_rsquared(df.iloc[:, indices], ix) for ix in range(len(indices))])
            max_vif = max(vif)

            if max_vif > threshold:
                max_index = np.where(vif == max_vif)[0][0]
                indices = np.delete(indices, max_index)

        return df.iloc[:, indices]

    @staticmethod
    def calc_ols_rsquared(df, idx):
        return OLS(df.iloc[:, idx], df.loc[:, np.arange(df.shape[1]) != idx]).fit().rsquared

    @staticmethod
    def calculate_residuals(df, correction_df):
        corrected_m = np.empty_like(df, dtype=np.float64)
        last_print_time = None
        n_tests = df.shape[0]
        for i in range(n_tests):
            now_time = int(time.time())
            if last_print_time is None or (now_time - last_print_time) >= 10 or (i + 1) == n_tests:
                last_print_time = now_time
                print("\t{}/{} ieQTLs analysed [{:.2f}%]".format((i + 1), n_tests, (100 / n_tests) * (i + 1)))

            ols = OLS(df.iloc[i, :], correction_df)
            results = ols.fit()
            # print(results.summary())
            corrected_m[i, :] = results.resid

        return pd.DataFrame(corrected_m, index=df.index, columns=df.columns)

    def pca(self, df, filename, sample_to_cohort, file_appendix="", plot_appendix=""):
        # samples should be on the columns and genes on the rows.
        zscores = (df - df.mean(axis=0)) / df.std(axis=0)
        pca = PCA(n_components=25)
        pca.fit(zscores)
        expl_variance = {"PC{}".format(i+1): pca.explained_variance_ratio_[i] * 100 for i in range(25)}
        components_df = pd.DataFrame(pca.components_)
        components_df.index = ["Comp{}".format(i + 1) for i, _ in enumerate(components_df.index)]
        components_df.columns = df.columns

        print("\tSaving file.")
        self.save_file(df=components_df, outpath=os.path.join(self.file_outdir, "{}.{}.PCAOverSamplesEigenvectors.txt.gz".format(filename, file_appendix)))

        print("Plotting PCA")
        plot_df = components_df.T
        plot_df["cohort"] = plot_df.index.map(sample_to_cohort)
        plot_df["cohort"] = plot_df["cohort"].fillna('NA')
        self.plot(df=plot_df, x="Comp1", y="Comp2", hue="cohort", palette=self.palette,
                  xlabel="PC1 [{:.2f}%]".format(expl_variance["PC1"]),
                  ylabel="PC2 [{:.2f}%]".format(expl_variance["PC2"]),
                  title="PCA - eigenvectors",
                  filename="eigenvectors_plot{}".format(plot_appendix))

    def plot(self, df, x="x", y="y", hue=None, palette=None, xlabel=None,
             ylabel=None, title="", filename="PCA_plot"):
        if xlabel is None:
            xlabel = x
        if ylabel is None:
            ylabel = y

        sns.set(rc={'figure.figsize': (12, 9)})
        sns.set_style("ticks")
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, gridspec_kw={"width_ratios": [0.9, 0.1]})
        sns.despine(fig=fig, ax=ax1)
        ax2.axis('off')

        sns.scatterplot(x=x,
                        y=y,
                        hue=hue,
                        data=df,
                        s=100,
                        linewidth=0,
                        legend=None,
                        palette=palette,
                        ax=ax1)

        ax1.set_title(title,
                      fontsize=20,
                      fontweight='bold')
        ax1.set_ylabel(ylabel,
                       fontsize=14,
                       fontweight='bold')
        ax1.set_xlabel(xlabel,
                       fontsize=14,
                       fontweight='bold')

        if palette is not None:
            handles = []
            for label, color in palette.items():
                if label in df[hue].values.tolist():
                    handles.append(mpatches.Patch(color=color, label=label))
            ax2.legend(handles=handles, loc="center")

        #fig.savefig(os.path.join(self.plot_outdir, "{}.pdf".format(filename)))
        fig.savefig(os.path.join(self.plot_outdir, "{}.png".format(filename)))
        plt.close()

    def print_arguments(self):
        print("Arguments:")
        print("  > Data: {}".format(self.data_path))
        print("  > Technical covariates: {}".format(self.tcov_path))
        print("  > GtE path: {}".format(self.gte_path))
        print("  >   GtE prefix: {}".format(self.gte_prefix))
        print("  >   Exclude: {}".format(self.exclude))
        print("  > Solo correction: {}".format(self.solo_correction))
        print("  > Plot output directory: {}".format(self.plot_outdir))
        print("  > File output directory: {}".format(self.file_outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
