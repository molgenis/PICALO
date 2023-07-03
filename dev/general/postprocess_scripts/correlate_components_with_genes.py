#!/usr/bin/env python3

"""
File:         correlate_components_with_genes.py
Created:      2021/05/25
Last Changed: 2022/05/17
Author:       M.Vochteloo

Copyright (C) 2020 University Medical Center Groningen.

A copy of the BSD 3-Clause "New" or "Revised" License can be found in the
LICENSE file in the root directory of this source tree.
"""

# Standard imports.
from __future__ import print_function
import argparse
import os

# Third party imports.
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import betainc
from statsmodels.stats import multitest

# Local application imports.

# Metadata
__program__ = "Correlate Components with Genes"
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
./correlate_components_with_genes.py -h

### BIOS ###

./correlate_components_with_genes.py \
    -c /groups/umcg-bios/tmp01/projects/PICALO/output/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/PICs.txt.gz \
    -g /groups/umcg-bios/tmp01/projects/PICALO/postprocess_scripts/force_normalise_matrix/2021-12-10-gene_read_counts_BIOS_and_LLD_passQC.tsv.SampleSelection.ProbesWithZeroVarianceRemoved.TMM.SampleSelection.ProbesWithZeroVarianceRemoved.Log2Transformed.ProbesCentered.SamplesZTransformed.CovariatesRemovedOLS_ForceNormalised.txt.gz \
    -gi /groups/umcg-bios/tmp01/projects/PICALO/data/ArrayAddressToSymbol.txt.gz \
    -avge /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/calc_avg_gene_expression/gene_read_counts_BIOS_and_LLD_passQC.tsv.SampleSelection.ProbesWithZeroVarianceRemoved.TMM.Log2Transformed.AverageExpression.txt.gz \
    -od 2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA \
    -of 2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA
    
### MetaBrain ###

./correlate_components_with_genes.py \
    -c /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/PICs.txt.gz \
    -g /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/postprocess_scripts/force_normalise_matrix/2022-04-13-MetaBrain.allCohorts.2020-02-16.TMM.freeze2dot1.SampleSelection.SampleSelection.ProbesWithZeroVarianceRemoved.Log2Transformed.ProbesCentered.SamplesZTransformed.CovariatesRemovedOLS_ForceNormalised.txt.gz \
    -gi /groups/umcg-biogen/tmp01/annotation/gencode.v32.primary_assembly.annotation.collapsedGenes.ProbeAnnotation.TSS.txt.gz \
    -avge /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/calc_avg_gene_expression/MetaBrain.allCohorts.2020-02-16.TMM.freeze2dot1.SampleSelection.ProbesWithZeroVarianceRemoved.Log2Transformed.AverageExpression.txt.gz \
    -od 2022-04-13-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA \
    -of 2022-04-13-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA

"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.components_path = getattr(arguments, 'components')
        self.genes_path = getattr(arguments, 'genes')
        self.gene_info_path = getattr(arguments, 'gene_info')
        self.avg_ge_path = getattr(arguments, 'average_gene_expression')
        outdir = getattr(arguments, 'out_directory')
        self.out_filename = getattr(arguments, 'out_filename')

        # Set variables.
        self.outdir = os.path.join(str(os.path.dirname(os.path.abspath(__file__))), 'correlate_components_with_genes', outdir)
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

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
        parser.add_argument("-c",
                            "--components",
                            type=str,
                            required=True,
                            help="The path to the components matrix.")
        parser.add_argument("-g",
                            "--genes",
                            type=str,
                            required=True,
                            help="The path to the gene expression matrix.")
        parser.add_argument("-gi",
                            "--gene_info",
                            type=str,
                            required=True,
                            help="The path to the gene info matrix.")
        parser.add_argument("-avge",
                            "--average_gene_expression",
                            type=str,
                            required=False,
                            default=None,
                            help="The path to the average gene expression "
                                 "matrix.")
        parser.add_argument("-od",
                            "--out_directory",
                            type=str,
                            required=False,
                            default=None,
                            help="The name of the output directory.")
        parser.add_argument("-of",
                            "--out_filename",
                            type=str,
                            required=False,
                            default="output",
                            help="The name of the output files.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        # Load data.
        print("Loading data.")
        comp_df = self.load_file(self.components_path, header=0, index_col=0)
        genes_df = self.load_file(self.genes_path, header=0, index_col=0, nrows=None)
        gene_info_df = self.load_file(self.gene_info_path, header=0, index_col=None)
        gene_dict = dict(zip(gene_info_df["ArrayAddress"], gene_info_df["Symbol"]))
        del gene_info_df

        print("Pre-processing data.")
        # Make sure order is the same.
        samples = set(comp_df.columns.tolist()).intersection(set(genes_df.columns.tolist()))
        comp_df = comp_df.loc[:, samples]
        genes_df = genes_df.loc[:, samples]

        # Safe the indices.
        components = comp_df.index.tolist()
        genes = genes_df.index.tolist()

        # Convert to numpy.
        comp_m = comp_df.to_numpy()
        genes_m = genes_df.to_numpy()
        del comp_df, genes_df

        print("Correlating.")
        corr_m, pvalue_m = self.corrcoef(genes_m.T, comp_m.T)

        print("Calculating FDR values.")
        fdr_m = np.empty_like(pvalue_m)
        for j in range(pvalue_m.shape[1]):
            fdr_m[:, j] = multitest.multipletests(pvalue_m[:, j], method='fdr_bh')[1]

        print("Calculating z-scores.")
        flip_m = np.ones_like(corr_m)
        flip_m[corr_m > 0] = -1
        tmp_pvalue_m = np.copy(pvalue_m)
        tmp_pvalue_m = tmp_pvalue_m / 2  # two sided test
        tmp_pvalue_m[tmp_pvalue_m > (1-1e-16)] = (1-1e-16)  # precision upper limit
        tmp_pvalue_m[tmp_pvalue_m < 2.4703282292062328e-324] = 2.4703282292062328e-324  # precision lower limit
        zscore_m = stats.norm.ppf(tmp_pvalue_m) * flip_m
        del tmp_pvalue_m

        print("Saving output.")
        for m, suffix in ([corr_m, "correlation_coefficient"],
                          [pvalue_m, "correlation_pvalue"],
                          [fdr_m, "correlation_FDR"],
                          [zscore_m, "correlation_zscore"]):
            df = pd.DataFrame(m, index=genes, columns=components)
            self.save_file(df=df,
                           outpath=os.path.join(self.outdir,
                                                "{}_{}.txt.gz".format(self.out_filename, suffix)),
                           index=True)
            del df

        print("Post-processing data.")
        corr_df = pd.DataFrame(np.hstack((corr_m, pvalue_m, fdr_m, zscore_m)),
                               columns=["{} r".format(comp) for comp in components] +
                                       ["{} pvalue".format(comp) for comp in components] +
                                       ["{} FDR".format(comp) for comp in components] +
                                       ["{} zscore".format(comp) for comp in components])
        corr_df.insert(0, "ProbeName", genes)
        corr_df.insert(1, 'HGNCName', corr_df["ProbeName"].map(gene_dict))
        file_appendix = ""
        if self.avg_ge_path is not None:
            avg_ge_df = self.load_file(self.avg_ge_path, header=0, index_col=0)
            avg_ge_dict = dict(zip(avg_ge_df.index, avg_ge_df["average"]))
            corr_df.insert(2, 'avgExpression', corr_df["ProbeName"].map(avg_ge_dict))
            file_appendix += "-avgExpressionAdded"
            del avg_ge_df

        print("Saving file.")
        print(corr_df)
        self.save_file(df=corr_df,
                       outpath=os.path.join(self.outdir, "{}_results{}.txt.gz".format(self.out_filename, file_appendix)),
                       index=False)
        corr_df.to_excel(os.path.join(self.outdir, "{}_results{}.xlsx".format(self.out_filename, file_appendix)))

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
    def corrcoef(m1, m2):
        """
        Pearson correlation over the columns.

        https://stackoverflow.com/questions/24432101/correlation-coefficients-and-p-values-for-all-pairs-of-rows-of-a-matrix
        """
        m1_dev = m1 - np.mean(m1, axis=0)
        m2_dev = m2 - np.mean(m2, axis=0)

        m1_rss = np.sum(m1_dev * m1_dev, axis=0)
        m2_rss = np.sum(m2_dev * m2_dev, axis=0)

        r = np.empty((m1_dev.shape[1], m2_dev.shape[1]), dtype=np.float64)
        for i in range(m1_dev.shape[1]):
            for j in range(m2_dev.shape[1]):
                r[i, j] = np.sum(m1_dev[:, i] * m2_dev[:, j]) / np.sqrt(m1_rss[i] * m2_rss[j])

        rf = r.flatten()
        df = m1.shape[0] - 2
        ts = rf * rf * (df / (1 - rf * rf))
        pf = betainc(0.5 * df, 0.5, df / (df + ts))
        p = pf.reshape(m1.shape[1], m2.shape[1])
        return r, p

    @staticmethod
    def save_file(df, outpath, header=True, index=False, sep="\t"):
        compression = 'infer'
        if outpath.endswith('.gz'):
            compression = 'gzip'

        df.to_csv(outpath, sep=sep, index=index, header=header,
                  compression=compression)
        print("\tSaved dataframe: {} "
              "with shape: {}".format(os.path.basename(outpath),
                                      df.shape))

    def print_arguments(self):
        print("Arguments:")
        print("  > Components: {}".format(self.components_path))
        print("  > Gene expression: {}".format(self.genes_path))
        print("  > Gene info: {}".format(self.gene_info_path))
        print("  > Average gene epxression path: {}".format(self.avg_ge_path))
        print("  > Output directory: {}".format(self.outdir))
        print("  > output filename: {}".format(self.out_filename))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
