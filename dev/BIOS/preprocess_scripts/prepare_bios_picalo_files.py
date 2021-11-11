#!/usr/bin/env python3

"""
File:         prepare_bios_picalo_files.py
Created:      2021/11/08
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
from pathlib import Path
import argparse
import os

# Third party imports.
import numpy as np
import pandas as pd

# Local application imports.

# Metadata
__program__ = "Prepare BIOS PICALO files"
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
./prepare_bios_picalo_files.py -h

./prepare_bios_picalo_files.py -eq /groups/umcg-bios/tmp01/projects/PICALO/data/2019-12-11-cis-eQTLsFDR0.05-ProbeLevel-CohortInfoRemoved-BonferroniAdded.txt.gz -ge /groups/umcg-bios/tmp01/projects/decon_optimizer/data/datasets_biosdata/brittexport -ex /groups/umcg-bios/tmp01/projects/BIOS_for_eQTLGenII/data/BIOS_EGCUT_for_eQTLGen/BIOS_only/eqtlpipeline_lld_backup150317/1-normalise/normalise/gene_read_counts_BIOS_and_LLD_passQC.tsv.SampleSelection.ProbesWithZeroVarianceRemoved.TMM.CPM.Log2Transformed.ProbesCentered.SamplesZTransformed.txt -pcpc /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/pre_process_bios_expression_matrix/BIOS-cis-noRNAPhenoNA-NoMDSOutlier-20RnaAlignment/data/gene_read_counts_BIOS_and_LLD_passQC.tsv.SampleSelection.ProbesWithZeroVarianceRemoved.TMM.CPM.Log2Transformed.ProbesCentered.SamplesZTransformed.CovariatesRemovedOLS.PCAOverSamplesEigenvectors.txt.gz -tc /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_phenotype_matrix/BIOS_CorrectionIncluded_RNA_AlignmentMetrics_andSex.txt.gz -m /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/preprocess_mds_file/BIOS-allchr-mds-BIOS-GTESubset-noRNAPhenoNA-noOutliers-VariantSubsetFilter.txt.gz -gte /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/filter_gte_file/BIOS_noRNAPhenoNA_NoMDSOutlier/GenotypeToExpression.txt.gz -o BIOS-cis-noRNAPhenoNA-NoMDSOutlier
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.eqtl_path = getattr(arguments, 'eqtl')
        self.geno_folder = getattr(arguments, 'genotype')
        self.expr_path = getattr(arguments, 'expression')
        self.post_corr_pcs_path = getattr(arguments, 'post_corr_pcs')
        self.tcov_path = getattr(arguments, 'technical_covariates')
        self.mds_path = getattr(arguments, 'mds')
        self.gte_path = getattr(arguments, 'genotype_to_expression')
        outdir = getattr(arguments, 'outdir')

        # Set variables.
        self.outdir = os.path.join(str(Path(__file__).parent.parent), 'prepare_bios_picalo_files', outdir)
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
        parser.add_argument("-eq",
                            "--eqtl",
                            type=str,
                            required=True,
                            help="The path to the replication eqtl matrix.")
        parser.add_argument("-ge",
                            "--genotype",
                            type=str,
                            required=True,
                            help="The path to the genotype directory")
        parser.add_argument("-ex",
                            "--expression",
                            type=str,
                            required=True,
                            help="The path to the deconvolution matrix")
        parser.add_argument("-pcpc",
                            "--post_corr_pcs",
                            type=str,
                            required=True,
                            help="The path to the post covariate"
                                 "correction expression PCs matrix")
        parser.add_argument("-tc",
                            "--technical_covariates",
                            type=str,
                            required=True,
                            help="The path to the technical covariates matrix.")
        parser.add_argument("-m",
                            "--mds",
                            type=str,
                            required=True,
                            help="The path to the mds matrix.")
        parser.add_argument("-gte",
                            "--genotype_to_expression",
                            type=str,
                            required=False,
                            default=None,
                            help="The path to the genotype-to-expression"
                                 " link matrix.")
        parser.add_argument("-o",
                            "--outdir",
                            type=str,
                            required=True,
                            help="The name of the output directory.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("Loading sample-to-dataset file")
        gte_df = self.load_file(self.gte_path, header=0, index_col=None)
        genotype_ids = gte_df.iloc[:, 0].tolist()
        rnaseq_ids = gte_df.iloc[:, 1].tolist()

        std_df = gte_df.loc[:, ["rnaseq_id", "dataset"]]
        std_df.columns = ["sample", "dataset"]
        self.save_file(df=std_df, outpath=os.path.join(self.outdir, "SampleToDataset.txt.gz"), index=False)
        del std_df

        print("Loading eQTL file.")
        eqtl_df = self.load_file(self.eqtl_path, header=0, index_col=None)

        print("Preparing genotype and expression file")
        geno_dfs = []
        allele_dfs = []
        for i in range(1, 23):
            geno_df = self.load_file(os.path.join(self.geno_folder, "chr{}".format(i), "GenotypeData.txt.gz"), header=0, index_col=0)
            geno_dfs.append(geno_df.iloc[:, 2:])
            allele_dfs.append(geno_df.loc[:, ["Alleles", "AltAllele"]].copy())
        geno_df = pd.concat(geno_dfs, axis=0)
        allele_df = pd.concat(allele_dfs, axis=0)
        geno_df = geno_df.groupby(geno_df.index).first()
        allele_df = allele_df.groupby(allele_df.index).first()
        print(geno_df)
        print(allele_df)

        unique_n = len(set(eqtl_df["SNPName"]))
        present_snps = set(geno_df.index)
        missing_snps = list(set([snp for snp in eqtl_df["SNPName"] if snp not in present_snps]))
        print("\t{} / {} SNPs found in genotype matrix.".format(unique_n - len(missing_snps), unique_n))
        if len(missing_snps) > 0:
            geno_df = pd.concat([geno_df, pd.DataFrame(np.nan, index=missing_snps, columns=geno_df.columns)], axis=0)
            allele_df = pd.concat([allele_df, pd.DataFrame(np.nan, index=missing_snps, columns=allele_df.columns)], axis=0)
        geno_df = geno_df.loc[eqtl_df["SNPName"], :]
        allele_df = allele_df.loc[eqtl_df["SNPName"], :]
        print(geno_df)
        print(allele_df)

        expr_df = self.load_file(self.expr_path, header=0, index_col=0)
        expr_df = expr_df.groupby(expr_df.index).first()
        print(expr_df)

        unique_n = len(set(eqtl_df["SNPName"]))
        present_genes = set(expr_df.index)
        missing_genes = list(set([gene for gene in eqtl_df["ProbeName"] if gene not in present_genes]))
        print("\t{} / {} genes found in expression matrix.".format(unique_n - len(missing_genes), unique_n))
        if len(missing_genes) > 0:
            expr_df = pd.concat([expr_df, pd.DataFrame(np.nan, index=missing_genes, columns=expr_df.columns)], axis=0)
        expr_df = expr_df.loc[eqtl_df["ProbeName"], :]
        print(expr_df)

        # Filter eQTL file on present data.
        mask = np.zeros(eqtl_df.shape[0], dtype=bool)
        for i in range(eqtl_df.shape[0]):
            if eqtl_df.iloc[i, 1] in present_snps and eqtl_df.iloc[i, 7] in present_genes:
                mask[i] = True
        present_eqtl_df = eqtl_df.loc[mask, :]
        geno_df = geno_df.loc[mask, :].loc[:, genotype_ids]
        geno_df.columns = rnaseq_ids
        allele_df = allele_df.loc[mask, :]
        expr_df = expr_df.loc[mask, :].loc[:, rnaseq_ids]

        self.save_file(df=present_eqtl_df, outpath=os.path.join(self.outdir, "BIOS_eQTLProbesFDR0.05-ProbeLevel-Available.txt.gz"), index=False)
        self.save_file(df=geno_df, outpath=os.path.join(self.outdir, "genotype_table.txt.gz"))
        self.save_file(df=allele_df, outpath=os.path.join(self.outdir, "genotype_alleles_table.txt.gz"))
        self.save_file(df=expr_df, outpath=os.path.join(self.outdir, "expression_table.txt.gz"))
        del eqtl_df, present_eqtl_df, geno_df, expr_df

        print("Preparing PCS after tech. cov. correction")
        pcpc_df = self.load_file(self.post_corr_pcs_path, header=0, index_col=0)
        print(pcpc_df)

        self.save_file(df=pcpc_df.iloc[:25, :].loc[:, rnaseq_ids], outpath=os.path.join(self.outdir, "first25PCComponents.txt.gz"))
        self.save_file(df=pcpc_df.iloc[:10, :].loc[:, rnaseq_ids], outpath=os.path.join(self.outdir, "first10PCComponents.txt.gz"))
        del pcpc_df

        print("Preparing technical covariates")
        tcov_df = self.load_file(self.tcov_path, header=0, index_col=0)
        mds_df = self.load_file(self.mds_path, header=0, index_col=0)
        print(tcov_df)
        print(mds_df)

        correction_df = tcov_df.merge(mds_df, left_index=True, right_index=True)
        self.save_file(df=correction_df.loc[rnaseq_ids, :].T, outpath=os.path.join(self.outdir, "technical_and_mds_covariates_table.txt.gz"))
        del correction_df

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

    def print_arguments(self):
        print("Arguments:")
        print("  > eQTL: {}".format(self.eqtl_path))
        print("  > Genotype: {}".format(self.geno_folder))
        print("  > Expression: {}".format(self.expr_path))
        print("  > Post-correction PCs: {}".format(self.post_corr_pcs_path))
        print("  > Technical covariates: {}".format(self.tcov_path))
        print("  > Genotype MDS components: {}".format(self.mds_path))
        print("  > Genotype-to-expression path: {}".format(self.gte_path))
        print("  > Output directory: {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
