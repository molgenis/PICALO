#!/usr/bin/env python3

"""
File:         prepare_bios_picalo_files.py
Created:      2021/11/08
Last Changed: 2021/11/17
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

./prepare_bios_picalo_files.py -eq /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_eqtl_file/EQTLGen_eQTLProbesFDR0.05-ProbeLevel.txt.gz -ge /groups/umcg-bios/tmp01/projects/decon_optimizer/data/datasets_biosdata/brittexport -ex /groups/umcg-bios/tmp01/projects/BIOS_for_eQTLGenII/data/BIOS_EGCUT_for_eQTLGen/BIOS_only/eqtlpipeline_lld_backup150317/1-normalise/normalise/gene_read_counts_BIOS_and_LLD_passQC.tsv.SampleSelection.ProbesWithZeroVarianceRemoved.TMM.CPM.Log2Transformed.ProbesCentered.SamplesZTransformed.txt -pcpc /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/pre_process_bios_expression_matrix/BIOS-cis-noRNAPhenoNA-NoMDSOutlier-20RnaAlignment/data/gene_read_counts_BIOS_and_LLD_passQC.tsv.SampleSelection.ProbesWithZeroVarianceRemoved.TMM.CPM.Log2Transformed.ProbesCentered.SamplesZTransformed.CovariatesRemovedOLS.PCAOverSamplesEigenvectors.txt.gz -tc /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_phenotype_matrix/BIOS_CorrectionIncluded_RNA_AlignmentMetrics_andSex.txt.gz -m /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/preprocess_mds_file/BIOS-allchr-mds-BIOS-GTESubset-noRNAPhenoNA-noOutliers-VariantSubsetFilter.txt.gz -gte /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/filter_gte_file/BIOS_noRNAPhenoNA_NoMDSOutlier/GenotypeToExpression.txt.gz -o BIOS-eQTLGen-cis-noRNAPhenoNA-NoMDSOutlier

./prepare_bios_picalo_files.py -eq /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_eqtl_file/BIOS_eQTLProbesFDR0.05-ProbeLevel.txt.gz -ge /groups/umcg-bios/tmp01/projects/decon_optimizer/data/datasets_biosdata/brittexport -ex /groups/umcg-bios/tmp01/projects/BIOS_for_eQTLGenII/data/BIOS_EGCUT_for_eQTLGen/BIOS_only/eqtlpipeline_lld_backup150317/1-normalise/normalise/gene_read_counts_BIOS_and_LLD_passQC.tsv.SampleSelection.ProbesWithZeroVarianceRemoved.TMM.CPM.Log2Transformed.ProbesCentered.SamplesZTransformed.txt -r /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_phenotype_matrix/BIOS_CorrectionIncluded_RNA_AlignmentMetrics.txt.gz -s /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_phenotype_matrix/BIOS_sex.txt.gz -m /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/preprocess_mds_file/BIOS-allchr-mds-BIOS-NoRNAPhenoNA-NoSexNA-NoMDSOutlier-VariantSubsetFilter.txt.gz -pcpc /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/pre_process_bios_expression_matrix/BIOS_NoRNAPhenoNA_NoSexNA_NoMDSOutlier_20RNAseqAlignemntMetrics/data/gene_read_counts_BIOS_and_LLD_passQC.tsv.SampleSelection.ProbesWithZeroVarianceRemoved.TMM.CPM.Log2Transformed.ProbesCentered.SamplesZTransformed.CovariatesRemovedOLS.PCAOverSamplesEigenvectors.txt.gz -gte /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/filter_gte_file/BIOS_NoRNAPhenoNA_NoSexNA_NoMDSOutlier/GenotypeToExpression.txt.gz -o BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMDSOutlier-20RNAseqAlignemntMetrics

./prepare_bios_picalo_files.py -eq /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_eqtl_file/BIOS_eQTLProbesFDR0.05-ProbeLevel.txt.gz -ge /groups/umcg-bios/tmp01/projects/decon_optimizer/data/datasets_biosdata/brittexport -ex /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/pre_process_bios_expression_matrix/BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_20RNAseqAlignemntMetrics/data/gene_read_counts_BIOS_and_LLD_passQC.tsv.SampleSelection.ProbesWithZeroVarianceRemoved.TMM.SampleSelection.ProbesWithZeroVarianceRemoved.Log2Transformed.ProbesCentered.SamplesZTransformed.txt.gz -r /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_phenotype_matrix/BIOS_CorrectionIncluded_RNA_AlignmentMetrics.txt.gz -s /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_phenotype_matrix/BIOS_sex.txt.gz -m /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/preprocess_mds_file/BIOS-allchr-mds-BIOS-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-VariantSubsetFilter.txt.gz -pcpc /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/pre_process_bios_expression_matrix/BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_20RNAseqAlignemntMetrics/data/gene_read_counts_BIOS_and_LLD_passQC.tsv.SampleSelection.ProbesWithZeroVarianceRemoved.TMM.SampleSelection.ProbesWithZeroVarianceRemoved.Log2Transformed.ProbesCentered.SamplesZTransformed.CovariatesRemovedOLS.PCAOverSamplesEigenvectors.txt.gz -gte /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/filter_gte_file/BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier/GenotypeToExpression.txt.gz -o BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-20RNAseqAlignemntMetrics

./prepare_bios_picalo_files.py -eq /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_eqtl_file/BIOS_eQTLProbesFDR0.05-ProbeLevel.txt.gz -ge /groups/umcg-bios/tmp01/projects/decon_optimizer/data/datasets_biosdata/brittexport -ex /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/pre_process_bios_expression_matrix/BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_AllRNAseqAlignemntMetrics/data/gene_read_counts_BIOS_and_LLD_passQC.tsv.SampleSelection.ProbesWithZeroVarianceRemoved.TMM.SampleSelection.ProbesWithZeroVarianceRemoved.Log2Transformed.ProbesCentered.SamplesZTransformed.txt.gz -r /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_phenotype_matrix/BIOS_RNA_AlignmentMetrics.txt.gz -s /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_phenotype_matrix/BIOS_sex.txt.gz -m /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/preprocess_mds_file/BIOS-allchr-mds-BIOS-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-VariantSubsetFilter.txt.gz -pcpc /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/pre_process_bios_expression_matrix/BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_AllRNAseqAlignemntMetrics/data/gene_read_counts_BIOS_and_LLD_passQC.tsv.SampleSelection.ProbesWithZeroVarianceRemoved.TMM.SampleSelection.ProbesWithZeroVarianceRemoved.Log2Transformed.ProbesCentered.SamplesZTransformed.CovariatesRemovedOLS.PCAOverSamplesEigenvectors.txt.gz -gte /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/filter_gte_file/BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier/GenotypeToExpression.txt.gz -o BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-AllRNAseqAlignemntMetrics

./prepare_bios_picalo_files.py -eq /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_eqtl_file/BIOS_eQTLProbesFDR0.05-ProbeLevel.txt.gz -ge /groups/umcg-bios/tmp01/projects/decon_optimizer/data/datasets_biosdata/brittexport -ex /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/pre_process_bios_expression_matrix/BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_AllRNAseqAlignemntMetricsNoFiltering/data/gene_read_counts_BIOS_and_LLD_passQC.tsv.SampleSelection.ProbesWithZeroVarianceRemoved.TMM.SampleSelection.ProbesWithZeroVarianceRemoved.Log2Transformed.ProbesCentered.SamplesZTransformed.txt.gz -r /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_phenotype_matrix/BIOS_RNA_AlignmentMetrics.txt.gz -s /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_phenotype_matrix/BIOS_sex.txt.gz -m /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/preprocess_mds_file/BIOS-allchr-mds-BIOS-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-VariantSubsetFilter.txt.gz -pcpc /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/pre_process_bios_expression_matrix/BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_AllRNAseqAlignemntMetricsNoFiltering/data/gene_read_counts_BIOS_and_LLD_passQC.tsv.SampleSelection.ProbesWithZeroVarianceRemoved.TMM.SampleSelection.ProbesWithZeroVarianceRemoved.Log2Transformed.ProbesCentered.SamplesZTransformed.CovariatesRemovedOLS.PCAOverSamplesEigenvectors.txt.gz -gte /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/filter_gte_file/BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier/GenotypeToExpression.txt.gz -o BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-AllRNAseqAlignemntMetricsNoFiltering

./prepare_bios_picalo_files.py -eq /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_eqtl_file/BIOS_eQTLProbesFDR0.05-ProbeLevel.txt.gz -ge /groups/umcg-bios/tmp01/projects/decon_optimizer/data/datasets_biosdata/brittexport -ex /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/pre_process_bios_expression_matrix/BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics/data/gene_read_counts_BIOS_and_LLD_passQC.tsv.SampleSelection.ProbesWithZeroVarianceRemoved.TMM.SampleSelection.ProbesWithZeroVarianceRemoved.Log2Transformed.ProbesCentered.SamplesZTransformed.txt.gz -s /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_phenotype_matrix/BIOS_sex.txt.gz -m /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/preprocess_mds_file/BIOS-allchr-mds-BIOS-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-VariantSubsetFilter.txt.gz -pcpc /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/pre_process_bios_expression_matrix/BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics/data/gene_read_counts_BIOS_and_LLD_passQC.tsv.SampleSelection.ProbesWithZeroVarianceRemoved.TMM.SampleSelection.ProbesWithZeroVarianceRemoved.Log2Transformed.ProbesCentered.SamplesZTransformed.CovariatesRemovedOLS.PCAOverSamplesEigenvectors.txt.gz -gte /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/filter_gte_file/BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier/GenotypeToExpression.txt.gz -o BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.eqtl_path = getattr(arguments, 'eqtl')
        self.geno_folder = getattr(arguments, 'genotype')
        self.expr_path = getattr(arguments, 'expression')
        self.rna_alignment_path = getattr(arguments, 'rna_alignment')
        self.sex_path = getattr(arguments, 'sex')
        self.mds_path = getattr(arguments, 'mds')
        self.post_corr_pcs_path = getattr(arguments, 'post_corr_pcs')
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
        parser.add_argument("-ra",
                            "--rna_alignment",
                            type=str,
                            required=False,
                            default=None,
                            help="The path to the RNAseq alignment metrics"
                                 " matrix.")
        parser.add_argument("-s",
                            "--sex",
                            type=str,
                            required=False,
                            default=None,
                            help="The path to the sex matrix.")
        parser.add_argument("-m",
                            "--mds",
                            type=str,
                            required=False,
                            default=None,
                            help="The path to the mds matrix.")
        parser.add_argument("-pcpc",
                            "--post_corr_pcs",
                            type=str,
                            required=True,
                            help="The path to the post covariate"
                                 "correction expression PCs matrix")
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
        self.save_file(df=std_df, outpath=os.path.join(self.outdir, "sample_to_dataset.txt.gz"), index=False)

        print("Creating dataset file.")
        dataset_sample_counts = list(zip(*np.unique(std_df.iloc[:, 1], return_counts=True)))
        dataset_sample_counts.sort(key=lambda x: -x[1])
        datasets = [csc[0] for csc in dataset_sample_counts]

        dataset_df = pd.DataFrame(0, index=std_df.iloc[:, 0], columns=datasets)
        for dataset in datasets:
            dataset_df.loc[(std_df.iloc[:, 1] == dataset).values, dataset] = 1
        dataset_df.index.name = "-"
        self.save_file(df=dataset_df, outpath=os.path.join(self.outdir, "datasets_table.txt.gz"))
        del std_df, dataset_df

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
        for i, (_, row) in enumerate(eqtl_df.iterrows()):
            if row["SNPName"] in present_snps and row["ProbeName"] in present_genes:
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

        for n_pcs in range(5, 105, 5):
            if n_pcs <= pcpc_df.shape[0]:
                self.save_file(df=pcpc_df.iloc[:n_pcs, :].loc[:, rnaseq_ids], outpath=os.path.join(self.outdir, "first{}ExpressionPCs.txt.gz".format(n_pcs)))
        del pcpc_df

        print("Preparing technical covariates")
        ram_df = None
        if self.rna_alignment_path is not None:
            ram_df = self.load_file(self.rna_alignment_path, header=0, index_col=0)
            self.save_file(df=ram_df.loc[rnaseq_ids, :].T, outpath=os.path.join(self.outdir, "rnaseq_alignment_metrics_table.txt.gz"))

        sex_df = None
        if self.sex_path is not None:
            sex_df = self.load_file(self.sex_path, header=0, index_col=0)
            self.save_file(df=sex_df.loc[rnaseq_ids, :].T, outpath=os.path.join(self.outdir, "sex_table.txt.gz"))

        mds_df = None
        if self.mds_path is not None:
            mds_df = self.load_file(self.mds_path, header=0, index_col=0)
            self.save_file(df=mds_df.loc[rnaseq_ids, :].T, outpath=os.path.join(self.outdir, "mds_table.txt.gz"))

        correction_df = None
        if ram_df is None:
            correction_df = ram_df

        if sex_df is not None:
            if correction_df is not None:
                correction_df = correction_df.merge(sex_df, left_index=True, right_index=True)
            else:
                correction_df = sex_df

        if mds_df is not None:
            if correction_df is not None:
                correction_df = correction_df.merge(mds_df, left_index=True, right_index=True)
            else:
                correction_df = mds_df

        correction_df = correction_df.loc[:, correction_df.std(axis=0) != 0]
        self.save_file(df=correction_df.loc[rnaseq_ids, :].T, outpath=os.path.join(self.outdir, "tech_covariates_with_interaction_df.txt.gz"))
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
        print("  > RNAseq alignment metrics: {}".format(self.rna_alignment_path))
        print("  > Sex: {}".format(self.sex_path))
        print("  > Genotype MDS: {}".format(self.mds_path))
        print("  > Post-correction PCs: {}".format(self.post_corr_pcs_path))
        print("  > Genotype-to-expression path: {}".format(self.gte_path))
        print("  > Output directory: {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
