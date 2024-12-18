#!/usr/bin/env python3

"""
File:         prepare_sceqtlgen_eqtl_files.py
Created:      2024/01/17
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
import os

# Third party imports.
import pandas as pd

# Local application imports.

# Metadata
__program__ = "Prepare sceQTLgen eQTL files"
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
./prepare_sceqtlgen_eqtl_files.py -h 
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.sceqtlgen_path = getattr(arguments, 'sceqtlgen')
        self.avg_ge_path = getattr(arguments, 'average_gene_expression')
        self.min_avg_expression = getattr(arguments, 'min_avg_expression')

        # Set variables.
        self.outdir = os.path.join(str(os.path.dirname(os.path.abspath(__file__))), 'prepare_sceqtlgen_eqtl_files')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        self.sceqtlgen_ct = ['CD4_T', 'CD8_T', 'NK', 'B', 'DC', 'Mono']

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
        parser.add_argument("-s",
                            "--sceqtlgen",
                            type=str,
                            required=True,
                            help="The path to the sceqtlgen eqtl matrix.")
        parser.add_argument("-avge",
                            "--average_gene_expression",
                            type=str,
                            required=False,
                            default=None,
                            help="The path to the average gene expression "
                                 "matrix.")
        parser.add_argument("-mae",
                            "--min_avg_expression",
                            type=float,
                            default=None,
                            help="The minimal average expression of a gene."
                                 "Default: None.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        file_appendix = ""
        avg_ge_dict = {}
        if self.min_avg_expression is not None:
            print("Loading average expression")
            avg_ge_df = self.load_file(self.avg_ge_path, header=0, index_col=0)
            avg_ge_dict = dict(zip(avg_ge_df.index, avg_ge_df["average"]))
            file_appendix = "_GT{}AvgExprFilter".format(self.min_avg_expression)

        eqtl_df_list = []
        for ct in self.sceqtlgen_ct:
            print("Processing {}".format(ct))

            ct_eqtl_path = glob.glob(self.sceqtlgen_path + "/" + ct + "*")[0]
            eqtl_df = self.load_file(ct_eqtl_path, header=0, index_col=None)
            print("\tN eQTLs: {:,}".format(eqtl_df.shape[0]))

            eqtl_df["ctSpecific"] = True
            for other_ct in self.sceqtlgen_ct:
                if other_ct == ct:
                    continue
                eqtl_df.loc[eqtl_df[other_ct + "_EffectDir"] & eqtl_df[other_ct + "_nomSig"], "ctSpecific"] = False
            eqtl_df = eqtl_df.loc[(eqtl_df["global_pvalue"] < 0.05) & (eqtl_df["ctSpecific"]), :]
            print("\tN cell-type specific eQTLs: {:,}".format(eqtl_df.shape[0]))

            if self.min_avg_expression is not None:
                eqtl_df['avgExpression'] = eqtl_df["ENSG"].map(avg_ge_dict)
                eqtl_df = eqtl_df.loc[eqtl_df['avgExpression'] > self.min_avg_expression, :]
                print("\tN bulk expressed eQTLs: {:,}".format(eqtl_df.shape[0]))

            # feature_id", "snp_id", "p_value", "beta", "beta_se", "empirical_feature_p_value", "feature_chromosome", "feature_start", "feature_end", "ENSG", "biotype", "n_samples", "n_e_samples", "alpha_param", "beta_param", "snp_chromosome", "snp_position", "assessed_allele", "call_rate", "maf", "hwe_p", "QTL", "z_score", "ds_z_scores", "ds_beta_deviation", "I_square", "global_pvalue
            eqtl_df["feature_center"] = (eqtl_df["feature_end"] - eqtl_df["feature_start"]) / 2
            eqtl_df["SNPType"] = "cis"
            eqtl_df = eqtl_df[["p_value", "snp_id", "snp_chromosome", "snp_position", "ENSG", "feature_chromosome", "feature_center", "SNPType", "assessed_allele", "z_score", "feature_id", "global_pvalue", "n_samples"]]
            eqtl_df.columns = ["PValue", "SNPName", "SNPChr", "SNPChrPos", "ProbeName", "ProbeChr", "ProbeCenterChrPos", "SNPType", "AlleleAssessed", "OverallZScore", "HGNCName", "FDR", "N samples"]
            # eqtl_df["ID"] = eqtl_df["SNPName"] + "_" + eqtl_df["ProbeName"]
            # print(eqtl_df)

            print("\tSaving output with shape: {}".format(eqtl_df.shape))
            if eqtl_df.shape[0] > 0:
                eqtl_df.to_csv(os.path.join(self.outdir, "sceQTLgen-eQTLProbesFDR0.05-ProbeLevel{}-{}.txt.gz".format(file_appendix, ct)),
                                  sep="\t",
                                  header=True,
                                  index=False,
                                  compression="gzip")

                eqtl_df.insert(0, "CellType", ct)
                eqtl_df_list.append(eqtl_df)

        print("Processing combined")
        eqtl_df = pd.concat(eqtl_df_list, axis=0)
        print("\tSaving output with shape: {}".format(eqtl_df.shape))
        eqtl_df.to_csv(os.path.join(self.outdir, "sceQTLgen-eQTLProbesFDR0.05-ProbeLevel{}.txt.gz".format(file_appendix)),
                       sep="\t",
                       header=True,
                       index=False,
                       compression="gzip")

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
        print("  > sceQTLgen eQTL: {}".format(self.sceqtlgen_path))
        print("  > Average expression path: {}".format(self.avg_ge_path))
        print("  > Minimal average expression: >{}".format(self.min_avg_expression))
        print("  > Output directory: {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
