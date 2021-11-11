#!/usr/bin/env python3

"""
File:         preprocess_mds_file.py
Created:      2020/10/26
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
import pandas as pd

# Local application imports.

# Metadata
__program__ = "Preprocess MDS file"
__author__ = "Martijn Vochteloo"
__maintainer__ = "Martijn Vochteloo"
__email__ = "m.vochteloo@st.hanze.nl"
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
./preprocess_mds_file.py -d ../data/MetaBrain-allchr-mds-noENA-dupsremoved.mds -gte /groups/umcg-biogen/tmp01/output/2019-11-06-FreezeTwoDotOne/2020-10-12-deconvolution/deconvolution/matrix_preparation/OLD/ContainsDuplicateSamples/CortexEUR-cis/combine_gte_files/GTE_combined.txt.gz

./preprocess_mds_file.py -d ../data/MetaBrain-allchr-mds-noENA-dupsremoved-outlierremoved.mds -gte /groups/umcg-biogen/tmp01/output/2019-11-06-FreezeTwoDotOne/2020-10-12-deconvolution/deconvolution/matrix_preparation/OLD/ContainsDuplicateSamples/CortexEUR-cis/combine_gte_files/GTE_combined.txt.gz

./preprocess_mds_file.py -d ../data/MetaBrain-allchr-mds-noENA-dupsremoved-outlierremoved-VariantFilter.mds -gte /groups/umcg-biogen/tmp01/output/2019-11-06-FreezeTwoDotOne/2020-10-12-deconvolution/deconvolution/matrix_preparation/OLD/ContainsDuplicateSamples/CortexEUR-cis/combine_gte_files/GTE_combined.txt.gz

./preprocess_mds_file.py -d ../data/MetaBrain-allchr-mds-dupsremoved-VariantFilter.mds -gte /groups/umcg-biogen/tmp01/output/2019-11-06-FreezeTwoDotOne/2020-10-12-deconvolution/deconvolution/matrix_preparation/OLD/ContainsDuplicateSamples/CortexEUR-cis/combine_gte_files/GTE_combined.txt.gz

./preprocess_mds_file.py -d /groups/umcg-bios/tmp01/resources/genotypes-hrc-imputed-vcf/2021-10-28-mds-all-VariantFilter/allchr-mds-VariantFilter.mds -gte /groups/umcg-bios/prm03/projects/BIOS_EGCUT_for_eQTLGen/BIOS_EGCUT/eqtlpipeline_bios_egcut_backup010517/GTE_LLDEEP_and_BIOS_last_related_removed_110417.txt -op BIOS-

./preprocess_mds_file.py -d /groups/umcg-bios/tmp01/projects/PICALO/tmpGonlStorage/2021-10-29-mds-BIOS-GTESubset-VariantFilter/allchr-mds-BIOS-GTESubset-VariantFilter.mds -gte /groups/umcg-bios/prm03/projects/BIOS_EGCUT_for_eQTLGen/BIOS_EGCUT/eqtlpipeline_bios_egcut_backup010517/GTE_LLDEEP_and_BIOS_last_related_removed_110417.txt -op BIOS-

./preprocess_mds_file.py -d /groups/umcg-bios/tmp01/projects/PICALO/tmpGonlStorage/2021-10-29-mds-BIOS-GTESubset-VariantSubsetFilter/allchr-mds-BIOS-GTESubset-VariantSubsetFilter.mds -gte /groups/umcg-bios/prm03/projects/BIOS_EGCUT_for_eQTLGen/BIOS_EGCUT/eqtlpipeline_bios_egcut_backup010517/GTE_LLDEEP_and_BIOS_last_related_removed_110417.txt -op BIOS-

./preprocess_mds_file.py -d /groups/umcg-bios/tmp01/projects/PICALO/tmpGonlStorage/2021-10-29-mds-BIOS-GTESubset-noOutlier-VariantSubsetFilter/allchr-mds-BIOS-GTESubset-noOutlier-VariantSubsetFilter.mds -gte /groups/umcg-bios/prm03/projects/BIOS_EGCUT_for_eQTLGen/BIOS_EGCUT/eqtlpipeline_bios_egcut_backup010517/GTE_LLDEEP_and_BIOS_last_related_removed_110417.txt -op BIOS-

./preprocess_mds_file.py -d /groups/umcg-bios/tmp01/projects/PICALO/tmpGonlStorage/2021-11-05-mds-BIOS-GTESubset-noRNAPhenoNA-VariantSubsetFilter/allchr-mds-BIOS-GTESubset-noRNAPhenoNA-VariantSubsetFilter.mds -gte /groups/umcg-bios/prm03/projects/BIOS_EGCUT_for_eQTLGen/BIOS_EGCUT/eqtlpipeline_bios_egcut_backup010517/GTE_LLDEEP_and_BIOS_last_related_removed_110417.txt -op BIOS-

./preprocess_mds_file.py -d /groups/umcg-bios/tmp01/projects/PICALO/tmpGonlStorage/2021-11-05-mds-BIOS-GTESubset-noRNAPhenoNA-noOutliers-VariantSubsetFilter/allchr-mds-BIOS-GTESubset-noRNAPhenoNA-noOutliers-VariantSubsetFilter.mds -gte /groups/umcg-bios/prm03/projects/BIOS_EGCUT_for_eQTLGen/BIOS_EGCUT/eqtlpipeline_bios_egcut_backup010517/GTE_LLDEEP_and_BIOS_last_related_removed_110417.txt -op BIOS-

./preprocess_mds_file.py -d /groups/umcg-bios/tmp01/resources/genotypes-hrc-imputed-vcf/2021-11-11-mds-NoRNAPhenoNA_NoSexNA-SamplesVariantFilter/allchr-mds-BIOS-NoRNAPhenoNA-NoSexNA-VariantSubsetFilter.mds -gte /groups/umcg-bios/tmp01/projects/PICALO/data/BIOS_GTE.txt.gz -op BIOS-

./preprocess_mds_file.py -d /groups/umcg-bios/tmp01/resources/genotypes-hrc-imputed-vcf/2021-11-11-mds-NoRNAPhenoNA-NoSexNA-NoMDSOutlier-SamplesVariantFilter/allchr-mds-BIOS-NoRNAPhenoNA-NoSexNA-NoMDSOutlier-VariantSubsetFilter.mds -gte /groups/umcg-bios/tmp01/projects/PICALO/data/BIOS_GTE.txt.gz -op BIOS-
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.data_path = getattr(arguments, 'data')
        self.gte_path = getattr(arguments, 'gene_to_exression')
        self.output_prefix = getattr(arguments, 'output_prefix')

        self.outdir = os.path.join(str(Path(__file__).parent.parent), 'preprocess_mds_file')
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
        parser.add_argument("-d",
                            "--data",
                            type=str,
                            required=True,
                            help="The path to the data matrix.")
        parser.add_argument("-gte",
                            "--gene_to_exression",
                            type=str,
                            required=True,
                            help="The path to the GtE file.")
        parser.add_argument("-op",
                            "--output_prefix",
                            type=str,
                            required=False,
                            default="",
                            help="The path to the GtE file.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("Loading data file")
        columns = None
        lines = []
        with open(self.data_path) as f:
            for i, line in enumerate(f):
                data = [x for x in line.rstrip().split(" ") if x != ""]

                if i == 0:
                    columns = data
                else:
                    lines.append(data)
        f.close()
        df = pd.DataFrame(lines, columns=columns)
        print(df)

        print("Loading GtE file")
        gte_df = pd.read_csv(self.gte_path, sep="\t", header=None, index_col=None)
        gte_dict = dict(zip(gte_df.iloc[:, 0], gte_df.iloc[:, 1]))

        print("Pre-process")
        df.set_index("IID", inplace=True)
        df.index.name = "-"
        df = df.loc[:, ["C1", "C2", "C3", "C4"]]

        print("Translating")
        df.index = [gte_dict[genotype_id] for genotype_id in df.index]
        print(df)

        print("Saving file")
        outpath = os.path.join(self.outdir, self.output_prefix + os.path.basename(self.data_path).split(".")[0] + ".txt.gz")
        df.to_csv(outpath, compression="gzip", sep="\t", header=True, index=True)
        print("\tSaved dataframe: {} "
                    "with shape: {}".format(os.path.basename(outpath),
                                            df.shape))

    def print_arguments(self):
        print("Arguments:")
        print("  > Data: {}".format(self.data_path))
        print("  > GtE: {}".format(self.gte_path))
        print("  > Output prefix: {}".format(self.output_prefix))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
