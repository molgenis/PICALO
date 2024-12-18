#!/usr/bin/env python3

"""
File:         vcf_to_dosage.py
Created:      2024/01/19
Last Changed:
Author:       M.Vochteloo

Copyright (C) 2020 University Medical Center Groningen.

A copy of the BSD 3-Clause "New" or "Revised" License can be found in the
LICENSE file in the root directory of this source tree.
"""

# Standard imports.
from __future__ import print_function
import argparse
import gzip
import os

# Third party imports.

# Local application imports.

# Metadata
__program__ = "VCF to Dosage"
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
./vcf_to_dosage.py -h
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.input = getattr(arguments, 'input')
        outdir = getattr(arguments, 'output')

        # Set variables.
        outdir = os.path.join(str(os.path.dirname(os.path.abspath(__file__))), 'vcf_to_dosage', outdir)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        self.output = os.path.join(outdir, os.path.basename(self.input).replace(".vcf.gz", ".tsv.gz"))

    @staticmethod
    def create_argument_parser():
        parser = argparse.ArgumentParser(prog=__program__,
                                         description=__description__,
                                         )

        # Add optional arguments.
        parser.add_argument("-v",
                            "--version",
                            action="version",
                            version="{} {}".format(__program__,
                                                   __version__),
                            help="show program's version number and exit.")
        parser.add_argument("-i",
                            "--input",
                            type=str,
                            default=None,
                            help="The input VCF file.")
        parser.add_argument("-o",
                            "--output",
                            type=str,
                            required=True,
                            help="The path to the output directory.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("Converting VCF to genotype dosage matrix")

        fh = gzip.open(self.input, 'rt')
        fho = gzip.open(self.output, 'wt')

        chrom_index = None
        pos_index = None
        # id_index = None
        ref_index = None
        alt_index = None
        format_index = None
        samples_index = None
        i = 0
        for i, line in enumerate(fh):
            if (i == 0) or (i % 1000 == 0):
                print("      parsed {:,} lines".format(i))

            # Skip comments.
            if line.startswith("##"):
                continue

            # Find query samples from header.
            values = line.strip("\n").split("\t")
            if line.startswith("#CHROM"):
                chrom_index = values.index("#CHROM")
                pos_index = values.index("POS")
                # id_index = values.index("ID")
                ref_index = values.index("REF")
                alt_index = values.index("ALT")
                format_index = values.index("FORMAT")
                for samples_index, column in enumerate(values):
                    if column not in ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"]:
                        break

                fho.write("\t".join(["SNP", "Alleles", "AltAllele"] + values[samples_index:]) + "\n")
                continue

            id_str = "{}:{}:{}:{}".format(values[chrom_index], values[pos_index], values[ref_index], values[alt_index])
            alleles = "{}/{}".format(values[ref_index], values[alt_index])
            alt_allele = values[alt_index]

            gt_index = values[format_index].split(":").index("GT")
            dosages = []
            for value in values[samples_index:]:
                gt_value = value.split(":")[gt_index]
                if gt_value == "./.":
                    dosages.append("-1.0")
                elif "/" in gt_value:
                    dosages.append(str(sum([float(dosage) for dosage in gt_value.split("/")])))
                elif "|" in gt_value:
                    dosages.append(str(sum([float(dosage) for dosage in gt_value.split("|")])))

            fho.write("\t".join([id_str, alleles, alt_allele] + dosages) + "\n")

        print("      parsed {:,} lines".format(i))

        fh.close()
        fho.close()

    def print_arguments(self):
        print("Arguments:")
        print("  > Input VCF: {}".format(self.input))
        print("  > Output dosage: {}".format(self.output))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
