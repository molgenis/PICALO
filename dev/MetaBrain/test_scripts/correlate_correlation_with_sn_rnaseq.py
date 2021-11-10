#!/usr/bin/env python3

"""
File:         Correlate_correlations_with_sn_rnaseq.py
Created:      2021/05/25
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
import glob
import pickle
import os

# Third party imports.
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Local application imports.

# Metadata
__program__ = "Correlate Correlations with SN-RNAseq"
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
./correlate_correlation_with_sn_rnaseq.py -h

"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.gc_path = getattr(arguments, 'gene_correlations')
        self.sn_indir = getattr(arguments, 'single_nucleus')
        self.gi_path = getattr(arguments, 'gene_info')
        self.ensg_col = getattr(arguments, 'ensg_column')
        self.hgnc_path = getattr(arguments, 'hgnc_column')
        self.out_filename = getattr(arguments, 'outfile')

        # Set variables.
        self.outdir = os.path.join(str(Path(__file__).parent.parent), 'plot')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        self.colormap = {
            "IN": "#56B4E9",
            "EX": "#0072B2",
            "OLI": "#009E73",
            "END": "#CC79A7",
            "MIC": "#E69F00",
            "AST": "#D55E00",
            "PER": "#808080",
            "OPC": "#F0E442"
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
        parser.add_argument("-gc",
                            "--gene_correlations",
                            type=str,
                            required=True,
                            help="The path to the gene correlations matrix.")
        parser.add_argument("-sn",
                            "--single_nucleus",
                            type=str,
                            required=True,
                            help="The path to the SN RNA-seq input directory.")
        parser.add_argument("-gi",
                            "--gene_info",
                            type=str,
                            required=True,
                            help="The path to the gene information matrix.")
        parser.add_argument("-ensg",
                            "--ensg_column",
                            type=str,
                            required=True,
                            help="The column in -gi / --gene_info matrix"
                                 "with the ENSG IDs.")
        parser.add_argument("-hgnc",
                            "--hgnc_column",
                            type=str,
                            required=True,
                            help="The column in -gi / --gene_info matrix"
                                 "with the HGNC names.")
        parser.add_argument("-o",
                            "--outfile",
                            type=str,
                            default="output",
                            help="The name of the outfile. Default: output.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("Load gene info.")
        gene_info_df = self.load_file(self.gi_path, header=0, index_col=0)
        gene_dict = dict(zip(gene_info_df[self.ensg_col], gene_info_df[self.hgnc_path]))
        del gene_info_df

        # with open("unique_genes_per_comp.pkl", "rb") as f:
        #     unique_genes = pickle.load(f)
        # f.close()
        genes = ['CCDC152', 'DUSP10', 'PDE4B', 'OLIG1', 'DLG2', 'TANC1', 'PTGDS', 'CRHBP', 'CLMP', 'UBASH3B', 'CPED1', 'MIR181A1HG', 'CUEDC1', 'DOCK3', 'VSTM2B', 'LOC105376244', 'LPAR6', 'CREB5', 'EGFR', 'ARFGEF3', 'CDH19', 'MAGI2', 'CD9', 'VAV2', 'SH3TC2', 'COLGALT2', 'HAPLN1', 'SELENBP1', 'ZNF385D', 'RAB3B', 'GNAI1', 'LOC101927967', 'APBB2', 'HDAC1', 'TRPC3', 'GSN', 'PCDH11Y', 'RAD51B', 'NXPH1', 'ACACB', 'ANK3', 'PLD1', 'ADAMTS17', 'ARHGAP42', 'KIAA1755', 'SLC24A3', 'PLEKHH1', 'BCAS1', 'ZNF608', 'LINC00844', 'CX3CR1', 'LOC339975', 'EDNRA', 'ZBTB20', 'CPM', 'ZFP36L1', 'SLCO3A1', 'FAM13C', 'PTPRK', 'SIK3', 'C5orf64', 'GRIN2D', 'TMC6', 'PKP2', 'CD82', 'PDE3A', 'LOC101928286', 'PRUNE2', 'HIF3A', 'LINC01314', 'DOCK11', 'PAPLN', 'LANCL1', 'ZDHHC9', 'TMEM98', 'MBP', 'APP', 'ANO1', 'DLX1', 'ZFHX4', 'RAPGEF5', 'LGI2', 'NMU', 'PPP2R2B', 'AK5', 'GLUL', 'NKX6-2', 'NDUFA4P1', 'SPOCK1', 'SAMD12', 'EYA4', 'NDRG1', 'DNM3', 'LINC00320', 'ATP6V0C', 'OLIG2', 'THSD7A', 'CRYAB', 'KLHL4', 'MAG', 'TTYH2', 'SOX2-OT', 'ROR1', 'SLC12A2', 'SH3D19', 'ENOSF1', 'EFS', 'SLC35F4', 'LOC100506725', 'ALCAM', 'STON2', 'ADGRG1', 'UGT8', 'LOC101929249', 'ZNF536', 'LINC00403', 'OPN5', 'AKAP6', 'LINC01105', 'LINC00609', 'FRYL', 'ZPBP', 'RND3', 'KIF13A', 'CSMD3', 'NALCN', 'RNF220', 'FAM134B', 'NOVA1', 'ARL4C', 'CERCAM', 'RNF219-AS1', 'SGK1', 'PNOC', 'CACNA2D1', 'H3F3AP4', 'LOC102724289', 'EZR', 'MTND4P12', 'NDRG2', 'HIP1', 'NCAM2', 'RFTN2', 'CSRP1', 'PCDH9', 'ATP8A1', 'FSTL1', 'SLC39A12', 'LOC101927121', 'ST18', 'CD22', 'GLDN', 'DSCAML1', 'TMEM144', 'KIAA1211', 'ERMN', 'ANKS1B', 'COBL', 'PTPRD', 'ST8SIA4', 'PAWR', 'LYPD6B', 'LHFPL3', 'HLA-A', 'FBXL7', 'MCC', 'S100B', 'PCSK6', 'LRP4', 'TMEFF2', 'AIF1L', 'CLDN11', 'ANLN', 'MAP4K5', 'NXPH2', 'EEPD1', 'SST', 'HS3ST3B1', 'LYPD4', 'ALK', 'ABCA2', 'SLC6A1', 'ARX', 'ERBB3', 'RASSF2', 'NLGN1', 'TLR4', 'MAML2', 'LOC101930275', 'KIRREL3', 'AGPAT4', 'ECE2', 'LOC105374460', 'KCNJ10', 'CNTNAP2', 'SOX10', 'LOC105376400', 'HEPACAM', 'DPYD', 'PIP4K2A', 'PTCHD4', 'COL9A2', 'TMEM165', 'FRMD4B', 'ZHX2', 'GPR37L1', 'LYN', 'PDE8A', 'CTTNBP2', 'DBNDD2', 'SLAIN1', 'ITPKB', 'PEX5L', 'MYRF', 'HEG1', 'PHLDB1', 'GAD2', 'PLA2G16', 'LOC105371352', 'KLF12', 'SULF1', 'SALL1', 'SLCO1C1', 'UNC5C', 'PLCL1', 'SOX2', 'RIN2', 'MAF', 'LPAR1', 'FAM124A', 'ZFAND4', 'GPM6B', 'WLS', 'GPRC5B', 'LOC105371255', 'PLCE1', 'PTRF', 'ASIC4', 'ARPP21', 'LOC102723503', 'FRMD5', 'MT2A', 'MYO10', 'VIP', 'SLC7A14', 'PTK7', 'DAAM2', 'SCD5', 'HS3ST4', 'ATP10B', 'GAD1', 'CDK18', 'FAM107B', 'STMN4', 'SLC24A2', 'ETNPPL', 'KLK6', 'CYP2J2', 'PRKX', 'KAZN', 'MYO15B', 'NCAM1', 'ARHGAP31', 'CECR2', 'IL1RAPL1', 'PXK', 'HHIP', 'JAM3', 'ADARB2', 'DGKD', 'CA2', 'LINC00982', 'NHSL1', 'NFASC', 'LOC105377183', 'LOC105375132', 'NKAIN4', 'MDGA2', 'CNDP1', 'TRIM2', 'NCKAP5', 'MARCKSL1', 'MEGF10', 'LOC101928622', 'PTN', 'LPGAT1', 'GPR37', 'DOCK9', 'PREX1', 'HECW2', 'PIEZO2', 'FGFR3', 'SPATA13', 'NOTCH1', 'FGFR2', 'CTGF', 'ANK1', 'NOS1', 'ROBO1', 'PHLPP1', 'GJA1', 'GALNT13', 'RAPGEF3', 'GRIK1', 'DOCK1', 'DLG1', 'CLDND1', 'SHROOM4', 'GRIP2', 'LOC100130370', 'EPS8', 'ATP1B2', 'LOC105375483', 'SLC35D2', 'SLC5A11', 'LOC255187', 'CXCL14', 'PKP4', 'PTMAP5', 'KIT', 'SLC44A1', 'LINC01322', 'TRIM59', 'XAF1', 'AASS', 'DMD', 'PPP1R14A', 'GAB1', 'MITF', 'LAMP2', 'DLX6', 'PARD3B', 'KCNH8', 'TMEM235', 'PDE1C', 'SEC14L5', 'PHF21B', 'ATP13A4', 'EFHD1', 'PLP1', 'RASD2', 'ACSBG1', 'ANKRD55', 'FRMPD1', 'SLC32A1', 'FOXO1', 'ADRA1A', 'ATP1A2', 'CLASP2', 'PLXNB1', 'LINC00639', 'APOD', 'OPALIN', 'LINC01170', 'TAC1', 'PON2', 'LOC101927437', 'GLI3', 'DOCK10', 'KCNAB3', 'F3', 'BACH2', 'LOC101927459', 'QKI', 'CTNNA3', 'PLXNB3', 'ENPP2', 'LOC102724312', 'CADM4', 'RGCC', 'DPP6', 'RPP25', 'KLHL32', 'FA2H', 'PMP2', 'TULP4', 'ELMO1', 'TMCC3', 'DLX6-AS1', 'TF', 'VRK2', 'LOC100507336', 'CMTM5', 'PCDH11X', 'LHX6', 'DNAJC6', 'ASPA', 'DNAH17', 'HTRA1', 'ITM2A', 'SLC1A3', 'SH3GL3', 'SEMA3E', 'CRH', 'VCAN', 'MIR219A2', 'RELN', 'LOC101927439', 'CLMN', 'PROX1', 'PCP4L1', 'APLP1', 'NRXN3', 'HAPLN2', 'ZCCHC24', 'PVRL2', 'LINC00499', 'KLHL5', 'RBMS3', 'PLLP', 'C10orf90', 'SLC22A15', 'PADI2', 'LOC645321', 'KCNJ16', 'CD81', 'CDC14A', 'LOC105369345', 'LOC101927199', 'TSC22D4', 'PREX2', 'MAP7', 'TMEM63A', 'NEAT1', 'SMOC1', 'GOLIM4', 'FOLH1', 'PLCXD3', 'MOB3B', 'AGAP1', 'PDK4', 'SLC25A18', 'COL4A5', 'ABCA8', 'CNTN2', 'CARNS1', 'KIF5C', 'B2M', 'CHD7', 'PALM2', 'ASPM', 'SPOCK3', 'PRR5L', 'ACSS1', 'Mar/01', 'MYO1D', 'EDIL3', 'PPAP2B', 'GSN-AS1', 'KCNIP1', 'SCD', 'PPP1R16B', 'SYNJ2', 'GRAMD3', 'CNTNAP4', 'WIF1', 'LPPR1', 'BTBD11', 'DOCK5', 'LOC105374461', 'MOBP', 'PITPNC1', 'TTLL7', 'CAMK2N2', 'Sep/10', 'SLC7A2', 'USP54', 'PMP22', 'CADM2', 'GPR149', 'SOX6', 'APCDD1', 'SLCO1A2', 'ZFP36L2', 'PPFIA2', 'ERBB4', 'SPP1', 'RNASE1', 'WIPF1', 'ZNF704', 'MTUS1', 'MAL', 'PPM1H', 'PSD4', 'RNF144B', 'ANO4', 'RGS12', 'NKX2-2', 'OAF', 'APOE', 'EYA2', 'PRDM16', 'PLEKHB1', 'PARD3', 'SEMA3B', 'LOC391359', 'GNG12-AS1', 'IGF1', 'DNER', 'MOG', 'LYPD6', 'SRCIN1', 'RNF144A', 'KANK1', 'CLIC4', 'TNS3', 'NDN', 'CYP27A1', 'LOC100128906', 'LOC105375415', 'NKAIN2', 'QDPR', 'TECRP1', 'LDB3', 'MAFB']

        # Load data.
        print("Loading data.")
        gc_df = self.load_file(self.gc_path, header=0, index_col=None)
        gc_df.index = gc_df["gene"]
        print(gc_df)

        data = {}
        cell_types = []
        for filepath in glob.glob(os.path.join(self.sn_indir, "*_expression.txt")):
            cell_type = os.path.basename(filepath).split("_")[0].upper()
            sn_df = self.load_file(filepath, header=0, index_col=0)
            sn_df = sn_df - sn_df.mean(axis=0)
            sn_df = sn_df.loc[:, sn_df.std(axis=0) != 0]
            data[cell_type] = sn_df
            cell_types.append(cell_type)

        print("Correlate")
        correlations = []
        indices = []
        for i in range(10):
            component = "component{}".format(i)
            # if "comp{}".format(i) not in unique_genes.keys():
            #     continue
            component_gc_df = gc_df.loc[(gc_df["component"] == component) & (gc_df["gene (HGNC)"].isin(genes)), :].copy()
            print(component_gc_df)
            component_gc_df.dropna(inplace=True)
            if component_gc_df.shape[0] == 0:
                continue

            print("\t{} {}".format(component, component_gc_df.shape))
            comp_results = []

            results = []
            for cell_type in cell_types:
                ct_df = data[cell_type]
                overlap = set(component_gc_df.index.tolist()).intersection(set(ct_df.index.tolist()))
                print(len(overlap))
                comp_ct_gc_df = component_gc_df.loc[overlap, :]
                comp_ct_df = ct_df.loc[overlap, :]

                coefficients = np.empty(comp_ct_df.shape[1])
                for i in range(comp_ct_df.shape[1]):
                    coef, p = stats.pearsonr(comp_ct_df.iloc[:, i], comp_ct_gc_df["correlation"])

                    coefficients[i] = coef

                mean_coef = np.mean(coefficients)
                results.append((cell_type, mean_coef, np.std(coefficients), abs(mean_coef)))
                comp_results.append(mean_coef)

            results.sort(key=lambda x: -x[3])
            for ss_cell_type, mean, std, _ in results:
                print("\t  {}: mean = {:.2f} std = {:.2f}".format(ss_cell_type, mean, std))
            print("")

            correlations.append(comp_results)
            indices.append(component)

        corr_df = pd.DataFrame(correlations, index=indices, columns=cell_types)
        cell_types.sort()
        corr_df = corr_df.loc[:, cell_types]
        print(corr_df)

        self.plot_heatmap(corr_df.T)

    @staticmethod
    def load_file(inpath, header, index_col, sep="\t", low_memory=True,
                  nrows=None, skiprows=None):
        df = pd.read_csv(inpath, sep=sep, header=header, index_col=index_col,
                         low_memory=low_memory, nrows=nrows, skiprows=skiprows)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(inpath),
                                      df.shape))
        return df

    def plot_heatmap(self, corr_df):
        cmap = sns.diverging_palette(246, 24, as_cmap=True)

        row_colors = [self.colormap[ct.split("_")[0]] for ct in corr_df.index]

        sns.set(color_codes=True)
        g = sns.clustermap(corr_df, cmap=cmap,
                           row_cluster=True, col_cluster=False,
                           yticklabels=True, xticklabels=True, square=True,
                           vmin=-1, vmax=1, annot=corr_df.round(2),
                           row_colors=row_colors, fmt='',
                           annot_kws={"size": 12, "color": "#000000"},
                           figsize=(12, 12))
        plt.setp(
            g.ax_heatmap.set_yticklabels(
                g.ax_heatmap.get_ymajorticklabels(),
                fontsize=16, rotation=0))
        plt.setp(
            g.ax_heatmap.set_xticklabels(
                g.ax_heatmap.get_xmajorticklabels(),
                fontsize=16, rotation=90))

        plt.tight_layout()
        g.savefig(os.path.join(self.outdir, "{}_SNRNASeq_corr_heatmap.png".format(self.out_filename)))
        plt.close()

    def print_arguments(self):
        print("Arguments:")
        print("  > Component - gene correlations data: {}".format(self.gc_path))
        print("  > SN-RNAseq input directory: {}".format(self.sn_indir))
        print("  > Gene information matrix: {}".format(self.gi_path))
        print("    > ENSG ID column: {}".format(self.ensg_col))
        print("    > HGNC column: {}".format(self.hgnc_path))
        print("  > Output directory {}".format(self.outdir))
        print("  > Output filename: {}".format(self.out_filename))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
