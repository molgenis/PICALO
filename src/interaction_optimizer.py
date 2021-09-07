"""
File:         interaction_optimizer.py
Created:      2021/03/25
Last Changed: 2021/08/30
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
import pandas as pd
import time
import os

# Third party imports.
import numpy as np
from statsmodels.stats import multitest

# Local application imports.
from src.force_normaliser import ForceNormaliser
from src.objects.ieqtl import IeQTL
from src.statistics import calc_eucledian_distance, calc_rss, calc_vertex_xpos, remove_covariates, remove_covariates_elementwise
from src.utilities import load_dataframe, save_dataframe
from src.visualiser import Visualiser


class InteractionOptimizer:
    def __init__(self, n_iterations, covariates, dataset_m, samples,
                 genotype_na, ieqtl_alpha, log):
        self.n_iterations = n_iterations
        self.covariates = covariates
        self.samples = samples
        self.genotype_na = genotype_na
        self.ieqtl_alpha = ieqtl_alpha
        self.log = log
        self.fn = ForceNormaliser(dataset_m=dataset_m,
                                  samples=samples,
                                  log=log)

        # set other variables.
        self.print_interval = 30

    def process(self, eqtl_m, geno_m, expr_m, covs_m, corr_m, corr_inter_m,
                outdir):
        pic_a = None
        cov = None
        results_df = None
        prev_included_ieqtls = (0, set())
        n_iterations_performed = 0
        n_iterations_with_no_change = 0
        info_m = np.empty((self.n_iterations, 5), dtype=np.float64)
        iterations_m = np.empty((self.n_iterations + 1, geno_m.shape[1]), dtype=np.float64)
        for iteration in range(n_iterations_performed, self.n_iterations):
            self.log.info("\t\tIteration: {}".format(iteration))

            start_time = int(time.time())

            if np.ndim(covs_m) == 1:
                pic_a = covs_m

            # Find the significant ieQTLs.
            if pic_a is None:
                self.log.info("\t\t  Finding covariate with most ieQTLs")

                cov = None
                n_hits = 0
                ieqtls = []
                hits_per_cov_data = []

                # Find which covariate has the highest number of ieQTLs.
                for cov_index in range(covs_m.shape[0]):
                    # Extract the covariate we are working on.
                    cova_a = covs_m[cov_index, :]

                    # Clean the expression matrix.
                    # TODO decide if we only remove the interaction vector or
                    # if we remove all technical stuff including interaction
                    # vector for every iteration. The latter should be batter
                    # but is way more slower.

                    iter_expr_m = remove_covariates_elementwise(y_m=expr_m, X_m=geno_m, a=cova_a)
                    # iter_expr_m = remove_covariates(y_m=expr_m,
                    #                                 X_m=np.hstack((corr_m, cova_a[:, np.newaxis])),
                    #                                 X_inter_m=corr_inter_m,
                    #                                 inter_m=geno_m,
                    #                                 include_inter_as_covariate=True)

                    # Force normalise the matrix.
                    iter_expr_m = self.fn.process(data=iter_expr_m)

                    # Find the significant ieQTLs.
                    cov_hits, cov_ieqtls, cov_results_df = self.get_ieqtls(
                        eqtl_m=eqtl_m,
                        geno_m=geno_m,
                        expr_m=iter_expr_m,
                        cova_a=cova_a,
                        cov=self.covariates[cov_index]
                    )

                    # Save hits.
                    hits_per_cov_data.append([self.covariates[cov_index], cov_hits])

                    if cov_hits > n_hits:
                        cov = self.covariates[cov_index]
                        n_hits = cov_hits
                        ieqtls = cov_ieqtls
                        pic_a = cova_a
                        results_df = cov_results_df
                    else:
                        del cov_ieqtls
                self.log.info("\t\t  Covariate '{}' will be used for this component.".format(cov))

                hits_per_cov_df = pd.DataFrame(hits_per_cov_data, columns=["Covariate", "N-ieQTLs"])
                save_dataframe(df=hits_per_cov_df,
                               outpath=os.path.join(outdir, "covariate_selection.txt.gz"),
                               header=True,
                               index=False,
                               log=self.log)
            else:
                self.log.info("\t\t  Finding ieQTLs")

                # Clean the expression matrix.
                # TODO decide if we only remove the interaction vector or
                # if we remove all technical stuff including interaction
                # vector for every iteration. The latter should be batter
                # but is way more slower.

                iter_expr_m = remove_covariates_elementwise(y_m=expr_m, X_m=geno_m, a=pic_a)
                # iter_expr_m = remove_covariates(y_m=expr_m,
                #                                 X_m=np.hstack((corr_m, pic_a[:, np.newaxis])),
                #                                 X_inter_m=corr_inter_m,
                #                                 inter_m=geno_m,
                #                                 include_inter_as_covariate=True)

                # Force normalise the matrix.
                iter_expr_m = self.fn.process(data=iter_expr_m)

                _, ieqtls, results_df = self.get_ieqtls(
                    eqtl_m=eqtl_m,
                    geno_m=geno_m,
                    expr_m=iter_expr_m,
                    cova_a=pic_a,
                    cov=cov)

            # Save results.
            save_dataframe(df=results_df,
                           outpath=os.path.join(outdir, "results_iteration{}_df.txt.gz".format(iteration)),
                           header=True,
                           index=False,
                           log=self.log)

            n_ieqtls = len(ieqtls)
            if n_ieqtls == 0:
                self.log.error("\t\t  No significant ieQTLs found\n")
                break

            # Optimize the interaction vector.
            optimized_pic_a = self.optimize_ieqtls(ieqtls)

            # Safe that interaction vector.
            if iteration == 0:
                iterations_m[iteration, :] = pic_a
            iterations_m[iteration + 1, :] = optimized_pic_a

            # # Visualise.
            # # TODO remove this temporary code to create interaction plots.
            # visualiser = Visualiser()
            # for ieqtl in ieqtls:
            #     if ieqtl.get_eqtl_id() == "ENSG00000019186.10:20:54173204:rs2248137:C_G":
            #         visualiser.plot(ieqtl, out_path=outdir, iteration=iteration, ocf=None)
            #         visualiser.plot(ieqtl, out_path=outdir, iteration=iteration, ocf=pic_a)

            # Calculate the Spearman correlation between the interaction
            # vector and the optimized vector.
            self.log.info("\t\t  Comparing the optimized vector with the previous vector:")
            eucl_dist = calc_eucledian_distance(x=pic_a, y=optimized_pic_a)
            rss = calc_rss(y=pic_a, y_hat=optimized_pic_a)
            self.log.info("\t\t\tEuclidean distance: {:.2f}".format(eucl_dist))
            self.log.info("\t\t\tRSS: {:.2f}".format(rss))

            if np.round(rss, decimals=2) == 0.:
                n_iterations_with_no_change += 1
            else:
                n_iterations_with_no_change = 0

            # Compare the included ieQTLs with the previous iteration.
            included_ieqtl_ids = {ieqtl.get_ieqtl_id() for ieqtl in ieqtls}
            pct_overlap = 0
            if prev_included_ieqtls[1]:
                overlap = prev_included_ieqtls[1].intersection(included_ieqtl_ids)
                n_overlap = len(overlap)
                pct_overlap = (100 / prev_included_ieqtls[0]) * n_overlap
                self.log.info("\t\t\tOverlap in included ieQTLs: {} [{:.2f}%]".format(n_overlap, pct_overlap))

            info_m[iteration, :] = np.array([len(ieqtls), len(ieqtls) - prev_included_ieqtls[0], pct_overlap, eucl_dist, rss])

            # Overwrite the variables for the next round.
            pic_a = optimized_pic_a
            prev_included_ieqtls = (n_ieqtls, included_ieqtl_ids)
            n_iterations_performed += 1

            # Check if we converged.
            # TODO: replace this by standard convergence cut-off based on log
            #  likelihood.
            if n_iterations_with_no_change >= 3:
                self.log.warning("Model converged")
                break

            # Print time.
            rt_min, rt_sec = divmod(int(time.time()) - start_time, 60)
            self.log.info("\t\t  Finished in {} minute(s) and "
                          "{} second(s)".format(int(rt_min),
                                                int(rt_sec)))

            self.log.info("")
            del ieqtls, optimized_pic_a

        # Save overview files.
        iteration_df = pd.DataFrame(iterations_m[:(n_iterations_performed + 1), :],
                                    index=["start"] + ["iteration{}".format(i) for i in range(n_iterations_performed)],
                                    columns=self.samples)
        save_dataframe(df=iteration_df,
                       outpath=os.path.join(outdir, "iteration_df.txt.gz"),
                       header=True,
                       index=True,
                       log=self.log)

        info_df = pd.DataFrame(info_m[:n_iterations_performed, :],
                               index=["iteration{}".format(i) for i in range(n_iterations_performed)],
                               columns=["N", "Diff", "Overlap %", "Euclidean distance", "RSS"])
        info_df.insert(0, "covariate", cov)
        save_dataframe(df=info_df,
                       outpath=os.path.join(outdir, "info_df.txt.gz"),
                       header=True,
                       index=True,
                       log=self.log)

        return pic_a

    def get_ieqtls(self, eqtl_m, geno_m, expr_m, cova_a, cov):
        n_eqtls = eqtl_m.shape[0]

        ieqtls = []
        results = []
        p_values = np.empty(n_eqtls, dtype=np.float64)
        for row_index in range(n_eqtls):
            snp, gene = eqtl_m[row_index, :]
            ieqtl = IeQTL(snp=snp,
                          gene=gene,
                          cov=cov,
                          genotype=geno_m[row_index, :],
                          covariate=cova_a,
                          expression=expr_m[row_index, :]
                          )
            ieqtl.compute()
            p_values[row_index] = ieqtl.p_value
            ieqtls.append(ieqtl)
            results.append([snp, gene, cov, ieqtl.n] + ieqtl.betas.tolist() + ieqtl.std.tolist() + [ieqtl.p_value])

        # Calculate the FDR.
        fdr_values = multitest.multipletests(p_values, method='fdr_bh')[1]

        # Calculate the number of significant hits.
        mask = fdr_values < self.ieqtl_alpha
        n_hits = np.sum(mask)

        self.log.info("\t\t\tCovariate: '{}' has {} significant ieQTLs".format(cov, n_hits))

        results_df = pd.DataFrame(results,
                                  columns=["SNP", "gene", "covariate", "N",
                                           "beta-intercept", "beta-genotype",
                                           "beta-covariate", "beta-interaction",
                                           "std-intercept", "std-genotype",
                                           "std-covariate", "std-interaction",
                                           "p-value"])
        results_df["FDR"] = fdr_values

        return n_hits, [ieqtl for ieqtl, include in zip(ieqtls, mask) if include], results_df

    def optimize_ieqtls(self, ieqtls):
        self.log.info("\t\t  Optimizing ieQTLs")

        coef_a_collection = []
        coef_b_collection = []
        for i, ieqtl in enumerate(ieqtls):
            coef_a, coef_b = ieqtl.get_mll_coef_representation(full_array=True)
            coef_a_collection.append(coef_a)
            coef_b_collection.append(coef_b)

        coef_a_sum = np.nansum(coef_a_collection, axis=0)
        coef_b_sum = np.nansum(coef_b_collection, axis=0)
        optimized_a = calc_vertex_xpos(a=coef_a_sum, b=coef_b_sum)

        return optimized_a
