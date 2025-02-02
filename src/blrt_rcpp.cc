// poLCAParallel
// Copyright (C) 2022 Sherman Lo

// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License along
// with this program; if not, write to the Free Software Foundation, Inc.,
// 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

#include "RcppArmadillo.h"
#include "blrt.h"

/**
 * Function to be exported to R, does bootstrap likelihood ratio test
 *
 * Does the bootstrap likelihood ratio test. Provide two fitted models, the
 * null model and the alt model which fits a different number of clusters.
 * Bootstrap samples are generated using the null model. The null model and alt
 * model are refitted on the bootstrap samples to investigate the log likelihood
 * ratio of the two models.
 *
 * Runs in parallel for each bootstrap sample, potentially high memory if the
 * data is large
 *
 * @param prior_null Null model, vector of prior probabilities for the null
 * model, probability data point is in cluster m NOT given responses
 * <ul>
 *   <li>dim 0: for each cluster</li>
 * </ul>
 * @param prob_null Null model, vector of estimated response probabilities for
 * each category, flatten list of matrices. Used as an initial value when
 * fitting onto the bootstrap sample.
 * <ul>
 *   <li>dim 0: for each outcome</li>
 *   <li>dim 1: for each cluster</li>
 *   <li>dim 2: for each category</li>
 * </ul>
 * @param n_cluster_null Null model, number of clusters fitted
 * @param prior_alt Alt model, vector of prior probabilities for the null
 * model, probability data point is in cluster m NOT given responses
 * <ul>
 *   <li>dim 0: for each cluster</li>
 * </ul>
 * @param prob_alt Alt model, vector of estimated response probabilities for
 * each category, flatten list of matrices. Used as an initial value when
 * fitting onto the bootstrap sample.
 * <ul>
 *   <li>dim 0: for each outcome</li>
 *   <li>dim 1: for each cluster</li>
 *   <li>dim 2: for each category</li>
 * </ul>
 * @param n_cluster_alt Alt model, number of clusters fitted
 * @param n_data Number of data points, used to bootstrap this many data
 * points
 * @param n_category Number of categories
 * @param n_outcomes Array of number of outcomes, for each category
 * @param n_bootstrap Number of bootstrap samples to generate
 * @param n_rep Number of initial values to try when fitting on the bootstrap
 * samples
 * @param n_thread Number of threads to use
 * @param max_iter Maximum number of iterations for EM algorithm
 * @param tolerance Tolerance for difference in log likelihood, used for
 * stopping condition
 * @param seed array of integers to seed rng
 * @return Rcpp::NumericVector array of bootstrap log likelihood ratios
 */
// [[Rcpp::export]]
Rcpp::NumericVector BlrtRcpp(Rcpp::NumericVector prior_null,
                             Rcpp::NumericVector prob_null, int n_cluster_null,
                             Rcpp::NumericVector prior_alt,
                             Rcpp::NumericVector prob_alt, int n_cluster_alt,
                             int n_data, int n_category,
                             Rcpp::IntegerVector n_outcomes, int n_bootstrap,
                             int n_rep, int n_thread, int max_iter,
                             double tolerance, Rcpp::IntegerVector seed) {
  int sum_outcomes = 0;  // calculate sum of number of outcomes
  int* n_outcomes_array = n_outcomes.begin();
  for (int i = 0; i < n_category; ++i) {
    sum_outcomes += n_outcomes_array[i];
  }

  // allocate memory for storing log likelihood ratios
  Rcpp::NumericVector ratio_array(n_bootstrap);

  polca_parallel::Blrt blrt(
      prior_null.begin(), prob_null.begin(), n_cluster_null, prior_alt.begin(),
      prob_alt.begin(), n_cluster_alt, n_data, n_category, n_outcomes.begin(),
      sum_outcomes, n_bootstrap, n_rep, n_thread, max_iter, tolerance,
      ratio_array.begin());

  std::seed_seq seed_seq(seed.begin(), seed.end());
  blrt.SetSeed(&seed_seq);
  blrt.Run();

  return ratio_array;
}
