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

#include <span>
#include <vector>

#include "RcppArmadillo.h"
#include "blrt.h"
#include "util.h"

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
Rcpp::NumericVector BlrtRcpp(
    Rcpp::NumericVector prior_null, Rcpp::NumericVector prob_null,
    std::size_t n_cluster_null, Rcpp::NumericVector prior_alt,
    Rcpp::NumericVector prob_alt, std::size_t n_cluster_alt, std::size_t n_data,
    std::size_t n_category, Rcpp::IntegerVector n_outcomes_int,
    std::size_t n_bootstrap, std::size_t n_rep, std::size_t n_thread,
    unsigned int max_iter, double tolerance, Rcpp::IntegerVector seed) {
  std::vector<std::size_t> n_outcomes_size_t(n_outcomes_int.begin(),
                                             n_outcomes_int.end());
  polca_parallel::NOutcomes n_outcomes(n_outcomes_size_t.data(),
                                       n_outcomes_size_t.size());

  // allocate memory for storing log likelihood ratios
  Rcpp::NumericVector ratio_array(n_bootstrap);

  polca_parallel::Blrt blrt(
      std::span<double>(prior_null.begin(), prior_null.size()),
      std::span<double>(prob_null.begin(), prob_null.size()), n_cluster_null,
      std::span<double>(prior_alt.begin(), prior_alt.size()),
      std::span<double>(prob_alt.begin(), prob_alt.size()), n_cluster_alt,
      n_data, n_category, n_outcomes, n_bootstrap, n_rep, n_thread, max_iter,
      tolerance, std::span<double>(ratio_array.begin(), ratio_array.size()));

  std::seed_seq seed_seq(seed.begin(), seed.end());
  blrt.SetSeed(seed_seq);
  blrt.Run();

  return ratio_array;
}
