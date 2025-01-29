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

#include <random>
#include <span>
#include <vector>

#include "RcppArmadillo.h"
#include "em_algorithm.h"
#include "em_algorithm_array.h"
#include "em_algorithm_nan.h"
#include "poLCA.c"
#include "util.h"

/**
 * Function to be exported to R, fit using the EM algorithm
 *
 * @param features: Design matrix of features, matrix with dimensions
 * <ul>
 *   <li>dim 0: for each data point</li>
 *   <li>dim 1: for each feature</li>
 * </ul>
 * @param responses Design matrix TRANSPOSED of responses, matrix containing
 * outcomes/responses for each category as integers 1, 2, 3, .... The matrix
 * has dimensions
 * <ul>
 *   <li>dim 0: for each category</li>
 *   <li>dim 1: for each data point</li>
 * </ul>
 * @param initial_prob Vector of initial response probabilities for each
 * outcome, category and cluster. Can be the return value of
 * poLCAParallel.vectorize.R. Flatten list in the following order
 * <ul>
 *   <li>dim 0: for each outcome</li>
 *   <li>dim 1: for each category</li>
 *   <li>dim 2: for each cluster</li>
 * </ul>
 * @param n_data number of data points
 * @param n_feature number of features
 * @param n_outcomes: vector, number of possible outcomes for each category
 * @param n_cluster: number of clusters, or classes, to fit
 * @param n_rep: number of repetitions
 * @param max_iter: maximum number of iterations for EM algorithm
 * @param tolerance: stop fitting when the change in log likelihood is less than
 * this
 * @param seed: array of integers to seed rng
 * @return a list containing
 * <ul>
 *   <li>[[1]]: matrix of posterior probabilities, dim 0: for each data
 *   point, dim 1: for each cluster</li>
 *   <li>[[2]]: matrix of prior probabilities, dim 0: for each data point,
 *   dim 1: for each cluster</li>
 *   <li>[[3]]: vector of estimated response probabilities, in the same
 *   format as initial_prob</li>
 *   <li>[[4]]: vector of regression coefficients</li>
 *   <li>[[5]]: log likelihood</li>
 *   <li>[[6]]: integer, which repetition achived the best fit</li>
 *   <li>[[7]]: number of iterations taken</li>
 *   <li>[[8]]: vector of initial response probabilities, in the same
 *   format as initial_prob, which achieved the best fit</li>
 *   <li>[[9]]: true if the em algorithm has to ever restart</li>
 * </ul>
 */
// [[Rcpp::export]]
Rcpp::List EmAlgorithmRcpp(Rcpp::NumericMatrix features,
                           Rcpp::IntegerMatrix responses,
                           Rcpp::NumericVector initial_prob, std::size_t n_data,
                           std::size_t n_feature,
                           Rcpp::IntegerVector n_outcomes_int,
                           std::size_t n_cluster, std::size_t n_rep, bool na_rm,
                           std::size_t n_thread, unsigned int max_iter,
                           double tolerance, Rcpp::IntegerVector seed) {
  std::vector<std::size_t> n_outcomes_size_t(n_outcomes_int.begin(),
                                             n_outcomes_int.end());
  polca_parallel::NOutcomes n_outcomes(n_outcomes_size_t.data(),
                                       n_outcomes_size_t.size());

  // allocate matrices to pass pointers to C++ code
  Rcpp::NumericMatrix posterior(n_data, n_cluster);
  Rcpp::NumericMatrix prior(n_data, n_cluster);
  Rcpp::NumericVector estimated_prob(n_outcomes.sum() * n_cluster);
  Rcpp::NumericVector regress_coeff(n_feature * (n_cluster - 1));
  Rcpp::NumericVector ln_l_array(n_rep);
  Rcpp::NumericVector best_initial_prob(n_outcomes.sum() * n_cluster);

  // fit using EM algorithm
  polca_parallel::EmAlgorithmArray fitter(
      std::span<double>(features.begin(), features.size()),
      std::span<int>(responses.begin(), responses.size()),
      std::span<double>(initial_prob.begin(), initial_prob.size()), n_data,
      n_feature, n_outcomes, n_cluster, n_rep, n_thread, max_iter, tolerance,
      std::span<double>(posterior.begin(), posterior.size()),
      std::span<double>(prior.begin(), prior.size()),
      std::span<double>(estimated_prob.begin(), estimated_prob.size()),
      std::span<double>(regress_coeff.begin(), regress_coeff.size()));

  std::seed_seq seed_seq(seed.begin(), seed.end());
  fitter.SetSeed(seed_seq);
  fitter.set_best_initial_prob(
      std::span<double>(best_initial_prob.begin(), best_initial_prob.size()));
  fitter.set_ln_l_array(
      std::span<double>(ln_l_array.begin(), ln_l_array.size()));

  bool is_regress = n_feature > 1;
  if (is_regress) {
    if (na_rm) {
      fitter.Fit<polca_parallel::EmAlgorithmRegress>();
    } else {
      fitter.Fit<polca_parallel::EmAlgorithmNanRegress>();
    }
  } else {
    if (na_rm) {
      fitter.Fit<polca_parallel::EmAlgorithm>();
    } else {
      fitter.Fit<polca_parallel::EmAlgorithmNan>();
    }
  }

  std::size_t best_rep_index = fitter.get_best_rep_index();
  unsigned int n_iter = fitter.get_n_iter();
  bool has_restarted = fitter.get_has_restarted();

  Rcpp::List to_return;
  to_return.push_back(posterior);
  to_return.push_back(prior);
  to_return.push_back(estimated_prob);
  to_return.push_back(regress_coeff);
  to_return.push_back(ln_l_array);
  to_return.push_back(best_rep_index + 1);
  to_return.push_back(n_iter);
  to_return.push_back(best_initial_prob);
  to_return.push_back(has_restarted);
  return to_return;
}

// Original author's likelihood
// (for some reason, cannot get the original C code to be regonised by R)
// [[Rcpp::export]]
Rcpp::NumericVector ylik(Rcpp::NumericVector probs, Rcpp::IntegerVector y,
                         int obs, int items, Rcpp::IntegerVector numChoices,
                         int classes) {
  Rcpp::NumericVector lik(obs * classes);
  ylik(probs.begin(), y.begin(), &obs, &items, numChoices.begin(), &classes,
       lik.begin());
  return lik;
}
