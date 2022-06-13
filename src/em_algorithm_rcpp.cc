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

#define ARMA_WARN_LEVEL 1

#include <memory>

#include "RcppArmadillo.h"
#include "em_algorithm_array.h"
#include "poLCA.c"

// EM FIT
// Fit using the EM algorithm
// Args:
// features: design matrix of features
// responses: design matrix transpose of responses
// initial_prob: vector of response probabilities for each cluster, flatten
// list of matrices, from the return value of poLCAParallel.vectorize.R
// flatten list of matrices
// dim 0: for each outcome
// dim 1: for each category
// dim 2: for each cluster
// n_data: number of data points
// n_feature: number of features
// n_category: number of categories
// n_outcomes: vector, number of possible responses for each category
// n_cluster: number of clusters, or classes, to fit
// n_rep: number of repetitions
// max_iter: maximum number of iterations for EM algorithm
// tolerance: stop fitting the log likelihood change less than this
// seed: array of integers to seed rng
// Return a list:
// posterior: matrix of posterior probabilities, dim 0: for each data point,
// dim 1: for each cluster
// prior: matrix of prior probabilities, dim 0: for each data point,
// dim 1: for each cluster
// estimated_prob: vector of estimated response probabilities, in the same
// format as initial_prob
// ln_l: log likelihood
// n_iter: number of iterations taken
// eflag: true if the em algorithm has to ever restart
// [[Rcpp::export]]
Rcpp::List EmAlgorithmRcpp(Rcpp::NumericMatrix features,
                           Rcpp::IntegerMatrix responses,
                           Rcpp::NumericVector initial_prob, int n_data,
                           int n_feature, int n_category,
                           Rcpp::IntegerVector n_outcomes, int n_cluster,
                           int n_rep, int n_thread, int max_iter,
                           double tolerance, Rcpp::IntegerVector seed) {
  int sum_outcomes = 0;  // calculate sum of number of outcomes
  int* n_outcomes_array = n_outcomes.begin();
  for (int i = 0; i < n_category; ++i) {
    sum_outcomes += n_outcomes_array[i];
  }

  // allocate matrices to pass pointers to C++ code
  Rcpp::NumericMatrix posterior(n_data, n_cluster);
  Rcpp::NumericMatrix prior(n_data, n_cluster);
  Rcpp::NumericVector estimated_prob(sum_outcomes * n_cluster);
  Rcpp::NumericVector regress_coeff(n_feature * (n_cluster - 1));
  Rcpp::NumericVector ln_l_array(n_rep);
  Rcpp::NumericVector best_initial_prob(sum_outcomes * n_cluster);

  // fit using EM algorithm
  polca_parallel::EmAlgorithmArray* fitter =
      new polca_parallel::EmAlgorithmArray(
          features.begin(), responses.begin(), initial_prob.begin(), n_data,
          n_feature, n_category, n_outcomes.begin(), sum_outcomes, n_cluster,
          n_rep, n_thread, max_iter, tolerance, posterior.begin(),
          prior.begin(), estimated_prob.begin(), regress_coeff.begin(),
          ln_l_array.begin());

  std::seed_seq seed_seq(seed.begin(), seed.end());
  fitter->SetSeed(&seed_seq);
  fitter->set_best_initial_prob(best_initial_prob.begin());

  fitter->Fit();

  int best_rep_index = fitter->get_best_rep_index();
  int n_iter = fitter->get_n_iter();
  bool has_restarted = fitter->get_has_restarted();

  delete fitter;

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
