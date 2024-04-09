// poLCAParallel
// Copyright (C) 2024 Sherman Lo

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
#include "standard_error.h"
#include "standard_error_regress.h"

// [[Rcpp::export]]
Rcpp::List StandardErrorRcpp(Rcpp::NumericVector features,
                             Rcpp::IntegerMatrix responses,
                             Rcpp::NumericVector probs,
                             Rcpp::NumericMatrix prior,
                             Rcpp::NumericMatrix posterior, int n_data,
                             int n_feature, int n_category,
                             Rcpp::IntegerVector n_outcomes, int n_cluster) {
  int sum_outcomes = 0;  // calculate sum of number of outcomes
  int* n_outcomes_array = n_outcomes.begin();
  for (int i = 0; i < n_category; ++i) {
    sum_outcomes += n_outcomes_array[i];
  }

  bool is_regress = n_feature > 1;
  int len_regress_coeff = n_feature * (n_cluster - 1);

  // allocate matrices to pass pointers to C++ code
  Rcpp::NumericVector prior_error(n_cluster);
  Rcpp::NumericVector probs_error(sum_outcomes * n_cluster);
  Rcpp::NumericMatrix regress_coeff_error(len_regress_coeff, len_regress_coeff);

  polca_parallel::StandardError* error;

  if (n_feature == 1) {
    error = new polca_parallel::StandardError(
        features.begin(), responses.begin(), probs.begin(), prior.begin(),
        posterior.begin(), n_data, n_feature, n_category, n_outcomes.begin(),
        sum_outcomes, n_cluster, prior_error.begin(), probs_error.begin(),
        regress_coeff_error.begin());
  } else {
    error = new polca_parallel::StandardErrorRegress(
        features.begin(), responses.begin(), probs.begin(), prior.begin(),
        posterior.begin(), n_data, n_feature, n_category, n_outcomes.begin(),
        sum_outcomes, n_cluster, prior_error.begin(), probs_error.begin(),
        regress_coeff_error.begin());
  }

  error->Calc();

  delete error;

  Rcpp::List to_return;
  to_return.push_back(prior_error);
  to_return.push_back(probs_error);
  to_return.push_back(regress_coeff_error);
  return to_return;
}
