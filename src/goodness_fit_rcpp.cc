
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

#include <array>
#include <map>
#include <memory>
#include <span>
#include <vector>

#include "RcppArmadillo.h"
#include "goodness_fit.h"
#include "util.h"

/**
 * Function to be exported to R, goodness of fit statistics
 *
 * Get goodness of fit statistics given fitted probabilities
 *
 * @param features: design matrix of features
 * @param prior: vector of prior probabilities, for each cluster
 * outcome_prob: vector of response probabilities for each cluster, flatten
 * list of matrices, from the return value of poLCA.vectorize.R,
 * flatten list of matrices
 * <ul>
 *   <li>dim 0: for each outcome</li>
 *   <li>dim 1: for each category</li>
 *   <li>dim 2: for each cluster</li>
 * </ul>
 * @param n_data number of data points
 * @param n_obs number of fully observed data points
 * @param n_outcomes vector, number of possible responses for each category
 * @param n_cluster number of clusters, or classes, to fit
 * @return a list containing:
 * <ul>
 *   <li>unique_freq_table: data frame of unique responses with their observed
 *   frequency and expected frequency</li>
 *   <li>ln_l_ratio</li>
 *   <li>chi_squared</li>
 * </ul>
 */
// [[Rcpp::export]]
Rcpp::List GoodnessFitRcpp(Rcpp::IntegerMatrix responses,
                           Rcpp::NumericVector prior,
                           Rcpp::NumericVector outcome_prob, std::size_t n_data,
                           std::size_t n_obs,
                           Rcpp::IntegerVector n_outcomes_int,
                           std ::size_t n_cluster) {
  std::vector<std::size_t> n_outcomes_size_t(n_outcomes_int.cbegin(),
                                             n_outcomes_int.cend());
  polca_parallel::NOutcomes n_outcomes(n_outcomes_size_t.data(),
                                       n_outcomes_size_t.size());
  std::size_t n_category = n_outcomes.size();

  // get observed and expected frequencies for each unique response
  // having problems doing static allocation and passing the pointer
  std::unique_ptr<std::map<std::vector<int>, polca_parallel::Frequency>>
      unique_freq = std::make_unique<
          std::map<std::vector<int>, polca_parallel::Frequency>>();
  GetUniqueObserved(std::span<int>(responses.begin(), responses.size()), n_data,
                    n_category, *unique_freq);
  GetExpected(std::span<double>(prior.begin(), prior.size()),
              std::span<double>(outcome_prob.begin(), outcome_prob.size()),
              n_obs, n_outcomes, n_cluster, *unique_freq);
  // get log likelihood ratio and chi squared statistics
  std::array<double, 2> stats = GetStatistics(*unique_freq, n_data);

  // transfer results from std::map unique_freq to a NumericMatrix
  // unique_freq_table
  // last two columns for observed and expected frequency
  std::size_t n_unique = unique_freq->size();
  Rcpp::NumericMatrix unique_freq_table(n_unique, n_category + 2);
  auto unique_freq_ptr = unique_freq_table.begin();

  std::size_t data_index = 0;
  for (auto iter = unique_freq->begin(); iter != unique_freq->end(); ++iter) {
    const std::vector<int>& response_i = iter->first;
    polca_parallel::Frequency frequency = iter->second;

    // copy over response
    for (std::size_t j = 0; j < n_category; ++j) {
      *std::next(unique_freq_ptr, j * n_unique + data_index) = response_i[j];
    }
    // copy over observed and expected frequency
    *std::next(unique_freq_ptr, n_category * n_unique + data_index) =
        static_cast<double>(frequency.observed);
    *std::next(unique_freq_ptr, (n_category + 1) * n_unique + data_index) =
        frequency.expected;
    ++data_index;
  }

  Rcpp::List to_return;
  to_return.push_back(unique_freq_table);
  to_return.push_back(stats[0]);
  to_return.push_back(stats[1]);

  return to_return;
}
