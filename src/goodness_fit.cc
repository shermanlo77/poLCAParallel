
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

#include "goodness_fit.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <map>
#include <span>
#include <stdexcept>
#include <vector>

#include "RcppArmadillo.h"
#include "em_algorithm.h"
#include "util.h"

void polca_parallel::GetUniqueObserved(
    std::span<const int> responses, std::size_t n_data, std::size_t n_category,
    std::map<std::vector<int>, Frequency>& unique_freq) {
  // iterate through each data point
  auto responses_iter = responses.begin();
  for (std::size_t i = 0; i < n_data; ++i) {
    bool fullyobserved = true;  // only considered fully observed responses
    auto response_span_i = std::span<const int>(responses_iter, n_category);

    for (int response_i_j : response_span_i) {
      if (response_i_j == 0) {
        fullyobserved = false;
        break;
      }
    }

    if (fullyobserved) {
      std::vector<int> response_copy_i(response_span_i.begin(),
                                       response_span_i.end());
      // add or update observation count
      try {
        ++unique_freq.at(response_copy_i).observed;
      } catch (std::out_of_range& e) {
        Frequency frequency;
        frequency.observed = 1;
        unique_freq.insert({response_copy_i, frequency});
      }
    }
    std::advance(responses_iter, n_category);
  }
}

void polca_parallel::GetExpected(
    std::span<const double> prior, std::span<const double> outcome_prob,
    std::size_t n_obs, polca_parallel::NOutcomes n_outcomes,
    std::size_t n_cluster, std::map<std::vector<int>, Frequency>& unique_freq) {
  const arma::Mat<double> outcome_prob_arma(
      const_cast<double*>(outcome_prob.data()), n_outcomes.sum(), n_cluster,
      false, true);

  // iterate through the map
  for (auto iter = unique_freq.begin(); iter != unique_freq.end(); ++iter) {
    // calculate likelihood
    std::vector<int> response_i = iter->first;
    std::span<int> response_i_span(response_i.data(), response_i.size());

    double total_p = 0.0;  // to be summed over all clusters

    // iterate through each cluster
    for (std::size_t m = 0; m < n_cluster; ++m) {
      auto outcome_prob_col = outcome_prob_arma.unsafe_col(m);
      // polca_parallel::PosteriorUnnormalize is located in em_algorithm
      total_p += polca_parallel::PosteriorUnnormalize(
          response_i_span, n_outcomes, outcome_prob_col, prior[m]);
    }

    iter->second.expected = total_p * static_cast<double>(n_obs);
  }
}

std::array<double, 2> polca_parallel::GetStatistics(
    const std::map<std::vector<int>, Frequency>& unique_freq,
    std::size_t n_data) {
  std::size_t n_unique = unique_freq.size();
  // store statistics for each unique response
  arma::Row<double> chi_squared_array(n_unique);
  arma::Row<double> ln_l_ratio_array(n_unique);
  arma::Row<double> expected_array(n_unique);

  // extract and calculate statistics for each unique response
  std::size_t index = 0;
  for (auto iter = unique_freq.cbegin(); iter != unique_freq.cend(); ++iter) {
    Frequency frequency = iter->second;
    double expected = frequency.expected;
    double observed = static_cast<double>(frequency.observed);

    double diff_squared = (expected - observed);
    diff_squared *= diff_squared;

    expected_array[index] = expected;
    chi_squared_array[index] = diff_squared / expected;
    ln_l_ratio_array[index] = observed * std::log(observed / expected);
    ++index;
  }
  // chi squared calculation also use unobserved responses
  double chi_squared =
      arma::sum(chi_squared_array) +
      (static_cast<double>(n_data) - arma::sum(expected_array));
  double ln_l_ratio = 2.0 * arma::sum(ln_l_ratio_array);

  return {ln_l_ratio, chi_squared};
}
