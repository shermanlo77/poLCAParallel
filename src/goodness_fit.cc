
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

#include <array>
#include <cmath>
#include <cstring>
#include <map>
#include <stdexcept>
#include <vector>

#include "RcppArmadillo.h"
#include "em_algorithm.h"

void polca_parallel::GetUniqueObserved(
    int* responses, std::size_t n_data, std::size_t n_category,
    std::map<std::vector<int>, Frequency>& unique_freq) {
  // iterate through each data point
  std::vector<int> response_i(n_category);
  bool fullyobserved;  // only considered fully observed responses
  for (std::size_t i = 0; i < n_data; ++i) {
    fullyobserved = true;
    std::memcpy(response_i.data(), responses,
                response_i.size() * sizeof(*responses));

    for (std::size_t j = 0; j < response_i.size(); ++j) {
      if (response_i.at(j) == 0) {
        fullyobserved = false;
        break;
      }
    }

    if (fullyobserved) {
      // add or update observation count
      try {
        ++unique_freq.at(response_i).observed;
      } catch (std::out_of_range& e) {
        Frequency frequency;
        frequency.observed = 1;
        unique_freq.insert({response_i, frequency});
      }
    }
    responses += n_category;
  }
}

void polca_parallel::GetExpected(
    double* prior, double* outcome_prob, std::size_t n_data, std::size_t n_obs,
    std::size_t n_category, std::size_t* n_outcomes, std::size_t n_cluster,
    std::map<std::vector<int>, Frequency>& unique_freq) {
  double total_p;
  double* outcome_prob_ptr;
  std::vector<int> response_i;

  // iterate through the map
  for (auto iter = unique_freq.begin(); iter != unique_freq.end(); ++iter) {
    // calculate likelihood
    response_i = iter->first;

    total_p = 0.0;  // to be summed over all clusters
    outcome_prob_ptr = outcome_prob;

    // iterate through each cluster
    for (std::size_t m = 0; m < n_cluster; ++m) {
      // polca_parallel::PosteriorUnnormalize is located in em_algorithm
      total_p += polca_parallel::PosteriorUnnormalize(
          response_i.data(), n_category, n_outcomes, &outcome_prob_ptr,
          prior[m]);
    }

    iter->second.expected = total_p * static_cast<double>(n_obs);
  }
}

std::array<double, 2> polca_parallel::GetStatistics(
    std::map<std::vector<int>, Frequency>& unique_freq, std::size_t n_data) {
  Frequency frequency;

  std::size_t n_unique = unique_freq.size();

  // store statistics for each unique response
  std::vector<double> chi_squared_array(n_unique);
  std::vector<double> ln_l_ratio_array(n_unique);
  std::vector<double> expected_array(n_unique);
  double expected;
  double observed;
  double diff_squared;

  // extract and calculate statistics for each unique response
  std::size_t index = 0;
  for (auto iter = unique_freq.begin(); iter != unique_freq.end(); ++iter) {
    frequency = iter->second;
    expected = frequency.expected;
    observed = static_cast<double>(frequency.observed);

    diff_squared = (expected - observed);
    diff_squared *= diff_squared;

    expected_array[index] = expected;
    chi_squared_array[index] = diff_squared / expected;
    ln_l_ratio_array[index] = observed * std::log(observed / expected);
    ++index;
  }

  double chi_squared;
  double ln_l_ratio;

  // chi squared calculation also use unobserved responses
  chi_squared =
      arma::sum(arma::Row<double>(chi_squared_array.data(), n_unique, false)) +
      (static_cast<double>(n_data) -
       arma::sum(arma::Row<double>(expected_array.data(), n_unique, false)));
  ln_l_ratio = 2.0 * arma::sum(arma::Row<double>(ln_l_ratio_array.data(),
                                                 n_unique, false));

  return {ln_l_ratio, chi_squared};
}
