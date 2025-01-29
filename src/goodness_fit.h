
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

#ifndef POLCAPARALLEL_SRC_GOODNESS_FIT_H_
#define POLCAPARALLEL_SRC_GOODNESS_FIT_H_

#include <array>
#include <map>
#include <span>
#include <vector>

#include "util.h"

namespace polca_parallel {

/**
 * For storing the observed and expected frequency, used for chi-squared test
 */
struct Frequency {
  std::size_t observed;
  double expected;
};

/**
 * Create a map of unique observations and their count
 *
 * Iterate through all the responses, then find and count unique combinations of
 * outcomes which were observed in the dataset. Results are stored in a map.
 * Observations are presented as a std::vector<int> of length n_category, each
 * element contains an int which represents the resulting outcome for each
 * category.
 *
 * @param responses Design matrix TRANSPOSED of responses, matrix containing
 * outcomes/responses
 * for each category as integers 1, 2, 3, .... The matrix has dimensions
 * <ul>
 *   <li>dim 0: for each category</li>
 *   <li>dim 1: for each data point</li>
 * </ul>
 * @param n_data number of data points
 * @param n_category number of categories
 * @param unique_freq map to write results to, a map with the following
 * <ul>
 *   <li>key: unique integer vector representing the observed responses</li>
 *   <li>value: Frequency with the observed attributed containing the number of
 *   times those unique responses were observed in the dataset</li>
 * </ul>
 */
void GetUniqueObserved(std::span<int> responses, std::size_t n_data,
                       std::size_t n_category,
                       std::map<std::vector<int>, Frequency>& unique_freq);

/**
 * Update a map of observed responses to contain expected frequencies
 *
 * Update the expected frequency in a map of <vector<int>, Frequency> by
 * modifying the value of Frequency.expected with the likelihood of that unique
 * reponse with multiplied by n_data
 *
 * @param prior vector of prior probabilities (probability in a cluster),
 * length n_cluster
 * @param outcome_prob Vector of estimated response probabilities, conditioned
 * on cluster, for each category. A flattened list in the following order
 * <ul>
 *   <li>dim 0: for each outcome</li>
 *   <li>dim 1: for each category</li>
 *   <li>dim 2: for each cluster</li>
 * </ul>
 * @param n_obs number of fully observed data points
 * @param n_outcomes array of integers, number of outcomes for each category,
 * array of length n_category
 * @param n_cluster number of clusters (or classes)
 * @param unique_freq map to update, a map with the following
 * <ul>
 *   <li>key: unique integer vector representing the observed responses</li>
 *   <li>value: Frequency, the expected attributed shall be modified</li>
 * </ul>
 */
void GetExpected(std::span<double> prior, std::span<double> outcome_prob,
                 std::size_t n_obs, NOutcomes n_outcomes, std::size_t n_cluster,
                 std::map<std::vector<int>, Frequency>& unique_freq);

/**
 * Get chi-squared and log-likelihood ratio statistics
 *
 * Calculate and return the chi-squared statistics and log-likelihood ratio
 *
 * @param unique_freq map of unique responses and their frequencies, both
 * observed and expected
 * @param n_data number of data points
 * @return std::array<double, 2> containing log-likelihood ratio and chi-squared
 * statistics
 */
std::array<double, 2> GetStatistics(
    std::map<std::vector<int>, Frequency>& unique_freq, std::size_t n_data);

}  // namespace polca_parallel

#endif  // POLCAPARALLEL_SRC_GOODNESS_FIT_H_
