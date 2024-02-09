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

#ifndef GOODNESS_FIT_H_
#define GOODNESS_FIT_H_

#include <math.h>

#include <array>
#include <map>
#include <stdexcept>
#include <vector>

#include "RcppArmadillo.h"

#include "em_algorithm.h"

namespace polca_parallel {

/**
 * For storing the observed and expected frequency, used for chi squared test
 */
struct Frequency {
  int observed;
  double expected;
};

/**
 * Iterate through all the responses, then find and count unique combinations of
 * outcomes which were observed in the dataset. Results are stored in a map.
 *
 * @param responses design matrix transpose of responses, matrix n_category x
 * n_data
 * @param n_data number of data points
 * @param n_category number of categories
 * @param unique_freq map to write results to, key: unique integer vectors,
 * value: Frequency containing the number of times it was observed in the
 * dataset
 */
void GetUniqueObserved(int* responses, int n_data, int n_category,
                       std::map<std::vector<int>, Frequency>* unique_freq);

/**
 * Update the expected frequency in a map of vector<int>:Frequency by modifying
 * the value of Frequency.expected with the likelihood of that unique
 * reponse with multiplied by n_data
 *
 * @param responses design matrix transpose of responses, matrix n_category x
 * n_data
 * @param prior vector of prior probabilities (probability in a cluster),
 * length n_cluster
 * @param outcome_prob array of outcome probabilities for each category and
 * cluster, flatten list of matrices
 * <ul>
 *   <li>dim 0: for each outcome</li>
 *   <li>dim 1: for each category</li>
 *   <li>dim 2: for each model</li>
 * </ul>
 * @param n_data number of data points
 * @param n_category number of categories
 * @param n_outcomes array of integers, number of outcomes for each category,
 * array of length n_category
 * @param n_cluster number of clusters (or classes)
 * @param unique_freq map to modify with results to, key: unique integer
 * vectors, value: Frequency with the expected frequency modified
 */
void GetExpected(int* responses, double* prior, double* outcome_prob,
                 int n_data, int n_category, int* n_outcomes, int n_cluster,
                 std::map<std::vector<int>, Frequency>* unique_freq);

/**
 * Get chi-squared and log likelihood ratio statistics
 *
 * Calculate and return the chi-squared statistics and log likelihood ratio
 *
 * @param unique_freq map of unique responses and their frequencies, both
 * observed and expected
 * @param n_data number of data points
 * @return std::array<double, 2> containing log likelihood ratio and chi squared
 * statistics
 */
std::array<double, 2> GetStatistics(
    std::map<std::vector<int>, Frequency>* unique_freq, int n_data);

}  // namespace polca_parallel

#endif  // GOODNESS_FIT_H_
