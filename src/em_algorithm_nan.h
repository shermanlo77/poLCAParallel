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

#ifndef EM_ALGORITHM_NAN_H_
#define EM_ALGORITHM_NAN_H_

#include <vector>

#include "RcppArmadillo.h"
#include "em_algorithm.h"
#include "em_algorithm_regress.h"

namespace polca_parallel {

/**
 * EM algorithm with NaN handling
 *
 * EM algorithm with NaN handling. NaN are encoded as zeros in reponses. The
 * methods responsible for probability estimation are overriden.
 *
 */
class EmAlgorithmNan : public polca_parallel::EmAlgorithm {
 protected:
  /** Temporary variable for summing posteriors for each category */
  std::vector<double> posterior_sum_;

 public:
  /**
   * EM algorithm with NaN handling
   *
   * @copydoc EmAlgorithm::EmAlgorithm
   */
  EmAlgorithmNan(double* features, int* responses, double* initial_prob,
                 int n_data, int n_feature, int n_category, int* n_outcomes,
                 int sum_outcomes, int n_cluster, int max_iter,
                 double tolerance, double* posterior, double* prior,
                 double* estimated_prob, double* regress_coeff);

 protected:
  /**
   * Overridden to handle and ignore reponse zero and modify posterior_sum
   *
   * @copydoc EmAlgorithm::WeightedSumProb
   */
  void WeightedSumProb(int cluster_index) override;

  /**
   * Overridden to estimate probabilities using posterior_sum
   *
   * @copydoc EmAlgorithm::NormalWeightedSumProb
   */
  void NormalWeightedSumProb(int cluster_index) override;
};

class EmAlgorithmNanRegress : public polca_parallel::EmAlgorithmRegress {
 protected:
  std::vector<double> posterior_sum_;

 public:
  EmAlgorithmNanRegress(double* features, int* responses, double* initial_prob,
                        int n_data, int n_feature, int n_category,
                        int* n_outcomes, int sum_outcomes, int n_cluster,
                        int max_iter, double tolerance, double* posterior,
                        double* prior, double* estimated_prob,
                        double* regress_coeff);

 protected:
  /**
   * Overridden to handle and ignore reponse zero and modify posterior_sum
   *
   * @copydoc EmAlgorithm::WeightedSumProb
   */
  void WeightedSumProb(int cluster_index) override;

  /**
   * Overridden to estimate probabilities using posterior_sum
   *
   * @copydoc EmAlgorithm::NormalWeightedSumProb
   */
  void NormalWeightedSumProb(int cluster_index) override;
};

/**
 * Static version of WeightedSumProb and used to override
 * EmAlgorithm::WeightedSumProb()
 *
 * Override so that is ignore response zero and do a cumulative sum of
 * posteriors for each category in posterior_sum
 *
 * @param cluster_index which cluster to consider
 * @param responses Design matrix TRANSPOSED of responses, matrix containing
 * outcomes/responses for each category as integers 1, 2, 3, .... The matrix
 * has dimensions
 * <ul>
 *   <li>dim 0: for each category</li>
 *   <li>dim 1: for each data point</li>
 * </ul>
 * @param n_data Number of data points
 * @param n_category Number of categories
 * @param n_outcomes Vector of number of outcomes for each category
 * @param sum_outcomes Sum of n_outcomes
 * @param posterior Design matrix of posterior probabilities (also called
 * responsibility). It's the probability a data point is in cluster m given
 * responses. The matrix has the following dimensions
 * <ul>
 *   <li>dim 0: for each data</li>
 *   <li>dim 1: for each cluster</li>
 * </ul>
 * @param estimated_prob Modified to contain the weighted sum of posteriors,
 * conditioned on cluster, for each category. A flattened list in the following
 * order
 * <ul>
 *   <li>dim 0: for each outcome</li>
 *   <li>dim 1: for each category</li>
 *   <li>dim 2: for each cluster</li>
 * </ul>
 * @param posterior_sum Modified to store the cumulative posterior sum for each
 * category
 */
void NanWeightedSumProb(int cluster_index, int* responses, int n_data,
                        int n_category, int* n_outcomes, int sum_outcomes,
                        double* posterior, double* estimated_prob,
                        std::vector<double>* posterior_sum);

/**
 * Static version of NormalWeightedSumProb and used to override
 * EmAlgorithm::NormalWeightedSumProb()
 *
 * Override so that it estimate probabilities using posterior_sum
 *
 * @param cluster_index which cluster to consider
 * @param n_category Number of categories
 * @param n_outcomes Vector of number of outcomes for each category
 * @param sum_outcomes Sum of n_outcomes
 * @param posterior_sum Vector which stores the resulting cumulative posterior
 * sum for each category
 * @param estimated_prob Modified to contain the estimated response
 * probabilities, conditioned on cluster, for each category. A flattened list in
 * the following order
 * <ul>
 *   <li>dim 0: for each outcome</li>
 *   <li>dim 1: for each category</li>
 *   <li>dim 2: for each cluster</li>
 * </ul>
 */
void NanNormalWeightedSumProb(int cluster_index, int n_category,
                              int* n_outcomes, int sum_outcomes,
                              std::vector<double>* posterior_sum,
                              double* estimated_prob);

}  // namespace polca_parallel

#endif  // EM_ALGORITHM_NAN_H_
