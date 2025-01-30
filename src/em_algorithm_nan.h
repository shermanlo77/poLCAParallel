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

#ifndef POLCAPARALLEL_SRC_EM_ALGORITHM_NAN_H_
#define POLCAPARALLEL_SRC_EM_ALGORITHM_NAN_H_

#include <vector>

#include "RcppArmadillo.h"
#include "em_algorithm.h"
#include "em_algorithm_regress.h"
#include "util.h"

namespace polca_parallel {

/**
 * Template class for EmAlgorithmNan and EmAlgorithmNanRegress
 *
 * Template class class for EmAlgorithmNan and EmAlgorithmNanRegress, which are
 * subclasses of EmAlgorithm and EmAlgorithmRegress respectively. Methods are
 * overridden so that they can handle response values of zero, the encoding of
 * Nan
 *
 * @tparam Either EmAlgorithm or EmAlgorithmRegress
 */
template <typename T>
class EmAlgorithmNanTemplate : public T {
 protected:
  /** Temporary variable for summing posteriors for each category */
  std::vector<double> posterior_sum_;

 public:
  /**
   * EM algorithm with NaN handling
   *
   * @copydoc EmAlgorithm::EmAlgorithm
   */
  EmAlgorithmNanTemplate(std::span<double> features, std::span<int> responses,
                         std::span<double> initial_prob, std::size_t n_data,
                         std::size_t n_feature, NOutcomes n_outcomes,
                         std::size_t n_cluster, unsigned int max_iter,
                         double tolerance, std::span<double> posterior,
                         std::span<double> prior,
                         std::span<double> estimated_prob,
                         std::span<double> regress_coeff);

 protected:
  /**
   * Overridden to handle and ignore reponse zero and modify posterior_sum
   *
   * @copydoc EmAlgorithm::WeightedSumProb
   */
  void WeightedSumProb(const std::size_t cluster_index) override;

  /**
   * Overridden to estimate probabilities using posterior_sum
   *
   * @copydoc EmAlgorithm::NormalWeightedSumProb
   */
  void NormalWeightedSumProb(const std::size_t cluster_index) override;

  [[nodiscard]] double PosteriorUnnormalize(
      std::span<const int> responses_i, double prior,
      const arma::Col<double>& estimated_prob) const override;
};

/**
 * EM algorithm with NaN handling
 *
 * EM algorithm with NaN handling. NaN are encoded as zeros in reponses. The
 * methods responsible for probability estimation are overriden.
 *
 */
class EmAlgorithmNan : public EmAlgorithmNanTemplate<EmAlgorithm> {
 public:
  /**
   * EM algorithm with NaN handling
   *
   * @copydoc EmAlgorithm::EmAlgorithm
   */
  EmAlgorithmNan(std::span<double> features, std::span<int> responses,
                 std::span<double> initial_prob, std::size_t n_data,
                 std::size_t n_feature, NOutcomes n_outcomes,
                 std::size_t n_cluster, unsigned int max_iter, double tolerance,
                 std::span<double> posterior, std::span<double> prior,
                 std::span<double> estimated_prob,
                 std::span<double> regress_coeff);
};

/**
 * EM algorithm for regression with NaN handling
 *
 * EM algorithm for regression  with NaN handling. NaN are encoded as zeros in
 * reponses. The methods responsible for probability estimation are overriden.
 *
 */
class EmAlgorithmNanRegress
    : public EmAlgorithmNanTemplate<EmAlgorithmRegress> {
 public:
  /**
   * EM algorithm for regression with NaN handling
   *
   * @copydoc EmAlgorithm::EmAlgorithmRegress
   */
  EmAlgorithmNanRegress(std::span<double> features, std::span<int> responses,
                        std::span<double> initial_prob, std::size_t n_data,
                        std::size_t n_feature, NOutcomes n_outcomes,
                        std::size_t n_cluster, unsigned int max_iter,
                        double tolerance, std::span<double> posterior,
                        std::span<double> prior,
                        std::span<double> estimated_prob,
                        std::span<double> regress_coeff);
};

/**
 * Static version of NanWeightedSumProb()
 *
 * Static version of NanWeightedSumProb() and the Nan version of
 * WeightedSumProb(). Used to override EmAlgorithm::WeightedSumProb()
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
 * @param n_outcomes Vector of number of outcomes for each category
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
void NanWeightedSumProb(const std::size_t cluster_index,
                        std::span<const int> responses,
                        std::span<const std::size_t> n_outcomes,
                        const arma::Mat<double>& posterior,
                        arma::Mat<double>& estimated_prob,
                        std::vector<double>& posterior_sum);

/**
 * Static version of NanNormalWeightedSumProb()
 *
 * Static version of NanNormalWeightedSumProb() and the Nan version of
 * NormalWeightedSumProb() Used to override EmAlgorithm::NormalWeightedSumProb()
 *
 * Override so that it estimate probabilities using posterior_sum
 *
 * @param cluster_index which cluster to consider
 * @param n_outcomes Vector of number of outcomes for each category
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
void NanNormalWeightedSumProb(const std::size_t cluster_index,
                              std::span<const std::size_t> n_outcomes,
                              std::vector<double>& posterior_sum,
                              arma::Mat<double>& estimated_prob);

}  // namespace polca_parallel

#endif  // POLCAPARALLEL_SRC_EM_ALGORITHM_NAN_H_
