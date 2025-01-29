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

#ifndef POLCAPARALLEL_SRC_BLRT_H_
#define POLCAPARALLEL_SRC_BLRT_H_

#include <memory>
#include <mutex>
#include <random>
#include <span>
#include <vector>

#include "em_algorithm.h"
#include "em_algorithm_array_serial.h"
#include "util.h"

namespace polca_parallel {

/**
 * Bootstrap likelihood ratio test (polca with no regression and no nan only
 * supported)
 *
 * Does the bootstrap likelihood ratio test. Provide two fitted models, the
 * null model and the alt model which fit a different number of clusters.
 * Bootstrap samples are generated using the null model. The null model and alt
 * model are refitted on the bootstrap samples to investigate the log-likelihood
 * ratio of the two models.
 *
 * Runs in parallel for each bootstrap sample, potentially high memory if the
 * data is large
 *
 * How to use: provide the null and alt models and an array to store the log
 * likelihood ratios to the constructor. Optionally, set the seed using
 * SetSeed(). Call Run() to collect bootstrap samples of the log-likelihood
 * ratios
 */
class Blrt {
 private:
  /**
   * Vector of probabilities, one for each cluster. Probability a data point
   * belongs to each cluster in the null model.
   */
  std::span<double> prior_null_;
  /**
   * Vector of estimated response probabilities, conditioned on cluster, for
   * each category, for the null model, flatten list in the order
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each category</li>
   *   <li>dim 2: for each cluster</li>
   * </ul>
   */
  std::span<double> prob_null_;
  /** Number of clusters fitted onto the null model */
  const std::size_t n_cluster_null_;
  /**
   * Vector of probabilities, one for each cluster. Probability a data point
   * belongs to each cluster in the alt model.
   */
  std::span<double> prior_alt_;
  /**
   * Vector of estimated response probabilities, conditioned on cluster, for
   * each category, for the alt model, flatten list in the order
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each category</li>
   *   <li>dim 2: for each cluster</li>
   * </ul>
   */
  std::span<double> prob_alt_;
  /** Number of clusters fitted onto the alt model */
  const std::size_t n_cluster_alt_;

  /** Number of data points */
  const std::size_t n_data_;
  /** Vector of the number of outcomes for each category */
  NOutcomes n_outcomes_;
  /** Number of bootstrap samples to run */
  const std::size_t n_bootstrap_;
  /** Number of initial values to try */
  const std::size_t n_rep_;
  /** Number of threads */
  const std::size_t n_thread_;
  /** Maximum number of iterations for EM algorithm */
  const unsigned int max_iter_;
  /** To provide to EmAlgorithm */
  const double tolerance_;

  /** What bootstrap sample is being worked on */
  std::size_t n_bootstrap_done_ = 0;
  /** Log-likelihood ratio for each bootstrap sample */
  std::span<double> ratio_array_;

  /** For locking n_bootstrap_done_ */
  std::mutex n_bootstrap_done_lock_;

  /** Array of seeds, for each bootstrap sample*/
  std::vector<unsigned> seed_array_;

 public:
  /**
   * @brief Construct a new Blrt object
   *
   * @param prior_null Null model, vector of prior probabilities for the null
   * model, the probability data point is in cluster m NOT given responses
   * <ul>
   *   <li>dim 0: for each cluster</li>
   * </ul>
   * @param prob_null Null model, vector of estimated response probabilities for
   * each category, flatten list of matrices. Used as an initial value when
   * fitting onto the bootstrap sample.
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each category</li>
   *   <li>dim 2: for each cluster</li>
   * </ul>
   * @param n_cluster_null Null model, number of clusters fitted
   * @param prior_alt Alt model, vector of prior probabilities for the null
   * model, the probability data point is in cluster m NOT given responses
   * <ul>
   *   <li>dim 0: for each cluster</li>
   * </ul>
   * @param prob_alt Alt model, vector of estimated response probabilities for
   * each category, flatten list of matrices. Used as an initial value when
   * fitting onto the bootstrap sample.
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each category</li>
   *   <li>dim 2: for each cluster</li>
   * </ul>
   * @param n_cluster_alt Alt model, number of clusters fitted
   * @param n_data Number of data points, used to bootstrap this many data
   * points
   * @param n_outcomes Array of number of outcomes, for each category
   * @param sum_outcomes Sum of all integers in n_outcomes
   * @param n_bootstrap Number of bootstrap samples to generate
   * @param n_rep Number of initial values to try when fitting on the bootstrap
   * samples
   * @param n_thread Number of threads to use
   * @param max_iter Maximum number of iterations for EM algorithm
   * @param tolerance Tolerance for difference in log-likelihood, used for
   * stopping condition
   * @param ratio_array To store results, array, the log-likelihood ratio for
   * each bootstrap sample
   */
  Blrt(std::span<double> prior_null, std::span<double> prob_null,
       std::size_t n_cluster_null, std::span<double> prior_alt,
       std::span<double> prob_alt, std::size_t n_cluster_alt,
       std::size_t n_data, NOutcomes n_outcomes, std::size_t n_bootstrap,
       std::size_t n_rep, std::size_t n_thread, unsigned int max_iter,
       double tolerance, std::span<double> ratio_array);

  /** Set the rng seed for each bootstrap sample */
  void SetSeed(std::seed_seq& seed);

  /** Do the bootstrap likelihood ratio test, output results to ratio_array_ */
  void Run();

 private:
  /** Do the bootstrap likelihood ratio test, to be run by a thread */
  void RunThread();

  /**
   * Generate a bootstrap sample
   *
   * @param prior Vector of prior probabilities for the null
   * model, probability data point is in cluster m NOT given responses
   * <ul>
   *   <li>dim 0: for each cluster</li>
   * </ul>
   * @param prob Vector of estimated response probabilities for
   * each category, flatten list of matrices. Used as an initial value when
   * fitting onto the bootstrap sample.
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each category</li>
   *   <li>dim 2: for each cluster</li>
   * </ul>
   * @param rng Random number generator
   * @param response To store results, design matrix transpose of responses
   * <ul>
   *   <li>dim 0: for each category</li>
   *   <li>dim 1: for each data point</li>
   * </ul>
   */
  void Bootstrap(std::span<double> prior, std::span<double> prob,
                 std::mt19937_64& rng, std::span<int> response);
};

}  // namespace polca_parallel

#endif  // POLCAPARALLEL_SRC_BLRT_H_
