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

#ifndef BLRT_H_
#define BLRT_H_

#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include "em_algorithm.h"
#include "em_algorithm_array_serial.h"

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
  double* prior_null_;
  /**
   * Vector of estimated response probabilities, conditioned on cluster, for
   * each category, for the null model, flatten list in the order
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each category</li>
   *   <li>dim 2: for each cluster</li>
   * </ul>
   */
  double* prob_null_;
  /** Number of clusters fitted onto the null model */
  int n_cluster_null_;
  /**
   * Vector of probabilities, one for each cluster. Probability a data point
   * belongs to each cluster in the alt model.
   */
  double* prior_alt_;
  /**
   * Vector of estimated response probabilities, conditioned on cluster, for
   * each category, for the alt model, flatten list in the order
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each category</li>
   *   <li>dim 2: for each cluster</li>
   * </ul>
   */
  double* prob_alt_;
  /** Number of clusters fitted onto the alt model */
  int n_cluster_alt_;

  /** Number of data points */
  int n_data_;
  /** Number of categories */
  int n_category_;
  /** Vector of the number of outcomes for each category */
  int* n_outcomes_;
  /** Sum of n_outcomes */
  int sum_outcomes_;
  /** Number of bootstrap samples to run */
  int n_bootstrap_;
  /** Number of initial values to try */
  int n_rep_;
  /** Number of threads */
  int n_thread_;
  /** Maximum number of iterations for EM algorithm */
  int max_iter_;
  /** To provide to EmAlgorithm */
  double tolerance_;

  /** What bootstrap sample is being worked on */
  int n_bootstrap_done_ = 0;
  /** Log-likelihood ratio for each bootstrap sample */
  double* ratio_array_;

  /** For locking n_bootstrap_done_ */
  std::unique_ptr<std::mutex> n_bootstrap_done_lock_;

  /** Array of seeds, for each bootstrap sample*/
  std::unique_ptr<unsigned[]> seed_array_;

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
   *   <li>dim 1: for each cluster</li>
   *   <li>dim 2: for each category</li>
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
   *   <li>dim 1: for each cluster</li>
   *   <li>dim 2: for each category</li>
   * </ul>
   * @param n_cluster_alt Alt model, number of clusters fitted
   * @param n_data Number of data points, used to bootstrap this many data
   * points
   * @param n_category Number of categories
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
  Blrt(double* prior_null, double* prob_null, int n_cluster_null,
       double* prior_alt, double* prob_alt, int n_cluster_alt, int n_data,
       int n_category, int* n_outcomes, int sum_outcomes, int n_bootstrap,
       int n_rep, int n_thread, int max_iter, double tolerance,
       double* ratio_array);

  /** Set the rng seed for each bootstrap sample */
  void SetSeed(std::seed_seq* seed);

  /** Do the bootstrap likelihood ratio test, output results to ratio_array_ */
  void Run();

 private:
  /** Do the bootstrap likelihood ratio test, to be run by a thread */
  void RunThread();

  /**
   * Bootstrap data
   * @param prior vector of probabilities, one for each cluster. Probability a
   * data point belongs to each cluster
   * @param prob Vector of estimated response probabilities, conditioned on
   * the cluster, for each category, flatten list in the order
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each category</li>
   *   <li>dim 2: for each cluster</li>
   * </ul>
   * @param n_cluster number of clusters, length of prior
   * @param rng random number generator
   * @param response output, bootstrapped responses/data, design matrix
   * TRANSPOSED of responses, matrix with dimensions
   * <ul>
   *   <li>dim 0: for each category</li>
   *   <li>dim 1: for each data point</li>
   * </ul>
   */
  void Bootstrap(double* prior, double* prob, int n_cluster,
                 std::mt19937_64* rng, int* response);
};

}  // namespace polca_parallel

#endif  // BLRT_H_
