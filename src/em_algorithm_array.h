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

#ifndef EM_ALGORITHM_ARRAY_H_
#define EM_ALGORITHM_ARRAY_H_

#include <memory>
#include <mutex>
#include <random>
#include <thread>

#include "em_algorithm.h"
#include "em_algorithm_regress.h"

namespace polca_parallel {

/**
 * For using EM algorithm with multiple inital probabilities, use to try to find
 * the global maximum. Each thread runs a repetition
 */
class EmAlgorithmArray {
 private:
  /** Features to provide EmAlgorithm with */
  double* features_;
  /** Reponses to provide EmAlgorithm with */
  int* responses_;
  /** Number of data points */
  int n_data_;
  /** Number of features */
  int n_feature_;
  /** Number of categories */
  int n_category_;
  /** Vector of number of outcomes for each category */
  int* n_outcomes_;
  /** Sum of n_outcomes */
  int sum_outcomes_;
  /** Number of clusters (classes in literature) to fit */
  int n_cluster_;
  /** Maximum number of iterations for EM algorithm */
  int max_iter_;
  /** To provide to EmAlgorithm */
  double tolerance_;
  /** To store posterior results */
  double* posterior_;
  /** To store prior results */
  double* prior_;
  /** To store estimated prob results */
  double* estimated_prob_;
  /** To store regression coefficient results */
  double* regress_coeff_;
  /** Optional, to store initial prob to obtain max likelihood */
  double* best_initial_prob_ = NULL;

  /** Number of initial values to try */
  int n_rep_;
  /** The best log likelihood found so far */
  double optimal_ln_l_;
  /** Number of iterations optimal fitter has done */
  int n_iter_;
  /** True if the EM algorithm has to ever restart */
  bool has_restarted_ = false;
  /**
   * Array of initial probabilities, each reptition uses sum_outcomes*n_cluster
   * probabilities
   */
  double* initial_prob_;
  /** Which initial value is being worked on */
  int n_rep_done_;
  /** Maximum log likelihood for each reptition */
  double* ln_l_array_ = NULL;
  /** Index of which inital value has the best log likelihood */
  int best_rep_index_;
  /** Number of threads */
  int n_thread_;

  /** For locking n_rep_done_ */
  std::mutex* n_rep_done_lock_;
  /** For locking optimal_ln_l_, best_rep_index_, n_iter_ and has_restarted_ */
  std::mutex* results_lock_;

 protected:
  /**
   * Array of seeds, for each repetition, used to seed each repetition, only
   * used if a run fails and needs to generate new initial values
   */
  std::unique_ptr<unsigned[]> seed_array_ = NULL;

 public:
  /**
   * Construct a new Em Algorithm Array object
   *
   * @param features Design matrix of features, matrix n_data x n_feature
   * @param responses Design matrix transpose of responses, matrix n_category x
   * n_data
   * @param initial_prob Vector of initial probabilities for each outcome, for
   * each category, for each cluster and for each repetition, flatten list of
   * matrices
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each category</li>
   *   <li>dim 2: for each cluster</li>
   *   <li>dim 3: for each repetition</li>
   * </ul>
   * @param n_data Number of data points
   * @param n_feature Number of features
   * @param n_category Number of categories
   * @param n_outcomes Array of number of outcomes, for each category
   * @param sum_outcomes Sum of all integers in n_outcomes
   * @param n_cluster Number of clusters to fit
   * @param n_rep Number of repetitions to do, length of dim 3 for initial_prob
   * @param n_thread Number of threads to use
   * @param max_iter Maximum number of iterations for EM algorithm
   * @param tolerance Tolerance for difference in log likelihood, used for
   * stopping condition
   * @param posterior To store results, design matrix of posterior probabilities
   * (also called responsibility), probability data point is in cluster
   * m given responses, matrix
   * <ul>
   *   <li>dim 0: for each data</li>
   *   <li>dim 1: for each cluster</li>
   * </ul>
   * @param prior To store results, design matrix of prior probabilities,
   * probability data point is in cluster m NOT given responses
   * <ul>
   *   <li>dim 0: for each data</li>
   *   <li>dim 1: for each cluster</li>
   * </ul>
   * @param estimated_prob To store results, vector of estimated response
   * probabilities for each category,
   * flatten list of matrices
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each cluster</li>
   *   <li>dim 2: for each category</li>
   * </ul>
   * @param regress_coeff To store results, vector length
   * n_features_*(n_cluster-1), linear regression coefficient in matrix
   * form, to be multiplied to the features and linked to the prior
   * using softmax
   */
  EmAlgorithmArray(double* features, int* responses, double* initial_prob,
                   int n_data, int n_feature, int n_category, int* n_outcomes,
                   int sum_outcomes, int n_cluster, int n_rep, int n_thread,
                   int max_iter, double tolerance, double* posterior,
                   double* prior, double* estimated_prob,
                   double* regress_coeff);

  ~EmAlgorithmArray();

  /**
   * Fit (in parallel) using EM algorithm. To be called right after
   * construction or after setting optional settings
   */
  void Fit();

  /** Set the member variable seed_array_ with seeds for each repetition */
  virtual void SetSeed(std::seed_seq* seed);

  /**
   * Set where to store initial probabilities (optional)
   *
   * @param best_initial_prob best_initial_prob to provide to EmAlgorithm
   * objects
   */
  void set_best_initial_prob(double* best_initial_prob);

  /** Set where to store the log likelihood for each iteration */
  void set_ln_l_array(double* ln_l_array);

  /** Get the index of the repetition with the highest log likelihood */
  int get_best_rep_index();

  /** Get the best log likelihood from all repetitions */
  double get_optimal_ln_l();

  /**
   * Get the number of EM iterations done for the repetition with the highest
   * log likelihood
   */
  int get_n_iter();

  /**
   * Return if at least one repetition had to restart, eg due to a singular
   * matrix
   */
  bool get_has_restarted();

 protected:
  /** Set the rng of a EmAlgorithm object given the rep_index */
  virtual void SetFitterRng(polca_parallel::EmAlgorithm* fitter, int rep_index);

  /**
   * Move ownership of a rng from a fitter back to here
   */
  virtual void MoveRngBackFromFitter(polca_parallel::EmAlgorithm* fitter);

 private:
  /**
   * A thread calls this. For each initial probability, fit using EM algorithm
   */
  void FitThread();
};

}  // namespace polca_parallel

#endif  // EM_ALGORITHM_ARRAY_H_
