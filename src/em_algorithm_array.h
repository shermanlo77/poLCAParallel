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

// EM ALGORITHM ARRAY
// For using EM algorithm for multiple inital probabilities, use to find the
// global maximum
// Each thread runs a repetition
class EmAlgorithmArray {
 private:
  double *features_;        // features to provide EmAlgorithm with
  int *responses_;          // reponses to provide EmAlgorithm with
  int n_data_;              // number of data points
  int n_feature_;           // number of features
  int n_category_;          // number of categories
  int *n_outcomes_;         // vector of number of outcomes for each category
  int sum_outcomes_;        // sum of n_outcomes
  int n_cluster_;           // number of clusters (classes in lit) to fit
  int max_iter_;            // maximum number of iterations for EM algorithm
  double tolerance_;        // to provide to EmAlgorithm
  double *posterior_;       // to store posterior results
  double *prior_;           // to store prior results
  double *estimated_prob_;  // to store estimated prob results
  double *regress_coeff_;   // to store regression coefficient results
  // optional, to store initial prob to obtain max likelihood
  double *best_initial_prob_ = NULL;

  int n_rep_;  // number of initial values to tries
  // the best log likelihood found so far
  double optimal_ln_l_;
  // number of iterations optimal fitter has done
  int n_iter_;
  // true if the EM algorithm has to ever restart
  bool has_restarted_ = false;
  // array of initial probabilities
  // each reptition uses sum_outcomes*n_cluster probabilities
  double *initial_prob_;
  int n_rep_done_;      // which initial value is being worked on
  double *ln_l_array_;  // maximum log likelihood for each reptition
  // index of which inital value has the best log likelihood
  int best_rep_index_;
  int n_thread_;  // number of threads

  // array of seeds, for each repetition
  std::unique_ptr<unsigned[]> seed_array_ = NULL;

  std::mutex *n_rep_done_lock_;  // for locking n_rep_done_
  // for locking optimal_ln_l_, best_rep_index_, n_iter_ and has_restarted_
  std::mutex *results_lock_;

 public:
  // CONSTRUCTOR
  // Parameters:
  // features: design matrix of features, matrix n_data x n_feature
  // responses: design matrix transpose of responses,
  // matrix n_category x n_data
  // initial_prob: vector of initial probabilities for each outcome,
  // for each category, for each cluster and for each repetition
  // flatten list of matrices
  // dim 0: for each outcome
  // dim 1: for each category
  // dim 2: for each cluster
  // dim 3: for each repetition
  // n_data: number of data points
  // n_feature: number of features
  // n_category: number of categories
  // n_outcomes: array of number of outcomes, for each category
  // sum_outcomes: sum of all integers in n_outcomes
  // n_cluster: number of clusters to fit
  // n_rep: number of repetitions to do, length of dim 3 for initial_prob
  // n_thread: number of threads to use
  // max_iter: maximum number of iterations for EM algorithm
  // tolerance: tolerance for difference in log likelihood, used for
  // stopping condition
  // posterior: to store results, design matrix of posterior probabilities
  // (also called responsibility), probability data point is in cluster
  // m given responses
  // matrix, dim 0: for each data, dim 1: for each cluster
  // prior: to store results, design matrix of prior probabilities,
  // probability data point is in cluster m NOT given responses
  // dim 0: for each data, dim 1: for each cluster
  // estimated_prob: to store results, vector of estimated response
  // probabilities for each category,
  // flatten list of matrices
  // dim 0: for each outcome
  // dim 1: for each cluster
  // dim 2: for each category
  // regress_coeff: to store results, vector length
  // n_features_*(n_cluster-1), linear regression coefficient in matrix
  // form, to be multiplied to the features and linked to the prior
  // using softmax
  // ln_l_array: to store results, vector, maxmimum log likelihood for each
  // repetition
  EmAlgorithmArray(double *features, int *responses, double *initial_prob,
                   int n_data, int n_feature, int n_category, int *n_outcomes,
                   int sum_outcomes, int n_cluster, int n_rep, int n_thread,
                   int max_iter, double tolerance, double *posterior,
                   double *prior, double *estimated_prob, double *regress_coeff,
                   double *ln_l_array);

  ~EmAlgorithmArray();

  // FIT USING EM (in parallel)
  // To be called right after construction
  void Fit();

  // Set Seed
  // Set the member variable seed_array_ with seeds for each repetition
  void SetSeed(std::seed_seq *seed);

  // Set where to store initial probabilities (optional)
  void set_best_initial_prob(double *best_initial_prob);

  int get_best_rep_index();

  int get_n_iter();

  bool get_has_restarted();

 private:
  // FIT THREAD
  // To be run by a thread(s)
  // For each initial probability, fit using EM algorithm
  void FitThread();
};

}  // namespace polca_parallel

#endif  // EM_ALGORITHM_ARRAY_H_
