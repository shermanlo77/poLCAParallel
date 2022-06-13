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

#ifndef EM_ALGORITHM_H_
#define EM_ALGORITHM_H_

#include <math.h>

#include <chrono>
#include <random>

#include "RcppArmadillo.h"

namespace polca_parallel {

// EM ALGORITHM
// For fitting using EM algorithm for a given initial value
// Member variables are made public for the sake of convenience so that
// EmAlgorithmArray can access and modify instances of EmAlgorithm
class EmAlgorithm {
 protected:
  double* features_;  // design matrix of features, matrix n_data x n_feature
  // design matrix transpose of responses, matrix n_category x n_data
  int* responses_;
  // vector of initial probabilities for each category and responses,
  // flatten list of matrices
  // dim 0: for each outcome
  // dim 1: for each category
  // dim 2: for each cluster
  double* initial_prob_;
  int n_data_;        // number of data points
  int n_feature_;     // number of features
  int n_category_;    // number of categories
  int* n_outcomes_;   // vector of number of outcomes for each category
  int sum_outcomes_;  // sum of n_outcomes
  int n_cluster_;     // number of clusters (classes in lit) to fit
  int max_iter_;      // maximum number of iterations for EM algorithm
  // tolerance for difference in log likelihood, used for stopping condition
  double tolerance_;
  // design matrix of posterior probabilities (also called responsibility)
  // probability data point is in cluster m given responses
  // matrix, dim 0: for each data, dim 1: for each cluster
  double* posterior_;
  // design matrix of prior probabilities, probability data point is in
  // cluster m NOT given responses
  // after calculations, it shall be in matrix form,
  // dim 0: for each data, dim 1: for each cluster
  // during the start and calculations, it may take on a different form,
  // use the method GetPrior() to get the prior for a data point and cluster
  double* prior_;
  // vector of estimated response probabilities, conditioned on cluster, for
  // each category,
  // flatten list of matrices
  // dim 0: for each outcome
  // dim 1: for each category
  // dim 2: for each cluster
  double* estimated_prob_;
  // vector length n_features_*(n_cluster-1), linear regression coefficient
  // in matrix form, to be multiplied to the features and linked to the
  // prior using softmax
  double* regress_coeff_;
  // vector of INITIAL response probabilities used to get the maximum log
  // likelihood, conditioned on cluster, for each category
  // this member variable is optional, set to NULL if not used
  // flatten list of matrices
  // dim 0: for each outcome
  // dim 1: for each category
  // dim 2: for each cluster
  double* best_initial_prob_ = NULL;

  // log likelihood, updated at each iteration of EM
  double ln_l_ = -INFINITY;
  // vector, for each data point, log likelihood for each data point
  // the log likelihood is the sum
  double* ln_l_array_;
  int n_iter_ = 0;  // number of iterations done right now
  // indicate if it needed to use new initial values during a fit, can happen
  // if a matrix is singular
  bool has_restarted_ = false;
  // seed for random number generator
  unsigned seed_ = std::chrono::system_clock::now().time_since_epoch().count();

 public:
  // for arguments for this constructor, see description of the member
  // variables
  // the following content pointed too shall modified:
  // posterior, prior, estimated_prob, regress_coeff
  EmAlgorithm(double* features, int* responses, double* initial_prob,
              int n_data, int n_feature, int n_category, int* n_outcomes,
              int sum_outcomes, int n_cluster, int max_iter, double tolerance,
              double* posterior, double* prior, double* estimated_prob,
              double* regress_coeff);

  virtual ~EmAlgorithm();

  // fit data to model using EM algorithm
  // data is provided through the constructor
  // important results are stored in the member variables:
  // posterior_
  // prior_
  // estimated_prob_
  // ln_l_array_
  // ln_l_
  // n_iter_
  void Fit();

  // Set where to store initial probabilities (optional)
  void set_best_initial_prob(double* best_initial_prob);

  double get_ln_l();

  int get_n_iter();

  bool get_has_restarted();

  void set_seed(unsigned seed);

 protected:
  // Reset parameters for a re-run
  // Reset the parameters estimated_prob_ with random values
  virtual void Reset(std::mt19937_64* rng,
                     std::uniform_real_distribution<double>* uniform);

  // Initalise prior probabilities
  // Modify the content of prior_ which contains prior probabilities for each
  // cluster
  virtual void InitPrior();

  // Adjust prior return value to matrix format
  virtual void FinalPrior();

  // Get prior during the EM algorithm
  virtual double GetPrior(int data_index, int cluster_index);

  // E step
  // update the posterior probabilities given the prior probabilities and
  // estimated response probabilities
  // modifies the member variables posterior_ and ln_l_array_
  // calculations for the E step also provides the elements for ln_l_array
  //
  // IMPORTANT DEV NOTES: p is iteratively being multiplied, underflow errors
  // may occur if the number of data points is large, say more than 300
  // consider summing over log probabilities  rather than multiplying
  // probabilities
  void EStep();

  // Check if the likelihood is invalid
  virtual bool IsInvalidLikelihood(double ln_l_difference);

  // M Step
  // update the prior probabilities and estimated response probabilities
  // given the posterior probabilities
  // modifies the member variables prior_ and estimated_prob_
  virtual bool MStep();

  // Estimate probability
  // updates and modify the member variable estimated_prob_ using the
  // posterior
  void EstimateProbability();

  // Weighted Sum for Outcome Probability Estimation
  // Calculates sum over data points of a observed outcome, weighted by the
  // posterior. This is done for all outcomes. The member variable
  // estimated_prob_ is updated with the results.
  // Parameters:
  // cluster_index: which cluster to consider
  void WeightedSumProb(int cluster_index);

  // Normalised Weighted Sum for Outcome Porbability Estimation
  // After calling WeightedSumProb, call this to normalise the weighted sum so
  // that the member variable estimated_prob_ contain estimated
  // probabilities for each outcome
  // Can be overridden as the sum of weights can be calculated differently
  // Parameters:
  // cluster_index: which cluster to consider
  virtual void NormalWeightedSumProb(int cluster_index);

  // Normalised Weighted Sum for Outcome Porbability Estimation
  // After calling WeightedSumProb, call this to normalise the weighted sum so
  // that the member variable estimated_prob_ contain estimated
  // probabilities for each outcome
  // Parameters:
  // cluster_index: which cluster to consider
  // normaliser: sum of weights
  void NormalWeightedSumProb(int cluster_index, double normaliser);
};

}  // namespace polca_parallel

#endif  // EM_ALGORITHM_H_
