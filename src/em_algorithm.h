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
#include <memory>
#include <random>

#include "RcppArmadillo.h"

namespace polca_parallel {

/**
 * Use the log sum of probabilities if the number of categories is equal or
 * greater than this
 **/
extern const int N_CATEGORY_SUMLOG;

/**
 * For fitting poLCA using EM algorithm for a given initial value.
 *
 * How to use:
 * <ul>
 *   <li>
 *     Pass the data, initial probabilities and other parameters to the
 *     constructor. Also in the constructor, pass an array to store the
 *     posterior and prior probabilities (for each cluster) and the estimated
 *     response probabilities
 *   </li>
 *   <li>
 *     Call optional methods such as set_best_initial_prob(), set_seed()
 *     and/or set_rng()
 *   </li>
 *   <li>
 *      Call Fit() to fit using the EM algorithm, results are stored in the
 *      provided arrays. The EM algorithm restarts with random initial values
 *      should it fail for some reason (more commonly in the regression model)
 *   </li>
 *   <li>
 *     Extract optional results using the methods get_ln_l(), get_n_iter()
 *     and/or get_has_restarted()
 *   </li>
 * </ul>
 */
class EmAlgorithm {
 protected:
  /** Design matrix of features, matrix n_data x n_feature */
  double* features_;
  /** Design matrix transpose of responses, matrix n_category x n_data */
  int* responses_;
  /**
   * Vector of initial probabilities for each category and responses,
   * flatten list of matrices
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each category</li>
   *   <li>dim 2: for each cluster</li>
   * </ul>
   */
  double* initial_prob_;
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
  /** Number of clusters to fit */
  int n_cluster_;
  /** Maximum number of iterations for EM algorithm */
  int max_iter_;
  /** Tolerance for difference in log likelihood, used for stopping condition */
  double tolerance_;
  /**
   * Design matrix of posterior probabilities (also called responsibility)
   * probability data point is in cluster m given responses
   * matrix
   * <ul>
   *   <li>dim 0: for each data</li>
   *   <li>dim 1: for each cluster</li>
   * </ul>
   */
  double* posterior_;
  /**
   * Design matrix of prior probabilities, probability data point is in
   * cluster m NOT given responses after calculations, it shall be in matrix
   * form with dimensions
   * <ul>
   *   <li>dim 0: for each data</li>
   *   <li>dim 1: for each cluster</li>
   * </ul>
   * During the start and calculations, it may take on a different form,
   * use the method GetPrior() to get the prior for a data point and cluster
   */
  double* prior_;
  /**
   * Vector of estimated response probabilities, conditioned on cluster, for
   * each category, flatten list of matrices
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each category</li>
   *   <li>dim 2: for each cluster</li>
   * </ul>
   */
  double* estimated_prob_;
  /**
   * Vector length n_features_*(n_cluster-1), linear regression coefficient
   * in matrix form, to be multiplied to the features and linked to the
   * prior using softmax
   */
  double* regress_coeff_;
  /**
   * Vector of INITIAL response probabilities used to get the maximum log
   * likelihood, this member variable is optional, set to NULL if not used
   * flatten list of matrices
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each category</li>
   *   <li>dim 2: for each cluster</li>
   * </ul>
   */
  double* best_initial_prob_ = NULL;

  /** Log likelihood, updated at each iteration of EM */
  double ln_l_ = -INFINITY;
  /**Vector, for each data point, log likelihood for each data point,
   * the total log likelihood is the sum
   */
  double* ln_l_array_;
  /** Number of iterations done right now */
  int n_iter_ = 0;
  /**Indicate if it needed to use new initial values during a fit, can happen
   * if a matrix is singular
   */
  bool has_restarted_ = false;
  /** Random number generator for generating new initial values if fail*/
  std::unique_ptr<std::mt19937_64> rng_;

 public:
  /**
   * Construct a new EM algorithm object
   *
   * Please see the description of the member variables for further information.
   * The following content pointed to shall be modified:
   * <ul>
   *   <li>posterior</li>
   *   <li>prior</li>
   *   <li>estimated_prob</li>
   * </ul>
   *
   * @param features Not used and ignored
   * @param responses Design matrix transpose of responses, one based
   * <ul>
   *   <li>dim 0: for each data point, length n_data</li>
   *   <li>dim 1: for each category, length n_category</li>
   * </ul>
   * @param initial_prob Vector of initial probabilities for each category and
   * responses, flatten list of matrices
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each category</li>
   *   <li>dim 2: for each cluster</li>
   * </ul>
   * @param n_data Number of data points
   * @param n_feature Number of features
   * @param n_category Number of categories
   * @param n_outcomes Vector of number of outcomes for each category
   * @param sum_outcomes Sum of n_outcomes
   * @param n_cluster Number of clusters to fit
   * @param max_iter Maximum number of iterations for EM algorithm
   * @param tolerance Tolerance for difference in log likelihood, used for
   * stopping condition
   * @param posterior Design matrix of posterior probabilities (also called
   * responsibility) probability data point is in cluster m given responses
   * matrix
   * <ul>
   *   <li>dim 0: for each data</li>
   *   <li>dim 1: for each cluster</li>
   * </ul>
   * @param prior Design matrix of prior probabilities, probability data point
   * is in cluster m NOT given responses after calculations, it shall be in
   * matrix form with dimensions
   * <ul>
   *   <li>dim 0: for each data</li>
   *   <li>dim 1: for each cluster</li>
   * </ul>
   * During the start and calculations, it may take on a different form,
   * use the method GetPrior() to get the prior for a data point and cluster
   * @param estimated_prob Vector of estimated response probabilities,
   * conditioned on cluster, for each category, flatten list of matrices
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each category</li>
   *   <li>dim 2: for each cluster</li>
   * </ul>
   * @param regress_coeff Not used and ignored
   */
  EmAlgorithm(double* features, int* responses, double* initial_prob,
              int n_data, int n_feature, int n_category, int* n_outcomes,
              int sum_outcomes, int n_cluster, int max_iter, double tolerance,
              double* posterior, double* prior, double* estimated_prob,
              double* regress_coeff);

  virtual ~EmAlgorithm();

  /**
   * Fit data to model using EM algorithm
   *
   * Data is provided through the constructor, important results are stored in
   * the member variables:
   * <ul>
   *   <li>posterior_</li>
   *   <li>prior_</li>
   *   <li>estimated_prob_</li>
   *   <li>ln_l_array_</li>
   *   <li>ln_l_</li>
   *   <li>n_iter_</li>
   * </ul>
   */
  void Fit();

  /**
   * Set where to store initial probabilities (optional)
   *
   * @param best_initial_prob Vector of INITIAL response probabilities used to
   * get the maximum log likelihood, flatten list of matrices
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each category</li>
   *   <li>dim 2: for each cluster</li>
   * </ul>
   */
  void set_best_initial_prob(double* best_initial_prob);

  /** Get the log likelihood */
  double get_ln_l();

  /** Get the number of iterations of EM done */
  int get_n_iter();

  /**
   * Indicate if it needed to use new initial values during a fit, can happen
   * if a matrix is singular
   */
  bool get_has_restarted();

  /** Set rng using a seed, for generating new random initial values */
  void set_seed(unsigned seed);

  /** Set rng by transferring ownership of an rng to here */
  void set_rng(std::unique_ptr<std::mt19937_64>* rng);

  /** Transfer ownership of rng back*/
  std::unique_ptr<std::mt19937_64> move_rng();

 protected:
  /**
   * Reset parameters for a re-run
   * Reset the parameters estimated_prob_ with random values
   * @param uniform required to generate random probabilities
   */
  virtual void Reset(std::uniform_real_distribution<double>* uniform);

  /**
   * Initalise prior probabilities
   *
   * Initalise the content of prior_ which contains prior probabilities for each
   * cluster, ready for the EM algorithm
   */
  virtual void InitPrior();

  /** Adjust prior return value to matrix format */
  virtual void FinalPrior();

  /** Get prior during the EM algorithm */

  /**
   * Get prior, for a specified data point and cluster, during the EM algorithm
   *
   * @param data_index
   * @param cluster_index
   * @return double prior
   */
  virtual double GetPrior(int data_index, int cluster_index);

  /**
   * Do E step, update the posterior probabilities given the prior probabilities
   * and estimated response probabilities. Modifies the member variables
   * posterior_ and ln_l_array_. Calculations from the E step also provides the
   * elements for ln_l_array_.
   */
  void EStep();

  /**
   * Calculate the unnormalize posterior using likelihood multiply by prior.
   * This is then assigned in posterior_.
   *
   * @param data_index
   * @param cluster_index
   * @param estimated_prob pointer to estimated probabilities for the
   * corresponding cluster. This is modified to point to the probabilities for
   * the next cluster.
   */
  void PosteriorUnnormalize(int data_index,
                              int cluster_index,
                              double** estimated_prob);

  /**
   * Check if the likelihood is invalid
   *
   * @param ln_l_difference the change in log likelihood after an iteration of
   * EM
   * @return true if the likelihood is invalid
   * @return false if the likelihood okay
   */
  virtual bool IsInvalidLikelihood(double ln_l_difference);

  /**
   * Do M step, update the prior probabilities and estimated response
   * probabilities given the posterior probabilities
   * modifies the member variables prior_ and estimated_prob_
   *
   * @return false
   */
  virtual bool MStep();

  /**
   * Estimate probability
   * updates and modify the member variable estimated_prob_ using the
   * posterior
   */
  void EstimateProbability();

  /**
   * Weighted sum for outcome probability estimation.
   * Calculates sum over data points of a observed outcome, weighted by the
   * posterior. This is done for all outcomes. The member variable
   * estimated_prob_ is updated with the results.
   *
   * @param cluster_index which cluster to consider
   */
  void WeightedSumProb(int cluster_index);

  /**
   * Normalised weighted sum for outcome probability estimation
   * After calling WeightedSumProb, call this to normalise the weighted sum so
   * that the member variable estimated_prob_ contain estimated
   * probabilities for each outcome
   * Can be overridden as the sum of weights can be calculated differently
   *
   * @param cluster_index which cluster to consider
   */
  virtual void NormalWeightedSumProb(int cluster_index);

  /**
   * Normalised Weighted Sum for Outcome Porbability Estimation
   * After calling WeightedSumProb, call this to normalise the weighted sum so
   * that the member variable estimated_prob_ contain estimated
   * probabilities for each outcome
   * @param cluster_index which cluster to consider
   * @param normaliser sum of weights
   */
  void NormalWeightedSumProb(int cluster_index, double normaliser);
};

/**
 * Calculates and returns the unnormalize posterior using likelihood multiply by
 * prior.
 *
 * It should be noted in the likelihood calculations, probabilities are
 * iteratively multiplied. However, to avoid underflow errors, a sum of log
 * probabilities is done instead if the number of categories is large. It
 * should be noted a sum of log is slower
 *
 * @param responses_i the responses for a given data point
 * @param n_catgeory
 * @param n_outcomes
 * @param estimated_prob pointer to estimated probabilities for the
 * corresponding cluster. This is modified to point to the probabilities for
 * the next cluster.
 * @param prior the prior for this data point and cluster
 * @return double the unnormalise posterior for this datap oint and cluster
 */
double PosteriorUnnormalize(int* responses_i,
                            int n_category,
                            int* n_outcomes,
                            double** estimated_prob,
                            double prior);

/**
 * Generate random response probabilities
 *
 * @param rng random number generator
 * @param uniform uniform (0, 1)
 * @param n_outcomes vector length n_category, number of outcomes for each
 * category
 * @param sum_outcomes sum of n_outcomes
 * @param n_category number of categories
 * @param n_cluster number of clusters
 * @param prob output, vector of random response probabilities, conditioned on
 * cluster, for each outcome, category and cluster
 * <ul>
 *   <li>dim 0: for each outcome</li>
 *   <li>dim 1: for each category</li>
 *   <li>dim 2: for each cluster</li>
 * </ul>
 */
void GenerateNewProb(std::mt19937_64* rng,
                     std::uniform_real_distribution<double>* uniform,
                     int* n_outcomes, int sum_outcomes, int n_category,
                     int n_cluster, double* prob);

}  // namespace polca_parallel

#endif  // EM_ALGORITHM_H_
