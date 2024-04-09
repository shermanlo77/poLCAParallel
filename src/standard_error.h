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

#ifndef STANDARD_ERROR_H
#define STANDARD_ERROR_H

#include <math.h>

#include "RcppArmadillo.h"

namespace polca_parallel {

class StandardError {
 protected:
  /** Design matrix of features, matrix n_data x n_feature */
  double* features_;
  /** Design matrix transpose of responses, matrix n_category x n_data */
  int* responses_;
  /**
   * Vector of probabilities for each category and responses,
   * flatten list of matrices
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each category</li>
   *   <li>dim 2: for each cluster</li>
   * </ul>
   */
  double* probs_;
  /**
   * Design matrix of prior probabilities, probability data point is in
   * cluster m NOT given responses after calculations, it shall be in matrix
   * form with dimensions
   * <ul>
   *   <li>dim 0: for each data</li>
   *   <li>dim 1: for each cluster</li>
   * </ul>
   */
  double* prior_;
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
  /**
   * Vector containing the standard error for the prior probabilities for each
   * cluster
   */
  double* prior_error_;
  /**
   * Vector containing the standard error for the outcome probabilities category
   * and cluster
   * flatten list of matrices
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each category</li>
   *   <li>dim 2: for each cluster</li>
   * </ul>
   */
  double* prob_error_;
  /** Covariance matrix of the regression coefficient */
  double* regress_coeff_error_;
  /** The size of the information matrix*/
  int info_size_;
  /** The width of the Jacobian matrix*/
  int jacobian_width_;

 public:
  /**
   * Construct a new StandardError object
   *
   * Call Calc() and the resulting errors will be saved to prior_error and
   * prob_error
   *
   * @param features Design matrix of features, matrix n_data x n_feature
   * @param responses Design matrix transpose of responses, matrix n_category x
   * n_data
   * @param probs Vector of probabilities for each outcome, for each category,
   * for each cluster flatten list of matrices
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each category</li>
   *   <li>dim 2: for each cluster</li>
   * </ul>
   * @param prior Design matrix of prior probabilities, probability data point
   * is in cluster m NOT given responses after calculations, it shall be in
   * matrix form with dimensions
   * <ul>
   *   <li>dim 0: for each data</li>
   *   <li>dim 1: for each cluster</li>
   * </ul>
   * @param posterior Design matrix of posterior probabilities (also called
   * responsibility) probability data point is in cluster m given responses
   * matrix
   * <ul>
   *   <li>dim 0: for each data</li>
   *   <li>dim 1: for each cluster</li>
   * </ul>
   * @param n_data Number of data points
   * @param n_feature Number of features, required to be 1
   * @param n_category Number of categories
   * @param n_outcomes Array of number of outcomes, for each category
   * @param sum_outcomes Sum of all integers in n_outcomes
   * @param n_cluster Number of clusters fitted
   * @param prior_error Vector containing the standard error for the prior
   * probabilities for each cluster
   * @param prob_error Vector containing the standard error for the outcome
   * probabilities category and cluster,
   * flatten list of matrices
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each category</li>
   *   <li>dim 2: for each cluster</li>
   * </ul>
   * @param regress_coeff_error Not used
   */
  StandardError(double* features, int* responses, double* probs, double* prior,
                double* posterior, int n_data, int n_feature, int n_category,
                int* n_outcomes, int sum_outcomes, int n_cluster,
                double* prior_error, double* prob_error,
                double* regress_coeff_error);

  /**
   * Calculate the standard errors
   *
   * Calculate the standard errors. Results are saved in the provided pointers
   * prior_error, prob_error and regress_coeff_error
   */
  virtual void Calc();

 protected:
  /**
   * Calculate the information matrix
   *
   * @param info pointer to save the information matrix
   */
  void CalcInfo(double* info);

  /**
   * Calculate the scores
   *
   * Calculate the scores and saves it as a transposed design matrix, at the
   * provided pointer score
   *
   * @param score pointer to save the scores
   * <ul>
   *   <li>dim 0: for each data point</li>
   *   <li>dim 1: for each parameter</li>
   * </ul>
   */
  void CalcScore(double* score);

  /**
   * Calculate the scores for the prior for all clusters except the zeroth one
   *
   * Calculate the scores for the prior for all clusters except the zeroth one
   * for all data points
   *
   * @param score MODIFIED where to save the results, modified so that *score is
   * shifited by n_data * (n_cluster - 1), ready for the next set of scores
   */
  virtual void CalcScorePrior(double** score);

  /**
   * Calculate the scores for the prior
   *
   * Calculate the scores for the prior for a given cluster for all data points.
   * It is the difference between the posterior and the prior
   *
   * @param cluster_index 1, 2, ..., cluster_index - 1
   * @param score pointer to save the results
   */
  void CalcScorePriorCol(int cluster_index, double* score_start);

  /**
   * Calculate the scores for ALL outcome probabilities (except zeroth outcome)
   *
   * Calculate the scores for outcome probabilities for all clusters, categories
   * and outcomes (except for the zeroth outcome) for all data points.
   *
   * @param score MODIFIED where to save the results, modified so that *score is
   * shifited by n_data * (n_cluster - 1), ready for the next set of scores
   */
  void CalcScoreProbs(double** score);

  /**
   * Calculate the scores for the outcome probabilities
   *
   * Calculate the scores for the outcome probabilities for a given category and
   * outcome for all data points.
   *
   * @param outcome_index 1, 2, ..., n_outcomes[category_index]
   * @param category_index 0, 1, 2, ..., n_category
   * @param cluster_index 0, 1, 2, ..., n_cluster
   * @param score pointer to save the results
   */
  void CalcScoreProbsCol(int outcome_index, int category_index,
                         int cluster_index, double* score);

  /**
   * Calculate the Jacobian matrix
   *
   * Calculate the Jacobian matrix, a block diagonal matrix and saves it in the
   * provided pointer
   *
   * @param jacobian pointer to save the Jacobian matrix
   */
  void CalcJacobian(double* jacobian);

  /**
   * Calculate the block matrix for the prior in the Jacobian matrix
   *
   * Calculate the block matrix for the prior in the Jacobian matrix and saves
   * it in the provided pointer
   *
   * @param jacobian_ptr MODIFIED, the start of the block matrix in the Jacobian
   * matrix, the calculated block matrix is saved in *jacobian_ptr.
   * *jacobian_ptr is modified so that it points to the start of the next block
   * matrix after calling this method
   */
  virtual void CalcJacobianPrior(double** jacobian_ptr);

  /**
   * Calculate all block matrices for the probabilities in the Jacobian matrix
   *
   * Calculate all block matrices for the probabilities in the Jacobian matrix
   * and saves it in the provided pointer
   *
   * @param category_index 0, 1, 2, ..., n_category
   * @param cluster_index 0, 1, 2, ..., n_cluster
   * @param jacobian_ptr MODIFIED, the start of the block matrix in the Jacobian
   * matrix, the calculated block matrices is saved in *jacobian_ptr.
   * *jacobian_ptr is modified so that it points to the start of the next block
   * matrix after calling this method
   */
  void CalcJacobianProbs(double** jacobian_ptr);

  /**
   * Calculate the block matrix for the probabilities in the Jacobian matrix
   *
   * Calculate the block matrix for the probabilities in the Jacobian matrix for
   * a given cluster category pair and saves it in the provided pointer
   *
   * @param category_index 0, 1, 2, ..., n_category
   * @param cluster_index 0, 1, 2, ..., n_cluster
   * @param jacobian_ptr MODIFIED, the start of the block matrix in the Jacobian
   * matrix, the calculated block matrix is saved in *jacobian_ptr.
   * *jacobian_ptr is modified so that it points to the start of the next block
   * matrix after calling this method
   */
  void CalcJacobianProbsBlock(int category_index, int cluster_index,
                              double** jacobian_ptr);

  /**
   * Extract errors of interest
   *
   * Extract errors of interest such as prior_error, prob_error and
   * regress_coeff_error_
   *
   * @param info the information matrix
   * @param jacobian the jacobian matrix
   */
  void ExtractError(double* info, double* jacobian);

  /**
   * Extract errors of interest given the inverse of the information matrix
   *
   * @param info the information matrix
   * @param jacobian the jacobian matrix
   */
  virtual void ExtractErrorGiveInfoInv(double* info_inv, double* jacobian);

  /**
   * Extract errors for the prior from a section of the covariance matrix
   *
   * @param covariance MODIFIED, pointer to the start of the block in the
   * covariance matrix for the prior probabilities. *covariance is modified
   * to point to the next diagonal element after this block in the covariance
   * matrix
   */
  void ExtractErrorPrior(double** covariance);

  /**
   * Extract errors for the probs from a section of the covariance matrix
   *
   * @param covariance MODIFIED, pointer to the start of the block in the
   * covariance matrix for the outcome probabilities. *covariance is modified
   * to point to the next diagonal element after this block in the covariance
   * matrix
   */
  void ExtractErrorProb(double** covaraince);
};

}  // namespace polca_parallel

#endif  // STANDARD_ERROR_H
