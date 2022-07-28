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

#ifndef EM_ALGORITHM_REGRESS_H_
#define EM_ALGORITHM_REGRESS_H_

#define ARMA_WARN_LEVEL 1

#include "RcppArmadillo.h"
#include "em_algorithm.h"

namespace polca_parallel {

/**
 * For fitting using EM algorithm for a given initial value, prior probabilities
 * are softmax. Member variables are made public for the sake of convenience so
 * that EmAlgorithmArray can access and modify instances of EmAlgorithm
 */
class EmAlgorithmRegress : public polca_parallel::EmAlgorithm {
 private:
  /** Number of parameters to estimate for the softmax */
  int n_parameters_;
  /** vector, length n_parameters_, gradient of the log likelihood */
  double* gradient_;
  /** matrix, n_parameters_ x n_parameters, hessian of the log likelihood */
  double* hessian_;

 public:
  /** @copydoc EmAlgorithmRegress::EmAlgorithm
   */
  EmAlgorithmRegress(double* features, int* responses, double* initial_prob,
                     int n_data, int n_feature, int n_category, int* n_outcomes,
                     int sum_outcomes, int n_cluster, int max_iter,
                     double tolerance, double* posterior, double* prior,
                     double* estimated_prob, double* regress_coeff);

  ~EmAlgorithmRegress() override;

 protected:
  void Reset(std::mt19937_64* rng,
             std::uniform_real_distribution<double>* uniform) override;

  void InitPrior() override;

  void FinalPrior() override;

  double GetPrior(int data_index, int cluster_index) override;

  bool IsInvalidLikelihood(double ln_l_difference) override;

  /**
   * Do M step, update the regression coefficient, prior probabilities and
   * estimated response probabilities given the posterior probabilities
   * modifies the member variables regress_coeff_, gradient_, hessian_, prior_
   * and estimated_prob_
   *
   * @return true if the solver cannot find a solution
   * @return false if successful
   */
  bool MStep() override;

  void NormalWeightedSumProb(int cluster_index) override;

 private:
  /** Initalise regress_coeff_ to all zero */
  void init_regress_coeff();

  /**
   * Calculate gradient of the log likelihood.
   * Updates the member variable gradient_
   */
  void CalcGrad();

  /**
   * Calculate hessian of the log likelihood.
   * Updates the member variable hessian_
   */
  void CalcHess();

  /**
   * Calculate one of the blocks of the hessian.
   * Updates the member variable hessian_ with one of the blocks.
   * The hessian consist of (n_cluster-1) by (n_cluster-1) blocks, each
   * corresponding to cluster 1, 2, 3, ..., n_cluster-1.
   *
   * @param cluster_index_0 row index of which block to work on
   * can take values of 0, 1, 2, ..., n_cluster-2
   * @param cluster_index_1 column index of which block to work on
   * can take values of 0, 1, 2, ..., n_cluster-2
   */
  void CalcHessSubBlock(int cluster_index_0, int cluster_index_1);

  /**
   * Calculate element of a block from the Hessian
   *
   * @param feature_index_0 row index
   * @param feature_index_1 column index
   * @param prior_post_inter vector of length n_data, dependent on pair of
   * clusters. Suppose r = posterior, pi = prior, u, v = cluster indexs.
   * For same cluster, r_u*(1-r_u) - pi_u(1-pi_u)
   * For different clusters, pi_u pi_v - r_u r_v
   * @return double value of an element of the Hessian
   */
  double CalcHessElement(int feature_index_0, int feature_index_1,
                         arma::Col<double>* prior_post_inter);

  /**
   * Get pointer of Hessian at specificed indexes
   *
   * Hessian is a block matrix, each rows/columns of block matrices correspond
   * to a cluster, and then each row/column of the block matrix correspond
   * to a feature. Use this method to get a pointer of a specified element
   * of the hessian matrix
   *
   * @param cluster_index_0 row index of block matrices
   * @param cluster_index_1 column index of block matrices
   * @param feature_index_0 row index within block matrix
   * @param feature_index_1 column index within block matrix
   * @return double* pointer to an element of the Hessian
   */
  double* HessianAt(int cluster_index_0, int cluster_index_1,
                    int feature_index_0, int feature_index_1);
};

}  // namespace polca_parallel

#endif  // EM_ALGORITHM_REGRESS_H_
