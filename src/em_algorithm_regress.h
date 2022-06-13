#ifndef EM_ALGORITHM_REGRESS_H_
#define EM_ALGORITHM_REGRESS_H_

#include "RcppArmadillo.h"
#include "em_algorithm.h"

namespace polca_parallel {

// EM ALGORITHM REGRESS
// For fitting using EM algorithm for a given initial value, prior probabilities
// are softmax
// Member variables are made public for the sake of convenience so that
// EmAlgorithmArray can access and modify instances of EmAlgorithm
class EmAlgorithmRegress : public polca_parallel::EmAlgorithm {
 private:
  int n_parameters_;  // number of parameters to estimate for the softmax
  // vector, length n_parameters_, gradient of the log likelihood
  double* gradient_;
  // matrix, n_parameters_ x n_parameters, hessian of the log likelihood
  double* hessian_;

 public:
  EmAlgorithmRegress(double* features, int* responses, double* initial_prob,
                     int n_data, int n_feature, int n_category, int* n_outcomes,
                     int sum_outcomes, int n_cluster, int max_iter,
                     double tolerance, double* posterior, double* prior,
                     double* estimated_prob, double* regress_coeff);

  ~EmAlgorithmRegress() override;

 protected:
  // Reset parameters for a re-run
  // Reset the parameters estimated_prob_ with random values and reset
  // regress_coeff to zero
  void Reset(std::mt19937_64* rng,
             std::uniform_real_distribution<double>* uniform) override;

  void InitPrior() override;

  void FinalPrior() override;

  double GetPrior(int data_index, int cluster_index) override;

  bool IsInvalidLikelihood(double ln_l_difference) override;

  // M Step
  // update the regression coefficient, prior probabilities and estimated
  // response probabilities given the posterior probabilities
  // modifies the member variables regress_coeff_, gradient_, hessian_, prior_
  // and estimated_prob_
  // Return:
  // true if the solver cannot find a solution
  // false if successful
  bool MStep() override;

  void NormalWeightedSumProb(int cluster_index) override;

 private:
  // Initalise regress_coeff_ to all zero
  void init_regress_coeff();

  // Calculate gradient of the log likelihood
  // Updates the member variable gradient_
  void CalcGrad();

  // Calculate hessian of the log likelihood
  // Updates the member variable hessian_
  void CalcHess();

  // Calculate one of the blocks of the hessian
  // Updates the member variable hessian_ with one of the blocks
  // The hessian consist of (n_cluster-1) by (n_cluster-1) blocks, each
  // corresponding to cluster 1, 2, 3, ..., n_cluster-1
  // Parameters:
  // cluster_index_0: row index of which block to work on
  // can take values of 0, 1, 2, ..., n_cluster-2
  // cluster_index_1: column index of which block to work on
  // can take values of 0, 1, 2, ..., n_cluster-2
  void CalcHessSubBlock(int cluster_index_0, int cluster_index_1);

  // Calculate element of a block from the Hessian
  // Parameters:
  // feature_index_0: row index
  // feature_index_1: column index
  // prior_post_inter: vector of length n_data, dependent on pair of
  // clusters. Suppose r = posterior, pi = prior, u, v = cluster indexs.
  // For same cluster, r_u*(1-r_u) - pi_u(1-pi_u)
  // For different clusters, pi_u pi_v - r_u r_v
  // Return:
  // value of an element of the Hessian
  double CalcHessElement(int feature_index_0, int feature_index_1,
                         arma::Col<double>* prior_post_inter);

  // Get pointer of Hessian at specificed indexes
  // Hessian is a block matrix, each rows/columns of block matrices correspond
  // to a cluster, and then each row/column of the block matrix correspond
  // to a feature. Use this method to get a pointer of a specified element
  // of the hessian matrix
  // Parameters:
  // cluster_index_0: row index of block matrices
  // cluster_index_1: column index of block matrices
  // feature_index_0: row index within block matrix
  // feature_index_1: column index within block matrix
  // Return:
  // pointer to an element of the Hessian
  double* HessianAt(int cluster_index_0, int cluster_index_1,
                    int feature_index_0, int feature_index_1);
};

}  // namespace polca_parallel

#endif  // EM_ALGORITHM_REGRESS_H_
