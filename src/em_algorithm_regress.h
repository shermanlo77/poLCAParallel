#ifndef EM_ALGORITHM_REGRESS_H
#define EM_ALGORITHM_REGRESS_H

#include "em_algorithm.h"

#include "RcppArmadillo.h"

using namespace arma;

// EM ALGORITHM REGRESS
// For fitting using EM algorithm for a given initial value, prior probabilities
    // are softmax
// Member variables are made public for the sake of convenience so that
    // EmAlgorithmArray can access and modify instances of EmAlgorithm
class EmAlgorithmRegress : public EmAlgorithm {

  private:
    int n_parameters_;  // number of parameters to estimate for the softmax
    // vector, length n_parameters_, gradient of the log likelihood
    double* gradient_;
    // matrix, n_parameters_ x n_parameters, hessian of the log likelihood
    double* hessian_;

  public:
    EmAlgorithmRegress(
        double* features,
        int* responses,
        double* initial_prob,
        int n_data,
        int n_feature,
        int n_category,
        int* n_outcomes,
        int sum_outcomes,
        int n_cluster,
        int max_iter,
        double tolerance,
        double* posterior,
        double* prior,
        double* estimated_prob,
        double* regress_coeff)
        : EmAlgorithm(
            features,
            responses,
            initial_prob,
            n_data,
            n_feature,
            n_category,
            n_outcomes,
            sum_outcomes,
            n_cluster,
            max_iter,
            tolerance,
            posterior,
            prior,
            estimated_prob,
            regress_coeff) {
      this->n_parameters_ = n_feature * (n_cluster - 1);
      this->gradient_ = new double[this->n_parameters_];
      this->hessian_ = new double[this->n_parameters_*this->n_parameters_];
      this->init_regress_coeff();
    }

  ~EmAlgorithmRegress() override {
    delete[] this->gradient_;
    delete[] this->hessian_;
  }

  protected:
    // Reset parameters for a re-run
    // Reset the parameters estimated_prob_ with random values and reset
      // regress_coeff to zero
    void Reset(std::mt19937_64* rng,
               std::uniform_real_distribution<double>* uniform) override {
      this->EmAlgorithm::Reset(rng, uniform);
      this->init_regress_coeff();
    }

    void InitPrior() override {
      // matrix multiply: eta = features x regress_coeff
        // then into softmax, ie prior probability proportional to exp(eta)
      // restrict regress_coeff for the 0th cluster to be 0
      Mat<double> features(
          this->features_, this->n_data_, this->n_feature_, false);
      Mat<double> regress_coeff(
          this->regress_coeff_, this->n_feature_, this->n_cluster_-1, false);
      Mat<double> prior = features * regress_coeff;
      prior = exp(prior);

      // for the 0th cluster, eta = 0, prior for 0th cluster propto 1
      for (int i=0; i<this->n_data_; i++) {
        this->prior_[i] = 1.0;
      }
      memcpy(this->prior_ + this->n_data_, prior.begin(),
             prior.size()*sizeof(double));

      // normalise so that prior_ are probabilities
      Mat<double> prior_arma(
          this->prior_, this->n_data_, this->n_cluster_, false);
      Col<double> normaliser = sum(prior_arma, 1);
      double* prior_ptr = this->prior_;
      for (int m=0; m<this->n_cluster_; m++) {
        for (int i=0; i<this->n_data_; i++) {
          *prior_ptr /= normaliser[i];
          prior_ptr++;
        }
      }
    }

    void FinalPrior() override {
      // do nothing, prior_ already in required format
    }

    double GetPrior(int data_index, int cluster_index) override {
      return this->prior_[this->n_data_*cluster_index + data_index];
    }

    bool IsInvalidLikelihood(double ln_l_difference) override {
      // comparing nan may be unclear, check nan first
      // check if the newton step decreases the log likelihood
      if (this->EmAlgorithm::IsInvalidLikelihood(ln_l_difference)) {
        return true;
      } else {
        return ln_l_difference < -1e-7;
      }
    }

    // M Step
    // update the regression coefficient, prior probabilities and estimated
      // response probabilities given the posterior probabilities
    // modifies the member variables regress_coeff_, gradient_, hessian_, prior_
      // and estimated_prob_
    // Return:
      // true if the solver cannot find a solution
      // false if successful
    bool MStep() override {
      // estimate outcome probabilities
      this->EstimateProbability();

      this->CalcGrad();
      this->CalcHess();

      // single Newton step
      Col<double> regress_coeff(this->regress_coeff_, this->n_parameters_,
          false);
      Col<double> gradient(this->gradient_, this->n_parameters_, false);
      Mat<double> hessian(this->hessian_, this->n_parameters_,
          this->n_parameters_, false);
      try {
        regress_coeff -= solve(hessian, gradient, solve_opts::likely_sympd);
      } catch (const std::runtime_error) {
        return true;
      }

      // using new regression coefficients, update priors
      this->InitPrior();

      return false;
    }

    void NormalWeightedSumProb(int cluster_index) override {
      // override as the normaliser cannot be calculated using prior
      // using sum of posterior instead
      Col<double> posterior(this->posterior_ + cluster_index*this->n_data_,
          this->n_data_, false);
      double normaliser = sum(posterior);
      this->EmAlgorithm::NormalWeightedSumProb(cluster_index, normaliser);
    }

  private:
    //Initalise regress_coeff_ to all zero
    void init_regress_coeff() {
      for (int i=0; i<this->n_parameters_; i++) {
        this->regress_coeff_[i] = 0.0;
      }
    }

    // Calculate gradient of the log likelihood
    // Updates the member variable gradient_
    void CalcGrad() {
      double* gradient = this->gradient_;
      for (int m=1; m<this->n_cluster_; m++) {
        Col<double>posterior_m(
            this->posterior_ + m*this->n_data_, this->n_data_, false);
        Col<double>prior_m(
            this->prior_ + m*this->n_data_, this->n_data_, false);
        Col<double> post_minus_prior = posterior_m - prior_m;
        for (int p=0; p<this->n_feature_; p++) {
          Col<double> x_p(
              this->features_ + p*this->n_data_, this->n_data_, false);
          *gradient = dot(x_p, post_minus_prior);
          gradient++;
        }
      }
    }

    // Calculate hessian of the log likelihood
    // Updates the member variable hessian_
    void CalcHess() {
      for (int cluster_j=0; cluster_j<this->n_cluster_-1; cluster_j++) {
        for (int cluster_i=cluster_j;
            cluster_i<this->n_cluster_-1; cluster_i++) {
          this->CalcHessSubBlock(cluster_i, cluster_j);
        }
      }
    }

    // Calculate one of the blocks of the hessian
    // Updates the member variable hessian_ with one of the blocks
    // The hessian consist of (n_cluster-1) by (n_cluster-1) blocks, each
      // corresponding to cluster 1, 2, 3, ..., n_cluster-1
    // Parameters:
      // cluster_index_0: row index of which block to work on
        // can take values of 0, 1, 2, ..., n_cluster-2
      // cluster_index_1: column index of which block to work on
        // can take values of 0, 1, 2, ..., n_cluster-2
    void CalcHessSubBlock(int cluster_index_0, int cluster_index_1) {

      // when retriving the prior and posterior, use cluster_index + 1 because
        // the hessian does not consider the 0th cluster as the regression
        // coefficient for the 0th cluster is set to zero
      Col<double> posterior0(
          this->posterior_ + (cluster_index_0+1)*this->n_data_,
          this->n_data_, false);
      Col<double> prior0(
          this->prior_ + (cluster_index_0+1)*this->n_data_,
          this->n_data_, false);

      // for the same cluster, copy over results as they will be modified
      bool is_same_cluster = cluster_index_0 == cluster_index_1;
      Col<double> posterior1(
          this->posterior_ + (cluster_index_1+1)*this->n_data_,
          this->n_data_, is_same_cluster);
      Col<double> prior1(
          this->prior_ + (cluster_index_1+1)*this->n_data_,
          this->n_data_, is_same_cluster);

      // Suppose r = posterior, pi = prior, u, v = cluster indexs
        // prior_post_inter is the following:
          // For same cluster, r_u*(1-r_u) - pi_u(1-pi_u)
          // For different clusters, pi_u pi_v - r_u r_v
      if (is_same_cluster) {
        posterior1 -= 1;
        prior1 -= 1;
      }
      Col<double> prior_post_inter = prior0 % prior1 - posterior0 % posterior1;

      double hess_element;
      // iterate through features i, j, working out the elements of the hessian
      // symmetric matrix, so loop over diagonal and lower triangle
      for (int j=0; j<this->n_feature_; j++) {
        for (int i=j; i<this->n_feature_; i++) {

          hess_element = CalcHessElement(i, j, &prior_post_inter);
          *this->HessianAt(cluster_index_0, cluster_index_1, i, j) =
              hess_element;

          // hessian and each block is symmetric
          // copy over values to their mirror
          if (i != j) {
            *this->HessianAt(cluster_index_0, cluster_index_1, j, i) =
                hess_element;
          }

          if (cluster_index_0 != cluster_index_1) {
            *this->HessianAt(cluster_index_1, cluster_index_0, i, j) =
                hess_element;
            if (i != j) {
              *this->HessianAt(cluster_index_1, cluster_index_0, j, i) =
                hess_element;
            }
          }
        }
      }
    }

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
                           Col<double>* prior_post_inter) {
      Col<double> feature0(this->features_ + feature_index_0*this->n_data_,
                           this->n_data_, false);
      Col<double> feature1(this->features_ + feature_index_1*this->n_data_,
                           this->n_data_, false);
      return sum(feature0 % feature1 % *prior_post_inter);
    }

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
                      int feature_index_0, int feature_index_1) {
      return this->hessian_
          + cluster_index_1*this->n_parameters_*this->n_feature_
          + feature_index_1*this->n_parameters_
          + cluster_index_0*this->n_feature_
          + feature_index_0;
    }
};

#endif
