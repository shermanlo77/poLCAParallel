#ifndef EM_ALGORITHM_ARRAY_H
#define EM_ALGORITHM_ARRAY_H

#include <thread>
#include <mutex>

#include "em_algorithm.h"

// EM ALGORITHM ARRAY
// For using EM algorithm for multiple inital probabilities, use to find the
  // global maximum
// Each thread runs a repetition
class EmAlgorithmArray {

  private:
    int n_rep_;  // number of initial values to tries
    // dummy EmAlgorithm object, for storing values for the best optima so far
    EmAlgorithm* optimal_fitter_;
    // array of initial probabilities
      // each reptition uses sum_outcomes*n_cluster probabilities
    double* initial_prob_;
    int n_initial_prob_done_;  // which initial value is being worked on
    double* ln_l_array_;  // maximum log likelihood for each reptition
    // index of which inital value has the best log likelihood
    int best_rep_index_;
    int n_thread_;  // number of threads

    std::mutex* initial_prob_lock_;  // for locking n_initial_prob_done_
    // for locking ln_l_array_ and optimal_fitter_
    std::mutex* optimial_fitter_lock_;

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
    EmAlgorithmArray(
        double* features,
        int* responses,
        double* initial_prob,
        int n_data,
        int n_feature,
        int n_category,
        int* n_outcomes,
        int sum_outcomes,
        int n_cluster,
        int n_rep,
        int n_thread,
        int max_iter,
        double tolerance,
        double* posterior,
        double* prior,
        double* estimated_prob,
        double* regress_coeff,
        double* ln_l_array) {

      this->n_rep_ = n_rep;

      // dummy em algorithm doesn't need initial probabilities, use NULL
      this->optimal_fitter_ = new EmAlgorithm(
          features,
          responses,
          NULL,
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
          regress_coeff
      );

      if (n_thread > n_rep) {
        n_thread = n_rep;
      }

      this->initial_prob_ = initial_prob;
      this->n_initial_prob_done_ = 0;
      this->optimal_fitter_->ln_l_ = -INFINITY;
      this->ln_l_array_ = ln_l_array;
      this->n_thread_ = n_thread;
      this->initial_prob_lock_ = new std::mutex();
      this->optimial_fitter_lock_ = new std::mutex();
    }

    ~EmAlgorithmArray() {
      delete this->optimal_fitter_;
      delete this->initial_prob_lock_;
      delete this->optimial_fitter_lock_;
    }

    // FIT USING EM (in parallel)
    // To be called right after construction
    void Fit() {
      // parallel run FitThread
      std::thread thread_array[this->n_thread_-1];
      for (int i=0; i<this->n_thread_-1; i++) {
        thread_array[i] = std::thread(&EmAlgorithmArray::FitThread, this);
      }
      // main thread run
      this->FitThread();
      // join threads
      for (int i=0; i<this->n_thread_-1; i++) {
        thread_array[i].join();
      }
    }

    // FIT THREAD
    // To be run by a thread(s)
    // For each initial probability, fit using EM algorithm
    void FitThread() {

      EmAlgorithm* fitter;
      bool is_working = true;
      // which initial probability this thread is working on
      int initial_prob_index;
      double ln_l;

      int n_data = this->optimal_fitter_->n_data_;
      int n_feature = this->optimal_fitter_->n_feature_;
      int sum_outcomes = this->optimal_fitter_->sum_outcomes_;
      int n_cluster = this->optimal_fitter_->n_cluster_;

      // allocate memory for this thread
      double* posterior = new double[n_data * n_cluster];
      double* prior = new double[n_data * n_cluster];
      double* estimated_prob = new double[sum_outcomes * n_cluster];
      double* regress_coeff = new double [n_feature * (n_cluster-1)];

      while (is_working) {

        // lock to retrive initial probability
        // shall be unlocked in both if and else branches
        this->initial_prob_lock_->lock();
        if (this->n_initial_prob_done_ < this->n_rep_) {
          initial_prob_index = this->n_initial_prob_done_;
          // increment for the next worker to work on
          this->n_initial_prob_done_++;
          this->initial_prob_lock_->unlock();

          // transfer pointer to data and where to store results
          // em fit
          fitter = new EmAlgorithm(
              this->optimal_fitter_->features_,
              this->optimal_fitter_->responses_,
              this->initial_prob_ + initial_prob_index*sum_outcomes*n_cluster,
              n_data,
              n_feature,
              this->optimal_fitter_->n_category_,
              this->optimal_fitter_->n_outcomes_,
              sum_outcomes,
              n_cluster,
              this->optimal_fitter_->max_iter_,
              this->optimal_fitter_->tolerance_,
              posterior,
              prior,
              estimated_prob,
              regress_coeff
          );
          fitter->Fit();
          ln_l = fitter->ln_l_;
          this->ln_l_array_[initial_prob_index] = ln_l;

          // copy results if log likelihood improved
          this->optimial_fitter_lock_->lock();
          if (ln_l > this->optimal_fitter_->ln_l_) {
            this->best_rep_index_ = initial_prob_index;
            this->optimal_fitter_->ln_l_ = ln_l;
            this->optimal_fitter_->n_iter_ = fitter->n_iter_;
            memcpy(this->optimal_fitter_->posterior_, posterior,
                n_data*n_cluster*sizeof(double));
            memcpy(this->optimal_fitter_->prior_, prior,
                n_data*n_cluster*sizeof(double));
            memcpy(this->optimal_fitter_->estimated_prob_, estimated_prob,
                sum_outcomes*n_cluster*sizeof(double));
            memcpy(this->optimal_fitter_->regress_coeff_, regress_coeff,
                n_feature*(n_cluster-1)*sizeof(double));
          }
          this->optimial_fitter_lock_->unlock();

          delete fitter;

        } else {
          // all initial values used, stop working
          this->initial_prob_lock_->unlock();
          is_working = false;
        }
      }

      delete[] posterior;
      delete[] prior;
      delete[] estimated_prob;
      delete[] regress_coeff;

    }

    int get_best_rep_index() {
      return this->best_rep_index_;
    }

    int get_n_iter() {
      return this->optimal_fitter_->n_iter_;
    }

};

#endif
