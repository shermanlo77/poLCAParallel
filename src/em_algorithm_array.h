#ifndef EM_ALGORITHM_ARRAY_H
#define EM_ALGORITHM_ARRAY_H

#include <thread>
#include <memory>
#include <mutex>

#include "em_algorithm.h"
#include "em_algorithm_regress.h"

// EM ALGORITHM ARRAY
// For using EM algorithm for multiple inital probabilities, use to find the
  // global maximum
// Each thread runs a repetition
class EmAlgorithmArray {

  private:

    double* features_;  // features to provide EmAlgorithm with
    int* responses_;  // reponses to provide EmAlgorithm with
    int n_data_;  // number of data points
    int n_feature_;  // number of features
    int n_category_;  // number of categories
    int* n_outcomes_;  // vector of number of outcomes for each category
    int sum_outcomes_;  // sum of n_outcomes
    int n_cluster_;  // number of clusters (classes in lit) to fit
    int max_iter_;  // maximum number of iterations for EM algorithm
    double tolerance_;  // to provide to EmAlgorithm
    double* posterior_;  // to store posterior results
    double* prior_;  // to store prior results
    double* estimated_prob_;  // to store estimated prob results
    double* regress_coeff_;  // to store regression coefficient results
    // optional, to store initial prob to obtain max likelihood
    double* best_initial_prob_ = NULL;

    int n_rep_;  // number of initial values to tries
    // the best log likelihood found so far
    double optimal_ln_l_;
    // number of iterations optimal fitter has done
    int n_iter_;
    // true if the EM algorithm has to ever restart
    bool has_restarted_ = false;
    // array of initial probabilities
      // each reptition uses sum_outcomes*n_cluster probabilities
    double* initial_prob_;
    int n_initial_prob_done_;  // which initial value is being worked on
    double* ln_l_array_;  // maximum log likelihood for each reptition
    // index of which inital value has the best log likelihood
    int best_rep_index_;
    int n_thread_;  // number of threads

    // array of seeds, for each repetition
    std::unique_ptr<unsigned[]> seed_array_ = NULL;

    std::mutex* initial_prob_lock_;  // for locking n_initial_prob_done_
    // for locking optimal_ln_l_, best_rep_index_, n_iter_ and has_restarted_
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

      this->features_ = features;
      this->responses_ = responses;
      this->n_data_ = n_data;
      this->n_feature_ = n_feature;
      this->n_category_ = n_category;
      this->n_outcomes_ = n_outcomes;
      this->sum_outcomes_ = sum_outcomes;
      this->n_cluster_ = n_cluster;
      this->max_iter_ = max_iter;
      this->tolerance_ = tolerance;
      this->posterior_ = posterior;
      this->prior_ = prior;
      this->estimated_prob_ = estimated_prob;
      this->regress_coeff_ = regress_coeff;

      if (n_thread > n_rep) {
        n_thread = n_rep;
      }

      this->initial_prob_ = initial_prob;
      this->n_initial_prob_done_ = 0;
      this->optimal_ln_l_ = -INFINITY;
      this->ln_l_array_ = ln_l_array;
      this->n_thread_ = n_thread;
      this->initial_prob_lock_ = new std::mutex();
      this->optimial_fitter_lock_ = new std::mutex();
    }

    ~EmAlgorithmArray() {
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

    // Set Seed
    // Set the member variable seed_array_ with seeds for each repetition
    void SetSeed(std::seed_seq* seed) {
      this->seed_array_ = std::unique_ptr<unsigned[]>(
          new unsigned[this->n_rep_]);
      unsigned* seed_array = this->seed_array_.get();
      seed->generate(seed_array, seed_array+this->n_rep_);
    }

    // Set where to store initial probabilities (optional)
    void set_best_initial_prob(double* best_initial_prob) {
      this->best_initial_prob_ = best_initial_prob;
    }

    int get_best_rep_index() {
      return this->best_rep_index_;
    }

    int get_n_iter() {
      return this->n_iter_;
    }

    bool get_has_restarted() {
      return this->has_restarted_;
    }
  
  private:
    // FIT THREAD
    // To be run by a thread(s)
    // For each initial probability, fit using EM algorithm
    void FitThread() {

      EmAlgorithm* fitter;
      bool is_working = true;
      // which initial probability this thread is working on
      int initial_prob_index;
      double ln_l;

      int n_data = this->n_data_;
      int n_feature = this->n_feature_;
      int sum_outcomes = this->sum_outcomes_;
      int n_cluster = this->n_cluster_;

      // allocate memory for this thread
      double* posterior = new double[n_data * n_cluster];
      double* prior = new double[n_data * n_cluster];
      double* estimated_prob = new double[sum_outcomes * n_cluster];
      double* regress_coeff = new double [n_feature * (n_cluster-1)];
      double* best_initial_prob;
      // allocate optional memory
      bool is_get_initial_prob = this->best_initial_prob_ != NULL;
      if (is_get_initial_prob) {
        best_initial_prob = new double[sum_outcomes * n_cluster];
      }

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
          if (n_feature == 1) {
            fitter = new EmAlgorithm(
                this->features_,
                this->responses_,
                this->initial_prob_ + initial_prob_index*sum_outcomes*n_cluster,
                n_data,
                n_feature,
                this->n_category_,
                this->n_outcomes_,
                sum_outcomes,
                n_cluster,
                this->max_iter_,
                this->tolerance_,
                posterior,
                prior,
                estimated_prob,
                regress_coeff);
          } else {
            fitter = new EmAlgorithmRegress(
                this->features_,
                this->responses_,
                this->initial_prob_ + initial_prob_index*sum_outcomes*n_cluster,
                n_data,
                n_feature,
                this->n_category_,
                this->n_outcomes_,
                sum_outcomes,
                n_cluster,
                this->max_iter_,
                this->tolerance_,
                posterior,
                prior,
                estimated_prob,
                regress_coeff);
          }
          if (this->seed_array_ != NULL) {
            fitter->set_seed(this->seed_array_[initial_prob_index]);
          }
          if (is_get_initial_prob) {
            fitter->set_best_initial_prob(best_initial_prob);
          }
          fitter->Fit();
          ln_l = fitter->get_ln_l();
          this->ln_l_array_[initial_prob_index] = ln_l;

          // copy results if log likelihood improved
          this->optimial_fitter_lock_->lock();
          this->has_restarted_ |= fitter->get_has_restarted();
          if (ln_l > this->optimal_ln_l_) {
            this->best_rep_index_ = initial_prob_index;
            this->optimal_ln_l_ = ln_l;
            this->n_iter_ = fitter->get_n_iter();
            memcpy(this->posterior_, posterior,
                n_data*n_cluster*sizeof(double));
            memcpy(this->prior_, prior,
                n_data*n_cluster*sizeof(double));
            memcpy(this->estimated_prob_, estimated_prob,
                sum_outcomes*n_cluster*sizeof(double));
            memcpy(this->regress_coeff_, regress_coeff,
                n_feature*(n_cluster-1)*sizeof(double));
            if (is_get_initial_prob) {
              memcpy(this->best_initial_prob_, best_initial_prob,
                sum_outcomes*n_cluster*sizeof(double));
            }
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
      if (is_get_initial_prob) {
        delete[] best_initial_prob;
      }

    }

};

#endif
