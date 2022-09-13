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

#include "em_algorithm_array.h"

polca_parallel::EmAlgorithmArray::EmAlgorithmArray(
    double* features, int* responses, double* initial_prob, int n_data,
    int n_feature, int n_category, int* n_outcomes, int sum_outcomes,
    int n_cluster, int n_rep, int n_thread, int max_iter, double tolerance,
    double* posterior, double* prior, double* estimated_prob,
    double* regress_coeff, bool is_regress) {
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
  this->is_regress_ = is_regress;

  if (n_thread > n_rep) {
    n_thread = n_rep;
  }

  this->initial_prob_ = initial_prob;
  this->n_rep_done_ = 0;
  this->optimal_ln_l_ = -INFINITY;
  this->n_thread_ = n_thread;
  this->n_rep_done_lock_ = new std::mutex();
  this->results_lock_ = new std::mutex();
}

polca_parallel::EmAlgorithmArray::~EmAlgorithmArray() {
  delete this->n_rep_done_lock_;
  delete this->results_lock_;
}

void polca_parallel::EmAlgorithmArray::Fit() {
  // parallel run FitThread
  std::thread thread_array[this->n_thread_ - 1];
  for (int i = 0; i < this->n_thread_ - 1; ++i) {
    thread_array[i] = std::thread(&EmAlgorithmArray::FitThread, this);
  }
  // main thread run
  this->FitThread();
  // join threads
  for (int i = 0; i < this->n_thread_ - 1; ++i) {
    thread_array[i].join();
  }
}

void polca_parallel::EmAlgorithmArray::SetSeed(std::seed_seq* seed) {
  this->seed_array_ = std::unique_ptr<unsigned[]>(new unsigned[this->n_rep_]);
  unsigned* seed_array = this->seed_array_.get();
  seed->generate(seed_array, seed_array + this->n_rep_);
}

void polca_parallel::EmAlgorithmArray::set_best_initial_prob(
    double* best_initial_prob) {
  this->best_initial_prob_ = best_initial_prob;
}

void polca_parallel::EmAlgorithmArray::set_ln_l_array(double* ln_l_array) {
  this->ln_l_array_ = ln_l_array;
}

int polca_parallel::EmAlgorithmArray::get_best_rep_index() {
  return this->best_rep_index_;
}

double polca_parallel::EmAlgorithmArray::get_optimal_ln_l() {
  return this->optimal_ln_l_;
}

int polca_parallel::EmAlgorithmArray::get_n_iter() { return this->n_iter_; }

bool polca_parallel::EmAlgorithmArray::get_has_restarted() {
  return this->has_restarted_;
}

void polca_parallel::EmAlgorithmArray::SetFitterRng(
    polca_parallel::EmAlgorithm* fitter, int rep_index) {
  if (this->seed_array_ != NULL) {
    fitter->set_seed(this->seed_array_[rep_index]);
  }
}

void polca_parallel::EmAlgorithmArray::MoveRngBackFromFitter(
    polca_parallel::EmAlgorithm* fitter) {
  // do nothing, no rng handelled here
}

void polca_parallel::EmAlgorithmArray::FitThread() {
  polca_parallel::EmAlgorithm* fitter;
  bool is_working = true;
  // which initial probability this thread is working on
  int rep_index;
  double ln_l;

  int n_data = this->n_data_;
  int n_feature = this->n_feature_;
  int sum_outcomes = this->sum_outcomes_;
  int n_cluster = this->n_cluster_;

  // allocate memory for this thread
  double* posterior = new double[n_data * n_cluster];
  double* prior = new double[n_data * n_cluster];
  double* estimated_prob = new double[sum_outcomes * n_cluster];
  double* regress_coeff = new double[n_feature * (n_cluster - 1)];
  double* best_initial_prob;
  // allocate optional memory
  bool is_get_initial_prob = this->best_initial_prob_ != NULL;
  if (is_get_initial_prob) {
    best_initial_prob = new double[sum_outcomes * n_cluster];
  }

  while (is_working) {
    // lock to retrive initial probability
    // shall be unlocked in both if and else branches
    this->n_rep_done_lock_->lock();
    if (this->n_rep_done_ < this->n_rep_) {
      rep_index = this->n_rep_done_;
      // increment for the next worker to work on
      ++this->n_rep_done_;
      this->n_rep_done_lock_->unlock();

      // transfer pointer to data and where to store results
      // em fit
      if (this->is_regress_) {
        fitter = new polca_parallel::EmAlgorithmRegress(
            this->features_, this->responses_,
            this->initial_prob_ + rep_index * sum_outcomes * n_cluster, n_data,
            n_feature, this->n_category_, this->n_outcomes_, sum_outcomes,
            n_cluster, this->max_iter_, this->tolerance_, posterior, prior,
            estimated_prob, regress_coeff);
      } else {
        fitter = new polca_parallel::EmAlgorithm(
            this->features_, this->responses_,
            this->initial_prob_ + rep_index * sum_outcomes * n_cluster, n_data,
            n_feature, this->n_category_, this->n_outcomes_, sum_outcomes,
            n_cluster, this->max_iter_, this->tolerance_, posterior, prior,
            estimated_prob, regress_coeff);
      }
      this->SetFitterRng(fitter, rep_index);
      if (is_get_initial_prob) {
        fitter->set_best_initial_prob(best_initial_prob);
      }
      fitter->Fit();
      ln_l = fitter->get_ln_l();
      if (this->ln_l_array_ != NULL) {
        this->ln_l_array_[rep_index] = ln_l;
      }

      // if ownership of rng transferred (if any) to fitter, get it back if
      // needed
      this->MoveRngBackFromFitter(fitter);

      // copy results if log likelihood improved
      this->results_lock_->lock();
      this->has_restarted_ |= fitter->get_has_restarted();
      if (ln_l > this->optimal_ln_l_) {
        this->best_rep_index_ = rep_index;
        this->optimal_ln_l_ = ln_l;
        this->n_iter_ = fitter->get_n_iter();
        memcpy(this->posterior_, posterior,
               n_data * n_cluster * sizeof(*this->posterior_));
        memcpy(this->prior_, prior, n_data * n_cluster * sizeof(*this->prior_));
        memcpy(this->estimated_prob_, estimated_prob,
               sum_outcomes * n_cluster * sizeof(*this->estimated_prob_));
        memcpy(this->regress_coeff_, regress_coeff,
               n_feature * (n_cluster - 1) * sizeof(*this->regress_coeff_));
        if (is_get_initial_prob) {
          memcpy(this->best_initial_prob_, best_initial_prob,
                 sum_outcomes * n_cluster * sizeof(*this->best_initial_prob_));
        }
      }
      this->results_lock_->unlock();

      delete fitter;
    } else {
      // all initial values used, stop working
      this->n_rep_done_lock_->unlock();
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
