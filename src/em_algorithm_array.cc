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

#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include "em_algorithm.h"
#include "em_algorithm_nan.h"
#include "em_algorithm_regress.h"

template void
polca_parallel::EmAlgorithmArray::Fit<polca_parallel::EmAlgorithm>();
template void
polca_parallel::EmAlgorithmArray::Fit<polca_parallel::EmAlgorithmRegress>();
template void
polca_parallel::EmAlgorithmArray::Fit<polca_parallel::EmAlgorithmNan>();
template void
polca_parallel::EmAlgorithmArray::Fit<polca_parallel::EmAlgorithmNanRegress>();

template void
polca_parallel::EmAlgorithmArray::FitThread<polca_parallel::EmAlgorithm>();
template void polca_parallel::EmAlgorithmArray::FitThread<
    polca_parallel::EmAlgorithmRegress>();
template void
polca_parallel::EmAlgorithmArray::FitThread<polca_parallel::EmAlgorithmNan>();
template void polca_parallel::EmAlgorithmArray::FitThread<
    polca_parallel::EmAlgorithmNanRegress>();

polca_parallel::EmAlgorithmArray::EmAlgorithmArray(
    double* features, int* responses, double* initial_prob, std::size_t n_data,
    std::size_t n_feature, std::size_t n_category, std::size_t* n_outcomes,
    std::size_t sum_outcomes, std::size_t n_cluster, std::size_t n_rep,
    std::size_t n_thread, unsigned int max_iter, double tolerance,
    double* posterior, double* prior, double* estimated_prob,
    double* regress_coeff)
    : features_(features),
      responses_(responses),
      n_data_(n_data),
      n_feature_(n_feature),
      n_category_(n_category),
      n_outcomes_(n_outcomes),
      sum_outcomes_(sum_outcomes),
      n_cluster_(n_cluster),
      max_iter_(max_iter),
      tolerance_(tolerance),
      posterior_(posterior),
      prior_(prior),
      estimated_prob_(estimated_prob),
      regress_coeff_(regress_coeff),
      n_rep_(n_rep),
      initial_prob_(initial_prob),
      n_thread_(std::min(n_thread, n_rep)),
      n_rep_done_lock_(std::make_unique<std::mutex>()),
      results_lock_(std::make_unique<std::mutex>()) {}

template <typename EmAlgorithmType>
void polca_parallel::EmAlgorithmArray::Fit() {
  // parallel run FitThread
  std::vector<std::thread> thread_array(this->n_thread_ - 1);
  for (std::size_t i = 0; i < this->n_thread_ - 1; ++i) {
    thread_array.at(i) =
        std::thread(&EmAlgorithmArray::FitThread<EmAlgorithmType>, this);
  }
  // main thread run
  this->FitThread<EmAlgorithmType>();
  // join threads
  for (std::size_t i = 0; i < this->n_thread_ - 1; ++i) {
    thread_array.at(i).join();
  }
}

void polca_parallel::EmAlgorithmArray::SetSeed(std::seed_seq* seed) {
  this->seed_array_ = std::make_unique<unsigned[]>(this->n_rep_);
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

std::size_t polca_parallel::EmAlgorithmArray::get_best_rep_index() {
  return this->best_rep_index_;
}

double polca_parallel::EmAlgorithmArray::get_optimal_ln_l() {
  return this->optimal_ln_l_;
}

unsigned int polca_parallel::EmAlgorithmArray::get_n_iter() {
  return this->n_iter_;
}

bool polca_parallel::EmAlgorithmArray::get_has_restarted() {
  return this->has_restarted_;
}

void polca_parallel::EmAlgorithmArray::SetFitterRng(
    polca_parallel::EmAlgorithm* fitter, std::size_t rep_index) {
  if (this->seed_array_) {
    fitter->set_seed(this->seed_array_[rep_index]);
  }
}

void polca_parallel::EmAlgorithmArray::MoveRngBackFromFitter(
    polca_parallel::EmAlgorithm* fitter) {
  // do nothing, no rng handelled here
}

template <typename EmAlgorithmType>
void polca_parallel::EmAlgorithmArray::FitThread() {
  std::size_t n_data = this->n_data_;
  std::size_t n_feature = this->n_feature_;
  std::size_t sum_outcomes = this->sum_outcomes_;
  std::size_t n_cluster = this->n_cluster_;

  // allocate memory for this thread
  std::vector<double> posterior(n_data * n_cluster);
  std::vector<double> prior(n_data * n_cluster);
  std::vector<double> estimated_prob(sum_outcomes * n_cluster);
  std::vector<double> regress_coeff(n_feature * (n_cluster - 1));
  std::vector<double> best_initial_prob(sum_outcomes * n_cluster);

  // transfer pointer to data and where to store results
  std::unique_ptr<polca_parallel::EmAlgorithm> fitter =
      std::make_unique<EmAlgorithmType>(
          this->features_, this->responses_, nullptr, n_data, n_feature,
          this->n_category_, this->n_outcomes_, sum_outcomes, n_cluster,
          this->max_iter_, this->tolerance_, posterior.data(), prior.data(),
          estimated_prob.data(), regress_coeff.data());
  if (this->best_initial_prob_) {
    fitter->set_best_initial_prob(best_initial_prob.data());
  }

  // which initial probability this thread is working on
  std::size_t rep_index;
  double ln_l;

  bool is_working = true;
  while (is_working) {
    // lock to retrive initial probability and other variables
    // lock is outside the if statement so that this->n_rep_done_ can be read
    // without modification from other threads
    // shall be unlocked in both if and else branches
    this->n_rep_done_lock_->lock();
    if (this->n_rep_done_ < this->n_rep_) {
      // increment for the next worker to work on
      rep_index = this->n_rep_done_++;

      this->n_rep_done_lock_->unlock();

      // set initial probability for this repetition
      fitter->NewRun(this->initial_prob_ +
                     rep_index * sum_outcomes * n_cluster);

      // each repetition uses their own rng
      this->SetFitterRng(fitter.get(), rep_index);

      fitter->Fit();
      ln_l = fitter->get_ln_l();
      if (this->ln_l_array_) {
        this->ln_l_array_[rep_index] = ln_l;
      }

      // if ownership of rng transferred (if any) to fitter, get it back if
      // needed
      this->MoveRngBackFromFitter(fitter.get());

      // copy results if log likelihood improved
      this->results_lock_->lock();
      this->has_restarted_ |= fitter->get_has_restarted();
      if (ln_l > this->optimal_ln_l_) {
        this->best_rep_index_ = rep_index;
        this->optimal_ln_l_ = ln_l;
        this->n_iter_ = fitter->get_n_iter();
        std::memcpy(this->posterior_, posterior.data(),
                    n_data * n_cluster * sizeof(*this->posterior_));
        std::memcpy(this->prior_, prior.data(),
                    n_data * n_cluster * sizeof(*this->prior_));
        std::memcpy(this->estimated_prob_, estimated_prob.data(),
                    sum_outcomes * n_cluster * sizeof(*this->estimated_prob_));
        std::memcpy(
            this->regress_coeff_, regress_coeff.data(),
            n_feature * (n_cluster - 1) * sizeof(*this->regress_coeff_));
        if (this->best_initial_prob_) {
          std::memcpy(
              this->best_initial_prob_, best_initial_prob.data(),
              sum_outcomes * n_cluster * sizeof(*this->best_initial_prob_));
        }
      }
      this->results_lock_->unlock();

    } else {
      // all initial values used, stop working
      this->n_rep_done_lock_->unlock();
      is_working = false;
    }
  }
}
