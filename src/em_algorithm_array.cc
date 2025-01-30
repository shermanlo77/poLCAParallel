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
#include <span>
#include <thread>
#include <vector>

#include "em_algorithm.h"
#include "em_algorithm_nan.h"
#include "em_algorithm_regress.h"
#include "util.h"

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
    std::span<const double> features, std::span<const int> responses,
    std::span<const double> initial_prob, std::size_t n_data,
    std::size_t n_feature, polca_parallel::NOutcomes n_outcomes,
    std::size_t n_cluster, std::size_t n_rep, std::size_t n_thread,
    unsigned int max_iter, double tolerance, std::span<double> posterior,
    std::span<double> prior, std::span<double> estimated_prob,
    std::span<double> regress_coeff)
    : features_(features),
      responses_(responses),
      n_data_(n_data),
      n_feature_(n_feature),
      n_outcomes_(n_outcomes),
      n_cluster_(n_cluster),
      max_iter_(max_iter),
      tolerance_(tolerance),
      posterior_(posterior),
      prior_(prior),
      estimated_prob_(estimated_prob),
      regress_coeff_(regress_coeff),
      n_rep_(n_rep),
      initial_prob_(initial_prob),
      n_thread_(std::min(n_thread, n_rep)) {}

template <typename EmAlgorithmType>
void polca_parallel::EmAlgorithmArray::Fit() {
  // parallel run FitThread
  std::vector<std::thread> thread_array(this->n_thread_ - 1);
  for (std::thread& thread : thread_array) {
    thread = std::thread(&EmAlgorithmArray::FitThread<EmAlgorithmType>, this);
  }
  // main thread run
  this->FitThread<EmAlgorithmType>();
  // join threads
  for (std::thread& thread : thread_array) {
    thread.join();
  }
}

void polca_parallel::EmAlgorithmArray::SetSeed(std::seed_seq& seed) {
  this->seed_array_ = std::make_unique<std::vector<unsigned>>(this->n_rep_);
  seed.generate(seed_array_->begin(), seed_array_->end());
}

void polca_parallel::EmAlgorithmArray::set_best_initial_prob(
    std::span<double> best_initial_prob) {
  this->best_initial_prob_ = best_initial_prob;
}

void polca_parallel::EmAlgorithmArray::set_ln_l_array(
    std::span<double> ln_l_array) {
  this->ln_l_array_ = ln_l_array;
}

std::size_t polca_parallel::EmAlgorithmArray::get_best_rep_index() const {
  return this->best_rep_index_;
}

double polca_parallel::EmAlgorithmArray::get_optimal_ln_l() const {
  return this->optimal_ln_l_;
}

unsigned int polca_parallel::EmAlgorithmArray::get_n_iter() const {
  return this->n_iter_;
}

bool polca_parallel::EmAlgorithmArray::get_has_restarted() const {
  return this->has_restarted_;
}

void polca_parallel::EmAlgorithmArray::SetFitterRng(
    std::size_t rep_index, polca_parallel::EmAlgorithm& fitter) {
  if (this->seed_array_) {
    fitter.set_seed(this->seed_array_->at(rep_index));
  }
}

void polca_parallel::EmAlgorithmArray::MoveRngBackFromFitter(
    polca_parallel::EmAlgorithm& fitter) {
  // do nothing, no rng handelled here
}

template <typename EmAlgorithmType>
void polca_parallel::EmAlgorithmArray::FitThread() {
  std::size_t n_data = this->n_data_;
  std::size_t n_feature = this->n_feature_;
  std::size_t n_cluster = this->n_cluster_;

  // allocate memory for this thread
  std::vector<double> posterior(n_data * n_cluster);
  std::vector<double> prior(n_data * n_cluster);
  std::vector<double> estimated_prob(this->n_outcomes_.sum() * n_cluster);
  std::vector<double> regress_coeff(n_feature * (n_cluster - 1));
  std::vector<double> best_initial_prob(this->n_outcomes_.sum() * n_cluster);

  // transfer pointer to data and where to store results
  std::unique_ptr<polca_parallel::EmAlgorithm> fitter;

  // which initial probability this thread is working on
  std::size_t rep_index;
  double ln_l;

  bool is_working = true;
  while (is_working) {
    // lock to retrive initial probability and other variables
    // lock is outside the if statement so that this->n_rep_done_ can be read
    // without modification from other threads
    // shall be unlocked in both if and else branches
    this->n_rep_done_lock_.lock();
    if (this->n_rep_done_ < this->n_rep_) {
      // increment for the next worker to work on
      rep_index = this->n_rep_done_++;

      this->n_rep_done_lock_.unlock();

      fitter = std::make_unique<EmAlgorithmType>(
          this->features_, this->responses_,
          this->initial_prob_.subspan(
              rep_index * this->n_outcomes_.sum() * n_cluster,
              this->n_outcomes_.sum() * n_cluster),
          n_data, n_feature, this->n_outcomes_, n_cluster, this->max_iter_,
          this->tolerance_,
          std::span<double>(posterior.begin(), posterior.size()),
          std::span<double>(prior.begin(), prior.size()),
          std::span<double>(estimated_prob.begin(), estimated_prob.size()),
          std::span<double>(regress_coeff.begin(), regress_coeff.size()));
      if (this->best_initial_prob_) {
        fitter->set_best_initial_prob(std::span<double>(
            best_initial_prob.begin(), best_initial_prob.size()));
      }

      // each repetition uses their own rng
      this->SetFitterRng(rep_index, *fitter);

      fitter->Fit();
      ln_l = fitter->get_ln_l();
      if (this->ln_l_array_) {
        this->ln_l_array_.value()[rep_index] = ln_l;
      }

      // if ownership of rng transferred (if any) to fitter, get it back if
      // needed
      this->MoveRngBackFromFitter(*fitter);

      // copy results if log likelihood improved
      this->results_lock_.lock();
      this->has_restarted_ |= fitter->get_has_restarted();
      if (ln_l > this->optimal_ln_l_) {
        this->best_rep_index_ = rep_index;
        this->optimal_ln_l_ = ln_l;
        this->n_iter_ = fitter->get_n_iter();

        std::copy(posterior.cbegin(), posterior.cend(),
                  this->posterior_.begin());
        std::copy(prior.cbegin(), prior.cend(), this->prior_.begin());
        std::copy(estimated_prob.cbegin(), estimated_prob.cend(),
                  this->estimated_prob_.begin());
        std::copy(regress_coeff.cbegin(), regress_coeff.cend(),
                  this->regress_coeff_.begin());
        if (this->best_initial_prob_) {
          std::copy(best_initial_prob.cbegin(), best_initial_prob.cend(),
                    this->best_initial_prob_.value().begin());
        }
      }
      this->results_lock_.unlock();

    } else {
      // all initial values used, stop working
      this->n_rep_done_lock_.unlock();
      is_working = false;
    }
  }
}
