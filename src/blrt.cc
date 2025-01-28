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

#include "blrt.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <random>
#include <span>
#include <thread>
#include <vector>

polca_parallel::Blrt::Blrt(
    std::span<double> prior_null, std::span<double> prob_null,
    std::size_t n_cluster_null, std::span<double> prior_alt,
    std::span<double> prob_alt, std::size_t n_cluster_alt, std::size_t n_data,
    std::size_t n_category, polca_parallel::NOutcomes n_outcomes,
    std::size_t n_bootstrap, std::size_t n_rep, std::size_t n_thread,
    unsigned int max_iter, double tolerance, std::span<double> ratio_array)
    : prior_null_(prior_null),
      prob_null_(prob_null),
      n_cluster_null_(n_cluster_null),
      prior_alt_(prior_alt),
      prob_alt_(prob_alt),
      n_cluster_alt_(n_cluster_alt),
      n_data_(n_data),
      n_category_(n_category),
      n_outcomes_(n_outcomes),
      n_bootstrap_(n_bootstrap),
      n_rep_(n_rep),
      n_thread_(n_thread),
      max_iter_(max_iter),
      tolerance_(tolerance),
      ratio_array_(ratio_array),
      seed_array_(n_bootstrap) {
  // default to random seeds
  std::seed_seq seed(
      {std::chrono::system_clock::now().time_since_epoch().count()});
  this->SetSeed(seed);
}

void polca_parallel::Blrt::SetSeed(std::seed_seq& seed) {
  seed.generate(this->seed_array_.begin(), this->seed_array_.end());
}

void polca_parallel::Blrt::Run() {
  std::vector<std::thread> thread_array(this->n_thread_ - 1);
  for (std::thread& thread : thread_array) {
    thread = std::thread(&Blrt::RunThread, this);
  }
  // main thread run
  this->RunThread();
  // join threads
  for (std::thread& thread : thread_array) {
    thread.join();
  }
}

void polca_parallel::Blrt::RunThread() {
  bool is_working = true;
  std::size_t i_bootstrap;

  // to store the bootstrap samples
  std::vector<int> bootstrap_data(this->n_data_ * this->n_category_);
  std::span<int> bootstrap_span(bootstrap_data.begin(), bootstrap_data.size());

  // allocate memory for storing initial values for the probabilities
  std::vector<double> init_prob_null(this->n_outcomes_.sum() *
                                     this->n_cluster_null_ * this->n_rep_);
  std::vector<double> init_prob_alt(this->n_outcomes_.sum() *
                                    this->n_cluster_alt_ * this->n_rep_);

  // use the fitted values as the initial values when fitting onto the bootstrap
  // samples
  std::copy(this->prob_null_.begin(), this->prob_null_.end(),
            init_prob_null.begin());
  std::copy(this->prob_alt_.begin(), this->prob_alt_.end(),
            init_prob_alt.begin());

  // allocate memory for all required arrays, a lot of them aren't used after
  // fitting
  std::span<double> features;
  std::vector<double> fitted_posterior_null(this->n_data_ *
                                            this->n_cluster_null_);
  std::vector<double> fitted_posterior_alt(this->n_data_ *
                                           this->n_cluster_alt_);
  std::vector<double> fitted_prior_null(this->n_data_ * this->n_cluster_null_);
  std::vector<double> fitted_prior_alt(this->n_data_ * this->n_cluster_alt_);
  std::vector<double> fitted_prob_null(this->n_cluster_null_ *
                                       this->n_outcomes_.sum());
  std::vector<double> fitted_prob_alt(this->n_cluster_alt_ *
                                      this->n_outcomes_.sum());
  std::vector<double> fitted_regress_coeff_null(this->n_cluster_null_ - 1);
  std::vector<double> fitted_regress_coeff_alt(this->n_cluster_alt_ - 1);

  while (is_working) {
    // lock to retrive n_bootstrap_done_
    // shall be unlocked in both if and else branches
    this->n_bootstrap_done_lock_.lock();
    if (this->n_bootstrap_done_ < this->n_bootstrap_) {
      // increment for the next worker to work on
      i_bootstrap = this->n_bootstrap_done_++;
      this->n_bootstrap_done_lock_.unlock();

      // instantiate a rng
      std::unique_ptr<std::mt19937_64> rng =
          std::make_unique<std::mt19937_64>(this->seed_array_.at(i_bootstrap));

      std::uniform_real_distribution<double> uniform(0.0, 1.0);

      // generate new initial values
      for (std::size_t i_rep = 1; i_rep < this->n_rep_; ++i_rep) {
        arma::Mat<double> init_prob_null_i(
            init_prob_null.data() +
                i_rep * this->n_outcomes_.sum() * this->n_cluster_null_,
            this->n_outcomes_.sum(), this->n_cluster_null_, false, true);

        arma::Mat<double> init_prob_alt_i(
            init_prob_alt.data() +
                i_rep * this->n_outcomes_.sum() * this->n_cluster_alt_,
            this->n_outcomes_.sum(), this->n_cluster_alt_, false, true);

        polca_parallel::GenerateNewProb(
            *rng, uniform, this->n_outcomes_, this->n_category_,
            this->n_cluster_null_, init_prob_null_i);

        polca_parallel::GenerateNewProb(*rng, uniform, this->n_outcomes_,
                                        this->n_category_, this->n_cluster_alt_,
                                        init_prob_alt_i);
      }

      // bootstrap data using null model
      this->Bootstrap(this->prior_null_, this->prob_null_,
                      this->n_cluster_null_, *rng, bootstrap_span);

      // null model fit
      polca_parallel::EmAlgorithmArraySerial null_model(
          features, bootstrap_span,
          std::span<double>(init_prob_null.begin(), init_prob_null.size()),
          this->n_data_, 1, this->n_category_, this->n_outcomes_,
          this->n_cluster_null_, this->n_rep_, this->max_iter_,
          this->tolerance_,
          std::span<double>(fitted_posterior_null.begin(),
                            fitted_posterior_null.size()),
          std::span<double>(fitted_prior_null.begin(),
                            fitted_prior_null.size()),
          std::span<double>(fitted_prob_null.begin(), fitted_prob_null.size()),
          std::span<double>(fitted_regress_coeff_null.begin(),
                            fitted_regress_coeff_null.size()));
      null_model.SetRng(&rng);
      null_model.Fit<polca_parallel::EmAlgorithm>();
      rng = null_model.MoveRng();

      // alt model fit
      polca_parallel::EmAlgorithmArraySerial alt_model(
          features, bootstrap_span,
          std::span<double>(init_prob_alt.begin(), init_prob_alt.size()),
          this->n_data_, 1, this->n_category_, this->n_outcomes_,
          this->n_cluster_alt_, this->n_rep_, this->max_iter_, this->tolerance_,
          std::span<double>(fitted_posterior_alt.begin(),
                            fitted_posterior_alt.size()),
          std::span<double>(fitted_prior_alt.begin(), fitted_prior_alt.size()),
          std::span<double>(fitted_prob_alt.begin(), fitted_prob_alt.size()),
          std::span<double>(fitted_regress_coeff_alt.begin(),
                            fitted_regress_coeff_alt.size()));
      alt_model.SetRng(&rng);
      alt_model.Fit<polca_parallel::EmAlgorithm>();
      rng = alt_model.MoveRng();

      // work out the log ratio, save it
      this->ratio_array_[i_bootstrap] =
          2 * (alt_model.get_optimal_ln_l() - null_model.get_optimal_ln_l());

    } else {
      // all bootstrap samples done, stop working
      this->n_bootstrap_done_lock_.unlock();
      is_working = false;
    }
  }
}

void polca_parallel::Blrt::Bootstrap(std::span<double> prior,
                                     std::span<double> prob,
                                     std::size_t n_cluster,
                                     std::mt19937_64& rng,
                                     std::span<int> response) {
  std::uniform_int_distribution<std::size_t> prior_dist(0, n_cluster - 1);
  std::uniform_real_distribution<double> uniform_dist(0, 1);

  auto response_iter = response.begin();

  for (std::size_t i_data = 0; i_data < this->n_data_; ++i_data) {
    std::size_t i_cluster = prior_dist(rng);  // select a random cluster
    // point to the corresponding probabilites for this random cluster
    auto prob_i = prob.begin();
    std::advance(prob_i, i_cluster * this->n_outcomes_.sum());

    for (std::size_t n_outcome : this->n_outcomes_) {
      double p_sum = 0.0;  // used to sum up probabilities for each outcome
      double rand_uniform = uniform_dist(rng);

      // use rand_uniform to randomly sample an outcome according to the
      // probabilities in prob_i_cluster[0], prob_i_cluster[1], ...
      // i_outcome is the randomly selected outcome
      std::size_t i_outcome = 0;
      while (rand_uniform > p_sum) {
        p_sum += *std::next(prob_i, i_outcome++);
      }
      *response_iter = i_outcome;  // response is one-based index

      // increment for the next category
      std::advance(prob_i, n_outcome);
      std::advance(response_iter, 1);
    }
  }
}
