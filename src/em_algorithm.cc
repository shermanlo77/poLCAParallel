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

#include "em_algorithm.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <random>
#include <span>
#include <utility>
#include <vector>

#include "RcppArmadillo.h"
#include "util.h"

polca_parallel::EmAlgorithm::EmAlgorithm(
    std::span<double> features, std::span<int> responses,
    std::span<double> initial_prob, std::size_t n_data, std::size_t n_feature,
    polca_parallel::NOutcomes n_outcomes, std::size_t n_cluster,
    unsigned int max_iter, double tolerance, std::span<double> posterior,
    std::span<double> prior, std::span<double> estimated_prob,
    std::span<double> regress_coeff)
    : responses_(responses),
      initial_prob_(initial_prob),
      n_data_(n_data),
      n_outcomes_(n_outcomes),
      n_cluster_(n_cluster),
      max_iter_(max_iter),
      tolerance_(tolerance),
      posterior_(posterior.data(), n_data, n_cluster, false, true),
      prior_(prior.data(), n_data, n_cluster, false, true),
      estimated_prob_(estimated_prob.data(), n_outcomes.sum(), n_cluster, false,
                      true),
      ln_l_array_(n_data) {
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  this->rng_ = std::make_unique<std::mt19937_64>(seed);
}

void polca_parallel::EmAlgorithm::Fit() {
  bool is_first_run = true;
  bool is_success = false;

  std::uniform_real_distribution<double> uniform(0.0, 1.0);

  while (!is_success) {
    if (is_first_run) {
      // copy initial prob to estimated prob
      std::copy(this->initial_prob_.begin(), this->initial_prob_.end(),
                this->estimated_prob_.begin());
    } else {
      // reach this condition if the first run has a problem
      // reset all required parameters
      this->Reset(uniform);
    }

    // make a copy initial probabilities if requested
    if (this->best_initial_prob_) {
      std::copy(this->estimated_prob_.begin(), this->estimated_prob_.end(),
                this->best_initial_prob_.value().begin());
    }

    double ln_l_before = -INFINITY;

    // initalise prior probabilities, for each cluster
    this->InitPrior();

    // do EM algorithm
    // assume successful until find error
    //
    // iteration goes:
    // initial E step (after which, n_iter=0, no effective iterations)
    // M step, E step (after which, n_iter=1, one effective iterations)
    // M step, E step (after which, n_iter=2, two effective iterations)
    // and so on until stopping condition
    // or when after an E step and n_iter=max_iter, making max_iter
    // effective iterations
    is_success = true;
    for (this->n_iter_ = 0; this->n_iter_ <= this->max_iter_; ++this->n_iter_) {
      // E step updates prior probabilities
      this->EStep();

      // E step updates ln_l_array_, use that to calculate log likelihood
      this->ln_l_ = arma::sum(this->ln_l_array_);

      // check for any errors
      double ln_l_difference = this->ln_l_ - ln_l_before;
      if (this->IsInvalidLikelihood(ln_l_difference)) {
        is_success = false;
        break;
      }

      // check stopping conditions
      if (ln_l_difference < this->tolerance_) {
        break;
      }
      if (this->n_iter_ == this->max_iter_) {
        break;
      }
      ln_l_before = this->ln_l_;

      // M step updates posterior probabilities and estimated probabilities
      // break if m step has an error
      if (this->MStep()) {
        is_success = false;
        break;
      }
    }
    is_first_run = false;
  }

  // reformat prior
  this->FinalPrior();
}

// Set where to store initial probabilities (optional)
void polca_parallel::EmAlgorithm::set_best_initial_prob(
    std::span<double> best_initial_prob) {
  this->best_initial_prob_ = best_initial_prob;
}

double polca_parallel::EmAlgorithm::get_ln_l() const { return this->ln_l_; }

unsigned int polca_parallel::EmAlgorithm::get_n_iter() const {
  return this->n_iter_;
}

bool polca_parallel::EmAlgorithm::get_has_restarted() const {
  return this->has_restarted_;
}

void polca_parallel::EmAlgorithm::set_seed(unsigned seed) {
  this->rng_ = std::make_unique<std::mt19937_64>(seed);
}

void polca_parallel::EmAlgorithm::set_rng(
    std::unique_ptr<std::mt19937_64> rng) {
  this->rng_ = std::move(rng);
}

std::unique_ptr<std::mt19937_64> polca_parallel::EmAlgorithm::move_rng() {
  return std::move(this->rng_);
}

void polca_parallel::EmAlgorithm::Reset(
    std::uniform_real_distribution<double>& uniform) {
  // generate random number for estimated_prob_
  this->has_restarted_ = true;
  polca_parallel::GenerateNewProb(this->n_outcomes_, this->n_cluster_, uniform,
                                  *this->rng_, this->estimated_prob_);
}

void polca_parallel::EmAlgorithm::InitPrior() {
  // prior probabilities are the same for all data points in this
  // implementation
  auto prior = this->prior_.begin();
  std::fill(prior, std::next(prior, this->n_cluster_),
            1.0 / static_cast<double>(this->n_cluster_));
}

void polca_parallel::EmAlgorithm::FinalPrior() {
  // Copying prior probabilities as each data point as the same prior
  auto prior = this->prior_.begin();
  std::vector<double> prior_copy(this->n_cluster_);
  std::copy(prior, std::next(prior, this->n_cluster_), prior_copy.begin());
  for (std::size_t m = 0; m < this->n_cluster_; ++m) {
    this->prior_.col(m).fill(prior_copy.at(m));
  }
}

double polca_parallel::EmAlgorithm::GetPrior(
    const std::size_t data_index, const std::size_t cluster_index) const {
  return this->prior_[cluster_index];
}

void polca_parallel::EmAlgorithm::EStep() {
  for (std::size_t i_data = 0; i_data < this->n_data_; ++i_data) {
    auto responses_i = this->responses_.subspan(
        i_data * this->n_outcomes_.size(), this->n_outcomes_.size());
    for (std::size_t i_cluster = 0; i_cluster < this->n_cluster_; ++i_cluster) {
      // access to posterior_ in this manner should result in cache misses
      // however PosteriorUnnormalize() is designed for cache efficiency
      double prior = this->GetPrior(i_data, i_cluster);
      auto estimated_prob = this->estimated_prob_.unsafe_col(i_cluster);
      this->posterior_[i_cluster * this->n_data_ + i_data] =
          this->PosteriorUnnormalize(responses_i, prior, estimated_prob);
    }
  }

  this->ln_l_array_ = arma::sum(this->posterior_, 1);  // row sum
  this->posterior_.each_col() /= this->ln_l_array_;  // normalise by the row sum
  this->ln_l_array_ = arma::log(this->ln_l_array_);  // log likelihood
}

double polca_parallel::EmAlgorithm::PosteriorUnnormalize(
    std::span<const int> responses_i, double prior,
    const arma::Col<double>& estimated_prob) const {
  return polca_parallel::PosteriorUnnormalize(responses_i, this->n_outcomes_,
                                              estimated_prob, prior);
}

bool polca_parallel::EmAlgorithm::IsInvalidLikelihood(
    double ln_l_difference) const {
  return std::isnan(this->ln_l_);
}

bool polca_parallel::EmAlgorithm::MStep() {
  // estimate prior
  // for this implementation, the mean posterior, taking the mean over data
  // points
  arma::Row<double> prior(this->prior_.begin(), this->n_cluster_, false, true);
  prior = arma::mean(this->posterior_, 0);

  // estimate outcome probabilities
  this->EstimateProbability();

  return false;
}

void polca_parallel::EmAlgorithm::EstimateProbability() {
  // set all estimated response probability to zero
  this->estimated_prob_.fill(0.0);
  // for each cluster
  for (std::size_t m = 0; m < this->n_cluster_; ++m) {
    // estimate outcome probabilities
    this->WeightedSumProb(m);
    this->NormalWeightedSumProb(m);
  }
}

void polca_parallel::EmAlgorithm::WeightedSumProb(
    const std::size_t cluster_index) {
  auto y = this->responses_.begin();
  // point to outcome probabilites for given cluster for the zeroth category
  arma::Col<double> estimated_prob_col =
      this->estimated_prob_.unsafe_col(cluster_index);

  for (double posterior_i : this->posterior_.unsafe_col(cluster_index)) {
    auto estimated_prob_iter = estimated_prob_col.begin();
    for (std::size_t n_outcome_j : this->n_outcomes_) {
      // selective summing of posterior
      *std::next(estimated_prob_iter, *y - 1) += posterior_i;
      // point to next category
      std::advance(y, 1);
      std::advance(estimated_prob_iter, n_outcome_j);
    }
  }
}

void polca_parallel::EmAlgorithm::NormalWeightedSumProb(
    const std::size_t cluster_index) {
  // in this implementation, normalise by n_data * prior
  //
  // note that the mean (over n data points) of posteriors is the prior
  //
  // by using the prior, you avoid having to do another sum
  this->NormalWeightedSumProb(
      cluster_index,
      static_cast<double>(this->n_data_) * this->prior_[cluster_index]);
}

void polca_parallel::EmAlgorithm::NormalWeightedSumProb(
    const std::size_t cluster_index, double normaliser) {
  // normalise by the sum of posteriors
  // calculations can be reused as the prior is the mean of posteriors
  // from the E step
  this->estimated_prob_.unsafe_col(cluster_index) /= normaliser;
}

template double polca_parallel::PosteriorUnnormalize<false>(
    std::span<const int> responses_i, std::span<const std::size_t> n_outcomes,
    const arma::Col<double>& estimated_prob, double prior);

template double polca_parallel::PosteriorUnnormalize<true>(
    std::span<const int> responses_i, std::span<const std::size_t> n_outcomes,
    const arma::Col<double>& estimated_prob, double prior);

template <bool is_check_zero>
double polca_parallel::PosteriorUnnormalize(
    std::span<const int> responses_i, std::span<const std::size_t> n_outcomes,
    const arma::Col<double>& estimated_prob, double prior) {
  // designed for cache efficiency here

  // used for calculating the posterior probability up to a constant
  // P(cluster m | Y^{(i)})
  double posterior;
  // for getting a response from responses_
  auto responses_i_it = responses_i.begin();

  bool use_sum_log = false;

  // used for likelihood calculation for a data point
  // P(Y^{(i)} | cluster m)
  double likelihood = 1;

  auto estimated_prob_it = estimated_prob.begin();

  // calculate conditioned on cluster m likelihood
  for (std::size_t n_outcome : n_outcomes) {
    int y = *responses_i_it;  // cache hit by accesing adjacent memory
    std::advance(responses_i_it, 1);
    // cache hit in estimated_prob by accesing memory n_outcomes + y -1 away

    if constexpr (is_check_zero) {
      if (y > 0) {
        likelihood *= *std::next(estimated_prob_it, y - 1);
      }
    } else {
      likelihood *= *std::next(estimated_prob_it, y - 1);
    }

    // increment to point to the next category
    std::advance(estimated_prob_it, n_outcome);

    // check for underflow
    if (likelihood < polca_parallel::EmAlgorithm::kUnderflowThreshold) {
      use_sum_log = true;
      break;
    }
  }

  // if underflow occured, use sum of logs instead
  // restart calculation
  if (!use_sum_log) {
    posterior = likelihood * prior;
  } else {
    double log_likelihood = 0;
    auto estimated_prob_it_2 = estimated_prob.begin();
    // for getting a response from responses_
    auto responses_i_it_2 = responses_i.begin();
    // calculate conditioned on cluster m likelihood
    for (std::size_t n_outcome : n_outcomes) {
      int y = *responses_i_it_2;  // cache hit by accesing adjacent memory
      std::advance(responses_i_it_2, 1);
      // cache hit in estimated_prob by accessing memory n_outcomes + y -1
      // away
      if constexpr (is_check_zero) {
        if (y > 0) {
          log_likelihood += std::log(*std::next(estimated_prob_it_2, y - 1));
        }
      } else {
        log_likelihood += std::log(*std::next(estimated_prob_it_2, y - 1));
      }
      // increment to point to the next category
      std::advance(estimated_prob_it_2, n_outcome);
    }
    posterior = log_likelihood + std::log(prior);
    posterior = std::exp(posterior);
  }

  return posterior;
}

void polca_parallel::GenerateNewProb(
    std::span<const size_t> n_outcomes, const std::size_t n_cluster,
    std::uniform_real_distribution<double>& uniform, std::mt19937_64& rng,
    arma::Mat<double>& prob) {
  for (auto& prob_i : prob) {
    prob_i = uniform(rng);
  }
  // normalise to probabilities
  for (std::size_t m = 0; m < n_cluster; ++m) {
    auto prob_col = prob.unsafe_col(m).begin();
    for (std::size_t n_outcome_i : n_outcomes) {
      arma::Col<double> prob_vector(prob_col, n_outcome_i, false, true);
      prob_vector /= arma::sum(prob_vector);
      std::advance(prob_col, n_outcome_i);
    }
  }
}
