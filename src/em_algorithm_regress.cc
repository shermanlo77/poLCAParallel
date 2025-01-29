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

#include "em_algorithm_regress.h"

#include <algorithm>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "util.h"

polca_parallel::EmAlgorithmRegress::EmAlgorithmRegress(
    std::span<double> features, std::span<int> responses,
    std::span<double> initial_prob, std::size_t n_data, std::size_t n_feature,
    polca_parallel::NOutcomes n_outcomes, std::size_t n_cluster,
    unsigned int max_iter, double tolerance, std::span<double> posterior,
    std::span<double> prior, std::span<double> estimated_prob,
    std::span<double> regress_coeff)
    : polca_parallel::EmAlgorithm(features, responses, initial_prob, n_data,
                                  n_feature, n_outcomes, n_cluster, max_iter,
                                  tolerance, posterior, prior, estimated_prob,
                                  regress_coeff),
      features_(features.data(), n_data, n_feature, false, true),
      regress_coeff_(regress_coeff.data(), n_feature, n_cluster - 1, false,
                     true),
      n_parameters_(n_feature * (n_cluster - 1)),
      gradient_(this->n_parameters_),
      hessian_(this->n_parameters_, this->n_parameters_) {
  this->init_regress_coeff();
}

void polca_parallel::EmAlgorithmRegress::Reset(
    std::uniform_real_distribution<double>& uniform) {
  this->polca_parallel::EmAlgorithm::Reset(uniform);
  this->init_regress_coeff();
}

void polca_parallel::EmAlgorithmRegress::InitPrior() {
  // matrix multiply: eta = features x regress_coeff
  // then into softmax, ie prior probability proportional to exp(eta)
  // restrict regress_coeff for the 0th cluster to be 0
  // for the 0th cluster, eta = 0, prior for 0th cluster propto 1
  this->prior_.col(0).fill(1.0);
  this->prior_.submat(0, 1, this->n_data_ - 1, this->n_cluster_ - 1) =
      arma::exp(this->features_ * this->regress_coeff_);

  // normalise so that prior_ are probabilities
  this->prior_.each_col() /= arma::sum(this->prior_, 1);
}

void polca_parallel::EmAlgorithmRegress::FinalPrior() {
  // do nothing, prior_ already in required format
}

double polca_parallel::EmAlgorithmRegress::GetPrior(std::size_t data_index,
                                                    std::size_t cluster_index) {
  return this->prior_[data_index + this->n_data_ * cluster_index];
}

bool polca_parallel::EmAlgorithmRegress::IsInvalidLikelihood(
    double ln_l_difference) {
  // comparing nan may be unclear, check nan first
  // check if the newton step decreases the log likelihood
  if (this->polca_parallel::EmAlgorithm::IsInvalidLikelihood(ln_l_difference)) {
    return true;
  } else {
    return ln_l_difference <
           polca_parallel::EmAlgorithmRegress::kMinLogLikelihoodDifference;
  }
}

bool polca_parallel::EmAlgorithmRegress::MStep() {
  // estimate outcome probabilities
  this->EstimateProbability();

  this->CalcGrad();
  this->CalcHess();

  // single Newton step
  try {
    auto result = arma::solve(this->hessian_, this->gradient_,
                              arma::solve_opts::likely_sympd);
    this->regress_coeff_ -=
        arma::reshape(std::move(result), arma::size(this->regress_coeff_));
  } catch (const std::runtime_error&) {
    return true;
  }

  // using new regression coefficients, update priors
  this->InitPrior();

  return false;
}

void polca_parallel::EmAlgorithmRegress::NormalWeightedSumProb(
    std::size_t cluster_index) {
  // override as the normaliser cannot be calculated using prior
  // using sum of posterior instead
  double normaliser = arma::sum(this->posterior_.unsafe_col(cluster_index));
  this->polca_parallel::EmAlgorithm::NormalWeightedSumProb(cluster_index,
                                                           normaliser);
}

void polca_parallel::EmAlgorithmRegress::init_regress_coeff() {
  this->regress_coeff_.fill(0.0);
}

void polca_parallel::EmAlgorithmRegress::CalcGrad() {
  auto gradient = this->gradient_.begin();
  for (std::size_t m = 1; m < this->n_cluster_; ++m) {
    auto posterior_m = this->posterior_.unsafe_col(m);
    auto prior_m = this->prior_.unsafe_col(m);
    arma::Col<double> post_minus_prior = posterior_m - prior_m;
    for (std::size_t p = 0; p < this->n_feature_; ++p) {
      *gradient = arma::dot(this->features_.unsafe_col(p), post_minus_prior);
      std::advance(gradient, 1);
    }
  }
}

void polca_parallel::EmAlgorithmRegress::CalcHess() {
  for (std::size_t cluster_j = 0; cluster_j < this->n_cluster_ - 1;
       ++cluster_j) {
    for (std::size_t cluster_i = cluster_j; cluster_i < this->n_cluster_ - 1;
         ++cluster_i) {
      this->CalcHessSubBlock(cluster_i, cluster_j);
    }
  }
}

void polca_parallel::EmAlgorithmRegress::CalcHessSubBlock(
    std::size_t cluster_index_0, std::size_t cluster_index_1) {
  // when retriving the prior and posterior, use cluster_index + 1 because
  // the hessian does not consider the 0th cluster as the regression
  // coefficient for the 0th cluster is set to zero

  auto posterior0 = this->posterior_.unsafe_col(cluster_index_0 + 1);
  auto prior0 = this->prior_.unsafe_col(cluster_index_0 + 1);

  // for the same cluster, copy over results as they will be modified
  bool is_same_cluster = cluster_index_0 == cluster_index_1;

  arma::Col<double> posterior1(
      this->posterior_.begin() + (cluster_index_1 + 1) * this->n_data_,
      this->n_data_, is_same_cluster);
  arma::Col<double> prior1(
      this->prior_.begin() + (cluster_index_1 + 1) * this->n_data_,
      this->n_data_, is_same_cluster);

  // Suppose r = posterior, pi = prior, u, v = cluster indexs
  // prior_post_inter is the following:
  // For same cluster, r_u*(1-r_u) - pi_u(1-pi_u)
  // For different clusters, pi_u pi_v - r_u r_v
  if (is_same_cluster) {
    posterior1 -= 1;
    prior1 -= 1;
  }
  arma::Col<double> prior_post_inter =
      prior0 % prior1 - posterior0 % posterior1;

  double hess_element;
  // iterate through features i, j, working out the elements of the hessian
  // symmetric matrix, so loop over diagonal and lower triangle
  for (std::size_t j = 0; j < this->n_feature_; ++j) {
    for (std::size_t i = j; i < this->n_feature_; ++i) {
      hess_element = CalcHessElement(i, j, prior_post_inter);
      this->AssignHessianAt(hess_element, cluster_index_0, cluster_index_1, i,
                            j);

      // hessian and each block is symmetric
      // copy over values to their mirror
      if (i != j) {
        this->AssignHessianAt(hess_element, cluster_index_0, cluster_index_1, j,
                              i);
      }

      if (cluster_index_0 != cluster_index_1) {
        this->AssignHessianAt(hess_element, cluster_index_1, cluster_index_0, i,
                              j);
        if (i != j) {
          this->AssignHessianAt(hess_element, cluster_index_1, cluster_index_0,
                                j, i);
        }
      }
    }
  }
}

double polca_parallel::EmAlgorithmRegress::CalcHessElement(
    std::size_t feature_index_0, std::size_t feature_index_1,
    arma::Col<double>& prior_post_inter) {
  return arma::sum(this->features_.unsafe_col(feature_index_0) %
                   this->features_.unsafe_col(feature_index_1) %
                   prior_post_inter);
}

void polca_parallel::EmAlgorithmRegress::AssignHessianAt(
    double hess_element, std::size_t cluster_index_0,
    std::size_t cluster_index_1, std::size_t feature_index_0,
    std::size_t feature_index_1) {
  std::size_t index = cluster_index_1 * this->n_parameters_ * this->n_feature_ +
                      feature_index_1 * this->n_parameters_ +
                      cluster_index_0 * this->n_feature_ + feature_index_0;
  this->hessian_[index] = hess_element;
}
