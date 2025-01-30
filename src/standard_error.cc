// poLCAParallel
// Copyright (C) 2024 Sherman Lo

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

#include "standard_error.h"

#include <algorithm>
#include <memory>
#include <span>
#include <vector>

#include "RcppArmadillo.h"
#include "error_solver.h"
#include "util.h"

polca_parallel::StandardError::StandardError(
    std::span<const double> features, std::span<const int> responses,
    std::span<const double> probs, std::span<double> prior,
    std::span<double> posterior, std::size_t n_data, std::size_t n_feature,
    polca_parallel::NOutcomes n_outcomes, std::size_t n_cluster,
    std::span<double> prior_error, std::span<double> prob_error,
    std::span<double> regress_coeff_error)
    : responses_(const_cast<int*>(responses.data()), n_data, n_outcomes.size(),
                 false, true),
      probs_(probs),
      prior_(prior.data(), n_data, n_cluster, false, true),
      posterior_(posterior.data(), n_data, n_cluster, false, true),
      n_data_(n_data),
      n_feature_(n_feature),
      n_outcomes_(n_outcomes),
      n_cluster_(n_cluster),
      prior_error_(prior_error),
      prob_error_(prob_error),
      regress_coeff_error_(regress_coeff_error),
      info_size_(n_feature_ * (n_cluster_ - 1) +
                 n_cluster_ * (n_outcomes.sum() - n_outcomes.size())),
      jacobian_width_(n_cluster_ + n_cluster_ * n_outcomes.sum()) {}

void polca_parallel::StandardError::Calc() {
  this->SmoothProbs();

  // calculate the score matrix
  arma::Mat<double> score(this->n_data_, this->info_size_);
  this->CalcScore(score);

  // calculate the Jacobian matrix
  arma::Mat<double> jacobian(this->info_size_, this->jacobian_width_);
  this->CalcJacobian(jacobian);

  // for solving equations, see error_solver.cc
  std::unique_ptr<polca_parallel::ErrorSolver> solver = this->InitErrorSolver();
  solver->Solve(score, jacobian);
}

void polca_parallel::StandardError::SmoothProbs() {
  if (this->smoother_) {
    this->smoother_->Smooth();
    this->probs_ = this->smoother_->get_probs();
    std::span<double> prior = this->smoother_->get_prior();
    this->prior_ = arma::Mat<double>(prior.data(), this->n_data_,
                                     this->n_cluster_, false, true);
    std::span<double> posterior = this->smoother_->get_posterior();
    this->posterior_ = arma::Mat<double>(posterior.data(), this->n_data_,
                                         this->n_cluster_, false, true);
  }
}

std::unique_ptr<polca_parallel::ErrorSolver>
polca_parallel::StandardError::InitErrorSolver() {
  return std::make_unique<polca_parallel::ScoreSvdSolver>(
      this->n_data_, this->n_feature_, this->n_outcomes_.sum(),
      this->n_cluster_, this->info_size_, this->jacobian_width_,
      this->prior_error_, this->prob_error_, this->regress_coeff_error_);
}

void polca_parallel::StandardError::CalcScore(arma::Mat<double>& score) const {
  // each call of CalcScorePrior and CalcScoreProbs fills in many of the columns
  // of the score design matrix
  // score is shifted automatically by these methods
  auto score_prior =
      score.cols(0, this->n_feature_ * (this->n_cluster_ - 1) - 1);
  this->CalcScorePrior(score_prior);
  auto score_probs =
      score.cols(this->n_feature_ * (this->n_cluster_ - 1), score.n_cols - 1);
  this->CalcScoreProbs(score_probs);
}

void polca_parallel::StandardError::CalcScorePrior(
    arma::subview<double>& score_prior) const {
  score_prior = this->posterior_.cols(1, this->n_cluster_ - 1) -
                this->prior_.cols(1, this->n_cluster_ - 1);
}

void polca_parallel::StandardError::CalcScoreProbs(
    arma::subview<double>& score_probs) const {
  // call CalcScoreProbsCol() for every cluster, category and outcome except
  // for the zeroth outcome
  std::size_t col = 0;
  auto prob = this->probs_.begin();
  for (std::size_t cluster_index = 0; cluster_index < this->n_cluster_;
       ++cluster_index) {
    // posterior for the given cluster
    auto posterior_i = this->posterior_.col(cluster_index);
    for (std::size_t category_index = 0;
         category_index < this->n_outcomes_.size(); ++category_index) {
      // response for the given category
      auto responses_j = this->responses_.col(category_index);
      std::advance(prob, 1);  // ignore the zeroth outcome
      for (std::size_t outcome_index = 1;
           outcome_index < this->n_outcomes_[category_index]; ++outcome_index) {
        auto score_col = score_probs.col(col++);
        this->CalcScoreProbsCol(outcome_index, *prob, responses_j, posterior_i,
                                score_col);
        std::advance(prob, 1);
      }
    }
  }
}

void polca_parallel::StandardError::CalcScoreProbsCol(
    std::size_t outcome_index, double prob,
    const arma::subview_col<int>& responses_j,
    const arma::subview_col<double>& posterior_i,
    arma::subview_col<double>& score_col) const {
  auto posterior_iter = posterior_i.cbegin();
  auto responses_iter = responses_j.cbegin();

  // boolean if the response is the same as the outcome
  // remember response is one index, not zero index
  // iterate for each data point
  for (auto& score_i : score_col) {
    if (*responses_iter > 0) {
      bool is_outcome =
          outcome_index == static_cast<std::size_t>(*responses_iter - 1);
      score_i = *posterior_iter * (static_cast<double>(is_outcome) - prob);
    } else {
      score_i = 0.0;
    }
    std::advance(posterior_iter, 1);
    std::advance(responses_iter, 1);
  }
}

void polca_parallel::StandardError::CalcJacobian(
    arma::Mat<double>& jacobian) const {
  // block matrix for prior
  auto jacobian_prior =
      jacobian.submat(0, 0, this->n_feature_ * (this->n_cluster_ - 1) - 1,
                      this->n_cluster_ - 1);
  this->CalcJacobianPrior(jacobian_prior);
  // block matrix for probs
  auto jacobian_probs = jacobian.submat(
      this->n_feature_ * (this->n_cluster_ - 1), this->n_cluster_,
      jacobian.n_rows - 1, jacobian.n_cols - 1);
  this->CalcJacobianProbs(jacobian_probs);
}

void polca_parallel::StandardError::CalcJacobianPrior(
    arma::subview<double>& jacobian_prior) const {
  // copy over the prior, they will be the same for all data points
  std::vector<double> prior(this->n_cluster_);
  for (std::size_t cluster_index = 0; cluster_index < this->n_cluster_;
       ++cluster_index) {
    prior[cluster_index] = this->prior_[cluster_index * this->n_data_];
  }
  this->CalcJacobianBlock(std::span<const double>(prior.cbegin(), prior.size()),
                          jacobian_prior);
}

void polca_parallel::StandardError::CalcJacobianProbs(
    arma::subview<double>& jacobian_probs) const {
  // block matrix for probs, one for each cluster and category pair
  std::size_t row_start = 0;
  std::size_t col_start = 0;
  auto probs = this->probs_.begin();
  for (std::size_t cluster_index = 0; cluster_index < this->n_cluster_;
       ++cluster_index) {
    for (std::size_t n_outcome : this->n_outcomes_) {
      auto jacobian_block =
          jacobian_probs.submat(row_start, col_start, row_start + n_outcome - 2,
                                col_start + n_outcome - 1);
      this->CalcJacobianBlock(std::span<const double>(probs, n_outcome),
                              jacobian_block);
      std::advance(probs, n_outcome);
      row_start += n_outcome - 1;
      col_start += n_outcome;
    }
  }
}

void polca_parallel::StandardError::CalcJacobianBlock(
    std::span<const double> probs,
    arma::subview<double>& jacobian_block) const {
  // dev notes: possible to do outer product of probs and then add to the off
  // diagonal, but note this method will commonly be used to create small
  // block matrices (ie n_prob typically be 2 or 3, the n_outcomes)
  auto jacobian = jacobian_block.begin();
  std::size_t n_prob = probs.size();
  // for each col
  for (std::size_t j = 0; j < n_prob; ++j) {
    // for each row
    for (std::size_t i = 1; i < n_prob; ++i) {
      *jacobian = -probs[i] * probs[j];
      if (i == j) {
        *jacobian += probs[i];
      }
      std::advance(jacobian, 1);
    }
  }
}
