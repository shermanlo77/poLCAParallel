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

polca_parallel::StandardError::StandardError(
    double* features, int* responses, double* probs, double* prior,
    double* posterior, int n_data, int n_feature, int n_category,
    int* n_outcomes, int sum_outcomes, int n_cluster, double* prior_error,
    double* prob_error, double* regress_coeff_error)
    : features_(features),
      responses_(responses),
      probs_(probs),
      prior_(prior),
      posterior_(posterior),
      n_data_(n_data),
      n_feature_(n_feature),
      n_category_(n_category),
      n_outcomes_(n_outcomes),
      sum_outcomes_(sum_outcomes),
      n_cluster_(n_cluster),
      prior_error_(prior_error),
      prob_error_(prob_error),
      regress_coeff_error_(regress_coeff_error),
      info_size_(n_feature_ * (n_cluster_ - 1) +
                 n_cluster_ * (sum_outcomes_ - n_category_)),
      jacobian_width_(n_cluster_ + n_cluster_ * sum_outcomes_) {}

void polca_parallel::StandardError::Calc() {
  this->SmoothProbs();

  // calcaulte the info matrix
  std::vector<double> info(this->info_size_ * this->info_size_);
  this->CalcInfo(info.data());

  // calculate the Jacobian matrix
  std::vector<double> jacobian(this->info_size_ * this->jacobian_width_);
  this->CalcJacobian(jacobian.data());

  this->ExtractError(info.data(), jacobian.data());
}

void polca_parallel::StandardError::SmoothProbs() {
  if (this->smoother_) {
    this->smoother_->Smooth();
    this->probs_ = this->smoother_->get_probs();
    this->prior_ = this->smoother_->get_prior();
    this->posterior_ = this->smoother_->get_posterior();
  }
}

void polca_parallel::StandardError::CalcInfo(double* info) {
  arma::Mat<double> score_arma(this->n_data_, this->info_size_);
  this->CalcScore(score_arma.memptr());
  arma::Mat<double> info_arma(info, this->info_size_, this->info_size_, false,
                              true);
  info_arma = score_arma.t() * score_arma;
}

void polca_parallel::StandardError::CalcScore(double* score) {
  // each call of CalcScorePrior and CalcScoreProbs fills in many of the columns
  // of the score design matrix
  // score is shifted automatically by these methods
  this->CalcScorePrior(&score);
  this->CalcScoreProbs(&score);
}

void polca_parallel::StandardError::CalcScorePrior(double** score) {
  int size = this->n_data_ * (this->n_cluster_ - 1);
  arma::Col<double> score_prior_arma(*score, size, false, true);
  arma::Col<double> prior_arma(this->prior_ + this->n_data_, size, false);
  arma::Col<double> posterior_arma(this->posterior_ + this->n_data_, size,
                                   false);
  score_prior_arma = posterior_arma - prior_arma;
  *score += size;
}

void polca_parallel::StandardError::CalcScoreProbs(double** score) {
  // call CalcScoreProbsCol() for every cluster, category and outcome except
  // for the zeroth outcome
  for (int cluster_index = 0; cluster_index < this->n_cluster_;
       ++cluster_index) {
    for (int category_index = 0; category_index < this->n_category_;
         ++category_index) {
      for (int outcome_index = 1;
           outcome_index < this->n_outcomes_[category_index]; ++outcome_index) {
        this->CalcScoreProbsCol(outcome_index, category_index, cluster_index,
                                *score);
        *score += this->n_data_;
      }
    }
  }
}

void polca_parallel::StandardError::CalcScoreProbsCol(int outcome_index,
                                                      int category_index,
                                                      int cluster_index,
                                                      double* score_start) {
  // start posterior for the given cluster
  double* posterior = this->posterior_ + cluster_index * this->n_data_;
  // start the response for the given category
  int* response = this->responses_ + category_index * this->n_data_;
  // start the prob for the given cluster, category and outcome triplet
  double* prob = this->probs_ + cluster_index * this->sum_outcomes_;
  for (int i = 0; i < category_index; ++i) {
    prob += this->n_outcomes_[i];
  }
  prob += outcome_index;

  // boolean if the response is the same as the outcome
  // remember response is one index, not zero index
  int is_outcome;
  // iterate for each data point
  // dev notes: it is possible to do an arma implementation using element-wise
  // equality and then case the umat to a Mat<double>
  for (double* score = score_start; score < score_start + this->n_data_;
       ++score) {
    is_outcome = outcome_index == (*response - 1);
    *score = *posterior * (static_cast<double>(is_outcome) - *prob);
    ++posterior;
    ++response;
  }
}

void polca_parallel::StandardError::CalcJacobian(double* jacobian) {
  std::fill(jacobian, jacobian + this->info_size_ * this->jacobian_width_, 0.0);
  // block matrix for prior
  this->CalcJacobianPrior(&jacobian);
  // block matrix for probs
  this->CalcJacobianProbs(&jacobian);
}

void polca_parallel::StandardError::CalcJacobianPrior(double** jacobian_ptr) {
  // copy over the prior, they will be the same for all data points
  double prior[this->n_cluster_];
  for (int cluster_index = 0; cluster_index < this->n_cluster_;
       ++cluster_index) {
    prior[cluster_index] = this->prior_[cluster_index * this->n_data_];
  }
  this->CalcJacobianBlock(prior, this->n_cluster_, jacobian_ptr);
}

void polca_parallel::StandardError::CalcJacobianProbs(double** jacobian_ptr) {
  // block matrix for probs, one for each cluster and category pair
  int n_outcome;
  double* probs = this->probs_;
  for (int cluster_index = 0; cluster_index < this->n_cluster_;
       ++cluster_index) {
    for (int category_index = 0; category_index < this->n_category_;
         ++category_index) {
      n_outcome = this->n_outcomes_[category_index];
      this->CalcJacobianBlock(probs, n_outcome, jacobian_ptr);
      probs += n_outcome;
    }
  }
}

void polca_parallel::StandardError::CalcJacobianBlock(double* probs, int n_prob,
                                                      double** jacobian_ptr) {
  // dev notes: possible to do outer product of probs and then add to the off
  // diagonal, but note this method will commonly be used to create small
  // block matrices (ie n_prob typically be 2 or 3, the n_outcomes)
  double* jacobian = *jacobian_ptr;
  // for each col
  for (int j = 0; j < n_prob; ++j) {
    // for each row
    for (int i = 1; i < n_prob; ++i) {
      *jacobian = -probs[i] * probs[j];
      if (i == j) {
        *jacobian += probs[i];
      }
      ++jacobian;
    }
    // shift to next column
    *jacobian_ptr += this->info_size_;
    jacobian = *jacobian_ptr;
  }
  // shift to the bottom right corner of the block matrix, ready for next block
  // matrix
  *jacobian_ptr += n_prob - 1;
}

void polca_parallel::StandardError::ExtractError(double* info,
                                                 double* jacobian) {
  arma::Mat<double> info_arma(info, this->info_size_, this->info_size_, false);

  // required to ensure symmetry, espically if using a preconditioner in further
  // development of this code
  info_arma = arma::symmatu(info_arma);

  arma::Col<double> eigval;
  arma::Mat<double> eigvec;
  arma::eig_sym(eigval, eigvec, info_arma);

  // remove small eigenvalues, use same tol as in pinv
  // required as info is usually ill-conditioned
  double tol = this->info_size_ * eigval[this->info_size_ - 1] *
               std::numeric_limits<double>::epsilon();
  // take the sqrt inverse for large eigenvalues
  for (double* eigval_i = eigval.begin(); eigval_i < eigval.end(); ++eigval_i) {
    if (*eigval_i < tol) {
      *eigval_i = 0.0;
    } else {
      *eigval_i = 1 / *eigval_i;
    }
  }
  this->ExtractErrorGivenEigen(&eigval, &eigvec, jacobian);
}

void polca_parallel::StandardError::ExtractErrorGivenEigen(
    arma::Col<double>* eigval_inv, arma::Mat<double>* eigvec,
    double* jacobian) {
  // extract errors for the prior and outcome probs
  arma::Mat<double> jac_arma(jacobian, this->info_size_, this->jacobian_width_,
                             false);
  // do root columns sum of squares, faster than full matrix multiplication
  arma::Row<double> std_err = arma::vecnorm(
      arma::diagmat(arma::sqrt(*eigval_inv)) * eigvec->t() * jac_arma, 2, 0);

  memcpy(this->prior_error_, std_err.memptr(),
         this->n_cluster_ * sizeof(*this->prior_error_));
  memcpy(this->prob_error_, std_err.memptr() + this->n_cluster_,
         this->sum_outcomes_ * this->n_cluster_ * sizeof(*this->prob_error_));
}
