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

#include "error_solver.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <span>

#include "RcppArmadillo.h"

polca_parallel::ErrorSolver::ErrorSolver(
    std::size_t n_data, std::size_t n_feature, std::size_t sum_outcomes,
    std::size_t n_cluster, std::size_t info_size, std::size_t jacobian_width,
    std::span<double> prior_error, std::span<double> prob_error,
    std::span<double> regress_coeff_error)
    : n_data_(n_data),
      n_feature_(n_feature),
      sum_outcomes_(sum_outcomes),
      n_cluster_(n_cluster),
      info_size_(info_size),
      jacobian_width_(jacobian_width),
      prior_error_(prior_error),
      prob_error_(prob_error),
      regress_coeff_error_(regress_coeff_error.data(),
                           n_feature * (n_cluster - 1),
                           n_feature * (n_cluster - 1), false, true) {}

polca_parallel::InfoEigenSolver::InfoEigenSolver(
    std::size_t n_data, std::size_t n_feature, std::size_t sum_outcomes,
    std::size_t n_cluster, std::size_t info_size, std::size_t jacobian_width,
    std::span<double> prior_error, std::span<double> prob_error,
    std::span<double> regress_coeff_error)
    : polca_parallel::ErrorSolver(n_data, n_feature, sum_outcomes, n_cluster,
                                  info_size, jacobian_width, prior_error,
                                  prob_error, regress_coeff_error) {}

void polca_parallel::InfoEigenSolver::Solve(const arma::Mat<double>& score,
                                            const arma::Mat<double>& jacobian) {
  arma::Mat<double> info_arma = score.t() * score;

  // required to ensure symmetry, especially if using a preconditioner in
  // further development of this code
  info_arma = arma::symmatu(info_arma);

  arma::Col<double> eigval;
  arma::Mat<double> eigvec;
  arma::eig_sym(eigval, eigvec, info_arma);

  // remove small eigenvalues, use same tol as in pinv
  // required as info is usually ill-conditioned
  // use std::numeric_limits<double>::epsilon() to reproduce pinv()
  double tol = static_cast<double>(this->info_size_) *
               eigval(this->info_size_ - 1) *
               std::numeric_limits<double>::epsilon();
  // take the sqrt inverse for large eigenvalues
  for (auto& eigval_i : eigval) {
    if (eigval_i < tol) {
      eigval_i = 0.0;
    } else {
      eigval_i = 1 / eigval_i;
    }
  }
  this->ExtractErrorGivenEigen(eigval, eigvec, jacobian);
}

void polca_parallel::InfoEigenSolver::ExtractErrorGivenEigen(
    const arma::Col<double>& eigval_inv, const arma::Mat<double>& eigvec,
    const arma::Mat<double>& jacobian) {
  // extract errors for the prior and outcome probs
  // do root columns sum of squares, faster than full matrix multiplication
  arma::Row<double> std_err = arma::vecnorm(
      arma::diagmat(arma::sqrt(eigval_inv)) * eigvec.t() * jacobian, 2, 0);

  std::copy_n(std_err.begin(), this->n_cluster_, this->prior_error_.begin());
  std::copy_n(std::next(std_err.begin(), this->n_cluster_),
              this->sum_outcomes_ * this->n_cluster_,
              this->prob_error_.begin());
}

polca_parallel::InfoEigenRegressSolver::InfoEigenRegressSolver(
    std::size_t n_data, std::size_t n_feature, std::size_t sum_outcomes,
    std::size_t n_cluster, std::size_t info_size, std::size_t jacobian_width,
    std::span<double> prior_error, std::span<double> prob_error,
    std::span<double> regress_coeff_error)
    : polca_parallel::InfoEigenSolver(
          n_data, n_feature, sum_outcomes, n_cluster, info_size, jacobian_width,
          prior_error, prob_error, regress_coeff_error) {}

void polca_parallel::InfoEigenRegressSolver::ExtractErrorGivenEigen(
    const arma::Col<double>& eigval_inv, const arma::Mat<double>& eigvec,
    const arma::Mat<double>& jacobian) {
  // extract errors for the prior and outcome probs
  this->InfoEigenSolver::ExtractErrorGivenEigen(eigval_inv, eigvec, jacobian);
  std::size_t size = this->n_feature_ * (this->n_cluster_ - 1);
  // then extract covariance matrix

  // make a copy of the submat which contains the dimensions for the
  // coefficients, this is used to create the covariance for the coefficients
  // no need to do full pinv(info) multiplication
  //
  // making a copy is faster than submat() * diagmat() * submat().t()
  arma::Mat<double> sub = eigvec.submat(0, 0, size - 1, this->info_size_ - 1);
  this->regress_coeff_error_ = sub * arma::diagmat(eigval_inv) * sub.t();
}

polca_parallel::ScoreSvdSolver::ScoreSvdSolver(
    std::size_t n_data, std::size_t n_feature, std::size_t sum_outcomes,
    std::size_t n_cluster, std::size_t info_size, std::size_t jacobian_width,
    std::span<double> prior_error, std::span<double> prob_error,
    std::span<double> regress_coeff_error)
    : polca_parallel::ErrorSolver(n_data, n_feature, sum_outcomes, n_cluster,
                                  info_size, jacobian_width, prior_error,
                                  prob_error, regress_coeff_error) {}

void polca_parallel::ScoreSvdSolver::Solve(const arma::Mat<double>& score,
                                           const arma::Mat<double>& jacobian) {
  // perhaps use a preconditioner like below
  // found to be unstable / doesn't reproduce similar results
  // eg error on probabilities can be much larger than 1.0
  // arma::Row<double> scale = 1 / arma::vecnorm(score_arma, "inf", 0);
  // score_arma = score_arma * arma::diagmat(scale);

  arma::Mat<double> u_mat;
  arma::Mat<double> v_mat;
  arma::Col<double> singular_values;
  arma::svd_econ(u_mat, singular_values, v_mat, score, "right");

  // use std::numeric_limits<float>::epsilon() as there were a few cases small
  // sigular values would get through this tol
  double tol = std::max<double>(this->info_size_, this->n_data_) *
               *singular_values.begin() *
               static_cast<double>(std::numeric_limits<float>::epsilon());
  for (auto& singular_val_i : singular_values) {
    if (singular_val_i < tol) {
      singular_val_i = 0.0;
    } else {
      singular_val_i = 1 / singular_val_i;
    }
  }

  this->ExtractErrorGivenEigen(singular_values, v_mat, jacobian);
}

void polca_parallel::ScoreSvdSolver::ExtractErrorGivenEigen(
    const arma::Col<double>& singular_inv, const arma::Mat<double>& v_mat,
    const arma::Mat<double>& jacobian) {
  // extract errors for the prior and outcome probs
  // do root columns sum of squares, faster than full matrix multiplication
  arma::Row<double> std_err =
      arma::vecnorm(arma::diagmat(singular_inv) * v_mat.t() * jacobian, 2, 0);

  std::copy_n(std_err.begin(), this->n_cluster_, this->prior_error_.begin());
  std::copy_n(std::next(std_err.begin(), this->n_cluster_),
              this->sum_outcomes_ * this->n_cluster_,
              this->prob_error_.begin());
}

polca_parallel::ScoreSvdRegressSolver::ScoreSvdRegressSolver(
    std::size_t n_data, std::size_t n_feature, std::size_t sum_outcomes,
    std::size_t n_cluster, std::size_t info_size, std::size_t jacobian_width,
    std::span<double> prior_error, std::span<double> prob_error,
    std::span<double> regress_coeff_error)
    : polca_parallel::ScoreSvdSolver(n_data, n_feature, sum_outcomes, n_cluster,
                                     info_size, jacobian_width, prior_error,
                                     prob_error, regress_coeff_error) {}

void polca_parallel::ScoreSvdRegressSolver::ExtractErrorGivenEigen(
    const arma::Col<double>& singular_inv, const arma::Mat<double>& v_mat,
    const arma::Mat<double>& jacobian) {
  // extract errors for the prior and outcome probs
  this->ScoreSvdSolver::ExtractErrorGivenEigen(singular_inv, v_mat, jacobian);

  std::size_t size = this->n_feature_ * (this->n_cluster_ - 1);
  // then extract covariance matrix

  // make a copy of the submat which contains the dimensions for the
  // coefficients, this is used to create the covariance for the coefficients
  // no need to do full pinv(info) multiplication
  //
  // making a copy is faster than submat().t() * diagmat() * submat()
  arma::Mat<double> sub = v_mat.submat(0, 0, size - 1, this->info_size_ - 1);
  this->regress_coeff_error_ =
      sub * arma::diagmat(singular_inv % singular_inv) * sub.t();
}
