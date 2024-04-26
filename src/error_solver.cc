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

polca_parallel::ErrorSolver::ErrorSolver(int n_data, int n_feature,
                                         int sum_outcomes, int n_cluster,
                                         int info_size, int jacobian_width,
                                         double* prior_error,
                                         double* prob_error,
                                         double* regress_coeff_error)
    : n_data_(n_data),
      n_feature_(n_feature),
      sum_outcomes_(sum_outcomes),
      n_cluster_(n_cluster),
      info_size_(info_size),
      jacobian_width_(jacobian_width),
      prior_error_(prior_error),
      prob_error_(prob_error),
      regress_coeff_error_(regress_coeff_error) {}

polca_parallel::InfoEigenSolver::InfoEigenSolver(
    int n_data, int n_feature, int sum_outcomes, int n_cluster, int info_size,
    int jacobian_width, double* prior_error, double* prob_error,
    double* regress_coeff_error)
    : polca_parallel::ErrorSolver(n_data, n_feature, sum_outcomes, n_cluster,
                                  info_size, jacobian_width, prior_error,
                                  prob_error, regress_coeff_error) {}

void polca_parallel::InfoEigenSolver::Solve(double* score, double* jacobian) {
  arma::Mat<double> score_arma(score, this->n_data_, this->info_size_, false);
  arma::Mat<double> info_arma = score_arma.t() * score_arma;

  // required to ensure symmetry, especially if using a preconditioner in
  // further development of this code
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

void polca_parallel::InfoEigenSolver::ExtractErrorGivenEigen(
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

polca_parallel::InfoEigenRegressSolver::InfoEigenRegressSolver(
    int n_data, int n_feature, int sum_outcomes, int n_cluster, int info_size,
    int jacobian_width, double* prior_error, double* prob_error,
    double* regress_coeff_error)
    : polca_parallel::InfoEigenSolver(
          n_data, n_feature, sum_outcomes, n_cluster, info_size, jacobian_width,
          prior_error, prob_error, regress_coeff_error) {}

void polca_parallel::InfoEigenRegressSolver::ExtractErrorGivenEigen(
    arma::Col<double>* eigval_inv, arma::Mat<double>* eigvec,
    double* jacobian) {
  // extract errors for the prior and outcome probs
  this->InfoEigenSolver::ExtractErrorGivenEigen(eigval_inv, eigvec, jacobian);
  int size = this->n_feature_ * (this->n_cluster_ - 1);
  // then extract covariance matrix
  arma::Mat<double> jac_arma(jacobian, this->info_size_, this->jacobian_width_,
                             false);

  arma::Mat<double> regress_coeff_error(this->regress_coeff_error_, size, size,
                                        false, true);

  // make a copy of the submat which contains the dimensions for the
  // coefficients, this is used to create the covariance for the coefficients
  // no need to do full pinv(info) multiplication
  //
  // making a copy is faster than submat() * diagmat() * submat().t()
  arma::Mat<double> sub = eigvec->submat(0, 0, size - 1, this->info_size_ - 1);
  regress_coeff_error = sub * arma::diagmat(*eigval_inv) * sub.t();
}
