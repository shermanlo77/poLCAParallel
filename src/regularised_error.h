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

#ifndef POLCAPARALLEL_SRC_REGULARISED_ERROR_H
#define POLCAPARALLEL_SRC_REGULARISED_ERROR_H

#include <span>

#include "standard_error.h"
#include "standard_error_regress.h"
#include "util.h"

namespace polca_parallel {

/**
 * For calculating the standard errors of the fitted poLCA parameters
 *
 * Uses a smoother to smooth the outcome probabilities. This increases numerical
 * stability by better conditioning the score matrix and prevent errors from
 * being exactly 0.0 or 1.0.
 * @copydoc StandardError
 *
 */
class RegularisedError : public polca_parallel::StandardError {
 public:
  /** @copydoc StandardError::StandardError */
  RegularisedError(std::span<const double> features,
                   std::span<const int> responses,
                   std::span<const double> probs, std::span<double> prior,
                   std::span<double> posterior, std::size_t n_data,
                   std::size_t n_feature, NOutcomes n_outcomes,
                   std::size_t n_cluster, std::span<double> prior_error,
                   std::span<double> prob_error,
                   std::span<double> regress_coeff_error);
};

/**
 * For calculating the standard errors of the fitted poLCA parameters
 *
 * Uses a smoother to smooth the outcome probabilities. This increases numerical
 * stability by better conditioning the score matrix and prevent errors from
 * being exactly 0.0 or 1.0.
 * @copydoc StandardErrorRegress
 *
 */
class RegularisedRegressError : public polca_parallel::StandardErrorRegress {
 public:
  /** @copydoc StandardErrorRegress::StandardErrorRegress */
  RegularisedRegressError(std::span<const double> features,
                          std::span<const int> responses,
                          std::span<const double> probs,
                          std::span<double> prior, std::span<double> posterior,
                          std::size_t n_data, std::size_t n_feature,
                          NOutcomes n_outcomes, std::size_t n_cluster,
                          std::span<double> prior_error,
                          std::span<double> prob_error,
                          std::span<double> regress_coeff_error);
};
}  // namespace polca_parallel

#endif  // POLCAPARALLEL_SRC_REGULARISED_ERROR_H
