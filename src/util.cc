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

#include "util.h"

#include <numeric>
#include <span>

polca_parallel::NOutcomes::NOutcomes(const std::size_t* data, std::size_t size)
    : std::span<const std::size_t>(data, size),
      sum_(std::accumulate(data, data + size, 0)) {}

std::size_t polca_parallel::NOutcomes::sum() const { return this->sum_; }
