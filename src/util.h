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

#ifndef POLCAPARALLEL_SRC_UTIL_H_
#define POLCAPARALLEL_SRC_UTIL_H_

#include <span>

namespace polca_parallel {

class NOutcomes : public std::span<const std::size_t> {
 private:
  const std::size_t sum_;

 public:
  NOutcomes(const std::size_t* data, std::size_t size);

  [[nodiscard]] std::size_t sum() const;
};

}  // namespace polca_parallel

#endif  // POLCAPARALLEL_SRC_EM_ALGORITHM_ARRAY_SERIAL_H_
