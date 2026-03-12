#include "kamaletdinov_r_bitwise_int/omp/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>

#include "kamaletdinov_r_bitwise_int/common/include/common.hpp"

namespace kamaletdinov_r_bitwise_int {

namespace {

void CountingSortByDigitSequential(std::vector<int> &arr, int exp) {
  const int n = static_cast<int>(arr.size());
  std::vector<int> output(n);
  std::array<int, 10> count = {};

  for (int i = 0; i < n; i++) {
    count[(arr[i] / exp) % 10]++;
  }

  for (int i = 1; i < 10; i++) {
    count[i] += count[i - 1];
  }

  for (int i = n - 1; i >= 0; i--) {
    const int digit = (arr[i] / exp) % 10;
    output[count[digit] - 1] = arr[i];
    count[digit]--;
  }

  arr.swap(output);
}

void CountingSortByDigitParallel(std::vector<int> &arr, int exp) {
  const int n = static_cast<int>(arr.size());
  if (n < 2048) {
    CountingSortByDigitSequential(arr, exp);
    return;
  }

  const int thread_count = omp_get_max_threads();
  std::vector<std::array<int, 10>> local_counts(thread_count);

#pragma omp parallel default(none) shared(arr, exp, local_counts, n) num_threads(thread_count)
  {
    const int tid = omp_get_thread_num();
    auto &current = local_counts[tid];
    current.fill(0);

#pragma omp for schedule(static)
    for (int i = 0; i < n; i++) {
      const int digit = (arr[i] / exp) % 10;
      current[digit]++;
    }
  }

  std::array<int, 10> global_count = {};
  for (int t = 0; t < thread_count; t++) {
    for (int d = 0; d < 10; d++) {
      global_count[d] += local_counts[t][d];
    }
  }

  for (int d = 1; d < 10; d++) {
    global_count[d] += global_count[d - 1];
  }

  std::array<int, 10> global_start = {};
  for (int d = 1; d < 10; d++) {
    global_start[d] = global_count[d - 1];
  }

  std::vector<std::array<int, 10>> thread_offsets(thread_count);
  for (int d = 0; d < 10; d++) {
    int offset = global_start[d];
    for (int t = 0; t < thread_count; t++) {
      thread_offsets[t][d] = offset;
      offset += local_counts[t][d];
    }
  }

  std::vector<int> output(n);

#pragma omp parallel default(none) shared(arr, exp, output, n, thread_count, thread_offsets) num_threads(thread_count)
  {
    const int tid = omp_get_thread_num();
    auto positions = thread_offsets[tid];

#pragma omp for schedule(static)
    for (int i = 0; i < n; i++) {
      const int digit = (arr[i] / exp) % 10;
      output[positions[digit]++] = arr[i];
    }
  }

  arr.swap(output);
}

void RadixSortPositiveParallel(std::vector<int> &arr) {
  if (arr.empty()) {
    return;
  }

  const int max_val = *std::ranges::max_element(arr);
  for (int exp = 1; max_val / exp > 0; exp *= 10) {
    CountingSortByDigitParallel(arr, exp);
    if (exp > max_val / 10) {
      break;
    }
  }
}

void BitwiseSortParallel(std::vector<int> &arr) {
  if (arr.size() <= 1) {
    return;
  }

  std::vector<int> neg;
  std::vector<int> pos;
  neg.reserve(arr.size());
  pos.reserve(arr.size());

  for (int value : arr) {
    if (value < 0) {
      neg.push_back(-value);
    } else {
      pos.push_back(value);
    }
  }

  RadixSortPositiveParallel(neg);
  RadixSortPositiveParallel(pos);

  std::size_t index = 0;
  for (int i = static_cast<int>(neg.size()) - 1; i >= 0; i--) {
    arr[index++] = -neg[i];
  }
  for (int value : pos) {
    arr[index++] = value;
  }
}

}  // namespace

KamaletdinovRBitwiseIntOMP::KamaletdinovRBitwiseIntOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool KamaletdinovRBitwiseIntOMP::ValidationImpl() {
  return GetInput() >= 0;
}

bool KamaletdinovRBitwiseIntOMP::PreProcessingImpl() {
  const int n = GetInput();
  data_.resize(n);

#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    data_[i] = (n / 2) - i;
  }

  return true;
}

bool KamaletdinovRBitwiseIntOMP::RunImpl() {
  BitwiseSortParallel(data_);
  return true;
}

bool KamaletdinovRBitwiseIntOMP::PostProcessingImpl() {
  const bool sorted = std::ranges::is_sorted(data_);
  GetOutput() = sorted ? GetInput() : 0;
  return true;
}

}  // namespace kamaletdinov_r_bitwise_int
