#ifndef GOODNESS_FIT_H_
#define GOODNESS_FIT_H_

#include <math.h>

#include <array>
#include <map>
#include <stdexcept>
#include <vector>

#include "RcppArmadillo.h"

namespace polca_parallel {

struct Frequency {
  int observed;
  double expected;
};

// GET UNIQUE OBSERVED
// Iterate through all the responses, then find and count unique combinations of
// outcomes which were observed in the dataset. Results are stored in a map.
// Parameters:
// responses: design matrix transpose of responses, matrix n_category x n_data
// n_data: number of data points
// n_category: number of categories
// unique_freq: map to write results to, key: unique integer vectors,
// value: Frequency containing the number of times it was observed in the
// dataset
void GetUniqueObserved(int* responses, int n_data, int n_category,
                       std::map<std::vector<int>, Frequency>* unique_freq);

// GET EXPECTED FREQUENCY
// Update the expected frequency in a map of vector<int>:Frequency by modifying
// the value of Frequency.expected with the likelihood of that unique
// reponse with multiplied by n_data
// Parameters:
// responses: design matrix transpose of responses, matrix n_category x n_data
// prior: vector of prior probabilities (probability in a cluster),
// length n_cluster
// outcome_prob: array of outcome probabilities for each category and
// cluster,
// flatten list of matrices
// dim 0: for each outcome
// dim 1: for each category
// dim 2: for each model
// n_data: number of data points
// n_category: number of categories
// n_outcomes: array of integers, number of outcomes for each category, array
// of length n_category
// n_cluster: number of clusters (or classes)
// unique_freq: map to modify with results to, key: unique integer vectors,
// value: Frequency with the expected frequency modified
void GetExpected(int* responses, double* prior, double* outcome_prob,
                 int n_data, int n_category, int* n_outcomes, int n_cluster,
                 std::map<std::vector<int>, Frequency>* unique_freq);

// GET CHI-SQUARED and LOG LIKELIHOOD RATIO STATISTICS
// Calculate and return the chi-squared statistics and log likelihood ratio
// Parameters:
// unique_freq: map of unique responses and their frequencies, both observed
// and expected
// n_data: number of data points
// Return:
// log likelihood ratio
// chi squared statistics
std::array<double, 2> GetStatistics(
    std::map<std::vector<int>, Frequency>* unique_freq, int n_data);

}  // namespace polca_parallel

#endif  // GOODNESS_FIT_H_
