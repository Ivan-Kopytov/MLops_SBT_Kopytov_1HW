#include <vector>
#include <tuple>
#include <stdexcept>

std::tuple<double, double> linear_regression(const std::vector<double>& X, const std::vector<double>& Y) {
    if (X.size() != Y.size() || X.empty()) {
        throw std::invalid_argument("Vectors X and Y must have the same non-zero size.");
    }

    size_t n = X.size();
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;

    for (size_t i = 0; i < n; ++i) {
        sum_x += X[i];
        sum_y += Y[i];
        sum_xy += X[i] * Y[i];
        sum_xx += X[i] * X[i];
    }

    double b1 = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    double b0 = (sum_y - b1 * sum_x) / n;

    return {b0, b1};
}