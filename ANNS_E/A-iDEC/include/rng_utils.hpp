
#ifndef _RNG_UTILS_HPP_
#define _RNG_UTILS_HPP_

#include <iterator>
#include <limits>
#include <chrono>
#include <random>

namespace ss::rng {
template<typename InputIt>
void generateRandomVector(InputIt f,
                          InputIt l,
                          typename std::iterator_traits<InputIt>::value_type max_range = std::numeric_limits<typename std::iterator_traits<
                              InputIt>::value_type>::max(),
                          unsigned seed = std::chrono::system_clock::now().time_since_epoch().count()) {
  typedef typename std::iterator_traits<InputIt>::value_type T;

  static_assert(std::is_arithmetic_v<T>);

  std::mt19937_64 eng(seed);
  if constexpr  (std::is_integral_v<T>) {
    std::uniform_int_distribution<T> dist(0, max_range);
    for (auto it = f; it != l; ++it) (*it) = dist(eng);
  } else {
    std::uniform_real_distribution<T> dist(0, max_range);
    for (auto it = f; it != l; ++it) (*it) = dist(eng);
  }
}

template<typename InputIt, typename RNG>
void generateRandomVector(InputIt f,
                          InputIt l,
                          RNG&& g) {
  typedef typename std::iterator_traits<InputIt>::value_type T;
  static_assert(std::is_arithmetic_v<T>);
  for (auto it = f; it != l; ++it) (*it) = g();
}

}
#endif // _RNG_UTILS_HPP_
