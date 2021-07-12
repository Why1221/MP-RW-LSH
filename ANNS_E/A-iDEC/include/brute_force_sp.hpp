
#ifndef _BRUTE_FORCE_SP_HPP_
#define _BRUTE_FORCE_SP_HPP_

#include <vector>
#include <srs_utils.hpp>
#include <distance_metric_sp.hpp>

using namespace ss::ann::srs;

namespace ss::ann::brute_force {

namespace flat_vector {
struct BruteForceHamming {
  explicit BruteForceHamming(size_t dim,
                             const std::vector<uint64_t> &raw_data)
      : _raw_data_ref(raw_data), _dim(dim), _N(_raw_data_ref.size() / _dim) {
    assert(_dim * _N == raw_data.size() && "dimension must fit");
  }

  res_pair_raw<unsigned> query(const uint64_t *q) const {
    res_pair_raw<unsigned> res{};
    res.id = -1;
    res.dist = std::numeric_limits<unsigned>::max();

    for (int idx = 0; idx < _N; ++idx) {
      auto dist = distance::hamming_distance(&_raw_data_ref[idx * _dim], q, _dim);
      if (dist < res.dist) {
        res.id = idx;
        res.dist = dist;
      }
    }
    return res;
  }

  void query(const uint64_t *q,
             unsigned k,
             std::vector<res_pair_raw<unsigned>> &heap) const {

    heap.clear();
    heap.reserve(k);

    res_pair_raw<unsigned> res{};

    for (int idx = 0; idx < _N; ++idx) {
      auto dist = distance::hamming_distance(&_raw_data_ref[idx * _dim], q, _dim);
      if (heap.size() < k) {
        res.id = idx;
        res.dist = dist;
        heap.push_back(res);
        std::push_heap(heap.begin(), heap.end());
      } else if (dist < heap.front().dist) {
        std::pop_heap(heap.begin(), heap.end());
        heap.pop_back();
        res.id = idx;
        res.dist = dist;
        heap.push_back(res);
        std::push_heap(heap.begin(), heap.end());
      }
    }
  }

  unsigned dim() const { return _dim * WORD_SIZE; }
  unsigned enc_dim() const { return _dim; }
 private:
  const std::vector<uint64_t> &_raw_data_ref;
  const unsigned _dim;
  const size_t _N;

  static constexpr unsigned WORD_SIZE = 64u;
};
}// end namespace flat_vector

struct BruteForceHamming {
  explicit BruteForceHamming(const std::vector<std::vector<uint64_t >> &raw_data)
      : _raw_data_ref(raw_data) {}

  res_pair_raw<unsigned> query(const uint64_t *q) const {
    res_pair_raw<unsigned> res{};
    res.id = -1;
    res.dist = std::numeric_limits<unsigned>::max();

    size_t idx = 0;
    for (const auto &data: _raw_data_ref) {
      auto dist = distance::hamming_distance(&data[0], q, data.size());
      if (dist < res.dist) {
        res.id = idx;
        res.dist = dist;
      }
      ++idx;
    }

    return res;
  }

  void query(const uint64_t *q, unsigned k, std::vector<res_pair_raw<unsigned>> &heap) const {
    heap.clear();
    heap.reserve(k);

    res_pair_raw<unsigned> res{};
    int idx = 0;
    for (const auto &data: _raw_data_ref) {
      auto dist = distance::hamming_distance(&data[0], q, data.size());
      if (heap.size() < k) {
        res.id = idx;
        res.dist = dist;
        heap.push_back(res);
        std::push_heap(heap.begin(), heap.end());
      } else if (dist < heap.front().dist) {
        std::pop_heap(heap.begin(), heap.end());
        heap.pop_back();
        res.id = idx;
        res.dist = dist;
        heap.push_back(res);
        std::push_heap(heap.begin(), heap.end());
      }
      ++idx;
    }
  }

 private:
  const std::vector<std::vector<uint64_t >> &_raw_data_ref;
};
}
#endif //_BRUTE_FORCE_SP_HPP_
