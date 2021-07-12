
#ifndef _SKETCH_SP_HPP_
#define _SKETCH_SP_HPP_

#include <memory>
#include <vector>

namespace ss::ann {
namespace sketch {

namespace detail {
static int _tmp_a[64], _tmp_b[64];
inline int16_t inner_product(uint64_t va, uint64_t vb, unsigned len);
inline int16_t inner_product(uint64_t va, uint64_t vb);
inline float inner_product(const float *data, uint64_t va);
inline float inner_product(const float *data, uint64_t va, unsigned len);
inline int16_t fake_range_sum(uint64_t zero_one_vec, unsigned c_range);
inline int16_t fake_range_sum(uint64_t zero_one_vec);
}// end namespace detail

struct GaussianSketchHamming {
  typedef uint64_t my_word_t;
  GaussianSketchHamming(unsigned m, unsigned d, unsigned seed) :
      _m(m),
      _dim(d),
      _real_dim_rv((_dim + WORD_SIZE - 1) / WORD_SIZE),
      _rem(_dim % WORD_SIZE),
      _seed(seed),
      _random_vectors(nullptr) {
    _generate_random_vectors(); // generate random vectors
  }

  void apply(const my_word_t *data, float *res) const {
    if (_rem == 0)
      _project_no_rem(data, res);
    else
      _project_with_rem(data, res);
  }
 private:
  void _project_no_rem(const my_word_t *data, float *res) const {
    for (unsigned si = 0; si < _m; ++si) {
      res[si] = 0;
      for (unsigned i = 0, j = 0; i < _real_dim_rv; ++i, j += WORD_SIZE) {
        res[si] += detail::inner_product(&_random_vectors[si * _dim + j], data[i]);
      }
    }
  }
  void _project_with_rem(const my_word_t *data, float *res) const {
    for (unsigned si = 0; si < _m; ++si) {
      res[si] = 0;
      unsigned i = 0;
      unsigned j = 0;
      for (; i < _real_dim_rv - 1; ++i, j += WORD_SIZE) {
        res[si] += detail::inner_product(&_random_vectors[si * _dim + j], data[i]);
      }
      res[si] += detail::inner_product(&_random_vectors[si * _dim + j], data[i], _rem);
    }
  }
  void _generate_random_vectors() {
    std::mt19937_64 eng(_seed);
    std::normal_distribution<float> distribution(0, 1);
    auto rng = std::bind(distribution, eng);

    _random_vectors = std::make_unique<float[]>(_m * _dim);
    for (size_t si = 0; si < _m; ++si)
      for (size_t sij = 0; sij < _dim; ++sij)
        _random_vectors[si * _dim + sij] = rng();
  }
  unsigned _m; // number of sketches for each data point
  unsigned _dim; // data dimension
  unsigned _real_dim_rv; // real dimension for random vectors
  unsigned _rem;
  unsigned _seed;// random seed
  std::unique_ptr<float[]> _random_vectors;// random vectors
  static constexpr size_t WORD_SIZE = 64;// use <<3u to convert # of bytes to # of bits
};

struct TugOfWarSketchHamming {
  typedef uint64_t my_word_t;
  TugOfWarSketchHamming(unsigned m, unsigned d, unsigned seed) :
      _m(m),
      _dim(d),
      _real_dim_rv((_dim + WORD_SIZE - 1) / WORD_SIZE),
      _rem(_dim % WORD_SIZE),
      _seed(seed),
      _random_vectors(nullptr) {
    _generate_random_vectors(); // generate random vectors
  }

  void apply(const my_word_t *data, int16_t *res) const {
    if (_rem == 0)
      _project_no_rem(data, res);
    else
      _project_with_rem(data, res);
  }
  void apply(const my_word_t *data, float *res) const {
    if (_rem == 0)
      _project_no_rem(data, res);
    else
      _project_with_rem(data, res);
  }
 private:
  template<typename T>
  void _project_no_rem(const my_word_t *data, T *res) const {
    for (unsigned si = 0; si < _m; ++si) {
      res[si] = 0;
      for (unsigned i = 0; i < _real_dim_rv; ++i) {
        res[si] += detail::inner_product(data[i], _random_vectors[si * _real_dim_rv + i]);
      }
    }
  }
  template<typename T>
  void _project_with_rem(const my_word_t *data, T *res) const {
    for (unsigned si = 0; si < _m; ++si) {
      res[si] = 0;
      unsigned i = 0;
      for (; i < _real_dim_rv - 1; ++i) {
        res[si] += detail::inner_product(data[i], _random_vectors[si * _real_dim_rv + i]);
      }
      res[si] += detail::inner_product(data[i], _random_vectors[si * _real_dim_rv + i], _rem);
    }
  }
  void _generate_random_vectors() {
    std::mt19937_64 eng(_seed);
    std::uniform_int_distribution<> distribution(0, 1);
    auto rng = std::bind(distribution, eng);

    std::bitset<WORD_SIZE> bits;
    _random_vectors = std::make_unique<my_word_t[]>(_m * _real_dim_rv);

    /// generate random 0-1 vectors
    for (size_t si = 0; si < _m; ++si) {
      size_t j = 0, c = 0;
      for (size_t sij = 0; sij < _dim; ++sij) {
        bits[j] = rng();
        ++j;
        if (j % WORD_SIZE == 0) {
          // The first bit of the bitset corresponds to the least significant digit of
          // the number and the last bit corresponds to the most significant digit.
          _random_vectors[si * _real_dim_rv + c] = bits.to_ullong();
          bits.reset();// reset bits
          j = 0;
          ++c;
        }
      }
      /// Pay attention
      if (j > 0) {
        _random_vectors[si * _real_dim_rv + c] = bits.to_ullong();
        bits.reset();// reset bits
      }
    }
  }
  unsigned _m; // number of sketches for each data point
  unsigned _dim; // data dimension
  unsigned _real_dim_rv; // real dimension for random vectors
  unsigned _rem;
  unsigned _seed;// random seed
  std::unique_ptr<my_word_t[]> _random_vectors;// random vectors
  static constexpr size_t WORD_SIZE = 64;// use <<3u to convert # of bytes to # of bits
};

///
/// \brief FeigenbaumSketchL1
///
/// This class implement the sketch propsoed in the following paper:
/// Joan Feigenbaum, Sampath Kannan, Martin J. Strauss, and Mahesh Viswanathan. 2003.
/// An Approximate L1-Difference Algorithm for Massive Data Streams. SIAM J. Comput. 32, 1 (January 2003),
/// 131-151. DOI: https://doi.org/10.1137/S0097539799361701
///
/// Note that we did not fully implement it. Hence our implementation might not as efficient and
/// scalable as the one described in the paper.
///
///
///
struct FeigenbaumSketchL1 {
  typedef uint64_t my_word_t;
  ///
  /// Constructor for FeigenbaumSketchL1
  ///
  /// \param m                # of sketch for each vector (currently, we only support unsigned-integral vector)
  /// \param d                dimension
  /// \param max_range        maximum range of each value in the vector
  /// \param seed             random seed
  FeigenbaumSketchL1(unsigned m, unsigned d, unsigned max_range, unsigned seed) :
      _m(m),
      _d(d),
      _max_range(max_range),
      _inner_dim((max_range + WORD_SIZE - 1) / WORD_SIZE),
      _seed(seed),
      _random_vectors(nullptr) {
    assert(MAX_RANGE_FSL1 >= max_range);
  }
  ///
  /// \brief apply
  ///
  /// !! Apply the sketch on a type T array storing the result in a int16_t array !!
  /// Note that it seems more natural to also add a template typename for the result type
  /// to make this API more scalar (fitting for more result type)! We choose only implement
  /// for int16_t result type, because we want to remind the user that, as mentioned before,
  /// this class has limitation on the maximum supported value for the value of each element
  /// in the input data.
  ///
  /// \tparam T           type for the input data
  ///
  /// \param data         input data (should have a dimenion of FeigenbaumSketchL1::_d)
  /// \param res          result
  template<typename T>
  void apply(const T *data, int16_t *res) const {
    return _project<T, int16_t>(data, res);
  }
  ///
  /// \brief apply
  ///
  /// This API is added because the Cover tree we used did only suport float type.
  /// TODO: avoid this API by templating cover tree
  template<typename T>
  void apply(const T *data, float *res) const {
    return _project<T, float>(data, res);
  }
 private:
  template<typename DataType, typename ResType>
  void _project(const DataType *data, ResType *res) const {
    /// generate m sketches
    for (unsigned si = 0; si < _m; ++si) {
      res[si] = 0;
      for (unsigned sij = 0; sij < _d; ++sij) {
        auto dj = data[sij];
        unsigned idx = si * _d * _inner_dim + sij * _inner_dim;
        while (dj >= WORD_SIZE) {
          res[si] += detail::fake_range_sum(_random_vectors[idx]);
          ++ idx;
          dj -= WORD_SIZE;
        }
        if (dj > 0) res[si] += detail::fake_range_sum(_random_vectors[idx], dj);
      }
    }
  }
  void _generate_random_vectors() {
    std::mt19937_64 eng(_seed);
    std::uniform_int_distribution<> distribution(0, 1);
    auto rng = std::bind(distribution, eng);

    std::bitset<WORD_SIZE> bits;
    _random_vectors = std::make_unique<my_word_t[]>(_m * _d * _inner_dim);

    /// generate random 0-1 vectors
    for (unsigned si = 0; si < _m; ++si) {
      /// each vector has _d elements (coordiantes), i.e., $\langle v_1, v_2, ..., v_d \rangle$
      for (unsigned sij = 0; sij < _d; ++sij) {
        unsigned j = 0;
        unsigned idx = si * _d * _inner_dim + sij * _inner_dim;
        /// each element corresponds to a sum of $\pm 1$, i.e., \sum_{i=1}^{v_i} r($\pm 1$)
        /// r($\pm 1$) means a random variable whose value can only be +1 or -1.
        /// This for loop is to generate instances of those random variables.
        for (unsigned sijk = 0; sijk < _inner_dim; ++sijk) {
          bits[j] = rng();
          ++j;
          if (j % WORD_SIZE == 0) {
            _random_vectors[idx] = bits.to_ullong();
            bits.reset();// reset bits
            j = 0;
            ++ idx;
          }
        }
        if (j > 0) {
          _random_vectors[idx] = bits.to_ullong();
          bits.reset();// reset bits
        }
      }
    }
  }
  unsigned _m;
  unsigned _d;
  unsigned _max_range;
  unsigned _inner_dim;
  unsigned _seed;
  std::unique_ptr<my_word_t[]> _random_vectors;// random vectors
  static constexpr unsigned MAX_RANGE_FSL1 = 8196;
  static constexpr unsigned WORD_SIZE = (sizeof(my_word_t) << 3u);
};

}// end namespace sketch
}// namespace ss::ann


namespace ss::ann::sketch::detail {
inline int16_t inner_product(uint64_t va, uint64_t vb, unsigned len) {
  uint64_t mask = ((1u) << (len - 1u));
  int16_t r = 0;
  for (size_t i = 0; i < len; ++i) {
    auto a = ((va & mask) == 0 ? (-1) : 1);
    auto b = ((vb & mask) == 0 ? (-1) : 1);
    r += a * b;
    mask >>= 1u;
  }
  return r;
}
inline int16_t inner_product(uint64_t va, uint64_t vb) {
  _tmp_a[0] = ((va & 0x1u) == 0 ? (-1) : 1);
  _tmp_b[0] = ((vb & 0x1u) == 0 ? (-1) : 1);

  _tmp_a[1] = ((va & 0x2u) == 0 ? (-1) : 1);
  _tmp_b[1] = ((vb & 0x2u) == 0 ? (-1) : 1);

  _tmp_a[2] = ((va & 0x4u) == 0 ? (-1) : 1);
  _tmp_b[2] = ((vb & 0x4u) == 0 ? (-1) : 1);

  _tmp_a[3] = ((va & 0x8u) == 0 ? (-1) : 1);
  _tmp_b[3] = ((vb & 0x8u) == 0 ? (-1) : 1);

  _tmp_a[4] = ((va & 0x10u) == 0 ? (-1) : 1);
  _tmp_b[4] = ((vb & 0x10u) == 0 ? (-1) : 1);

  _tmp_a[5] = ((va & 0x20u) == 0 ? (-1) : 1);
  _tmp_b[5] = ((vb & 0x20u) == 0 ? (-1) : 1);

  _tmp_a[6] = ((va & 0x40u) == 0 ? (-1) : 1);
  _tmp_b[6] = ((vb & 0x40u) == 0 ? (-1) : 1);

  _tmp_a[7] = ((va & 0x80u) == 0 ? (-1) : 1);
  _tmp_b[7] = ((vb & 0x80u) == 0 ? (-1) : 1);

  _tmp_a[8] = ((va & 0x100u) == 0 ? (-1) : 1);
  _tmp_b[8] = ((vb & 0x100u) == 0 ? (-1) : 1);

  _tmp_a[9] = ((va & 0x200u) == 0 ? (-1) : 1);
  _tmp_b[9] = ((vb & 0x200u) == 0 ? (-1) : 1);

  _tmp_a[10] = ((va & 0x400u) == 0 ? (-1) : 1);
  _tmp_b[10] = ((vb & 0x400u) == 0 ? (-1) : 1);

  _tmp_a[11] = ((va & 0x800u) == 0 ? (-1) : 1);
  _tmp_b[11] = ((vb & 0x800u) == 0 ? (-1) : 1);

  _tmp_a[12] = ((va & 0x1000u) == 0 ? (-1) : 1);
  _tmp_b[12] = ((vb & 0x1000u) == 0 ? (-1) : 1);

  _tmp_a[13] = ((va & 0x2000u) == 0 ? (-1) : 1);
  _tmp_b[13] = ((vb & 0x2000u) == 0 ? (-1) : 1);

  _tmp_a[14] = ((va & 0x4000u) == 0 ? (-1) : 1);
  _tmp_b[14] = ((vb & 0x4000u) == 0 ? (-1) : 1);

  _tmp_a[15] = ((va & 0x8000u) == 0 ? (-1) : 1);
  _tmp_b[15] = ((vb & 0x8000u) == 0 ? (-1) : 1);

  _tmp_a[16] = ((va & 0x10000u) == 0 ? (-1) : 1);
  _tmp_b[16] = ((vb & 0x10000u) == 0 ? (-1) : 1);

  _tmp_a[17] = ((va & 0x20000u) == 0 ? (-1) : 1);
  _tmp_b[17] = ((vb & 0x20000u) == 0 ? (-1) : 1);

  _tmp_a[18] = ((va & 0x40000u) == 0 ? (-1) : 1);
  _tmp_b[18] = ((vb & 0x40000u) == 0 ? (-1) : 1);

  _tmp_a[19] = ((va & 0x80000u) == 0 ? (-1) : 1);
  _tmp_b[19] = ((vb & 0x80000u) == 0 ? (-1) : 1);

  _tmp_a[20] = ((va & 0x100000u) == 0 ? (-1) : 1);
  _tmp_b[20] = ((vb & 0x100000u) == 0 ? (-1) : 1);

  _tmp_a[21] = ((va & 0x200000u) == 0 ? (-1) : 1);
  _tmp_b[21] = ((vb & 0x200000u) == 0 ? (-1) : 1);

  _tmp_a[22] = ((va & 0x400000u) == 0 ? (-1) : 1);
  _tmp_b[22] = ((vb & 0x400000u) == 0 ? (-1) : 1);

  _tmp_a[23] = ((va & 0x800000u) == 0 ? (-1) : 1);
  _tmp_b[23] = ((vb & 0x800000u) == 0 ? (-1) : 1);

  _tmp_a[24] = ((va & 0x1000000u) == 0 ? (-1) : 1);
  _tmp_b[24] = ((vb & 0x1000000u) == 0 ? (-1) : 1);

  _tmp_a[25] = ((va & 0x2000000u) == 0 ? (-1) : 1);
  _tmp_b[25] = ((vb & 0x2000000u) == 0 ? (-1) : 1);

  _tmp_a[26] = ((va & 0x4000000u) == 0 ? (-1) : 1);
  _tmp_b[26] = ((vb & 0x4000000u) == 0 ? (-1) : 1);

  _tmp_a[27] = ((va & 0x8000000u) == 0 ? (-1) : 1);
  _tmp_b[27] = ((vb & 0x8000000u) == 0 ? (-1) : 1);

  _tmp_a[28] = ((va & 0x10000000u) == 0 ? (-1) : 1);
  _tmp_b[28] = ((vb & 0x10000000u) == 0 ? (-1) : 1);

  _tmp_a[29] = ((va & 0x20000000u) == 0 ? (-1) : 1);
  _tmp_b[29] = ((vb & 0x20000000u) == 0 ? (-1) : 1);

  _tmp_a[30] = ((va & 0x40000000u) == 0 ? (-1) : 1);
  _tmp_b[30] = ((vb & 0x40000000u) == 0 ? (-1) : 1);

  _tmp_a[31] = ((va & 0x80000000u) == 0 ? (-1) : 1);
  _tmp_b[31] = ((vb & 0x80000000u) == 0 ? (-1) : 1);

  _tmp_a[32] = ((va & 0x100000000u) == 0 ? (-1) : 1);
  _tmp_b[32] = ((vb & 0x100000000u) == 0 ? (-1) : 1);

  _tmp_a[33] = ((va & 0x200000000u) == 0 ? (-1) : 1);
  _tmp_b[33] = ((vb & 0x200000000u) == 0 ? (-1) : 1);

  _tmp_a[34] = ((va & 0x400000000u) == 0 ? (-1) : 1);
  _tmp_b[34] = ((vb & 0x400000000u) == 0 ? (-1) : 1);

  _tmp_a[35] = ((va & 0x800000000u) == 0 ? (-1) : 1);
  _tmp_b[35] = ((vb & 0x800000000u) == 0 ? (-1) : 1);

  _tmp_a[36] = ((va & 0x1000000000u) == 0 ? (-1) : 1);
  _tmp_b[36] = ((vb & 0x1000000000u) == 0 ? (-1) : 1);

  _tmp_a[37] = ((va & 0x2000000000u) == 0 ? (-1) : 1);
  _tmp_b[37] = ((vb & 0x2000000000u) == 0 ? (-1) : 1);

  _tmp_a[38] = ((va & 0x4000000000u) == 0 ? (-1) : 1);
  _tmp_b[38] = ((vb & 0x4000000000u) == 0 ? (-1) : 1);

  _tmp_a[39] = ((va & 0x8000000000u) == 0 ? (-1) : 1);
  _tmp_b[39] = ((vb & 0x8000000000u) == 0 ? (-1) : 1);

  _tmp_a[40] = ((va & 0x10000000000u) == 0 ? (-1) : 1);
  _tmp_b[40] = ((vb & 0x10000000000u) == 0 ? (-1) : 1);

  _tmp_a[41] = ((va & 0x20000000000u) == 0 ? (-1) : 1);
  _tmp_b[41] = ((vb & 0x20000000000u) == 0 ? (-1) : 1);

  _tmp_a[42] = ((va & 0x40000000000u) == 0 ? (-1) : 1);
  _tmp_b[42] = ((vb & 0x40000000000u) == 0 ? (-1) : 1);

  _tmp_a[43] = ((va & 0x80000000000u) == 0 ? (-1) : 1);
  _tmp_b[43] = ((vb & 0x80000000000u) == 0 ? (-1) : 1);

  _tmp_a[44] = ((va & 0x100000000000u) == 0 ? (-1) : 1);
  _tmp_b[44] = ((vb & 0x100000000000u) == 0 ? (-1) : 1);

  _tmp_a[45] = ((va & 0x200000000000u) == 0 ? (-1) : 1);
  _tmp_b[45] = ((vb & 0x200000000000u) == 0 ? (-1) : 1);

  _tmp_a[46] = ((va & 0x400000000000u) == 0 ? (-1) : 1);
  _tmp_b[46] = ((vb & 0x400000000000u) == 0 ? (-1) : 1);

  _tmp_a[47] = ((va & 0x800000000000u) == 0 ? (-1) : 1);
  _tmp_b[47] = ((vb & 0x800000000000u) == 0 ? (-1) : 1);

  _tmp_a[48] = ((va & 0x1000000000000u) == 0 ? (-1) : 1);
  _tmp_b[48] = ((vb & 0x1000000000000u) == 0 ? (-1) : 1);

  _tmp_a[49] = ((va & 0x2000000000000u) == 0 ? (-1) : 1);
  _tmp_b[49] = ((vb & 0x2000000000000u) == 0 ? (-1) : 1);

  _tmp_a[50] = ((va & 0x4000000000000u) == 0 ? (-1) : 1);
  _tmp_b[50] = ((vb & 0x4000000000000u) == 0 ? (-1) : 1);

  _tmp_a[51] = ((va & 0x8000000000000u) == 0 ? (-1) : 1);
  _tmp_b[51] = ((vb & 0x8000000000000u) == 0 ? (-1) : 1);

  _tmp_a[52] = ((va & 0x10000000000000u) == 0 ? (-1) : 1);
  _tmp_b[52] = ((vb & 0x10000000000000u) == 0 ? (-1) : 1);

  _tmp_a[53] = ((va & 0x20000000000000u) == 0 ? (-1) : 1);
  _tmp_b[53] = ((vb & 0x20000000000000u) == 0 ? (-1) : 1);

  _tmp_a[54] = ((va & 0x40000000000000u) == 0 ? (-1) : 1);
  _tmp_b[54] = ((vb & 0x40000000000000u) == 0 ? (-1) : 1);

  _tmp_a[55] = ((va & 0x80000000000000u) == 0 ? (-1) : 1);
  _tmp_b[55] = ((vb & 0x80000000000000u) == 0 ? (-1) : 1);

  _tmp_a[56] = ((va & 0x100000000000000u) == 0 ? (-1) : 1);
  _tmp_b[56] = ((vb & 0x100000000000000u) == 0 ? (-1) : 1);

  _tmp_a[57] = ((va & 0x200000000000000u) == 0 ? (-1) : 1);
  _tmp_b[57] = ((vb & 0x200000000000000u) == 0 ? (-1) : 1);

  _tmp_a[58] = ((va & 0x400000000000000u) == 0 ? (-1) : 1);
  _tmp_b[58] = ((vb & 0x400000000000000u) == 0 ? (-1) : 1);

  _tmp_a[59] = ((va & 0x800000000000000u) == 0 ? (-1) : 1);
  _tmp_b[59] = ((vb & 0x800000000000000u) == 0 ? (-1) : 1);

  _tmp_a[60] = ((va & 0x1000000000000000u) == 0 ? (-1) : 1);
  _tmp_b[60] = ((vb & 0x1000000000000000u) == 0 ? (-1) : 1);

  _tmp_a[61] = ((va & 0x2000000000000000u) == 0 ? (-1) : 1);
  _tmp_b[61] = ((vb & 0x2000000000000000u) == 0 ? (-1) : 1);

  _tmp_a[62] = ((va & 0x4000000000000000u) == 0 ? (-1) : 1);
  _tmp_b[62] = ((vb & 0x4000000000000000u) == 0 ? (-1) : 1);

  _tmp_a[63] = ((va & 0x8000000000000000u) == 0 ? (-1) : 1);
  _tmp_b[63] = ((vb & 0x8000000000000000u) == 0 ? (-1) : 1);

#ifdef DEBUG_INNER_PRODUCT
  spdlog::debug("_tmp_a = {}", ss::io::stringtify(_tmp_a, _tmp_a + 64));
  spdlog::debug("_tmp_b = {}", ss::io::stringtify(_tmp_a, _tmp_a + 64));
#endif

  return _tmp_a[0] * _tmp_b[0] + _tmp_a[1] * _tmp_b[1] + _tmp_a[2] * _tmp_b[2] + _tmp_a[3] * _tmp_b[3]
      + _tmp_a[4] * _tmp_b[4] + _tmp_a[5] * _tmp_b[5] + _tmp_a[6] * _tmp_b[6] + _tmp_a[7] * _tmp_b[7]
      + _tmp_a[8] * _tmp_b[8] + _tmp_a[9] * _tmp_b[9] + _tmp_a[10] * _tmp_b[10] + _tmp_a[11] * _tmp_b[11]
      + _tmp_a[12] * _tmp_b[12] + _tmp_a[13] * _tmp_b[13] + _tmp_a[14] * _tmp_b[14] + _tmp_a[15] * _tmp_b[15]
      + _tmp_a[16] * _tmp_b[16] + _tmp_a[17] * _tmp_b[17] + _tmp_a[18] * _tmp_b[18] + _tmp_a[19] * _tmp_b[19]
      + _tmp_a[20] * _tmp_b[20] + _tmp_a[21] * _tmp_b[21] + _tmp_a[22] * _tmp_b[22] + _tmp_a[23] * _tmp_b[23]
      + _tmp_a[24] * _tmp_b[24] + _tmp_a[25] * _tmp_b[25] + _tmp_a[26] * _tmp_b[26] + _tmp_a[27] * _tmp_b[27]
      + _tmp_a[28] * _tmp_b[28] + _tmp_a[29] * _tmp_b[29] + _tmp_a[30] * _tmp_b[30] + _tmp_a[31] * _tmp_b[31]
      + _tmp_a[32] * _tmp_b[32] + _tmp_a[33] * _tmp_b[33] + _tmp_a[34] * _tmp_b[34] + _tmp_a[35] * _tmp_b[35]
      + _tmp_a[36] * _tmp_b[36] + _tmp_a[37] * _tmp_b[37] + _tmp_a[38] * _tmp_b[38] + _tmp_a[39] * _tmp_b[39]
      + _tmp_a[40] * _tmp_b[40] + _tmp_a[41] * _tmp_b[41] + _tmp_a[42] * _tmp_b[42] + _tmp_a[43] * _tmp_b[43]
      + _tmp_a[44] * _tmp_b[44] + _tmp_a[45] * _tmp_b[45] + _tmp_a[46] * _tmp_b[46] + _tmp_a[47] * _tmp_b[47]
      + _tmp_a[48] * _tmp_b[48] + _tmp_a[49] * _tmp_b[49] + _tmp_a[50] * _tmp_b[50] + _tmp_a[51] * _tmp_b[51]
      + _tmp_a[52] * _tmp_b[52] + _tmp_a[53] * _tmp_b[53] + _tmp_a[54] * _tmp_b[54] + _tmp_a[55] * _tmp_b[55]
      + _tmp_a[56] * _tmp_b[56] + _tmp_a[57] * _tmp_b[57] + _tmp_a[58] * _tmp_b[58] + _tmp_a[59] * _tmp_b[59]
      + _tmp_a[60] * _tmp_b[60] + _tmp_a[61] * _tmp_b[61] + _tmp_a[62] * _tmp_b[62] + _tmp_a[63] * _tmp_b[63];
}
inline float inner_product(const float *data, uint64_t va, unsigned len) {
  uint64_t mask = ((1u) << (len - 1u));
  float r = 0;
  for (size_t i = 0; i < len; ++i) {
    if ((va & mask) != 0)
      r += data[i];
    mask >>= 1u;
  }
  return r;
}
inline float inner_product(const float *data, uint64_t va) {
  _tmp_a[0] = ((va & 0x1u) == 0 ? 0 : 1);
  _tmp_a[1] = ((va & 0x2u) == 0 ? 0 : 1);
  _tmp_a[2] = ((va & 0x4u) == 0 ? 0 : 1);
  _tmp_a[3] = ((va & 0x8u) == 0 ? 0 : 1);
  _tmp_a[4] = ((va & 0x10u) == 0 ? 0 : 1);
  _tmp_a[5] = ((va & 0x20u) == 0 ? 0 : 1);
  _tmp_a[6] = ((va & 0x40u) == 0 ? 0 : 1);
  _tmp_a[7] = ((va & 0x80u) == 0 ? 0 : 1);
  _tmp_a[8] = ((va & 0x100u) == 0 ? 0 : 1);
  _tmp_a[9] = ((va & 0x200u) == 0 ? 0 : 1);
  _tmp_a[10] = ((va & 0x400u) == 0 ? 0 : 1);
  _tmp_a[11] = ((va & 0x800u) == 0 ? 0 : 1);
  _tmp_a[12] = ((va & 0x1000u) == 0 ? 0 : 1);
  _tmp_a[13] = ((va & 0x2000u) == 0 ? 0 : 1);
  _tmp_a[14] = ((va & 0x4000u) == 0 ? 0 : 1);
  _tmp_a[15] = ((va & 0x8000u) == 0 ? 0 : 1);
  _tmp_a[16] = ((va & 0x10000u) == 0 ? 0 : 1);
  _tmp_a[17] = ((va & 0x20000u) == 0 ? 0 : 1);
  _tmp_a[18] = ((va & 0x40000u) == 0 ? 0 : 1);
  _tmp_a[19] = ((va & 0x80000u) == 0 ? 0 : 1);
  _tmp_a[20] = ((va & 0x100000u) == 0 ? 0 : 1);
  _tmp_a[21] = ((va & 0x200000u) == 0 ? 0 : 1);
  _tmp_a[22] = ((va & 0x400000u) == 0 ? 0 : 1);
  _tmp_a[23] = ((va & 0x800000u) == 0 ? 0 : 1);
  _tmp_a[24] = ((va & 0x1000000u) == 0 ? 0 : 1);
  _tmp_a[25] = ((va & 0x2000000u) == 0 ? 0 : 1);
  _tmp_a[26] = ((va & 0x4000000u) == 0 ? 0 : 1);
  _tmp_a[27] = ((va & 0x8000000u) == 0 ? 0 : 1);
  _tmp_a[28] = ((va & 0x10000000u) == 0 ? 0 : 1);
  _tmp_a[29] = ((va & 0x20000000u) == 0 ? 0 : 1);
  _tmp_a[30] = ((va & 0x40000000u) == 0 ? 0 : 1);
  _tmp_a[31] = ((va & 0x80000000u) == 0 ? 0 : 1);
  _tmp_a[32] = ((va & 0x100000000u) == 0 ? 0 : 1);
  _tmp_a[33] = ((va & 0x200000000u) == 0 ? 0 : 1);
  _tmp_a[34] = ((va & 0x400000000u) == 0 ? 0 : 1);
  _tmp_a[35] = ((va & 0x800000000u) == 0 ? 0 : 1);
  _tmp_a[36] = ((va & 0x1000000000u) == 0 ? 0 : 1);
  _tmp_a[37] = ((va & 0x2000000000u) == 0 ? 0 : 1);
  _tmp_a[38] = ((va & 0x4000000000u) == 0 ? 0 : 1);
  _tmp_a[39] = ((va & 0x8000000000u) == 0 ? 0 : 1);
  _tmp_a[40] = ((va & 0x10000000000u) == 0 ? 0 : 1);
  _tmp_a[41] = ((va & 0x20000000000u) == 0 ? 0 : 1);
  _tmp_a[42] = ((va & 0x40000000000u) == 0 ? 0 : 1);
  _tmp_a[43] = ((va & 0x80000000000u) == 0 ? 0 : 1);
  _tmp_a[44] = ((va & 0x100000000000u) == 0 ? 0 : 1);
  _tmp_a[45] = ((va & 0x200000000000u) == 0 ? 0 : 1);
  _tmp_a[46] = ((va & 0x400000000000u) == 0 ? 0 : 1);
  _tmp_a[47] = ((va & 0x800000000000u) == 0 ? 0 : 1);
  _tmp_a[48] = ((va & 0x1000000000000u) == 0 ? 0 : 1);
  _tmp_a[49] = ((va & 0x2000000000000u) == 0 ? 0 : 1);
  _tmp_a[50] = ((va & 0x4000000000000u) == 0 ? 0 : 1);
  _tmp_a[51] = ((va & 0x8000000000000u) == 0 ? 0 : 1);
  _tmp_a[52] = ((va & 0x10000000000000u) == 0 ? 0 : 1);
  _tmp_a[53] = ((va & 0x20000000000000u) == 0 ? 0 : 1);
  _tmp_a[54] = ((va & 0x40000000000000u) == 0 ? 0 : 1);
  _tmp_a[55] = ((va & 0x80000000000000u) == 0 ? 0 : 1);
  _tmp_a[56] = ((va & 0x100000000000000u) == 0 ? 0 : 1);
  _tmp_a[57] = ((va & 0x200000000000000u) == 0 ? 0 : 1);
  _tmp_a[58] = ((va & 0x400000000000000u) == 0 ? 0 : 1);
  _tmp_a[59] = ((va & 0x800000000000000u) == 0 ? 0 : 1);
  _tmp_a[60] = ((va & 0x1000000000000000u) == 0 ? 0 : 1);
  _tmp_a[61] = ((va & 0x2000000000000000u) == 0 ? 0 : 1);
  _tmp_a[62] = ((va & 0x4000000000000000u) == 0 ? 0 : 1);
  _tmp_a[63] = ((va & 0x8000000000000000u) == 0 ? 0 : 1);

#ifdef DEBUG_INNER_PRODUCT
  spdlog::debug("_tmp_a = {}", ss::io::stringtify(_tmp_a, _tmp_a + 64));
#endif
  return _tmp_a[0] * data[0] + _tmp_a[1] * data[1] + _tmp_a[2] * data[2] + _tmp_a[3] * data[3] + _tmp_a[4] * data[4]
      + _tmp_a[5] * data[5] + _tmp_a[6] * data[6] + _tmp_a[7] * data[7] + _tmp_a[8] * data[8] + _tmp_a[9] * data[9]
      + _tmp_a[10] * data[10] + _tmp_a[11] * data[11] + _tmp_a[12] * data[12] + _tmp_a[13] * data[13]
      + _tmp_a[14] * data[14] + _tmp_a[15] * data[15] + _tmp_a[16] * data[16] + _tmp_a[17] * data[17]
      + _tmp_a[18] * data[18] + _tmp_a[19] * data[19] + _tmp_a[20] * data[20] + _tmp_a[21] * data[21]
      + _tmp_a[22] * data[22] + _tmp_a[23] * data[23] + _tmp_a[24] * data[24] + _tmp_a[25] * data[25]
      + _tmp_a[26] * data[26] + _tmp_a[27] * data[27] + _tmp_a[28] * data[28] + _tmp_a[29] * data[29]
      + _tmp_a[30] * data[30] + _tmp_a[31] * data[31] + _tmp_a[32] * data[32] + _tmp_a[33] * data[33]
      + _tmp_a[34] * data[34] + _tmp_a[35] * data[35] + _tmp_a[36] * data[36] + _tmp_a[37] * data[37]
      + _tmp_a[38] * data[38] + _tmp_a[39] * data[39] + _tmp_a[40] * data[40] + _tmp_a[41] * data[41]
      + _tmp_a[42] * data[42] + _tmp_a[43] * data[43] + _tmp_a[44] * data[44] + _tmp_a[45] * data[45]
      + _tmp_a[46] * data[46] + _tmp_a[47] * data[47] + _tmp_a[48] * data[48] + _tmp_a[49] * data[49]
      + _tmp_a[50] * data[50] + _tmp_a[51] * data[51] + _tmp_a[52] * data[52] + _tmp_a[53] * data[53]
      + _tmp_a[54] * data[54] + _tmp_a[55] * data[55] + _tmp_a[56] * data[56] + _tmp_a[57] * data[57]
      + _tmp_a[58] * data[58] + _tmp_a[59] * data[59] + _tmp_a[60] * data[60] + _tmp_a[61] * data[61]
      + _tmp_a[62] * data[62] + _tmp_a[63] * data[63];
}

namespace constants {
constexpr uint64_t BITMASKS[] = {
    0x0, 0x1, 0x3, 0x7, 0xF, 0x1F, 0x3F, 0x7F, 0xFF,
    0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF, 0x3FFF, 0x7FFF, 0xFFFF,
    0x1FFFF, 0x3FFFF, 0x7FFFF, 0xFFFFF, 0x1FFFFF, 0x3FFFFF, 0x7FFFFF, 0xFFFFFF,
    0x1FFFFFF, 0x3FFFFFF, 0x7FFFFFF, 0xFFFFFFF, 0x1FFFFFFF, 0x3FFFFFFF, 0x7FFFFFFF, 0xFFFFFFFF,
    0x1FFFFFFFF, 0x3FFFFFFFF, 0x7FFFFFFFF, 0xFFFFFFFFF, 0x1FFFFFFFFF, 0x3FFFFFFFFF, 0x7FFFFFFFFF, 0xFFFFFFFFFF,
    0x1FFFFFFFFFF, 0x3FFFFFFFFFF, 0x7FFFFFFFFFF, 0xFFFFFFFFFFF, 0x1FFFFFFFFFFF, 0x3FFFFFFFFFFF, 0x7FFFFFFFFFFF,
    0xFFFFFFFFFFFF,
    0x1FFFFFFFFFFFF, 0x3FFFFFFFFFFFF, 0x7FFFFFFFFFFFF, 0xFFFFFFFFFFFFF, 0x1FFFFFFFFFFFFF, 0x3FFFFFFFFFFFFF,
    0x7FFFFFFFFFFFFF, 0xFFFFFFFFFFFFFF,
    0x1FFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFF, 0x1FFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF,
    0x7FFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF
};
}// end namespace constants

inline int16_t fake_range_sum(uint64_t zero_one_vec, unsigned c_range) {
  auto c = constants::BITMASKS[c_range];
  int16_t r = __builtin_popcountll(c & zero_one_vec) * 2;
  return (r - c_range);
}
inline int16_t fake_range_sum(uint64_t zero_one_vec) {
  int16_t r = __builtin_popcountll(zero_one_vec) * 2;
  return (r - 64);
}
}// end namespace ss::ann::sketch
#endif //_SKETCH_SP_HPP_
