#include <utility>

#ifndef _STRING_EMBED_HPP_
#define _STRING_EMBED_HPP_

#include <array>
#include <limits>
#include <chrono>
#include <bitset>
#include <random>
#include <memory>
#include <functional>
#include <tuple>
#include <functional>
#include <list>

//#include <spdlog/spdlog.h>

#include "sketch_sp.h"

#include "popcnt.hpp"
#ifdef USE_ROLLING_HASH
#include <rollinghashcpp/rabinkarphash.h>
/// TODO adding support for rolling hash
#endif
//#define DEBUG_CGK
#ifdef DEBUG_CGK
#include <boost/dynamic_bitset.hpp>
#endif

//#define DEBUG_GRAM
//#define DEBUG_CGKTOW
//#define DEBUG_GRAM_FEIGENBAUM
//#define DEBUG_CC_FEIGENBAUM
//#define DEBUG_CGKLSH

#if defined(DEBUG)
#include <io.hpp>
using namespace ss::io;
#endif

namespace ss::ann {

namespace embed {

namespace detail {
inline int fake_range_sum(uint8_t zero_one_vec, unsigned c_range);
inline int fake_range_sum(uint8_t zero_one_vec);

//inline int popcnt(uint64_t v);
inline void q_gram_block64(const char *s,
                           unsigned q,
                           std::unordered_map<std::string, int> &dict,
                           std::vector<int> &res);

inline void
q_gram_block8(const char *s, unsigned q, std::unordered_map<std::string, int> &dict, std::vector<int> &res);
template<typename T>
inline void q_gram_block8(const char *s,
                          unsigned q,
                          std::unordered_map<std::string, int> &dict,
                          T *res);
template<typename T>
inline void q_gram_block64(const char *s,
                           unsigned q,
                           std::unordered_map<std::string, int> &dict,
                           T *res);
}// end namespace detail
///
/// \brief CGKEmbed
///
///
/// This class implements the CGK embedding, the detail can be found in the following paper.
/// Diptarka Chakraborty, Elazar Goldenberg, and Michal Koucký. 2016. Streaming algorithms for
/// embedding and computing edit distance in the low distance regime. In Proceedings of the forty-eighth
/// annual ACM symposium on Theory of Computing (STOC '16). ACM, New York, NY, USA, 712-725.
/// DOI: https://doi.org/10.1145/2897518.2897577
///
/// Note that this implementation is a slight modification from this one provided in
/// https://github.com/kedayuge/Embedjoin
///
///
struct CGKEmbed {
  ///
  /// \brief Constructor for CGKEmbed
  ///
  ///
  /// \param first                      header for the alphabet
  /// \param alphabet_size              size for the alphabet
  /// \param max_len                    maximum length for the embedded (resulting) string
  ///                                   Note that in the paper, this value should at 3*N, where
  ///                                   N is the length of the input string.
  ///
  /// \param n_embeddings               # of embeddings, we add this parameter, because usually a
  ///                                   single embedding might not be enough in real-application.
  /// \param seed                       Random seed
  ///
  CGKEmbed(const char *first,
           unsigned alphabet_size,
           unsigned max_len,
           unsigned n_embeddings = 1u,
           unsigned seed = std::chrono::system_clock::now().time_since_epoch().count()) :
      _alphabet_size(alphabet_size),
//      _un_used(std::numeric_limits<int>::max()),
      _N(max_len),
      _m(n_embeddings),
      _seed(seed),
      _dict(),
      _jump()
#ifdef DEBUG_CGK
  , _hash(n_embeddings * alphabet_size, boost::dynamic_bitset<>(_N))
#endif
  {
    assert(alphabet_size < MAX_ALPHABET_SIZE);
    unsigned i = 0;
    for (; i < _dict.size(); ++i) _dict[i] = EMPTY;
    i = 0;
    for (; i < alphabet_size; ++i) _dict[*(first + i)] = i;

    _generate_hash();
  }

  template<typename RandomEngine>
  CGKEmbed(const char *first,
           unsigned alphabet_size,
           unsigned max_len,
           unsigned n_embeddings,
           RandomEngine &eng) :
      _alphabet_size(alphabet_size),
//      _un_used(std::numeric_limits<int>::max()),
      _N(max_len),
      _m(n_embeddings),
      _seed(0),
      _dict(),
      _jump()
#ifdef DEBUG_CGK
  , _hash(n_embeddings * alphabet_size, boost::dynamic_bitset<>(_N))
#endif
  {
    assert(alphabet_size < MAX_ALPHABET_SIZE);
    unsigned i = 0;
    for (; i < _dict.size(); ++i) _dict[i] = EMPTY;
    i = 0;
    for (; i < alphabet_size; ++i) _dict[*(first + i)] = i;

    _generate_hash(eng);
  }
  ///
  int map(const char ch) const {
    return _dict[ch];
  }

  unsigned alphabet_size() const {
    return _alphabet_size;
  }

  unsigned num_cgk() const {
    return _m;
  }

  const std::array<int, 256> &dict() const {
    return _dict;
  }
  ///
  /// \brief apply
  ///
  /// Apply CGK embedding on the string #str.
  ///
  /// \param str
  /// \param len
  /// \param res
  void apply(const char *str, unsigned len, std::string *res) const {
    const unsigned ul = std::min(3 * len, _N);
    for (unsigned k = 0; k < _m; ++k) {
      res[k].reserve(ul);
      unsigned idx = k * _N * _alphabet_size;
      for (unsigned j = 0, i = 0; j < ul && i < len; ++i) {
        auto ch_ind = _dict[str[i]];
        //if (ch_ind >= _alphabet_size) throw std::runtime_error("The input string is not valid!!");
        auto r = _jump[idx + ch_ind * _N + j] + 1;
        res[k].insert(res[k].end(), std::min(r, (int) (ul - res[k].size())), str[i]);
        j += r;
      }
    }
  }

#ifdef DEBUG_CGK

  void apply_naive(const char *str, unsigned len, std::string *res) const {
    const unsigned ul = std::min(3 * len, _N);
    for (unsigned k = 0; k < _m; ++k) {
      res[k].reserve(ul);
      unsigned i = 0;
      for (unsigned j = 0; j < ul && i < len; ++j) {
        auto ch_ind = _dict[str[i]];
        res[k].push_back(str[i]);
        i += _hash[k * _alphabet_size + ch_ind][j];
      }
    }
  }

#endif

  void apply(const std::string &str, std::string *res) const {
    apply(str.c_str(), str.size(), res);
  }

 private:
  template<typename RandomEngine>
  void _generate_hash(RandomEngine &eng) {
    std::uniform_int_distribution<> distribution(0, 1);
    auto rng = std::bind(distribution, eng);

    _jump.resize(_alphabet_size * _N * _m, 0u);
    ///std::vector<uint8_t > hash(_N, 0u);
    int rb = 0;
    for (unsigned si = 0; si < _m; ++si) {
      for (unsigned t = 0; t < _alphabet_size; ++t) {
        unsigned idx = (si * _alphabet_size + t) * _N;
        /// Note that _jump actually stores how many consecutive zeros (including itself) after it
        int cnt = 0;
#ifdef DEBUG_CGK
        auto &h = _hash[si * _alphabet_size + t];
#endif
        for (int d = _N - 1; d >= 0; --d) {
          rb = rng();/// generate random bit
#ifdef DEBUG_CGK
          h[d] = rb;
#endif
          if (rb == 0) ++cnt;
          else cnt = 0;
          _jump[idx + d] = cnt;
        }
      }
    }
  }
  void _generate_hash() {
    std::mt19937_64 eng(_seed);
    _generate_hash(eng);
  }

  const unsigned _alphabet_size;
  const unsigned _N;
  const unsigned _m;
  unsigned _seed;
  std::array<int, 256> _dict;
  std::vector<int> _jump;

  static constexpr int EMPTY = -1;
#ifdef DEBUG_CGK
  std::vector<boost::dynamic_bitset<>> _hash;
#endif
  static constexpr int MAX_ALPHABET_SIZE = (1u << ((sizeof(char) << 3u) - 1));
};

namespace constants {
constexpr unsigned MAX_DICT_SIZE = 8196;
}// end namespace constants

///
/// \brief QGramDense
///
/// Using q-gram sto "embed" a string into a $\ell_1$ space.
/// The coordinates of the $\ell_1$ are (q-gram$_1$, q-gram$_2$m, ..., q-gram$_d$),
/// element value for the corresponding coordinate of the embedded $\ell_1$ vector
/// just the number of occurrences of the q-gram in the string.
///
/// By "Dense", we mean that most coordinates should have non-zero values. That is, either q
/// should be small enough, or the string should be long enough.
///
struct QGramDense {
  ///
  /// \brief constructor for QGramDense
  ///
  ///
  /// \param first                 Pointer points to the first element in the alphabet
  /// \param alphabet_size         alphabet size
  /// \param q                     gram size
  /// \param wrap_around           whether wrap around the input string
  QGramDense(const char *first,
             unsigned alphabet_size,
             unsigned q,
             bool wrap_around) : _q(q),
                                 _wrap_around(wrap_around),
                                 _gdim(static_cast<unsigned >(std::pow(alphabet_size, q))),
                                 _dict{}
#ifdef DEBUG_GRAM
  , _inverse_index(_gdim, "")
#endif
  {
//    assert(_gdim < constants::MAX_DICT_SIZE);
    std::string str(_q, ' ');
    unsigned cnt = 0;
    _generate_dict(first, alphabet_size, 0, str, cnt);
    assert(cnt == _gdim);
#ifdef DEBUG_GRAM
    for (const auto &map : _dict) _inverse_index[map.second] = map.first;
#endif
  }

#ifdef DEBUG_GRAM

  std::vector<std::string> inverse_map() const {
    return _inverse_index;
  }

#endif

  ///
  /// \brief apply
  ///
  /// Aplly the QGramDense on the string #str
  ///
  /// \param str           pointer to the first element of string #str (to apply QGramDense to)
  /// \param len           string len
  /// \return              the $\ell_1$ vector after embedding
  std::vector<int> apply(const char *str, unsigned len) const {
    std::vector<int> res(_gdim, 0);
    _apply_impl_wo(str, len, res);
    if (_wrap_around) _apply_impl_wrap_part(str, len, res);
    return res;
  }

  template<typename T>
  void apply(const char *str, unsigned len, T *res) const {
    _apply_impl_wo(str, len, res);
    if (_wrap_around) _apply_impl_wrap_part(str, len, res);
  }

  unsigned dim() const { return _gdim; }

  ///
  /// \brief apply
  ///
  /// Aplly the QGramDense on the string #str
  ///
  /// \param str     string #str to apply QGramDense to
  /// \return        the $\ell_1$ vector after embedding
  std::vector<int> apply(const std::string &str) const {
    return apply(str.c_str(), str.size());
  }

  std::unordered_map<std::string, int> dict() const {
    return _dict;
  }

  int id(const std::string &s) const { return _dict.at(s); }

#ifdef DEBUG_GRAM

  std::vector<int> apply_no_op(const char *str, unsigned len) const {
    std::vector<int> res(_gdim, 0);
    _apply_impl_no_op(str, len, res);
    if (_wrap_around) _apply_impl_wrap_part(str, len, res);
    return res;
  }

#endif

 private:
#ifdef DEBUG_GRAM

  void _apply_impl_no_op(const char *str, unsigned len, std::vector<int> &res) const {
    unsigned ub = len - _q + 1;
    unsigned i = 0;
    for (; i < ub; ++i) ++res[_dict[std::string(str + i, _q)]];
  }

#endif

  void _apply_impl_wo(const char *str, unsigned len, std::vector<int> &res) const {
    unsigned ub = len - _q + 1;
    unsigned i = 0;
    for (; i + 64 <= ub; i += 64) detail::q_gram_block64(str + i, _q, _dict, res);
    for (; i + 8 <= ub; i += 8) detail::q_gram_block8(str + i, _q, _dict, res);
    for (; i < ub; ++i) ++res[_dict[std::string(str + i, _q)]];
  }

  template<typename T>
  void _apply_impl_wo(const char *str, unsigned len, T *res) const {
    unsigned ub = len - _q + 1;
    unsigned i = 0;
    for (; i + 64 <= ub; i += 64) detail::q_gram_block64(str + i, _q, _dict, res);
    for (; i + 8 <= ub; i += 8) detail::q_gram_block8(str + i, _q, _dict, res);
    for (; i < ub; ++i) ++res[_dict[std::string(str + i, _q)]];
  }

  template<typename T>
  void _apply_impl_wrap_part(const char *str, unsigned len, T *res) const {
    unsigned i = len - _q + 1;
    for (unsigned k = 0; k < (_q - 1); ++k) {
      std::string tmp(_q, ' ');
      unsigned ks = i + k;
      for (unsigned j = 0; j < _q; ++j) tmp[j] = str[(ks + j) % len];
      ++res[_dict[tmp]];
    }
  }

  void _apply_impl_wrap_part(const char *str, unsigned len, std::vector<int> &res) const {
    unsigned i = len - _q + 1;
    for (unsigned k = 0; k < (_q - 1); ++k) {
      std::string tmp(_q, ' ');
      unsigned ks = i + k;
      for (unsigned j = 0; j < _q; ++j) tmp[j] = str[(ks + j) % len];
      ++res[_dict[tmp]];
    }
  }

  void _generate_dict(const char *first, unsigned s, unsigned level, std::string &str, unsigned &cnt) {
    if (level == _q) {
#ifdef DEBUG_GRAM
      spdlog::debug("Got string: {} => {}", str, cnt);
#endif
      _dict[str] = cnt;
      ++cnt;
      return;
    }
    for (unsigned i = 0; i < s; ++i) {
      str[level] = first[i];
      _generate_dict(first, s, level + 1, str, cnt);
    }
  }

  unsigned _q;
  bool _wrap_around;
  unsigned _gdim;
  mutable std::unordered_map<std::string, int> _dict;
#ifdef DEBUG_GRAM
  std::vector<std::string> _inverse_index;
#endif
};

using Feigenbaum=sketch::FeigenbaumSketchL1;
///
/// \brief QGramDenseFeigenbaum
///
/// This class implements a simple scheme to embed a string into a L1 space.
/// More precisely, it first uses q-gram to embed the input string into a
/// a vector in L1 space, where each coordinate correspond to the number of
/// occurrences of a certain q-gram in the string. However, when q is large or
/// the alphabet size is large, the dimension of the resulted L1 space has very
/// high dimension. Then, it uses Feigenbaum sketch to reduce the dimension (
/// embed a higher-dimensional L1 space to a lower-dimensional one).
///
/// Note that if your strings have very small alphabet size and you small gram size (i.e., value of q)
/// QGramDense should be enough.
///
struct QGramDenseFeigenbaum {
  ///
  /// \brief constructor for QGramDenseFeigenbaum
  ///
  /// \param first                  pointer points to the first alphabet
  /// \param alphabet_size          alphabet size
  /// \param q                      gram size
  /// \param m                      number of sketches to be used (for each data)
  /// \param max_range              maximum length of the input strings (This parameter is needed
  ///                               just because we implement a simplified version of Feigenbaum
  ///                               L1 sketch.
  /// \param seed                   random seed
  /// \param wrap_around            whether or not to do wrap-around on the input string when
  ///                               calculating its q-grams.
  ///
  /// \sa QGramDense::QGramDense() and FeigenbaumSketchL1::FeigenbaumSketchL1()
  ///
  QGramDenseFeigenbaum(const char *first,
                       unsigned alphabet_size,
                       unsigned q,
                       unsigned m,
                       unsigned max_str_len,
                       unsigned seed,
                       bool wrap_around = false) : gram(first, alphabet_size, q, wrap_around),
                                                   feigenbaum(m, gram.dim(), max_str_len, seed) {

  }
  template<typename ResType>
  void apply(const char *data, unsigned len, ResType *res) const {
    auto gram_counts = gram.apply(data, len);
#ifdef DEBUG_GRAM_FEIGENBAUM
    spdlog::debug("grams counts: {}", stringtify(gram_counts.cbegin(), gram_counts.cend()));
#endif
    feigenbaum.apply(&gram_counts[0], res);
  }
  QGramDense gram;
  Feigenbaum feigenbaum;
};

namespace detail {
inline int64_t random_walk_block64(const char *s, const uint64_t *hash, unsigned wi, const int *dict, unsigned enc_dim);
}// end namespace detail
///
/// \brief TugOfWarSketchHamming
///
/// This class implements the tug of war sketch, which was first proposed in the following paper.
///
/// Alon, N., Matias, Y. and Szegedy, M., 1999. The space complexity of approximating the frequency moments.
/// Journal of Computer and system sciences, 58(1), pp.137-147.
///
/// However, here we use it to estimate Hamming distance instead of the frequency moments. The basic idea is
/// as follows. Given a d-dimension vector <$v_1,v_2,...,v_d$>, we using certain 4-wise hash functions to map
/// each $v_i$ to $\pm 1$ and the sum them up.
struct TugOfWarSketchHamming {
  ///
  /// \brief constructor for TugOfWarSketchHamming
  ///
  /// In the implementation of this class, we assume that the input string in stored into multiple blocks,
  /// where each block has a fixed block size. This implmentation choice is to fit the requirements of
  /// class CGKTOW.
  ///
  /// \param alphabets           alphabets
  /// \param alphabet_size       alphabet size
  /// \param m                   number of sketches to be used
  /// \param block_size          block size
  /// \param n_block             number of blocks
  /// \param seed                random seed
  ///
  TugOfWarSketchHamming(const char *alphabets,
                        unsigned alphabet_size,
                        unsigned m,
                        unsigned block_size,
                        unsigned n_block,
                        unsigned seed) :
      _m(m),
      _db(block_size),
      _nb(n_block),
      _alphabet_size(alphabet_size + 1),
      _enc_dim_eb((_db + WORD_SIZE - 1) / WORD_SIZE),
      _seed(seed),
      _dict{},
      _hash{} {
    for (auto &d : _dict) d = EMPTY;/// initialization
    for (unsigned i = 0; i < alphabet_size; ++i) _dict[alphabets[i]] = i;
    _generate_random_vectors();
  }

  template<typename RandomEngine>
  TugOfWarSketchHamming(const char *alphabets,
                        unsigned alphabet_size,
                        unsigned m,
                        unsigned block_size,
                        unsigned n_block,
                        RandomEngine &eng):      _m(m),
                                                 _db(block_size),
                                                 _nb(n_block),
                                                 _alphabet_size(alphabet_size + 1),
                                                 _enc_dim_eb((_db + WORD_SIZE - 1) / WORD_SIZE),
                                                 _seed(0),
                                                 _dict{},
                                                 _hash{} {
    for (auto &d : _dict) d = EMPTY;/// initialization
    for (unsigned i = 0; i < alphabet_size; ++i) _dict[alphabets[i]] = i;
    _generate_random_vectors(eng);
  }

  ///
  /// \brief apply
  ///
  /// \tparam T                   Result type (should be arithmetic type)
  ///
  /// \param data                Input data
  /// \param res                 result
  template<typename T>
  void apply(const std::string *data, T *res) const {
    for (unsigned k = 0; k < _m; ++k) {
      res[k] = 0;
      for (unsigned b = 0; b < _nb; ++b) {
        /// get the corresponding hash function
        const uint64_t *hash_k = &_hash[k * _nb * _alphabet_size * _enc_dim_eb + b * _alphabet_size * _enc_dim_eb];
        unsigned len = data[b].size();
        T tmp = T(0);
        if (len % WORD_SIZE == 0) {
          _apply_each_no_rem(data[b].c_str(), len, hash_k, tmp);
          res[k] += tmp;
        } else {
          _apply_each_rem(data[b].c_str(), len, hash_k, tmp);
          res[k] += tmp;
        }
      }
    }
  }

 private:

  template<typename T>
  void _apply_each_no_rem(const char *s, unsigned s_len, const uint64_t *hash_k, T &loc) const {
    loc = 0;
    unsigned wi = 0;
    int len = s_len;
    unsigned idx = 0;
    /// pay special attention
    while (len >= WORD_SIZE) {
      loc += detail::random_walk_block64(s + idx, hash_k, wi, &_dict[0], _enc_dim_eb);
      len -= WORD_SIZE;
      idx += WORD_SIZE;
      ++wi;
    }
    /// Note here padding is because when doing CGK embedding, we do not
    /// make all string to the same length. However, when computing the
    /// Hamming distance, we need all strings have the same length.
    int padding = (_db - s_len);
    if (padding > 0) loc += _handle_padding(hash_k, wi, padding);
  }

  template<typename T>
  void _apply_each_rem(const char *s, unsigned s_len, const uint64_t *hash_k, T &loc) const {
    loc = 0;
    unsigned wi = 0;
    int len = s_len;
    unsigned idx = 0;
    while (len >= WORD_SIZE) {
      loc += detail::random_walk_block64(s + idx, hash_k, wi, &_dict[0], _enc_dim_eb);
      len -= WORD_SIZE;
      idx += WORD_SIZE;
      ++wi;
    }
    uint64_t mask = 1ull;
    while (len > 0) {
      loc += ((hash_k[_dict[*(s + idx)] * _enc_dim_eb + wi] & mask) == 0 ? (-1) : 1);
      --len;
      ++idx;
      mask <<= 1u;
    }
    int align64 = WORD_SIZE - len;
    int rem = (_db - s_len);
    auto i_last = std::min(align64, rem);
    /// pay attention
    for (unsigned i = 0; i < i_last; ++i, --rem) {
      loc += (hash_k[(_alphabet_size - 1) * _enc_dim_eb + wi] & mask) == 0 ? (-1) : 1;
      mask <<= 1u;
    }
    if (rem > 0) loc += _handle_padding(hash_k, wi + 1, rem);
  }

  int16_t _handle_padding(const uint64_t *hash_k, unsigned wi, int padding) const {
    int16_t loc = 0;
    while (padding >= 64) {
      loc += 2 * ss::ann::helper::popcount(hash_k[(_alphabet_size - 1) * _enc_dim_eb + wi]) - 64;
      padding -= 64;
      ++wi;
    }
    if (padding > 0) {
      loc += 2 * ss::ann::helper::popcount(hash_k[(_alphabet_size - 1) * _enc_dim_eb + wi]) - padding;
    }
    return loc;
  }

  template<typename RandomEngine>
  void _generate_random_vectors(RandomEngine &eng) {
    std::uniform_int_distribution<> distribution(0, 1);
    auto rng = std::bind(distribution, eng);

    _hash.resize(_m * _alphabet_size * _nb * _enc_dim_eb);
    std::bitset<WORD_SIZE> bits;

    for (unsigned k = 0; k < _m; ++k) {
      for (unsigned bi = 0; bi < _nb; ++bi) {
        for (unsigned t = 0; t < _alphabet_size; ++t) {
          unsigned h_ind = k * _nb * _alphabet_size * _enc_dim_eb + bi * _alphabet_size * _enc_dim_eb + t * _enc_dim_eb;
          unsigned c = 0;
          for (unsigned d = 0; d < _db; ++d) {
            bits[c] = rng();
            ++c;
            if (c == WORD_SIZE) {
              _hash[h_ind] = bits.to_ullong();
              bits.reset();
              ++h_ind;
              c = 0;
            }
          }
          if (c > 0) {
            _hash[h_ind] = bits.to_ullong();
            bits.reset();
          }
        }
      }
    }
  }
  void _generate_random_vectors() {
    std::mt19937_64 eng(_seed);
    _generate_random_vectors(eng);
  }
  unsigned _m;
  unsigned _db;
  unsigned _nb;
  unsigned _enc_dim_eb;
  unsigned _alphabet_size;
  unsigned _seed;
  std::array<int, 256> _dict;
  std::vector<uint64_t> _hash;

  static constexpr unsigned WORD_SIZE = (sizeof(uint64_t) << 3u);
  static constexpr int EMPTY = -1;
};

///
/// \brief CGKTOW
///
/// This class implements a simple scheme (building block for string match) combining CGK embedding and
/// tug of war sketch. The basic idea is to use CGK embedding embeds strings in editting space
/// to those in Hamming space. However, such strings usually has very high dimension, so we then use
/// tug of war sketch to further embed them into Euclidean space.
///
struct CGKTOW {
  CGKTOW(const char *first,
         unsigned alphabet_size,
         unsigned n_cgk,
         unsigned max_len,
         unsigned n_sketch,
         unsigned seed) : cgk_ptr(nullptr), tow_ptr(nullptr) {
    std::mt19937_64 eng(seed);
    cgk_ptr = std::make_unique<CGKEmbed>(first, alphabet_size, 3 * max_len, n_cgk, eng);
    tow_ptr = std::make_unique<TugOfWarSketchHamming>(first, alphabet_size, n_sketch, 3 * max_len, n_cgk, eng);
  }

  template<typename ResType>
  void apply(const char *data, unsigned len, ResType *res) const {
    std::vector<std::string> hamming_vecs(cgk_ptr->num_cgk());
    cgk_ptr->apply(data, len, &hamming_vecs[0]);
#ifdef DEBUG_CGKTOW
    std::string sep(80, '-');
    spdlog::debug("{}", sep);
    for(const auto& h_str : hamming_vecs)
          spdlog::debug("{}", h_str);
    spdlog::debug("{}", sep);
#endif
    tow_ptr->apply(&hamming_vecs[0], res);
  }
  std::unique_ptr<CGKEmbed> cgk_ptr;
  std::unique_ptr<TugOfWarSketchHamming> tow_ptr;
};

///
/// \brief CContextsDenseFeigenbaum
///
struct CContextsDenseFeigenbaum {

  /// customized hash function for CContextsDenseFeigenbaum
  /// which is used to indexing the hash functions for Feigenbaum sketch
  struct SpTupleHash {
    std::size_t operator()(std::tuple<unsigned/* char id */,
                                      unsigned/* occurrences */,
                                      bool /* (context's) before (windows) or not */> const &t) const noexcept {
      unsigned key = (std::get<2>(t) ? 0 : 1);
      key <<= 16u;
      key |= std::get<1>(t);
      key <<= 8u;
      key |= std::get<0>(t);
      return std::hash<unsigned>{}(key);
    }
  };
  ///
  /// \brief Contexts
  ///
  struct Contexts {
    std::vector<int> before;/// grams for before window
    std::vector<int> after;/// grams for after window
  };

  static constexpr unsigned MAX_CHAR_SIZE = 256;/// 1B char
  static constexpr unsigned MAX_STR_LEN = (1u << 31u) - 1u;///
  static constexpr unsigned MAX_GRAM_NUM = 8196;

  ///
  /// \brief CharContext
  ///
  /// This is a customized version of CContexts
  ///
  struct CharContext {
    //////////////////////// member variables /////////////////////////////
    unsigned window_size;          // window size
    unsigned gram_size;            // gram size, i.e., value for q
    unsigned num_char;             // alphabet size
    int char_id[MAX_CHAR_SIZE];    // id for each alphabet
    unsigned cdim;                 // dimension after performing q-gram
    bool wrap_around;              // whether performing wrap-around for the input string
    bool wrap_around_in;           // whether performing wrap-around within the windows
    /// Remarks. Usually wrap around or white-padding (i.e., padding q - 1 white letters that are
    ///          not included in the alphabets at the beginning and ending of the string) are used to "normalize"
    ///          the impact of errors, since errors occur at the first/last few letters will have less impact because
    ///          they will occur in less grams, if those schemes are not used. In the future version, we will try to implement
    ///          the white padding also. White padding seems to be more robust than wrap around.
    /// TODO replace wrap around with white-padding
    ///
    mutable std::unordered_map<std::string, int> gram_id;/// make sure can be accessed by "[]"
    static constexpr int EMPTY = -1;

    ///
    /// \brief constructor for CharContext
    ///
    /// \sa CContexts
    CharContext(const char *first,
                unsigned alphabet_size,
                unsigned w,
                unsigned q,
                bool wrap_around,
                bool wrap_around_in) : window_size(w), gram_size(q), num_char(alphabet_size),
                                       char_id{},
                                       cdim(unsigned(std::pow(alphabet_size, q))),
                                       wrap_around(wrap_around), wrap_around_in(wrap_around_in) {
      assert(cdim <= MAX_GRAM_NUM);
      std::fill(char_id, char_id + MAX_CHAR_SIZE, EMPTY);
      for (unsigned i = 0; i < alphabet_size; ++i) char_id[*(first + i)] = i;
      std::string str(gram_size, ' ');
      unsigned cnt = 0;
      _generate_dict(first, alphabet_size, 0, str, cnt);
      assert(cnt == cdim);
    }

    Contexts apply_cconext_impl(const char *str, unsigned len, unsigned loc) const {
      std::string window(window_size, ' ');
      if (loc < window_size) {
        for (int i = loc - 1, c = window_size - 1; c >= 0; --c, --i) window[c] = str[(i + len) % len];
      } else {
        for (int i = loc - 1, c = window_size - 1; c >= 0; --c, --i) window[c] = str[i];
      }
      Contexts cont{};
      cont.before.resize(cdim, 0);
      _apply_qgram_impl_wo(window.c_str(), window_size, cont.before);
      if (wrap_around_in) _apply_qgram_impl_wrap_part(window.c_str(), window_size, cont.before);

      if (loc + window_size >= len) {
        for (unsigned i = loc + 1, c = 0; c < window_size; ++c, ++i) window[c] = str[i % len];
      } else {
        for (unsigned i = loc + 1, c = 0; c < window_size; ++c, ++i) window[c] = str[i];
      }
      cont.after.resize(cdim, 0);
      _apply_qgram_impl_wo(window.c_str(), window_size, cont.after);
      if (wrap_around_in) _apply_qgram_impl_wrap_part(window.c_str(), window_size, cont.after);

      return cont;
    }

   private:
    void _apply_qgram_impl_wo(const char *str, unsigned len, std::vector<int> &res) const {
      unsigned ub = len - gram_size + 1;
      unsigned i = 0;
      for (; i < ub; ++i) ++res[gram_id[std::string(str + i, gram_size)]];
    }

    void _apply_qgram_impl_wrap_part(const char *str, unsigned len, std::vector<int> &res) const {
      unsigned i = len - gram_size + 1;
      for (unsigned k = 0; k < (gram_size - 1); ++k) {
        std::string tmp(gram_size, ' ');
        unsigned ks = i + k;
        for (unsigned j = 0; j < gram_size; ++j) tmp[j] = str[(ks + j) % len];
        ++res[gram_id[tmp]];
      }
    }
    void _generate_dict(const char *first, unsigned s, unsigned level, std::string &str, unsigned &cnt) {
      if (level == gram_size) {
        gram_id[str] = cnt;
        ++cnt;
        return;
      }
      for (unsigned i = 0; i < s; ++i) {
        str[level] = first[i];
        _generate_dict(first, s, level + 1, str, cnt);
      }
    }
  };// CharContext

  ///
  /// \brief ContextFeigenbaum
  ///
  /// This class implements customized version of FeigenbaumSketchL1.
  ///
  struct ContextFeigenbaum {
    ////////////////////////////////member variables//////////////////////////////////////////////////////
    typedef uint8_t my_word_t;
    typedef std::tuple<unsigned /* char id */, unsigned /* occurrences */, bool /* before or not */>
        feigen_hash_index_t;
    unsigned num_sketches;
    unsigned random_seed;
    std::mt19937_64 eng;
    int num_hashes_each_char;
    int num_char;
    int dim;
    int max_range;
    int inner_dim;
    mutable std::unordered_map<feigen_hash_index_t, std::vector<my_word_t>, SpTupleHash> hash;
    static constexpr unsigned WORD_SIZE = (sizeof(my_word_t) << 3u);
    ///////////////////////////////////////////////////////////////////////////////////////////////////////

    ContextFeigenbaum(unsigned m, unsigned seed) :
        num_sketches(m),
        random_seed(seed),
        eng(seed),
        num_hashes_each_char(-1),
        num_char(-1),
        dim(-1),
        max_range(-1),
        inner_dim(-1),
        hash() {
    }

    const std::vector<my_word_t> &get_hash_func(unsigned ch_id, bool before, unsigned loc) const {
      auto h_id = std::make_tuple(ch_id, loc, before);
      if (!hash.count(h_id)) return generate_random_hash(h_id);
      return hash[h_id];
    }

    unsigned get_hash_func_id(unsigned k, unsigned d) const {
      return k * dim * inner_dim + d * inner_dim;
    }

    int eval_hash(const my_word_t *hash_func, unsigned key) const {
      int h = 0;
      auto idx = 0;
      while (key >= WORD_SIZE) {
        h += detail::fake_range_sum(hash_func[idx]);
        key -= WORD_SIZE;
        ++idx;
      }
      if (key > 0)h += detail::fake_range_sum(hash_func[idx], key);
      return h;
    }

    const std::vector<my_word_t> &generate_random_hash(const std::tuple<unsigned, unsigned, bool> &h_id) const {

#ifdef DEBUG_CC_FEIGENBAUM
      spdlog::debug("generate new hash function for: char id: {}, occ: {}, before: {}",
          std::get<0>(h_id), std::get<1>(h_id), std::get<2>(h_id));
#endif
      std::uniform_int_distribution<> distribution(0, 1);
      auto rng = std::bind(distribution, eng);
      std::bitset<WORD_SIZE> bits;

      hash[h_id] = std::vector<my_word_t>(num_sketches * dim * inner_dim, 0u);

      auto &new_hash_func = hash[h_id];
      for (unsigned k = 0; k < num_sketches; ++k) {
        for (unsigned d = 0; d < dim; ++d) {
          unsigned j = 0;
          auto idx = k * dim * inner_dim + d * inner_dim;/// pay attention
          for (unsigned r = 0; r < max_range; ++r) {
            bits[j] = rng();
            ++j;
            if (j % WORD_SIZE == 0) {
              new_hash_func[idx] = bits.to_ulong();
              bits.reset();// reset bits
              j = 0;
              ++idx;
            }
          }
          if (j > 0) {
            new_hash_func[idx] = bits.to_ulong();
            bits.reset();// reset bits
          }
        }
      }

      return new_hash_func;
    }
  };// ContextFeigenbaum

  ///
  /// \brief CContextsDenseFeigenbaum
  ///
  /// \param first
  /// \param alphabet_size
  /// \param max_len
  /// \param m
  /// \param w
  /// \param q
  /// \param seed
  /// \param wrap_around
  /// \param wrap_around_in
  CContextsDenseFeigenbaum(const char *first,
                           unsigned alphabet_size,
                           unsigned max_len,
                           unsigned m,
                           unsigned w,
                           unsigned q,
                           unsigned seed = std::chrono::system_clock::now().time_since_epoch().count(),
                           bool wrap_around = false,
                           bool wrap_around_in = false) : _char_context(first,
                                                                        alphabet_size,
                                                                        w,
                                                                        q,
                                                                        wrap_around,
                                                                        wrap_around_in),
                                                          _feigenbaum(m, seed) {
    assert(max_len <= MAX_STR_LEN && "Sorry! This implementation can not scale to very long strings!");
    _feigenbaum.dim = static_cast<int>(std::pow(alphabet_size, q));
    _feigenbaum.num_hashes_each_char = max_len;
    _feigenbaum.max_range = w;
    _feigenbaum.inner_dim = (w + ContextFeigenbaum::WORD_SIZE - 1) / ContextFeigenbaum::WORD_SIZE;
    _feigenbaum.num_char = alphabet_size;
  }

  template<typename ResType>
  void apply(const char *data, unsigned len, ResType *res) {
    std::vector<int> cnt(_char_context.num_char, 0);
    std::fill(res, res + _feigenbaum.num_sketches, 0);// init

    const unsigned ls = (_char_context.wrap_around ? 0 : _char_context.window_size);
    const unsigned le = (_char_context.wrap_around ? len : (len - _char_context.window_size));

    Contexts cont;
    for (unsigned loc = ls; loc < le; ++loc) {
      auto t = _char_context.char_id[data[loc]];
      auto tk = cnt[t];
      ++cnt[t]; // TODO: remove location dependencies to reduce index space (otherwise too many hash functions
      // would be generated)
      if (loc == ls || _char_context.wrap_around_in) {
        cont = _char_context.apply_cconext_impl(data, len, loc);
      } else {
        //// avoid recalculating q-grams for the two windows
        //// TODO !! figure out how to do this for the wrap around cases !!
        auto before_out =
            _char_context.gram_id[std::string(data + loc - 1 - _char_context.window_size, _char_context.gram_size)];
        auto before_in =
            _char_context.gram_id[std::string(data + loc - _char_context.gram_size, _char_context.gram_size)];
        auto after_out = _char_context.gram_id[std::string(data + loc, _char_context.gram_size)];
        auto after_in =
            _char_context.gram_id[std::string(data + loc + _char_context.window_size - _char_context.gram_size + 1,
                                              _char_context.gram_size)];
        --cont.before[before_out];
        ++cont.before[before_in];
        --cont.after[after_out];
        ++cont.after[after_in];
      }
      const auto &hash_func_before = _feigenbaum.get_hash_func(t, true, tk);
      const auto &hash_func_after = _feigenbaum.get_hash_func(t, false, tk);

#ifdef DEBUG_CC_FEIGENBAUM
      spdlog::debug("loc: {}, char: {}, before: {}, after: {}", loc,
          data[loc],
          stringtify(cont.before.cbegin(), cont.before.cend()),
          stringtify(cont.after.cbegin(), cont.after.cend()));
#endif
      for (unsigned k = 0; k < _feigenbaum.num_sketches; ++k) {
        // pay attention ==> res[k] = 0;//
        for (unsigned d = 0; d < cont.before.size(); ++d) {
          if (cont.before[d] > 0) {
            auto hf_id = _feigenbaum.get_hash_func_id(k, d);
            res[k] += _feigenbaum.eval_hash(&hash_func_before[hf_id], cont.before[d]);
          }
        }
        for (unsigned d = 0; d < cont.after.size(); ++d) {
          if (cont.after[d] > 0) {
            auto hf_id = _feigenbaum.get_hash_func_id(k, d);
            res[k] += _feigenbaum.eval_hash(&hash_func_after[hf_id], cont.after[d]);
          }
        }
      }
    }
  }

 private:
  CharContext _char_context;
  ContextFeigenbaum _feigenbaum;
};

///
/// \brief CContextsDenseFeigenbaum
///
/// Removed location dependencies
struct CContextsDenseFeigenbaumV2 {

  /// customized hash function for CContextsDenseFeigenbaum
  /// which is used to indexing the hash functions for Feigenbaum sketch
  struct SpTupleHash {
    std::size_t operator()(std::tuple<unsigned/* char id */,
                                      unsigned/* occurrences */,
                                      bool /* (context's) before (windows) or not */> const &t) const noexcept {
      unsigned key = (std::get<2>(t) ? 0 : 1);
      key <<= 16u;
      key |= std::get<1>(t);
      key <<= 8u;
      key |= std::get<0>(t);
      return std::hash<unsigned>{}(key);
    }
  };
  ///
  /// \brief Contexts
  ///
  struct Contexts {
    std::vector<int> before;/// grams for before window
    std::vector<int> after;/// grams for after window
  };

  static constexpr unsigned MAX_CHAR_SIZE = 256;/// 1B char
  static constexpr unsigned MAX_STR_LEN = (1u << 31u) - 1u;///
  static constexpr unsigned MAX_GRAM_NUM = 8196;

  ///
  /// \brief CharContext
  ///
  /// This is a customized version of CContexts
  ///
  struct CharContext {
    //////////////////////// member variables /////////////////////////////
    unsigned window_size;          // window size
    unsigned gram_size;            // gram size, i.e., value for q
    unsigned num_char;             // alphabet size
    int char_id[MAX_CHAR_SIZE];    // id for each alphabet
    unsigned cdim;                 // dimension after performing q-gram
    bool wrap_around;              // whether performing wrap-around for the input string
    bool wrap_around_in;           // whether performing wrap-around within the windows
    /// Remarks. Usually wrap around or white-padding (i.e., padding q - 1 white letters that are
    ///          not included in the alphabets at the beginning and ending of the string) are used to "normalize"
    ///          the impact of errors, since errors occur at the first/last few letters will have less impact because
    ///          they will occur in less grams, if those schemes are not used. In the future version, we will try to implement
    ///          the white padding also. White padding seems to be more robust than wrap around.
    /// TODO replace wrap around with white-padding
    ///
    mutable std::unordered_map<std::string, int> gram_id;/// make sure can be accessed by "[]"
    static constexpr int EMPTY = -1;

    ///
    /// \brief constructor for CharContext
    ///
    /// \sa CContexts
    CharContext(const char *first,
                unsigned alphabet_size,
                unsigned w,
                unsigned q,
                bool wrap_around,
                bool wrap_around_in) : window_size(w), gram_size(q), num_char(alphabet_size),
                                       char_id{},
                                       cdim(unsigned(std::pow(alphabet_size, q))),
                                       wrap_around(wrap_around), wrap_around_in(wrap_around_in) {
      assert(cdim <= MAX_GRAM_NUM);
      std::fill(char_id, char_id + MAX_CHAR_SIZE, EMPTY);
      for (unsigned i = 0; i < alphabet_size; ++i) char_id[*(first + i)] = i;
      std::string str(gram_size, ' ');
      unsigned cnt = 0;
      _generate_dict(first, alphabet_size, 0, str, cnt);
      assert(cnt == cdim);
    }

    Contexts apply_cconext_impl(const char *str, unsigned len, unsigned loc) const {
      std::string window(window_size, ' ');
      if (loc < window_size) {
        for (int i = loc - 1, c = window_size - 1; c >= 0; --c, --i) window[c] = str[(i + len) % len];
      } else {
        for (int i = loc - 1, c = window_size - 1; c >= 0; --c, --i) window[c] = str[i];
      }
      Contexts cont{};
      cont.before.resize(cdim, 0);
      _apply_qgram_impl_wo(window.c_str(), window_size, cont.before);
      if (wrap_around_in) _apply_qgram_impl_wrap_part(window.c_str(), window_size, cont.before);

      if (loc + window_size >= len) {
        for (unsigned i = loc + 1, c = 0; c < window_size; ++c, ++i) window[c] = str[i % len];
      } else {
        for (unsigned i = loc + 1, c = 0; c < window_size; ++c, ++i) window[c] = str[i];
      }
      cont.after.resize(cdim, 0);
      _apply_qgram_impl_wo(window.c_str(), window_size, cont.after);
      if (wrap_around_in) _apply_qgram_impl_wrap_part(window.c_str(), window_size, cont.after);

      return cont;
    }

   private:
    void _apply_qgram_impl_wo(const char *str, unsigned len, std::vector<int> &res) const {
      unsigned ub = len - gram_size + 1;
      unsigned i = 0;
      for (; i < ub; ++i) ++res[gram_id[std::string(str + i, gram_size)]];
    }

    void _apply_qgram_impl_wrap_part(const char *str, unsigned len, std::vector<int> &res) const {
      unsigned i = len - gram_size + 1;
      for (unsigned k = 0; k < (gram_size - 1); ++k) {
        std::string tmp(gram_size, ' ');
        unsigned ks = i + k;
        for (unsigned j = 0; j < gram_size; ++j) tmp[j] = str[(ks + j) % len];
        ++res[gram_id[tmp]];
      }
    }
    void _generate_dict(const char *first, unsigned s, unsigned level, std::string &str, unsigned &cnt) {
      if (level == gram_size) {
        gram_id[str] = cnt;
        ++cnt;
        return;
      }
      for (unsigned i = 0; i < s; ++i) {
        str[level] = first[i];
        _generate_dict(first, s, level + 1, str, cnt);
      }
    }
  };// CharContext

  ///
  /// \brief ContextFeigenbaum
  ///
  /// This class implements customized version of FeigenbaumSketchL1.
  ///
  struct ContextFeigenbaum {
    ////////////////////////////////member variables//////////////////////////////////////////////////////
    typedef uint8_t my_word_t;
    typedef std::tuple<unsigned /* char id */, unsigned /* occurrences */, bool /* before or not */>
        feigen_hash_index_t;
    unsigned num_sketches;
    unsigned random_seed;
    std::mt19937_64 eng;
    int num_hashes_each_char;
    int num_char;
    int dim;
    int max_range;
    int inner_dim;
    mutable std::unordered_map<feigen_hash_index_t, std::vector<my_word_t>, SpTupleHash> hash;
    static constexpr unsigned WORD_SIZE = (sizeof(my_word_t) << 3u);
    ///////////////////////////////////////////////////////////////////////////////////////////////////////

    ContextFeigenbaum(unsigned m, unsigned seed) :
        num_sketches(m),
        random_seed(seed),
        eng(seed),
        num_hashes_each_char(-1),
        num_char(-1),
        dim(-1),
        max_range(-1),
        inner_dim(-1),
        hash() {
    }

    const std::vector<my_word_t> &get_hash_func(unsigned ch_id, bool before, unsigned loc) const {
      auto h_id = std::make_tuple(ch_id, loc, before);
      if (!hash.count(h_id)) return generate_random_hash(h_id);
      return hash[h_id];
    }

    unsigned get_hash_func_id(unsigned k, unsigned d) const {
      return k * dim * inner_dim + d * inner_dim;
    }

    int eval_hash(const my_word_t *hash_func, unsigned key) const {
      int h = 0;
      auto idx = 0;
      while (key >= WORD_SIZE) {
        h += detail::fake_range_sum(hash_func[idx]);
        key -= WORD_SIZE;
        ++idx;
      }
      if (key > 0)h += detail::fake_range_sum(hash_func[idx], key);
      return h;
    }

    const std::vector<my_word_t> &generate_random_hash(const std::tuple<unsigned, unsigned, bool> &h_id) const {

#ifdef DEBUG_CC_FEIGENBAUM
      spdlog::debug("generate new hash function for: char id: {}, occ: {}, before: {}",
          std::get<0>(h_id), std::get<1>(h_id), std::get<2>(h_id));
#endif
      std::uniform_int_distribution<> distribution(0, 1);
      auto rng = std::bind(distribution, eng);
      std::bitset<WORD_SIZE> bits;

      hash[h_id] = std::vector<my_word_t>(num_sketches * dim * inner_dim, 0u);

      auto &new_hash_func = hash[h_id];
      for (unsigned k = 0; k < num_sketches; ++k) {
        for (unsigned d = 0; d < dim; ++d) {
          unsigned j = 0;
          auto idx = k * dim * inner_dim + d * inner_dim;/// pay attention
          for (unsigned r = 0; r < max_range; ++r) {
            bits[j] = rng();
            ++j;
            if (j % WORD_SIZE == 0) {
              new_hash_func[idx] = bits.to_ulong();
              bits.reset();// reset bits
              j = 0;
              ++idx;
            }
          }
          if (j > 0) {
            new_hash_func[idx] = bits.to_ulong();
            bits.reset();// reset bits
          }
        }
      }

      return new_hash_func;
    }
  };// ContextFeigenbaum

  ///
  /// \brief CContextsDenseFeigenbaum
  ///
  /// \param first
  /// \param alphabet_size
  /// \param max_len
  /// \param m
  /// \param w
  /// \param q
  /// \param seed
  /// \param wrap_around
  /// \param wrap_around_in
  CContextsDenseFeigenbaumV2(const char *first,
                             unsigned alphabet_size,
                             unsigned max_len,
                             unsigned m,
                             unsigned w,
                             unsigned q,
                             unsigned seed = std::chrono::system_clock::now().time_since_epoch().count(),
                             bool wrap_around = false,
                             bool wrap_around_in = false) : _char_context(first,
                                                                          alphabet_size,
                                                                          w,
                                                                          q,
                                                                          wrap_around,
                                                                          wrap_around_in),
                                                            _feigenbaum(m, seed) {
    assert(max_len <= MAX_STR_LEN && "Sorry! This implementation can not scale to very long strings!");
    _feigenbaum.dim = static_cast<int>(std::pow(alphabet_size, q));
    _feigenbaum.num_hashes_each_char = max_len;
    _feigenbaum.max_range = w;
    _feigenbaum.inner_dim = (w + ContextFeigenbaum::WORD_SIZE - 1) / ContextFeigenbaum::WORD_SIZE;
    _feigenbaum.num_char = alphabet_size;
  }

  template<typename ResType>
  void apply(const char *data, unsigned len, ResType *res) {
    std::vector<int> cnt(_char_context.num_char, 0);
    std::fill(res, res + _feigenbaum.num_sketches, 0);// init

    const unsigned ls = (_char_context.wrap_around ? 0 : _char_context.window_size);
    const unsigned le = (_char_context.wrap_around ? len : (len - _char_context.window_size));

    Contexts cont;
    for (unsigned loc = ls; loc < le; ++loc) {
      auto t = _char_context.char_id[data[loc]];
      auto tk = cnt[t];
      //++cnt[t]; // TODO: remove location dependencies to reduce index space (otherwise too many hash functions
      // would be generated)
      if (loc == ls || _char_context.wrap_around_in) {
        cont = _char_context.apply_cconext_impl(data, len, loc);
      } else {
        //// avoid recalculating q-grams for the two windows
        //// TODO !! figure out how to do this for the wrap around cases !!
        auto before_out =
            _char_context.gram_id[std::string(data + loc - 1 - _char_context.window_size, _char_context.gram_size)];
        auto before_in =
            _char_context.gram_id[std::string(data + loc - _char_context.gram_size, _char_context.gram_size)];
        auto after_out = _char_context.gram_id[std::string(data + loc, _char_context.gram_size)];
        auto after_in =
            _char_context.gram_id[std::string(data + loc + _char_context.window_size - _char_context.gram_size + 1,
                                              _char_context.gram_size)];
        --cont.before[before_out];
        ++cont.before[before_in];
        --cont.after[after_out];
        ++cont.after[after_in];
      }
      const auto &hash_func_before = _feigenbaum.get_hash_func(t, true, tk);
      const auto &hash_func_after = _feigenbaum.get_hash_func(t, false, tk);

#ifdef DEBUG_CC_FEIGENBAUM
      spdlog::debug("loc: {}, char: {}, before: {}, after: {}", loc,
          data[loc],
          stringtify(cont.before.cbegin(), cont.before.cend()),
          stringtify(cont.after.cbegin(), cont.after.cend()));
#endif
      for (unsigned k = 0; k < _feigenbaum.num_sketches; ++k) {
        // pay attention ==> res[k] = 0;//
        for (unsigned d = 0; d < cont.before.size(); ++d) {
          if (cont.before[d] > 0) {
            auto hf_id = _feigenbaum.get_hash_func_id(k, d);
            res[k] += _feigenbaum.eval_hash(&hash_func_before[hf_id], cont.before[d]);
          }
        }
        for (unsigned d = 0; d < cont.after.size(); ++d) {
          if (cont.after[d] > 0) {
            auto hf_id = _feigenbaum.get_hash_func_id(k, d);
            res[k] += _feigenbaum.eval_hash(&hash_func_after[hf_id], cont.after[d]);
          }
        }
      }
    }
  }

 private:
  CharContext _char_context;
  ContextFeigenbaum _feigenbaum;
};

///
/// \brief CContextsDenseFeigenbaum
///
struct CContextsDenseFeigenbaumV3 {

  /// customized hash function for CContextsDenseFeigenbaum
  /// which is used to indexing the hash functions for Feigenbaum sketch
  struct SpTupleHash {
    std::size_t operator()(std::tuple<unsigned/* char id */,
                                      unsigned/* occurrences */,
                                      bool /* (context's) before (windows) or not */> const &t) const noexcept {
      unsigned key = (std::get<2>(t) ? 0 : 1);
      key <<= 16u;
      key |= std::get<1>(t);
      key <<= 8u;
      key |= std::get<0>(t);
      return std::hash<unsigned>{}(key);
    }
  };
  ///
  /// \brief Contexts
  ///
  struct Contexts {
    std::vector<int> before;/// grams for before window
    std::vector<int> after;/// grams for after window
  };

  static constexpr unsigned MAX_CHAR_SIZE = 256;/// 1B char
  static constexpr unsigned MAX_STR_LEN = (1u << 31u) - 1u;///
  static constexpr unsigned MAX_GRAM_NUM = 8196;

  ///
  /// \brief CharContext
  ///
  /// This is a customized version of CContexts
  ///
  struct CharContext {
    //////////////////////// member variables /////////////////////////////
    unsigned window_size;          // window size
    unsigned gram_size;            // gram size, i.e., value for q
    unsigned num_char;             // alphabet size
    int char_id[MAX_CHAR_SIZE];    // id for each alphabet
    unsigned cdim;                 // dimension after performing q-gram
    bool wrap_around;              // whether performing wrap-around for the input string
    bool wrap_around_in;           // whether performing wrap-around within the windows
    /// Remarks. Usually wrap around or white-padding (i.e., padding q - 1 white letters that are
    ///          not included in the alphabets at the beginning and ending of the string) are used to "normalize"
    ///          the impact of errors, since errors occur at the first/last few letters will have less impact because
    ///          they will occur in less grams, if those schemes are not used. In the future version, we will try to implement
    ///          the white padding also. White padding seems to be more robust than wrap around.
    /// TODO replace wrap around with white-padding
    ///
    mutable std::unordered_map<std::string, int> gram_id;/// make sure can be accessed by "[]"
    static constexpr int EMPTY = -1;

    ///
    /// \brief constructor for CharContext
    ///
    /// \sa CContexts
    CharContext(const char *first,
                unsigned alphabet_size,
                unsigned w,
                unsigned q,
                bool wrap_around,
                bool wrap_around_in) : window_size(w), gram_size(q), num_char(alphabet_size),
                                       char_id{},
                                       cdim(unsigned(std::pow(alphabet_size, q))),
                                       wrap_around(wrap_around), wrap_around_in(wrap_around_in) {
      assert(cdim <= MAX_GRAM_NUM);
      std::fill(char_id, char_id + MAX_CHAR_SIZE, EMPTY);
      for (unsigned i = 0; i < alphabet_size; ++i) char_id[*(first + i)] = i;
      std::string str(gram_size, ' ');
      unsigned cnt = 0;
      _generate_dict(first, alphabet_size, 0, str, cnt);
      assert(cnt == cdim);
    }

    Contexts apply_cconext_impl(const char *str, unsigned len, unsigned loc) const {
      std::string window(window_size, ' ');
      if (loc < window_size) {
        for (int i = loc - 1, c = window_size - 1; c >= 0; --c, --i) window[c] = str[(i + len) % len];
      } else {
        for (int i = loc - 1, c = window_size - 1; c >= 0; --c, --i) window[c] = str[i];
      }
      Contexts cont{};
      cont.before.resize(cdim, 0);
      _apply_qgram_impl_wo(window.c_str(), window_size, cont.before);
      if (wrap_around_in) _apply_qgram_impl_wrap_part(window.c_str(), window_size, cont.before);

      if (loc + window_size >= len) {
        for (unsigned i = loc + 1, c = 0; c < window_size; ++c, ++i) window[c] = str[i % len];
      } else {
        for (unsigned i = loc + 1, c = 0; c < window_size; ++c, ++i) window[c] = str[i];
      }
      cont.after.resize(cdim, 0);
      _apply_qgram_impl_wo(window.c_str(), window_size, cont.after);
      if (wrap_around_in) _apply_qgram_impl_wrap_part(window.c_str(), window_size, cont.after);

      return cont;
    }

   private:
    void _apply_qgram_impl_wo(const char *str, unsigned len, std::vector<int> &res) const {
      unsigned ub = len - gram_size + 1;
      unsigned i = 0;
      for (; i < ub; ++i) ++res[gram_id[std::string(str + i, gram_size)]];
    }

    void _apply_qgram_impl_wrap_part(const char *str, unsigned len, std::vector<int> &res) const {
      unsigned i = len - gram_size + 1;
      for (unsigned k = 0; k < (gram_size - 1); ++k) {
        std::string tmp(gram_size, ' ');
        unsigned ks = i + k;
        for (unsigned j = 0; j < gram_size; ++j) tmp[j] = str[(ks + j) % len];
        ++res[gram_id[tmp]];
      }
    }
    void _generate_dict(const char *first, unsigned s, unsigned level, std::string &str, unsigned &cnt) {
      if (level == gram_size) {
        gram_id[str] = cnt;
        ++cnt;
        return;
      }
      for (unsigned i = 0; i < s; ++i) {
        str[level] = first[i];
        _generate_dict(first, s, level + 1, str, cnt);
      }
    }
  };// CharContext

  ///
  /// \brief ContextFeigenbaum
  ///
  /// This class implements customized version of FeigenbaumSketchL1.
  ///
  struct ContextFeigenbaum {
    ////////////////////////////////member variables//////////////////////////////////////////////////////
    typedef uint8_t my_word_t;
    typedef std::tuple<unsigned /* char id */, unsigned /* occurrences */, bool /* before or not */>
        feigen_hash_index_t;
    unsigned num_sketches;
    unsigned random_seed;
    std::mt19937_64 eng;
    int num_hashes_each_char;
    int num_char;
    int dim;
    int max_range;
    int inner_dim;
    mutable std::unordered_map<feigen_hash_index_t, std::vector<my_word_t>, SpTupleHash> hash;
    static constexpr unsigned WORD_SIZE = (sizeof(my_word_t) << 3u);
    ///////////////////////////////////////////////////////////////////////////////////////////////////////

    ContextFeigenbaum(unsigned m, unsigned seed) :
        num_sketches(m),
        random_seed(seed),
        eng(seed),
        num_hashes_each_char(-1),
        num_char(-1),
        dim(-1),
        max_range(-1),
        inner_dim(-1),
        hash() {
    }

    const std::vector<my_word_t> &get_hash_func(unsigned ch_id, bool before, unsigned loc) const {
      auto h_id = std::make_tuple(ch_id, loc, before);
      if (!hash.count(h_id)) return generate_random_hash(h_id);
      return hash[h_id];
    }

    unsigned get_hash_func_id(unsigned k, unsigned d) const {
      return k * dim * inner_dim + d * inner_dim;
    }

    int eval_hash(const my_word_t *hash_func, unsigned key) const {
      int h = 0;
      auto idx = 0;
      while (key >= WORD_SIZE) {
        h += detail::fake_range_sum(hash_func[idx]);
        key -= WORD_SIZE;
        ++idx;
      }
      if (key > 0)h += detail::fake_range_sum(hash_func[idx], key);
      return h;
    }

    const std::vector<my_word_t> &generate_random_hash(const std::tuple<unsigned, unsigned, bool> &h_id) const {

#ifdef DEBUG_CC_FEIGENBAUM
      spdlog::debug("generate new hash function for: char id: {}, occ: {}, before: {}",
          std::get<0>(h_id), std::get<1>(h_id), std::get<2>(h_id));
#endif
      std::uniform_int_distribution<> distribution(0, 1);
      auto rng = std::bind(distribution, eng);
      std::bitset<WORD_SIZE> bits;

      hash[h_id] = std::vector<my_word_t>(num_sketches * dim * inner_dim, 0u);

      auto &new_hash_func = hash[h_id];
      for (unsigned k = 0; k < num_sketches; ++k) {
        for (unsigned d = 0; d < dim; ++d) {
          unsigned j = 0;
          auto idx = k * dim * inner_dim + d * inner_dim;/// pay attention
          for (unsigned r = 0; r < max_range; ++r) {
            bits[j] = rng();
            ++j;
            if (j % WORD_SIZE == 0) {
              new_hash_func[idx] = bits.to_ulong();
              bits.reset();// reset bits
              j = 0;
              ++idx;
            }
          }
          if (j > 0) {
            new_hash_func[idx] = bits.to_ulong();
            bits.reset();// reset bits
          }
        }
      }

      return new_hash_func;
    }
  };// ContextFeigenbaum

  ///
  /// \brief CContextsDenseFeigenbaum
  ///
  /// \param first
  /// \param alphabet_size
  /// \param max_len
  /// \param m
  /// \param w
  /// \param q
  /// \param seed
  /// \param wrap_around
  /// \param wrap_around_in
  CContextsDenseFeigenbaumV3(const char *first,
                             unsigned alphabet_size,
                             unsigned max_len,
                             unsigned m,
                             unsigned w,
                             unsigned q,
                             unsigned seed = std::chrono::system_clock::now().time_since_epoch().count(),
                             bool wrap_around = false,
                             bool wrap_around_in = false) : _char_context(first,
                                                                          alphabet_size,
                                                                          w,
                                                                          q,
                                                                          wrap_around,
                                                                          wrap_around_in),
                                                            _feigenbaum(m, seed) {
    assert(max_len <= MAX_STR_LEN && "Sorry! This implementation can not scale to very long strings!");
    _feigenbaum.dim = static_cast<int>(std::pow(alphabet_size, q));
    _feigenbaum.num_hashes_each_char = max_len;
    _feigenbaum.max_range = w;
    _feigenbaum.inner_dim = (w + ContextFeigenbaum::WORD_SIZE - 1) / ContextFeigenbaum::WORD_SIZE;
    _feigenbaum.num_char = alphabet_size;
  }

  template<typename ResType>
  void apply(const char *data, unsigned len, ResType *res) {

    const unsigned ls = (_char_context.wrap_around ? 0 : _char_context.window_size);
    const unsigned le = (_char_context.wrap_around ? len : (len - _char_context.window_size));

    Contexts cont;
    std::vector<std::vector<int>> before_bundle(_char_context.num_char, std::vector<int>(_char_context.cdim, 0));
    std::vector<std::vector<int>> after_bundle(_char_context.num_char, std::vector<int>(_char_context.cdim, 0));

    for (unsigned loc = ls; loc < le; ++loc) {
      auto t = _char_context.char_id[data[loc]];
//      auto tk = cnt[t];
////      ++cnt[t]; // TODO: remove location dependencies to reduce index space (otherwise too many hash functions
//      // would be generated)
      if (loc == ls || _char_context.wrap_around_in) {
        cont = _char_context.apply_cconext_impl(data, len, loc);
      } else {
        //// avoid recalculating q-grams for the two windows
        //// TODO !! figure out how to do this for the wrap around cases !!
        auto before_out =
            _char_context.gram_id[std::string(data + loc - 1 - _char_context.window_size, _char_context.gram_size)];
        auto before_in =
            _char_context.gram_id[std::string(data + loc - _char_context.gram_size, _char_context.gram_size)];
        auto after_out = _char_context.gram_id[std::string(data + loc, _char_context.gram_size)];
        auto after_in =
            _char_context.gram_id[std::string(data + loc + _char_context.window_size - _char_context.gram_size + 1,
                                              _char_context.gram_size)];
        --cont.before[before_out];
        ++cont.before[before_in];
        --cont.after[after_out];
        ++cont.after[after_in];
      }

      for (unsigned gi = 0; gi < cont.before.size(); ++gi) {
        before_bundle[t][gi] += cont.before[gi];
        after_bundle[t][gi] += cont.after[gi];
      }

    }

    std::fill(res, res + _feigenbaum.num_sketches, 0);// init
    for (unsigned t = 0; t < before_bundle.size(); ++t) {
      const auto &hash_func_before = _feigenbaum.get_hash_func(t, true, 0);
      const auto &hash_func_after = _feigenbaum.get_hash_func(t, false, 0);
      for (unsigned k = 0; k < _feigenbaum.num_sketches; ++k) {
        for (unsigned d = 0; d < before_bundle[t].size(); ++d) {
          if (before_bundle[t][d] > 0) {
            auto hf_id = _feigenbaum.get_hash_func_id(k, d);
            res[k] += _feigenbaum.eval_hash(&hash_func_before[hf_id], before_bundle[t][d]);
          }
        }
        for (unsigned d = 0; d < after_bundle[t].size(); ++d) {
          if (after_bundle[t][d] > 0) {
            auto hf_id = _feigenbaum.get_hash_func_id(k, d);
            res[k] += _feigenbaum.eval_hash(&hash_func_after[hf_id], after_bundle[t][d]);
          }
        }
      }
    }
  }

 private:
  CharContext _char_context;
  ContextFeigenbaum _feigenbaum;
};

///
/// \brief CGKBitSamplingHash
///
/// This class implements the scheme proposed in the following paper.
/// Zhang, Haoyu, and Qin Zhang. “EmbedJoin: Efficient Edit Similarity Joins via Embeddings.”
/// In Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining
/// - KDD ’17, 585–94. Halifax, NS, Canada: ACM Press, 2017. https://doi.org/10.1145/3097983.3098003.
///
/// Most lines of codes are (slightly) modified from the source codes provided by the authors.
///
struct CGKBitSamplingHash {
  ///
  /// \brief constructor for CGKBitSamplingHash
  ///
  ///
  /// \param first                  alphabets
  /// \param alphabet_size          alphabet size
  /// \param num_cgk                number of CGK embedding for each string
  /// \param num_hashes             number of LSH hashes
  /// \param num_sample_bits        number of sampling "bits" for the LSH
  /// \param max_len                maximum length for the input string
  /// \param seed                   random seed
  CGKBitSamplingHash(const char *first,
                     unsigned alphabet_size,
                     unsigned num_cgk,
                     unsigned num_hashes,
                     unsigned num_sample_bits,
                     unsigned max_len,
                     unsigned seed) :
      _m(num_hashes),
      _dim(3 * max_len),
      _n_samples(num_sample_bits),
      _seed(seed),
      _M(10003u),
      _mem(-1),
      _changed(false),
      _hash{},
      _hash_params(num_sample_bits, 0),
      _buckets(num_cgk, std::vector<std::list<unsigned>>(num_hashes * _M, std::list<unsigned>())),
      _cgk_ptr(nullptr) {
    std::mt19937_64 eng(_seed);
    _generate_hash(eng);
    _cgk_ptr = std::make_unique<CGKEmbed>(first, alphabet_size, max_len, num_cgk, eng);
  }

  void insert(const std::string &key, unsigned value) {
    /// Using CGK embeddings get new strings in Hamming space
    std::vector<std::string> strs(_cgk_ptr->num_cgk());
    _cgk_ptr->apply(key.c_str(), key.size(), &strs[0]);

    /// Handle each string
    for (unsigned si = 0; si < strs.size(); ++si) {
      const auto &str = strs[si];
      const auto &dict = _cgk_ptr->dict();

      for (unsigned k = 0; k < _m; ++k) {
        unsigned h = 0;

        unsigned idx_s = k * _n_samples;/// index for first-level hash (i.e., LSH)
        for (unsigned i = 0; i < _n_samples; ++i) {
          auto j = _hash[idx_s + i];/// which "bit" to sample
          if (j >= str.size()) h += _cgk_ptr->alphabet_size() * _hash_params[i];
          else h += dict[str[j]] * _hash_params[i];
        }

        h = (h % _M);/// index for second-level hash
#ifdef DEBUG_CGKLSH
        spdlog::debug("bucket id for {} is {}", key, h);
#endif
        _buckets[si][k * _M + h].push_back(value);
      }
    }

    _changed = true;
  }

  std::vector<unsigned> at(const std::string &key) const {
    const auto &dict = _cgk_ptr->dict();
    std::vector<unsigned> candidates;
    candidates.reserve(_m * 10);// avoid frequent memory allocation

    std::vector<std::string> strs(_cgk_ptr->num_cgk());
    _cgk_ptr->apply(key.c_str(), key.size(), &strs[0]);

    for (unsigned si = 0; si < strs.size(); ++si) {
      const auto &str = strs[si];
      for (unsigned k = 0; k < _m; ++k) {
        unsigned h = 0;
        unsigned idx_s = k * _n_samples;
        for (unsigned i = 0; i < _n_samples; ++i) {
          auto j = _hash[idx_s + i];
          if (j >= str.size()) h += _cgk_ptr->alphabet_size() * _hash_params[i];
          else h += dict[str[j]] * _hash_params[i];
        }
        h = (h % _M);
#ifdef DEBUG_CGKLSH
        spdlog::debug("bucket id for {} is {}", key, h);
#endif
        const auto &bucket = _buckets[si][k * _M + h];
        if (!bucket.empty()) candidates.insert(candidates.end(), bucket.begin(), bucket.end());
      }
    }

    std::sort(candidates.begin(), candidates.end());
    auto last = std::unique(candidates.begin(), candidates.end());
    candidates.erase(last, candidates.end());
    return candidates;
  }

  size_t index_size() {
    if (_changed || _mem == -1) {
      size_t base_mem = _m * _cgk_ptr->num_cgk() * _M * PTR_SIZE;//
      size_t extra_mem = 0;

      for (unsigned k = 0; k < _cgk_ptr->num_cgk(); ++k)
        for (unsigned i = 0; i < _m; ++i)
          for (unsigned j = 0; j < _M; ++j)
            extra_mem += _buckets[k][i * _M + j].empty() ? 0 : (_buckets[k][i * _M + j].size() * LNODE_SIZE);

      _mem = base_mem + extra_mem;
    }
    return _mem;
  }
  /// debug only
  size_t bucket_size(unsigned cgk_i, unsigned hash_j, unsigned bucket_k) const {
    return _buckets[cgk_i][hash_j * _M + bucket_k].size();
  }
  /// debug only
  size_t bucket_capacity() const {
    return _M;
  }

 private:
  template<typename RandomEngine>
  void _generate_hash(RandomEngine &eng) {
    std::uniform_int_distribution<> dist(0, _dim);
    auto rng = std::bind(dist, eng);

    _hash.resize(_m * _n_samples);
    for (unsigned k = 0; k < _m; ++k)
      for (unsigned i = 0; i < _n_samples; ++i)
        _hash[k * _n_samples + i] = rng();

    std::uniform_int_distribution<>::param_type p(0, _M - 2);
    dist.param(p);
    for (unsigned i = 0; i < _n_samples; ++i) _hash_params[i] = dist(eng);

#ifdef DEBUG_CGKLSH
    spdlog::debug("hash_param: {}", stringtify(_hash_params.begin(), _hash_params.end()));
#endif
  }

//  void _generate_hash() {
//    std::mt19937_64 eng(_seed);
//    _generate_hash(eng);
//  }
  unsigned _m;
  unsigned _dim;
  unsigned _n_samples;
  unsigned _seed;
  unsigned _M; // for second level hash
  long long _mem;
  bool _changed;
  std::vector<unsigned> _hash;
  std::vector<unsigned> _hash_params;
  std::vector<std::vector<std::list<unsigned>>> _buckets;
  //
  std::unique_ptr<CGKEmbed> _cgk_ptr;

  static constexpr unsigned PTR_SIZE = 4;
  static constexpr unsigned LNODE_SIZE = sizeof(unsigned) + 2 * PTR_SIZE;// std::list implements doubly-linked list
};

namespace detail {
/// TODO add support to other compiler
//inline int popcnt(uint64_t v) {
//  //return __builtin_popcountll(v);
//}

inline int64_t random_walk_block64(const char *s,
                                   const uint64_t *hash,
                                   unsigned wi,
                                   const int *dict,
                                   unsigned enc_dim) {
  return ((hash[dict[*(s + 0)] * enc_dim + wi] & 0x1ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 1)] * enc_dim + wi] & 0x2ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 2)] * enc_dim + wi] & 0x4ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 3)] * enc_dim + wi] & 0x8ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 4)] * enc_dim + wi] & 0x10ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 5)] * enc_dim + wi] & 0x20ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 6)] * enc_dim + wi] & 0x40ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 7)] * enc_dim + wi] & 0x80ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 8)] * enc_dim + wi] & 0x100ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 9)] * enc_dim + wi] & 0x200ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 10)] * enc_dim + wi] & 0x400ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 11)] * enc_dim + wi] & 0x800ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 12)] * enc_dim + wi] & 0x1000ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 13)] * enc_dim + wi] & 0x2000ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 14)] * enc_dim + wi] & 0x4000ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 15)] * enc_dim + wi] & 0x8000ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 16)] * enc_dim + wi] & 0x10000ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 17)] * enc_dim + wi] & 0x20000ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 18)] * enc_dim + wi] & 0x40000ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 19)] * enc_dim + wi] & 0x80000ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 20)] * enc_dim + wi] & 0x100000ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 21)] * enc_dim + wi] & 0x200000ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 22)] * enc_dim + wi] & 0x400000ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 23)] * enc_dim + wi] & 0x800000ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 24)] * enc_dim + wi] & 0x1000000ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 25)] * enc_dim + wi] & 0x2000000ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 26)] * enc_dim + wi] & 0x4000000ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 27)] * enc_dim + wi] & 0x8000000ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 28)] * enc_dim + wi] & 0x10000000ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 29)] * enc_dim + wi] & 0x20000000ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 30)] * enc_dim + wi] & 0x40000000ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 31)] * enc_dim + wi] & 0x80000000ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 32)] * enc_dim + wi] & 0x100000000ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 33)] * enc_dim + wi] & 0x200000000ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 34)] * enc_dim + wi] & 0x400000000ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 35)] * enc_dim + wi] & 0x800000000ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 36)] * enc_dim + wi] & 0x1000000000ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 37)] * enc_dim + wi] & 0x2000000000ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 38)] * enc_dim + wi] & 0x4000000000ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 39)] * enc_dim + wi] & 0x8000000000ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 40)] * enc_dim + wi] & 0x10000000000ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 41)] * enc_dim + wi] & 0x20000000000ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 42)] * enc_dim + wi] & 0x40000000000ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 43)] * enc_dim + wi] & 0x80000000000ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 44)] * enc_dim + wi] & 0x100000000000ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 45)] * enc_dim + wi] & 0x200000000000ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 46)] * enc_dim + wi] & 0x400000000000ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 47)] * enc_dim + wi] & 0x800000000000ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 48)] * enc_dim + wi] & 0x1000000000000ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 49)] * enc_dim + wi] & 0x2000000000000ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 50)] * enc_dim + wi] & 0x4000000000000ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 51)] * enc_dim + wi] & 0x8000000000000ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 52)] * enc_dim + wi] & 0x10000000000000ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 53)] * enc_dim + wi] & 0x20000000000000ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 54)] * enc_dim + wi] & 0x40000000000000ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 55)] * enc_dim + wi] & 0x80000000000000ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 56)] * enc_dim + wi] & 0x100000000000000ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 57)] * enc_dim + wi] & 0x200000000000000ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 58)] * enc_dim + wi] & 0x400000000000000ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 59)] * enc_dim + wi] & 0x800000000000000ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 60)] * enc_dim + wi] & 0x1000000000000000ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 61)] * enc_dim + wi] & 0x2000000000000000ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 62)] * enc_dim + wi] & 0x4000000000000000ull) == 0 ? (-1) : 1)
      + ((hash[dict[*(s + 63)] * enc_dim + wi] & 0x8000000000000000ull) == 0 ? (-1) : 1);
}
}// end namespace detail

namespace detail {

namespace constants {
static constexpr uint8_t BitMask8[] = {
    0x0, 0x1, 0x3, 0x7, 0xF, 0x1F, 0x3F, 0x7F, 0xFF
};
static constexpr unsigned char BitsSetTable256[256] =
    {
#   define B2(n) n,     n+1,     n+1,     n+2
#   define B4(n) B2(n), B2(n+1), B2(n+1), B2(n+2)
#   define B6(n) B4(n), B4(n+1), B4(n+1), B4(n+2)
        B6(0), B6(1), B6(1), B6(2)
    };

}// end namespace constants

inline int fake_range_sum(uint8_t zero_one_vec, unsigned c_range) {
  uint8_t c = constants::BitMask8[c_range];
  int r = constants::BitsSetTable256[c & zero_one_vec] * 2;
  return (r - c_range);
}
inline int fake_range_sum(uint8_t zero_one_vec) {
  int r = constants::BitsSetTable256[zero_one_vec] * 2;
  return (r - 8);
}

template<typename T>
inline void q_gram_block64(const char *s,
                           unsigned q,
                           std::unordered_map<std::string, int> &dict,
                           T *res) {
  ++res[dict[std::string(s + 0, q)]];
  ++res[dict[std::string(s + 1, q)]];
  ++res[dict[std::string(s + 2, q)]];
  ++res[dict[std::string(s + 3, q)]];
  ++res[dict[std::string(s + 4, q)]];
  ++res[dict[std::string(s + 5, q)]];
  ++res[dict[std::string(s + 6, q)]];
  ++res[dict[std::string(s + 7, q)]];
  ++res[dict[std::string(s + 8, q)]];
  ++res[dict[std::string(s + 9, q)]];
  ++res[dict[std::string(s + 10, q)]];
  ++res[dict[std::string(s + 11, q)]];
  ++res[dict[std::string(s + 12, q)]];
  ++res[dict[std::string(s + 13, q)]];
  ++res[dict[std::string(s + 14, q)]];
  ++res[dict[std::string(s + 15, q)]];
  ++res[dict[std::string(s + 16, q)]];
  ++res[dict[std::string(s + 17, q)]];
  ++res[dict[std::string(s + 18, q)]];
  ++res[dict[std::string(s + 19, q)]];
  ++res[dict[std::string(s + 20, q)]];
  ++res[dict[std::string(s + 21, q)]];
  ++res[dict[std::string(s + 22, q)]];
  ++res[dict[std::string(s + 23, q)]];
  ++res[dict[std::string(s + 24, q)]];
  ++res[dict[std::string(s + 25, q)]];
  ++res[dict[std::string(s + 26, q)]];
  ++res[dict[std::string(s + 27, q)]];
  ++res[dict[std::string(s + 28, q)]];
  ++res[dict[std::string(s + 29, q)]];
  ++res[dict[std::string(s + 30, q)]];
  ++res[dict[std::string(s + 31, q)]];
  ++res[dict[std::string(s + 32, q)]];
  ++res[dict[std::string(s + 33, q)]];
  ++res[dict[std::string(s + 34, q)]];
  ++res[dict[std::string(s + 35, q)]];
  ++res[dict[std::string(s + 36, q)]];
  ++res[dict[std::string(s + 37, q)]];
  ++res[dict[std::string(s + 38, q)]];
  ++res[dict[std::string(s + 39, q)]];
  ++res[dict[std::string(s + 40, q)]];
  ++res[dict[std::string(s + 41, q)]];
  ++res[dict[std::string(s + 42, q)]];
  ++res[dict[std::string(s + 43, q)]];
  ++res[dict[std::string(s + 44, q)]];
  ++res[dict[std::string(s + 45, q)]];
  ++res[dict[std::string(s + 46, q)]];
  ++res[dict[std::string(s + 47, q)]];
  ++res[dict[std::string(s + 48, q)]];
  ++res[dict[std::string(s + 49, q)]];
  ++res[dict[std::string(s + 50, q)]];
  ++res[dict[std::string(s + 51, q)]];
  ++res[dict[std::string(s + 52, q)]];
  ++res[dict[std::string(s + 53, q)]];
  ++res[dict[std::string(s + 54, q)]];
  ++res[dict[std::string(s + 55, q)]];
  ++res[dict[std::string(s + 56, q)]];
  ++res[dict[std::string(s + 57, q)]];
  ++res[dict[std::string(s + 58, q)]];
  ++res[dict[std::string(s + 59, q)]];
  ++res[dict[std::string(s + 60, q)]];
  ++res[dict[std::string(s + 61, q)]];
  ++res[dict[std::string(s + 62, q)]];
  ++res[dict[std::string(s + 63, q)]];
}

inline void q_gram_block64(const char *s,
                           unsigned q,
                           std::unordered_map<std::string, int> &dict,
                           std::vector<int> &res) {
  ++res[dict[std::string(s + 0, q)]];
  ++res[dict[std::string(s + 1, q)]];
  ++res[dict[std::string(s + 2, q)]];
  ++res[dict[std::string(s + 3, q)]];
  ++res[dict[std::string(s + 4, q)]];
  ++res[dict[std::string(s + 5, q)]];
  ++res[dict[std::string(s + 6, q)]];
  ++res[dict[std::string(s + 7, q)]];
  ++res[dict[std::string(s + 8, q)]];
  ++res[dict[std::string(s + 9, q)]];
  ++res[dict[std::string(s + 10, q)]];
  ++res[dict[std::string(s + 11, q)]];
  ++res[dict[std::string(s + 12, q)]];
  ++res[dict[std::string(s + 13, q)]];
  ++res[dict[std::string(s + 14, q)]];
  ++res[dict[std::string(s + 15, q)]];
  ++res[dict[std::string(s + 16, q)]];
  ++res[dict[std::string(s + 17, q)]];
  ++res[dict[std::string(s + 18, q)]];
  ++res[dict[std::string(s + 19, q)]];
  ++res[dict[std::string(s + 20, q)]];
  ++res[dict[std::string(s + 21, q)]];
  ++res[dict[std::string(s + 22, q)]];
  ++res[dict[std::string(s + 23, q)]];
  ++res[dict[std::string(s + 24, q)]];
  ++res[dict[std::string(s + 25, q)]];
  ++res[dict[std::string(s + 26, q)]];
  ++res[dict[std::string(s + 27, q)]];
  ++res[dict[std::string(s + 28, q)]];
  ++res[dict[std::string(s + 29, q)]];
  ++res[dict[std::string(s + 30, q)]];
  ++res[dict[std::string(s + 31, q)]];
  ++res[dict[std::string(s + 32, q)]];
  ++res[dict[std::string(s + 33, q)]];
  ++res[dict[std::string(s + 34, q)]];
  ++res[dict[std::string(s + 35, q)]];
  ++res[dict[std::string(s + 36, q)]];
  ++res[dict[std::string(s + 37, q)]];
  ++res[dict[std::string(s + 38, q)]];
  ++res[dict[std::string(s + 39, q)]];
  ++res[dict[std::string(s + 40, q)]];
  ++res[dict[std::string(s + 41, q)]];
  ++res[dict[std::string(s + 42, q)]];
  ++res[dict[std::string(s + 43, q)]];
  ++res[dict[std::string(s + 44, q)]];
  ++res[dict[std::string(s + 45, q)]];
  ++res[dict[std::string(s + 46, q)]];
  ++res[dict[std::string(s + 47, q)]];
  ++res[dict[std::string(s + 48, q)]];
  ++res[dict[std::string(s + 49, q)]];
  ++res[dict[std::string(s + 50, q)]];
  ++res[dict[std::string(s + 51, q)]];
  ++res[dict[std::string(s + 52, q)]];
  ++res[dict[std::string(s + 53, q)]];
  ++res[dict[std::string(s + 54, q)]];
  ++res[dict[std::string(s + 55, q)]];
  ++res[dict[std::string(s + 56, q)]];
  ++res[dict[std::string(s + 57, q)]];
  ++res[dict[std::string(s + 58, q)]];
  ++res[dict[std::string(s + 59, q)]];
  ++res[dict[std::string(s + 60, q)]];
  ++res[dict[std::string(s + 61, q)]];
  ++res[dict[std::string(s + 62, q)]];
  ++res[dict[std::string(s + 63, q)]];
}

template<typename T>
inline void q_gram_block8(const char *s,
                          unsigned q,
                          std::unordered_map<std::string, int> &dict,
                          T *res) {
#ifdef DEBUG_GRAM
  spdlog::debug("first gram: {}, its id: {}", std::string(s + 0, q), dict[std::string(s + 0, q)]);
#endif
  ++res[dict[std::string(s + 0, q)]];
  ++res[dict[std::string(s + 1, q)]];
  ++res[dict[std::string(s + 2, q)]];
  ++res[dict[std::string(s + 3, q)]];
  ++res[dict[std::string(s + 4, q)]];
  ++res[dict[std::string(s + 5, q)]];
  ++res[dict[std::string(s + 6, q)]];
  ++res[dict[std::string(s + 7, q)]];
}
inline void q_gram_block8(const char *s,
                          unsigned q,
                          std::unordered_map<std::string, int> &dict,
                          std::vector<int> &res) {
#ifdef DEBUG_GRAM
  spdlog::debug("first gram: {}, its id: {}", std::string(s + 0, q), dict[std::string(s + 0, q)]);
#endif
  ++res[dict[std::string(s + 0, q)]];
  ++res[dict[std::string(s + 1, q)]];
  ++res[dict[std::string(s + 2, q)]];
  ++res[dict[std::string(s + 3, q)]];
  ++res[dict[std::string(s + 4, q)]];
  ++res[dict[std::string(s + 5, q)]];
  ++res[dict[std::string(s + 6, q)]];
  ++res[dict[std::string(s + 7, q)]];
}
}// end namespace detail
}// end namespace embed

} // end namespace ss::ann

#endif // _STRING_EMBED_HPP_
