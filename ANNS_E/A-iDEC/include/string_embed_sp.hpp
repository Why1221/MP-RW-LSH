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

#include <spdlog/spdlog.h>
#define DEBUG_CGK
#ifdef DEBUG_CGK
#include <boost/dynamic_bitset.hpp>
#endif

#define DEBUG_GRAM

namespace ss::ann {

namespace embed {

namespace detail {
inline void q_gram_block64(const char *s,
                           unsigned q,
                           std::unordered_map<std::string, int> &dict,
                           std::vector<int> &res);
inline void q_gram_block8(const char *s, unsigned q, std::unordered_map<std::string, int> &dict, std::vector<int> &res);
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
      _un_used(std::numeric_limits<int>::max()),
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
    for (; i < _dict.size(); ++i) _dict[i] = _un_used;
    i = 0;
    for (; i < alphabet_size; ++i) _dict[*(first + i)] = i;

    _generate_hash();
  }
  ///
  int map(const char ch) const {
    return _dict[ch];
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
  void _generate_hash() {
    std::mt19937_64 eng(_seed);
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
  const unsigned _alphabet_size;
  const int _un_used;
  const unsigned _N;
  const unsigned _m;
  unsigned _seed;
  std::array<int, 256> _dict;
  std::vector<int> _jump;
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
    assert(_gdim < constants::MAX_DICT_SIZE);
    std::string str(_q, ' ');
    unsigned cnt = 0;
    _generate_dict(first, alphabet_size, 0, str, cnt);
    assert(cnt == _gdim);
#ifdef DEBUG_GRAM
    for(const auto& map : _dict) _inverse_index[map.second] = map.first;
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
  int id(const std::string& s) const { return _dict.at(s); }
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

struct CContextsDense {
  struct Contexts {
    std::vector<int> before;
    std::vector<int> after;
  };
  CContextsDense(const char *first,
                 unsigned alphabet_size,
                 unsigned w,
                 unsigned q,
                 bool wrap_around,
                 bool wrap_around_in) : _w(w),
                                        _q(q),
                                        _wrap(wrap_around),
                                        _wrap_in(wrap_around_in),
                                        _gdim(static_cast<unsigned >(std::pow(alphabet_size, q))),
                                        _dict()
#ifdef DEBUG_GRAM
      , _inverse_index(_gdim, "")
#endif
  {
    assert(_gdim < constants::MAX_DICT_SIZE);
    std::string str(_q, ' ');
    unsigned cnt = 0;
    _generate_dict(first, alphabet_size, 0, str, cnt);
    assert(cnt == _gdim);
#ifdef DEBUG_GRAM
    for(const auto& map : _dict) _inverse_index[map.second] = map.first;
#endif
  }
  Contexts apply(const char *str, unsigned len, unsigned loc) const {
    return _apply_impl(str, len, loc);
  }
  Contexts apply(const std::string &str, unsigned loc) const {
    return apply(str.c_str(), str.size(), loc);
  }
  std::unordered_map<std::string, int> dict() const {
    return _dict;
  }
#ifdef DEBUG_GRAM
  std::vector<std::string> inverse_map() const {
    return _inverse_index;
  }
#endif
 private:
  Contexts _apply_impl(const char *str, unsigned len, unsigned loc) const {
    std::string window(_w, ' ');
    if (loc < _w) {
      if (!_wrap) throw std::runtime_error("Value for loc is invalid!");
      for (int i = loc - 1, c = _w - 1; c >= 0; --c, --i) window[c] = str[(i + len) % len];
    } else {
      for (int i = loc - 1, c = _w - 1; c >= 0; --c, --i) window[c] = str[i];
    }
    Contexts cont;
    cont.before.resize(_gdim, 0);
    _apply_qgram_impl_wo(window.c_str(), _w, cont.before);
    if (_wrap_in) _apply_qgram_impl_wrap_part(window.c_str(), _w, cont.before);

    if (loc + _w >= len) {
      if (!_wrap) throw std::runtime_error("Value for loc is invalid!");
      for (int i = loc + 1, c = 0; c < _w; ++c, ++i) window[c] = str[i % len];
    } else {
      for (int i = loc + 1, c = 0; c < _w; ++c, ++i) window[c] = str[i];
    }
    cont.after.resize(_gdim, 0);
    _apply_qgram_impl_wo(window.c_str(), _w, cont.after);
    if (_wrap_in) _apply_qgram_impl_wrap_part(window.c_str(), _w, cont.after);

    return cont;
  }

  void _apply_qgram_impl_wo(const char *str, unsigned len, std::vector<int> &res) const {
    unsigned ub = len - _q + 1;
    unsigned i = 0;
    for (; i < ub; ++i) ++res[_dict[std::string(str + i, _q)]];
  }
  void _apply_qgram_impl_wrap_part(const char *str, unsigned len, std::vector<int> &res) const {
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
      _dict[str] = cnt;
      ++cnt;
      return;
    }
    for (unsigned i = 0; i < s; ++i) {
      str[level] = first[i];
      _generate_dict(first, s, level + 1, str, cnt);
    }
  }
  unsigned _w;
  unsigned _q;
  bool _wrap;
  bool _wrap_in;
  unsigned _gdim;
  mutable std::unordered_map<std::string, int> _dict;
#ifdef DEBUG_GRAM
  std::vector<std::string> _inverse_index;
#endif
};

namespace detail {
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
