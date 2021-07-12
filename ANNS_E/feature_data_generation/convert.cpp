#include <utility>

#include <array>
#include <limits>
#include <chrono>
#include <bitset>
#include <random>
#include <memory>
#include <functional>
#include <tuple>
#include <unordered_map>
#include <functional>
#include <list>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <srs_utils.hpp>// result pair
#include <edlib/edlib.h>
//#include "string_feature.h"

#include <spdlog/spdlog.h>
#include <boost/timer.hpp>
#include <boost/filesystem.hpp>
#include <boost/progress.hpp>
// #include <spdlog/spdlog.h>// logging
//#include <srs/SRSCoverTree.h>
// #include <srs_utils.hpp>// result pair


// #include <io.hpp>
// #include <distance_metric_sp.hpp>
// #include <string_feature_sp.hpp>

// typedef boost::timer Timer;
// typedef boost::progress_display ProgressBar;
#include <boost/filesystem.hpp>
#include <highfive/H5File.hpp>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>



typedef boost::progress_display ProgressBar;
typedef boost::timer Timer;
using namespace HighFive;

// unsigned q = 5;
unsigned max_str_len = 35213;



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


///
/// \brief QGramDense5
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

///
/// \brief CContextsDenseFeigenbaum
///
struct CContextsDense {

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
  /// \brief CContextsDense
  ///
  /// \param first
  /// \param alphabet_size
  /// \param max_len
  /// \param m
  /// \param w
  /// \param q
  /// \param wrap_around
  /// \param wrap_around_in
  CContextsDense(const char *first,
                             unsigned alphabet_size,
                             unsigned max_len,
                             unsigned w,
                             unsigned q,
                             bool wrap_around = false,
                             bool wrap_around_in = false) : _char_context(first,
                                                                          alphabet_size,
                                                                          w,
                                                                          q,
                                                                          wrap_around,
                                                                          wrap_around_in) {
    assert(max_len <= MAX_STR_LEN && "Sorry! This implementation can not scale to very long strings!");
  }

  std::vector<int> apply(const char *data, unsigned len) {

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

    std::vector<int> res(_char_context.cdim, 0);// init

    for (unsigned t = 0; t < before_bundle.size(); ++t) {
      for (unsigned d = 0; d < before_bundle[t].size(); ++d) {
          if (before_bundle[t][d] > 0) {
            res[d] += before_bundle[t][d];
          }
      }
      for (unsigned d = 0; d < after_bundle[t].size(); ++d) {
          if (after_bundle[t][d] > 0) {
            res[d] += after_bundle[t][d];
          }
        }
    }

    // std::fill(res, res + _feigenbaum.num_sketches, 0);// init
    // for (unsigned t = 0; t < before_bundle.size(); ++t) {
    //   const auto &hash_func_before = _feigenbaum.get_hash_func(t, true, 0);
    //   const auto &hash_func_after = _feigenbaum.get_hash_func(t, false, 0);
    //   for (unsigned k = 0; k < _feigenbaum.num_sketches; ++k) {
    //     for (unsigned d = 0; d < before_bundle[t].size(); ++d) {
    //       if (before_bundle[t][d] > 0) {
    //         auto hf_id = _feigenbaum.get_hash_func_id(k, d);
    //         res[k] += _feigenbaum.eval_hash(&hash_func_before[hf_id], before_bundle[t][d]);
    //       }
    //     }
    //     for (unsigned d = 0; d < after_bundle[t].size(); ++d) {
    //       if (after_bundle[t][d] > 0) {
    //         auto hf_id = _feigenbaum.get_hash_func_id(k, d);
    //         res[k] += _feigenbaum.eval_hash(&hash_func_after[hf_id], after_bundle[t][d]);
    //       }
    //     }
    //   }
    // }
    return res;
  }

 private:
  CharContext _char_context;
};

int main()
{
    std::vector<std::string> raw_data;
    std::vector<std::string> queries;
    File file("enron-utf8-vlen.h5", File::ReadOnly);
    // we get the dataset
    DataSet dataset = file.getDataSet("base");
    dataset.read(raw_data);
    DataSet query = file.getDataSet("query");
    query.read(queries);
    unsigned n = raw_data.size();
    unsigned nq = queries.size();

    // const std::vector<std::string> _raw_data(raw_data);
    // const std::vector<std::string> _queries(queries);
    

    //std::string alphabets = "W0AS 8C7DEXG34JFL2HK6VMIZYB591QPTORU";
    std::string alphabets = "0128REQUST MANGOC59HIYBLPFDKJ7V3W4X6Z";
    //std::string alphabets = "WASCDEGFLHKVMIZYBQPTORU";
    //std::string alphabets = "GACT";
    std::vector<int> _feature_data;
    std::vector<int> _feature_queires;
    


    unsigned w = 12;
    unsigned q = 1;
    unsigned cdim = std::pow(alphabets.size(), q);
    QGramDense gram(alphabets.c_str(), alphabets.size(), q, false);
    CContextsDense ccontext(alphabets.c_str(), alphabets.size(), max_str_len, w, q, false, false);

    
    // timer.restart();
    // {
    // ProgressBar progress(n);
    // for (auto i = 0; i < n;++i) {
    //    ccontext.apply(raw_data[i].c_str(), raw_data[i].size());
    //    ++progress;
    // }
    // }
    // std::cout << "Time to convert original data is: " << timer.elapsed() << " seconds"  << std::endl;

    Timer timer; 

  //   timer.restart();
  //   {
  //   //ProgressBar progress(n);
  //   for (auto i = 0; i < nq;++i) {
  //     //std::cout << i << std::endl;
  //      auto temp = gram.apply(queries[i].c_str(), queries[i].size());
  //      //++progress;
  //   }
  //   }
  //   std::cout << "Time to convert query data is: " << timer.elapsed() << " seconds"  << std::endl;


  // for (auto i = 0; i<raw_data.size();i++) {
  //     // std::cout << "test" << std::endl;
  //     // std::cout << raw_data[1376].c_str() << std::endl;
  //      auto temp = gram.apply(raw_data[i].c_str(), raw_data[i].size());
  //     _feature_data.insert(_feature_data.end(), temp.begin(), temp.end());
  //   }

  // for (auto i = 0; i<queries.size();i++) {
  //      auto temp = gram.apply(queries[i].c_str(), queries[i].size());
  //     _feature_queires.insert(_feature_queires.end(), temp.begin(), temp.end());
  //   }
  

  //   std::ofstream MyFile("feature_data_enron.txt");
  //   for (auto i=0;i<_feature_data.size();++i){
  //     MyFile << _feature_data[i] << " ";
  //   }


  //   std::ofstream MyFile2("feature_queries_enron.txt");
  //   for (auto i=0;i<_feature_queires.size();++i){
  //     MyFile2 << _feature_queires[i] << " ";
  //   }

unsigned k = 100;
unsigned t = 10400;


    //calculating edit distance

    res_pair_raw<unsigned> res{};
    std::vector<int> results;
    std::ifstream fin("result.txt");
    
    
    std::vector<int> q_result;
    int element;
    while (fin >> element)
    {
        q_result.push_back(element);
    }

   

    std::cout << q_result.size() << std::endl;

    timer.restart();
    double time = 0;
for (auto j=0;j<nq;++j){
  unsigned len = queries[j].size();
  res.id = -1;
  res.dist = std::numeric_limits<unsigned>::max();
    for (int i = 0; i < k; ++i) {
      int idx = q_result[i+j*t];
      if (idx == -1) 
      {
          continue;
      }
      else{
      const auto &t_data = raw_data[idx];
      auto d_len = t_data.size();
      auto len_diff = (len > d_len ? (len - d_len) : (d_len - len));
      if (len_diff < res.dist) {
        EdlibAlignResult result = edlibAlign(t_data.c_str(), d_len, queries[j].c_str(), len, edlibDefaultAlignConfig());
        unsigned dist = result.editDistance; 
        if (dist >= 0 && dist < res.dist) {
          res.id = idx;
          res.dist = dist;
        }
      }
      }
    }
    results.push_back(res.dist);
}
    time = timer.elapsed();
    std::cout << "total time used: " << time << "seconds " << std::endl;

    std::ofstream MyFile("result_distance.txt");
    for (auto i=0;i<results.size();++i){
     MyFile << results[i] << std::endl;
    }
    MyFile.close();


    

    return 0;
}