#ifndef _DISTANCE_METRIC_HPP_
#define _DISTANCE_METRIC_HPP_
#define USE_EDLIB

#ifdef USE_EDLIB
/// 3rd party library for calculating edit distance
#include <edlib/edlib.h>
#endif

#include <popcnt.hpp>
#include <functional>
#include <iterator>

/// various distance metrics
namespace detail {
inline unsigned hamming_distance_block8(const char *sa, const char *sb);
inline unsigned hamming_distance_block64(const char *sa, const char *sb);
}// end namespace detail
///
/// \brief hamming distance (for {0,1}^d)
///
/// \param va, vb             Hamming vectors between which the distance is calculated
/// \param dim                dimension of the Hamming vector (note that here dimension is
///                           not the raw dimension in Hamming space, but the one after encoding
///                           i.e., roughly d / 64, where d is the raw dimension.
/// \return                   Hamming distance between #va and $vb
inline unsigned hamming_distance(const uint64_t *va, const uint64_t *vb, unsigned dim);

///
/// \brief hamming distance (for $\Sigma^d$ where $\Sigma$ is the alphabet set)
///
/// Note that the two string can have different lengths, as we consider both strings
/// are from $(\Sigma \cup \{\bot \})^d$ with $d =\infty$. That is, we can cosider each string is with
/// infinite length padding with a special character $\bot$.
///
/// \param sa           string a
/// \param la           length of string a
/// \param sb           string b
/// \param lb           length of string b
/// \return
inline unsigned hamming_distance(const char *sa, size_t la, const char *sb, size_t lb);
///
/// \brief edit distance
///
///
/// \tparam InputIt                  InputIt must meet the requirements of LegacyInputIterator.
///
/// If macro USE_EDLIB is not set, the API will use naive dynamic programming (with space optimization)
/// to calculate the edit distance; otherwise the more efficient method provided by the 3rd party library
/// will be used. From simple measurements, we found that the method implemented by edlib is much faster
/// than the naive dp based method. So unless you have good reason to not use edlib, you are suggested to
/// use the edlib's edit distance calculation.
///
/// \param a_first, a_last           the range for string a
/// \param b_first, b_last           the range for string b
/// \return                          edit distance between string a & b
template<typename InputIt>
inline unsigned int edit_distance(InputIt a_first,
                                  InputIt a_last,
                                  InputIt b_first,
                                  InputIt b_last);
///
/// \brief C-style edit distance API
///
///
/// \tparam CharType              char type
/// \param a                      string a's header
/// \param sa                     string a's length
/// \param b                      string b's header
/// \param sb                     string b's length
/// \return
template<typename CharType>
inline unsigned int edit_distance(const CharType *a,
                                  size_t sa,
                                  const CharType *b,
                                  size_t sb);

inline unsigned hamming_distance(const uint64_t *va, const uint64_t *vb, unsigned dim) {
  unsigned d = 0u;
  for (unsigned i = 0; i < dim; ++i)
    //d += __builtin_popcountll(va[i] ^ vb[i]);
    d += popcount(va[i] ^ vb[i]);
  return d;
}

inline unsigned hamming_distance(const char *sa, size_t la, const char *sb, size_t lb) {
  if (la > lb) return hamming_distance(sb, lb, sa, la);
  unsigned d = 0;
  unsigned offset = 0;
  unsigned rem = la;
  while (rem >= 64) {
    d += detail::hamming_distance_block64(sa + offset, sb + offset);
    rem -= 64;
    offset += 64;
  }
  while (rem >= 8) {
    d += detail::hamming_distance_block8(sa + offset, sb + offset);
    rem -= 8;
    offset += 8;
  }
  for (;offset < la;++ offset) d += (sa[offset] == sb[offset] ? 0 : 1);
  return d + (lb - la);
}
template<typename InputIt>
inline unsigned int edit_distance(InputIt a_first,
                                  InputIt a_last,
                                  InputIt b_first,
                                  InputIt b_last) {
  typedef typename std::iterator_traits<InputIt>::difference_type my_size_t;
  my_size_t sa = std::distance(a_first, a_last);
  my_size_t sb = std::distance(b_first, b_last);
#ifndef USE_EDLIB
  if (sa < sb)
    return edit_distance_dp(b_first, b_last, a_first, a_last);
  if (sb == 0)
    return sa;

  auto *row_c = new unsigned int[sb + 1];
  auto *row_p = new unsigned int[sb + 1];

  size_t k = 0;
  for (; k < sb + 1; ++k)
    row_p[k] = k;
  row_c[0] = 0;

  InputIt b_it, a_it = a_first;
  for (size_t i = 1; i < sa + 1; ++i, ++a_it) {
    b_it = b_first;
    row_c[0] = i;
    for (size_t j = 1; j < sb + 1; ++j, ++b_it) {
      row_c[j] = std::min(std::min(row_c[j - 1], row_p[j]) + 1, row_p[j - 1] + (*a_it == *b_it ? 0 : 1));
    }
    std::copy(row_c, row_c + sb + 1, row_p);
  }

  unsigned int d = row_c[sb];
  delete[] row_c;
  delete[] row_p;

  return d;
#else
  typedef typename std::iterator_traits<InputIt>::value_type CharType;
  EdlibAlignResult result = edlibAlign(&(*a_first), sa, &(*b_first), sb, edlibDefaultAlignConfig());
  unsigned int d = result.editDistance;
  edlibFreeAlignResult(result);
  return d;
#endif

}

template<typename CharType>
inline unsigned int edit_distance(const CharType *a, size_t sa, const CharType *b, size_t sb) {
#ifdef USE_EDLIB
  EdlibAlignResult result = edlibAlign(a, sa, b, sb, edlibDefaultAlignConfig());
  unsigned int d = result.editDistance;
  edlibFreeAlignResult(result);
  return d;
#else
  return edit_distance(a, a + sa, b, b + sb);
#endif
}

template<typename CharType>
inline int edit_distance(const CharType *a, size_t sa, const CharType *b, size_t sb, unsigned k) {
#ifdef USE_EDLIB
  auto config = edlibDefaultAlignConfig();
  config.k = (k < std::max(sa, sb) ? k : (-1));// ignore useless threshold
  EdlibAlignResult result = edlibAlign(a, sa, b, sb, config);
  int d = result.editDistance;
  edlibFreeAlignResult(result);
  return d;
#else
  /// ignore k
  return edit_distance(a, a + sa, b, b + sb);
#endif
}

namespace detail {
inline unsigned hamming_distance_block8(const char *sa, const char *sb) {
  return (sa[0] == sb[0] ? 0 : 1) + (sa[1] == sb[1] ? 0 : 1) + (sa[2] == sb[2] ? 0 : 1) + (sa[3] == sb[3] ? 0 : 1)
      + (sa[4] == sb[4] ? 0 : 1) + (sa[5] == sb[5] ? 0 : 1) + (sa[6] == sb[6] ? 0 : 1) + (sa[7] == sb[7] ? 0 : 1);
}
inline unsigned hamming_distance_block64(const char *sa, const char *sb) {
  return (sa[0] == sb[0] ? 0 : 1) + (sa[1] == sb[1] ? 0 : 1) + (sa[2] == sb[2] ? 0 : 1) + (sa[3] == sb[3] ? 0 : 1)
      + (sa[4] == sb[4] ? 0 : 1) + (sa[5] == sb[5] ? 0 : 1) + (sa[6] == sb[6] ? 0 : 1) + (sa[7] == sb[7] ? 0 : 1)
      + (sa[8] == sb[8] ? 0 : 1) + (sa[9] == sb[9] ? 0 : 1) + (sa[10] == sb[10] ? 0 : 1) + (sa[11] == sb[11] ? 0 : 1)
      + (sa[12] == sb[12] ? 0 : 1) + (sa[13] == sb[13] ? 0 : 1) + (sa[14] == sb[14] ? 0 : 1)
      + (sa[15] == sb[15] ? 0 : 1) + (sa[16] == sb[16] ? 0 : 1) + (sa[17] == sb[17] ? 0 : 1)
      + (sa[18] == sb[18] ? 0 : 1) + (sa[19] == sb[19] ? 0 : 1) + (sa[20] == sb[20] ? 0 : 1)
      + (sa[21] == sb[21] ? 0 : 1) + (sa[22] == sb[22] ? 0 : 1) + (sa[23] == sb[23] ? 0 : 1)
      + (sa[24] == sb[24] ? 0 : 1) + (sa[25] == sb[25] ? 0 : 1) + (sa[26] == sb[26] ? 0 : 1)
      + (sa[27] == sb[27] ? 0 : 1) + (sa[28] == sb[28] ? 0 : 1) + (sa[29] == sb[29] ? 0 : 1)
      + (sa[30] == sb[30] ? 0 : 1) + (sa[31] == sb[31] ? 0 : 1) + (sa[32] == sb[32] ? 0 : 1)
      + (sa[33] == sb[33] ? 0 : 1) + (sa[34] == sb[34] ? 0 : 1) + (sa[35] == sb[35] ? 0 : 1)
      + (sa[36] == sb[36] ? 0 : 1) + (sa[37] == sb[37] ? 0 : 1) + (sa[38] == sb[38] ? 0 : 1)
      + (sa[39] == sb[39] ? 0 : 1) + (sa[40] == sb[40] ? 0 : 1) + (sa[41] == sb[41] ? 0 : 1)
      + (sa[42] == sb[42] ? 0 : 1) + (sa[43] == sb[43] ? 0 : 1) + (sa[44] == sb[44] ? 0 : 1)
      + (sa[45] == sb[45] ? 0 : 1) + (sa[46] == sb[46] ? 0 : 1) + (sa[47] == sb[47] ? 0 : 1)
      + (sa[48] == sb[48] ? 0 : 1) + (sa[49] == sb[49] ? 0 : 1) + (sa[50] == sb[50] ? 0 : 1)
      + (sa[51] == sb[51] ? 0 : 1) + (sa[52] == sb[52] ? 0 : 1) + (sa[53] == sb[53] ? 0 : 1)
      + (sa[54] == sb[54] ? 0 : 1) + (sa[55] == sb[55] ? 0 : 1) + (sa[56] == sb[56] ? 0 : 1)
      + (sa[57] == sb[57] ? 0 : 1) + (sa[58] == sb[58] ? 0 : 1) + (sa[59] == sb[59] ? 0 : 1)
      + (sa[60] == sb[60] ? 0 : 1) + (sa[61] == sb[61] ? 0 : 1) + (sa[62] == sb[62] ? 0 : 1)
      + (sa[63] == sb[63] ? 0 : 1);
}
}// end namespace detail
#endif //_DISTANCE_METRIC_HPP_
