
#ifndef _IO_HPP_
#define _IO_HPP_

#include <string>
#include <vector>
#include <array>
#include <iterator>
#include <map>
#include <unordered_map>

#ifdef USE_FMTLIB
#include <fmt/format.h>
#endif

namespace ss::io {

namespace detail {
template<typename Scalar>
constexpr std::string_view default_format() {
#ifdef USE_FMTLIB
  return "{}";
#else
  return "";
#endif
}

template<typename T>
struct StringtifyScalarOrStringImpl {
  static_assert(std::is_scalar_v<T> || std::is_same_v<std::remove_cv_t <T>, std::string>);
  static std::string apply(const T &s, const char *format) {
#ifdef USE_FMTLIB
    if (format == nullptr) return fmt::format(default_format<T>(), s);
    else return fmt::format(format, s);
#else
    return std::to_string(s);
#endif
  }
};

template<typename FType, typename SType>
struct StringtifyPairImpl {
  static std::string apply(const std::pair<FType, SType> &p, const char *first_format, const char *second_format) {
    std::string s("(");
    s += StringtifyScalarOrStringImpl<FType>::apply(p.first, first_format);
    s += ", " + StringtifyScalarOrStringImpl<SType>::apply(p.second, second_format) + ")";
    return s;
  }
};

template<typename ForwardIt, bool is_map>
struct StringtifyIteratorImpl;

template<typename ForwardIt>
struct StringtifyIteratorImpl<ForwardIt, false> {
  static std::string apply(ForwardIt first, ForwardIt last, const char *format) {
    typedef typename std::iterator_traits<ForwardIt>::value_type T;
    std::string s = "[";
    for (auto it = first; it != last; ++it) {
      s += StringtifyScalarOrStringImpl<T>::apply(*it, format) + ", ";
    }
    s[s.size() - 2] = ']';
    s.pop_back();
    return s;
  }
};

template<typename ForwardIt>
struct StringtifyIteratorImpl<ForwardIt, true> {
  static std::string apply(ForwardIt first, ForwardIt last, const char *key_format, const char *value_format) {
    typedef typename std::iterator_traits<ForwardIt>::value_type T;
    typedef typename T::first_type Key;
    typedef typename T::second_type Value;

    std::string s = "{";
    for (auto it = first; it != last; ++it) {
      s += StringtifyScalarOrStringImpl<Key>::apply(it->first, key_format) + ": ";
      if constexpr (std::is_scalar_v<Value> || std::is_same_v<Value, std::string>) s += StringtifyScalarOrStringImpl<
            Value>::apply(it->second, value_format);
      else s += StringtifyIteratorImpl<typename Value::const_iterator, false>::apply(it->second.cbegin(),
                                                                                     it->second.cend(),
                                                                                     value_format);
      s += ", ";
    }
    s[s.size() - 2] = '}';
    s.pop_back();
    return s;
  }
};

}// end namespace detail

template<typename Scalar>
constexpr std::string_view default_format() {
#ifdef USE_FMTLIB
  return "{}";
#else
  return "";
#endif
}

template<typename Scalar>
std::string stringtify(Scalar s, const char *format = nullptr) {
  static_assert(std::is_scalar_v<Scalar> || std::is_same_v<std::string, std::remove_cv_t <Scalar>>);
#ifdef USE_FMTLIB
  if (format == nullptr) return fmt::format(default_format<Scalar>(), s);
  else return fmt::format(format, s);
#else
  return std::to_string(s);
#endif
}

template<typename ForwardIt>
std::string stringtify(ForwardIt first, ForwardIt last, const char *scalar_format = nullptr) {
  // static_assert(std::is_scalar_v<typename std::iterator_traits<ForwardIt>::value_type> );
  std::string s = "[";
  for (auto it = first; it != last; ++it) {
    s += stringtify(*it, scalar_format) + ", ";
  }
  s[s.size() - 2] = ']';
  s.pop_back();
  return s;
}

template<typename Key, typename Value>
std::string stringtify(const std::map<Key, Value> &o_map,
                       const char *key_format = nullptr,
                       const char *value_scalar_format = nullptr) {
  typedef typename std::map<Key, Value>::const_iterator CIt;
  return detail::StringtifyIteratorImpl<CIt, true>::apply(o_map.cbegin(), o_map.cend(), key_format, value_scalar_format);
}

template<typename Key, typename Value>
std::string stringtify(const std::unordered_map<Key, Value> &unordered_map,
                       const char *key_format = nullptr,
                       const char *value_scalar_format = nullptr,
                       bool sort = true) {
  typedef typename std::unordered_map<Key, Value>::const_iterator CIt;
  if (sort) {
    std::map<Key, Value> o_map(unordered_map.cbegin(), unordered_map.cend());
    return stringtify<Key, Value>(o_map, key_format, value_scalar_format);
  } else {
    return detail::StringtifyIteratorImpl<CIt, true>::apply(unordered_map.cbegin(), unordered_map.cend(), key_format, value_scalar_format);
  }
}

}
#endif // _IO_HPP_
