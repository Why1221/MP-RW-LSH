/*
 *   This file is part of SRS project.
 *
 *   SRS is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   SRS is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with SRS. If not, see <http://www.gnu.org/licenses/>.
 *
 *   Created by: Yifang Sun, Jianbin Qin
 *   Last modified by: Yifang Sun, Jianbin Qin
 */

#ifndef _SRS_UTILS_HPP_
#define _SRS_UTILS_HPP_


namespace ss::ann::srs {
template<typename T>
struct res_pair_raw {
  int id;
  T dist;
  bool operator>(const res_pair_raw<T> &) const;
  bool operator>=(const res_pair_raw<T> &) const;
  bool operator==(const res_pair_raw<T> &) const;
  bool operator<=(const res_pair_raw<T> &) const;
  bool operator<(const res_pair_raw<T> &) const;
};

template<typename T>
bool res_pair_raw<T>::operator>(const res_pair_raw<T> &n) const {
  return (dist > n.dist);
}
template<typename T>
bool res_pair_raw<T>::operator>=(const res_pair_raw<T> &n) const {
  return (dist >= n.dist);
}
template<typename T>
bool res_pair_raw<T>::operator==(const res_pair_raw<T> &n) const {
  return (dist == n.dist);
}
template<typename T>
bool res_pair_raw<T>::operator<=(const res_pair_raw<T> &n) const {
  return (dist <= n.dist);
}
template<typename T>
bool res_pair_raw<T>::operator<(const res_pair_raw<T> &n) const {
  return (dist < n.dist);
}

template<>
bool res_pair_raw<long long>::operator>(
    const res_pair_raw<long long> &n) const {
  return (dist > n.dist);
}
template<>
bool res_pair_raw<long long>::operator>=(
    const res_pair_raw<long long> &n) const {
  return (dist >= n.dist);
}
template<>
bool res_pair_raw<long long>::operator==(
    const res_pair_raw<long long> &n) const {
  return (dist == n.dist);
}
template<>
bool res_pair_raw<long long>::operator<=(
    const res_pair_raw<long long> &n) const {
  return (dist <= n.dist);
}
template<>
bool res_pair_raw<long long>::operator<(
    const res_pair_raw<long long> &n) const {
  return (dist < n.dist);
}

template<>
bool res_pair_raw<double>::operator>(const res_pair_raw<double> &n) const {
  return (dist > n.dist);
}
template<>
bool res_pair_raw<double>::operator>=(const res_pair_raw<double> &n) const {
  return (dist >= n.dist);
}
template<>
bool res_pair_raw<double>::operator==(const res_pair_raw<double> &n) const {
  return (dist == n.dist);
}
template<>
bool res_pair_raw<double>::operator<=(const res_pair_raw<double> &n) const {
  return (dist <= n.dist);
}
template<>
bool res_pair_raw<double>::operator<(const res_pair_raw<double> &n) const {
  return (dist < n.dist);
}

template<typename T>
struct type_name {
  static const char *name() {
    return "double";
  }  // fixme
};
template<>
struct type_name<int> {
  static const char *name() {
    return "int";
  }
};
template<>
struct type_name<float> {
  static const char *name() {
    return "float";
  }
};
template<>
struct type_name<double> {
  static const char *name() {
    return "double";
  }
};
template<>
struct type_name<long long> {
  static const char *name() {
    return "long long";
  }
};

template<typename T>
struct type_format {
  static const char *format() {
    return "%s";
  }  // fixme
};
template<>
struct type_format<int> {
  static const char *format() {
    return "%d";
  }
};
template<>
struct type_format<float> {
  static const char *format() {
    return "%f";
  }
};
template<>
struct type_format<double> {
  static const char *format() {
    return "%f";
  }
};
template<>
struct type_format<long long> {
  static const char *format() {
    return "%lld";
  }
};
}// end namespace ss::ann::srs

#endif //_SRS_UTILS_HPP_
