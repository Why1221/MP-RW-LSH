#ifndef SRSINMEMORY_H_
#define SRSINMEMORY_H_

#include <stdio.h>
#include <vector>
#include <algorithm>

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

#endif