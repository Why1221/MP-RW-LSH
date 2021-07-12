
#ifndef _POPCNT_HPP_
#define _POPCNT_HPP_

#if defined(_MSC_VER)
#include <intrin.h>// for __popcnt64
#elif defined(__clang__)
#include <popcntintrin.h> // for __builtin_popcountll
#endif

#include <cstdint> // include this header for uint64_t
#include "stdlib.h"

// Counting bits set, Brian Kernighan's way
// from https://graphics.stanford.edu/~seander/bithacks.html#ConditionalSetOrClearBitsWithoutBranching
inline int __popcount_bk(uint64_t v) {
  int c; // c accumulates the total bits set in v
  for (c = 0; v; c++) {
    v &= v - 1; // clear the least significant bit set
  }
  return c;
}
inline int popcount(uint64_t v) {
//  Visual Studio       _MSC_VER
//  gcc                 __GNUC__
//  clang               __clang__
//  emscripten          __EMSCRIPTEN__ (for asm.js and webassembly)
//  MinGW 32            __MINGW32__
//  MinGW-w64 32bit     __MINGW32__
//  MinGW-w64 64bit     __MINGW64__
#if defined(_MSC_VER)
#if defined(_M_X64 )
  return __popcnt64(v);
#else
  unsigned h32b = (v >> 32u) & 0xFFFFFFFFu;
  unsigned l32b = (v & 0xFFFFFFFFu);
  return __popcnt(h32b) + __popcnt(l32b);
#endif
#elif defined(__clang__) || defined(__GNUC__)
  return __builtin_popcountll(v);
#else
  return __popcount_bk(v);
#endif
}
#endif //_POPCNT_HPP_
