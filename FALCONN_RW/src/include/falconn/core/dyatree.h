#ifndef DYATREE_H
#define DYATREE_H
// #define DEBUG

#include <random>
#include <string>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <vector>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <limits>
#include <numeric>
#include "wyhash/wyhash32.h"
//#include <boost/align/aligned_allocator.hpp>

#include <chrono>


namespace falconn{
namespace core{

template <typename UniverseT = unsigned>
class DyaSimTree{
private:
    const unsigned _level;
    const UniverseT _universe;
    std::vector<unsigned> _seedl, _seedr;
    const unsigned _seed_root, _seed_w;
    static constexpr float sqrt2 = 1.4142135623730951;
    const float _denom;

public:
// minhash should be in [0, 1], so 2.0 is always large enough
    DyaSimTree(unsigned level, UniverseT universe, std::mt19937& rng): 
        _level(level), _universe(universe), _seedl(level+1), _seedr(level+1), _seed_root(rng()), 
        _seed_w(rng()), _denom(std::powf(2.f, -(float)_level/2.0f))
    {
        assert(!std::numeric_limits<UniverseT>::is_signed); // only support unsigned type
        assert(level == 8 * sizeof(UniverseT) || (level < 8 * sizeof(UniverseT) && universe <= (1u << level)));

        for (int idx = 0; idx <= _level; ++idx){
            _seedl[idx] = rng();
            _seedr[idx] = rng();
        }

    }

    float range_sum(UniverseT x){
        // if (start < 0 || end >= _max || start > end) return false;
        if (x > _universe) return -1.f;
        float val = wy2gau(_seed_root) * (float)x * _denom; // root level
        return val + rs_recur(x, _seed_w, _level, _denom);
    }

private:
    // input interval [start, end], outside interval size 1<<level, 
    float rs_recur(UniverseT x, unsigned hash, unsigned level, float denom){
        if (x == 0 || level == 0) return 0;

        float val;

        unsigned n = 1u<<(level-1);
        if (x>n){
            x = 2 * n - x;
            val = wy2gau(hash) * (float)x * denom;
            val += rs_recur(x, wyhash32(&hash, sizeof(unsigned), _seedr[level]), level-1, denom * sqrt2);
        } else {
            val = wy2gau(hash) * (float)x * denom;
            val += rs_recur(x, wyhash32(&hash, sizeof(unsigned), _seedl[level]), level-1, denom * sqrt2);
        }
        return val;
    }
};
}
} // namespace 
#endif 
