
#ifndef _SKETCH_SP_HPP_
#define _SKETCH_SP_HPP_

#include <vector>
#include <bitset>

namespace ss::ann {

namespace string_sketch {

namespace detail {
inline int16_t random_walk_block64(const char *s, const uint64_t *hash, unsigned wi, const int *dict, unsigned enc_dim);
}// end namespace detail

struct BitSamplingHash {
  BitSamplingHash(unsigned num_hashes,
               unsigned num_sample_bits,
               unsigned dim,
               unsigned seed) : _m(num_hashes),
                                _dim(dim),
                                _n_samples(num_sample_bits),
                                _seed(seed) {
    _generate_hash();
  }

  void apply(const char * str, unsigned len, std::string* res) const {
     for(unsigned k = 0;k < _m;++ k) {
       res[k].reserve(_n_samples);
       unsigned idx_s = k * _n_samples;
       for(unsigned i = 0;i < _n_samples;++ k) res[k].push_back(str[_hash[idx_s + i]]);
     }
  }

  std::vector<std::string> apply(const std::string& str) const {
    std::vector<std::string> res(_m);
    apply(str.c_str(), str.size(), &res[0]);
  }

 private:
  void _generate_hash() {
    std::mt19937_64 eng(_seed);
    std::uniform_int_distribution<> dist(0, _dim);
    auto rng = std::bind(dist, eng);

    _hash.resize(_m * _n_samples);

    for (unsigned k = 0; k < _m; ++k)
      for (unsigned i = 0; i < _n_samples; ++i)
        _hash[k * _n_samples + i] = rng();

  }
  unsigned _m;
  unsigned _dim;
  unsigned _n_samples;
  unsigned _seed;
  std::vector<unsigned> _hash;
};

typedef uint64_t my_word_t;
static constexpr unsigned WORD_SIZE = (sizeof(my_word_t) << 3u);


///
/// \brief TugOfWarSketchHamming
///
///
///
struct TugOfWarSketchHamming {
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
    for (auto &d : _dict) d = -1;
    for (unsigned i = 0; i < alphabet_size; ++i) _dict[alphabets[i]] = i;
    _generate_random_vectors();
  }

  void apply(const std::string *data, int16_t *res) const {
    for (unsigned k = 0; k < _m; ++k) {
      res[k] = 0;
      for (unsigned b = 0; b < _nb; ++b) {
           const uint64_t *hash_k = &_hash[k * _nb * _alphabet_size  * _enc_dim_eb + b * _alphabet_size * _enc_dim_eb];
           unsigned len = data[b].size();
          if(len % WORD_SIZE == 0) res[k] += _apply_each_no_rem(data[b].c_str(), len, hash_k);
          else res[k] += _apply_each_rem(data[b].c_str(), len, hash_k);
      }
    }
  }

 private:
  int16_t _apply_each_no_rem(const char * s, unsigned s_len, const uint64_t* hash_k) const {
    int16_t loc = 0;
    unsigned wi = 0;
    int len = s_len;
    while (len >= 64) {
      loc += detail::random_walk_block64(s + len, hash_k, wi, &_dict[0], _enc_dim_eb);
      len -= 64;
      ++ wi;
    }
    int rem = (_db - s_len);
    if (rem > 0) loc += _handle_padding(hash_k, wi, rem);
    return loc;
  }
  int16_t _apply_each_rem(const char * s, unsigned s_len, const uint64_t* hash_k) const {
    int16_t loc = 0;
    unsigned wi = 0;
    int len = s_len;
    while (len >= 64) {
      loc += detail::random_walk_block64(s + len, hash_k, wi, &_dict[0], _enc_dim_eb);
      len -= 64;
      ++ wi;
    }
    uint64_t mask = 1ull;
    int align64 = 64 - len;
    while (len >= 0) {
      loc += (hash_k[_dict[*(s + len)] * _enc_dim_eb + wi] & mask) == 0 ? (-1) : 1;
      -- len;
      mask <<= 1u;
    }
    int rem = (_db - s_len);
    for(unsigned i = 0;i < std::min(align64, rem);++ i, --rem) {
      loc += (hash_k[(_alphabet_size - 1) * _enc_dim_eb + wi] & mask) == 0 ? (-1) : 1;
      mask <<= 1u;
    }
    if (rem > 0) loc += _handle_padding(hash_k, wi + 1, rem);
    return loc;
  }
  int16_t _handle_padding(const uint64_t *hash_k, unsigned wi, int rem) const {
    int16_t loc = 0;
    while(rem >= 64) {
      loc += 2 * __builtin_popcount(hash_k[(_alphabet_size - 1) * _enc_dim_eb + wi]) - 64;
      rem -= 64;
      ++ wi;
    }
    if (rem > 0) loc += 2 * __builtin_popcount(hash_k[(_alphabet_size - 1) * _enc_dim_eb + wi]) - rem;
    return loc;
  }
  void _generate_random_vectors() {
    std::mt19937_64 eng(_seed);
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
            bits.set(c);
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
  unsigned _m;
  unsigned _db;
  unsigned _nb;
  unsigned _enc_dim_eb;
  unsigned _alphabet_size;
  unsigned _seed;
  std::array<int, 256> _dict;
  std::vector<uint64_t> _hash;
};

namespace detail {
inline int16_t random_walk_block64(const char *s,
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
}// end namespace string_sketch

}// end namespace ss:ann

#endif //_SKETCH_SP_HPP_
