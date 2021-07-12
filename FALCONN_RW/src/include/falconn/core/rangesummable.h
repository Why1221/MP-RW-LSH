#ifndef __RANGE_SUMMABLE_H__
#define __RANGE_SUMMABLE_H__

#include <cstdint>
#include <ctime>
#include <iterator>
#include <memory>
#include <random>
#include <vector>
#include <limits>
#include<algorithm>

#include "Eigen/Dense"

#include "data_storage.h"
#include "lsh_query_new.h"
#include "multiprobe.h"
#include "gaussian_hash.h"
#include "dyatree.h"
#include "wyhash/wyhash32.h"

  // Generate the hash_vector for one seed and length is universe
  // Currently use float type for simplicity
  // The function change for Gaussian

namespace falconn {
namespace core {


template <typename CoordinateType = float, typename HashType = uint32_t>
class RangeSummableGaussian
    : public GaussianHashBase<
          RangeSummableGaussian<CoordinateType, HashType>,
          Eigen::Matrix<CoordinateType, Eigen::Dynamic, 1, Eigen::ColMajor>,
          CoordinateType, HashType> {
 public:
  typedef Eigen::Matrix<CoordinateType, Eigen::Dynamic, 1, Eigen::ColMajor>
      DerivedVectorT;
  typedef Eigen::Matrix<CoordinateType, Eigen::Dynamic, Eigen::Dynamic,
                        Eigen::ColMajor>
      MatrixType;

  const DerivedVectorT& get_translation() const {return translation_;}
// For convinience we add universe here, though not needed
  RangeSummableGaussian(int dim, int_fast32_t k, int_fast32_t l, int_fast32_t universe,
                      uint_fast64_t seed, float w, int_fast32_t id_width)
      :GaussianHashBase<RangeSummableGaussian<CoordinateType, HashType>,
                           DerivedVectorT, CoordinateType, HashType>(dim, k, l, w, id_width), 
      seed_(seed), gen_(seed_),universe_(universe), dya_trees_(l * k * dim), hash_seed_(gen_) {

    std::cauchy_distribution<CoordinateType> cauchy(0.0, 1.0); 
    std::uniform_real_distribution<CoordinateType> uniform_dist(0.0, this->bucket_width_);
// dim: data dimension
    hyperplanes_.resize(this->k_ * this->l_, this->dim_);
    translation_.resize(this->k_ * this->l_);

    for (int jj = 0; jj < this->k_ * this->l_; ++jj){
      translation_(jj) = uniform_dist(gen_);
    }

    for (int ii = 0; ii < this->l_ * this->k_ * this->dim_; ++ii){
      dya_trees_.emplace_back(31, universe, gen_); 
        // L hash tables * K hash functions per table * D input dimenstions
    }
  }
// (Ax+b)/w
  void get_multiplied_vector_all_tables(const DerivedVectorT& point,
                                        DerivedVectorT* res) const {
    int dst_idx = 0, re_idx = 0;
    for (int li = 0; li < this->l_; ++li){
      for (int ki = 0; ki < this->k_; ++ki){
        float result = 0.f;
        for (int dj = 0; dj < this->dim_; ++dj){
          unsigned idx = (unsigned) point[dj];
          result += dya_trees_[dst_idx++];
        }
        (*res)[re_idx++] = result;
      }
    }
  }

  void get_multiplied_vector_single_table(const DerivedVectorT& point,
                                          int_fast32_t l,
                                          DerivedVectorT* res) const {
    int dst_idx = l*this->k_*this->dim_;
    for (int ki = 0; ki < this->k_; ++ki){
      float result = 0.f;
      for (int dj = 0; dj < this->dim_; ++dj){
        unsigned idx = (unsigned) point[dj];
        result += dya_trees_[dst_idx++];
      }
      (*res)[ki] = result;
    }
  }

  void hash(const DerivedVectorT& point, std::vector<HashType>* result,
            DerivedVectorT* tmp_hash_vector = nullptr) const {
    bool allocated = false;
    if (tmp_hash_vector == nullptr) {
      allocated = true;
      tmp_hash_vector = new DerivedVectorT(this->k_ * this->l_);
    }

    get_multiplied_vector_all_tables(point, tmp_hash_vector);

    std::vector<HashType>& res = *result;
    std::vector<int> res_rounded(this->k_, 0);
    if (res.size() != static_cast<size_t>(this->l_)) {
      res.resize(this->l_);
    }
    for (int_fast32_t ii = 0; ii < this->l_; ++ii) {
      for (int_fast32_t jj = 0; jj < this->k_; ++jj) {
        res_rounded[jj] =  hash_round((*tmp_hash_vector)[ii * this->k_ + jj], this->bucket_id_width_);
      }
      res[ii] = wyhash32(res_rounded.data(), this->k_ * sizeof(int), hash_seed_);
    }

    if (allocated) {
      delete tmp_hash_vector;
    }
  }

  void hash_to_bucket(const DerivedVectorT& hash_vec, std::vector<HashType>& res) const{
    if (res.size() != static_cast<size_t>(this->l_)) {
      res.resize(this->l_);
    }
    std::vector<int> res_rounded(this->k_, 0);
    for (int_fast32_t ii = 0; ii < this->l_; ++ii) {
      for (int_fast32_t jj = 0; jj < this->k_; ++jj) {
        res_rounded[jj] =  hash_round((*hash_vec)[ii * this->k_ + jj], this->bucket_id_width_);
      }
      res[ii] = wyhash32(res_rounded.data(), this->k_ * sizeof(int), hash_seed_);
    }
  }

public:
  uint_fast32_t hash_seed_; // multiprobe also needs this

private:
  // ToW hashed Matrix A
  std::vector<DyaSimTree<unsigned>> dya_trees_;
      // uniform vector b
  DerivedVectorT translation_;
  std::mt19937_64 gen_; // Put seed inside? Probably?
  uint_fast64_t seed_;
  int_fast32_t universe_;
};

     

}  // namespace core
}  // namespace falconn

#endif
