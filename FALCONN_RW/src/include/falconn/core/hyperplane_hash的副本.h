#ifndef __GAUSSIAN_HASH_H__
#define __GAUSSIAN_HASH_H__

#include <cstdint>
#include <ctime>
#include <iterator>
#include <memory>
#include <random>
#include <vector>
#include <limits>

#include <Eigen/Dense>

#include "data_storage.h"
#include "heap.h"
#include "lsh_function_helpers.h"

namespace falconn {
namespace core {

template <typename CoordinateType>
inline int_fast32_t hash_round(CoordinateType coor, int_fast32_t levels){
  int_fast32_t half_bucket_num = 1 << (levels-1);
  int_fast32_t bucket_id = static_cast<int_fast32_t>(std::floor(coor)) + half_bucket_num;
  if (bucket_id < 0){
    bucket_id = 0;
  } else if (bucket_id >= 2 * half_bucket_num){
    bucket_id = 2 * half_bucket_num - 1;
  }
  return bucket_id;
}

// Base class for both the dense and spare gaussian hash classes.
// The derived classes only have to define how to multiply an input vector
// with the gaussian weights. Everything else (next step in the hash computation,
// probing, etc.) is done in the base class.
// template parameters: Derived - the derived class that inherits this base class
// VectorT - type of data points  CoordinateType - type of gaussian weights
// HashT - type of hashed values (id of hash buckets)
template <typename Derived, typename VectorT, typename CoordinateType = float,
          typename HashT = uint32_t>
class GaussianHashBase {
 private:
  class MultiProbeLookup;

 public:
  typedef VectorT VectorType;
  typedef HashT HashType;
  // vertical vector (x,y,...)T    store column by column
  typedef Eigen::Matrix<CoordinateType, Eigen::Dynamic, 1, Eigen::ColMajor>
      TransformedVectorType;
      // temporarily store transformation data
  typedef void* TransformationTmpData;
  typedef Eigen::Matrix<CoordinateType, Eigen::Dynamic, Eigen::Dynamic,
                        Eigen::ColMajor>
      MatrixType;

  // defined in lsh_function_helpers.h
  typedef HashObjectQuery<Derived> Query;

  class HashTransformation {
   public:
    HashTransformation(const Derived& parent) : parent_(parent) {}

    // apply y=(Ax+b)/w
    // This is hash transformation without quantization
    void apply(const VectorT& v, TransformedVectorType* result) const {
      parent_.get_multiplied_vector_all_tables(v, result);
    }

   private:
   // reference to the hash transformation
    const Derived& parent_;
  };

  // TODO: specialize template for faster batch hyperplane setup (if the batch
  // vector type is also an Eigen matrix, we can just do a single matrix-matrix
  // multiplication.)
  template <typename BatchVectorType>
  class BatchHash {
   public:
    BatchHash(const Derived& parent)
        : parent_(parent), tmp_vector_(parent.get_k()) {}

    // hash a set of points, using the hash functions of the l-th table
    void batch_hash_single_table(const BatchVectorType& points, int_fast32_t l,
                                 std::vector<HashType>* res) {
      int_fast64_t nn = points.size();
      if (static_cast<int_fast64_t>(res->size()) != nn) {
        res->resize(nn);
      }

      typename BatchVectorType::FullSequenceIterator iter =
          points.get_full_sequence();
      for (int_fast64_t ii = 0; ii < nn; ++ii) {
        parent_.get_multiplied_vector_single_table(iter.get_point(), l,
                                                   &tmp_vector_);
        (*res)[ii] = compute_hash_single_table(tmp_vector_);
        ++iter;
      }
    }

   private:
    const Derived& parent_;
    TransformedVectorType tmp_vector_;
  };

  void reserve_transformed_vector_memory(TransformedVectorType* tv) const {
    tv->resize(k_ * l_);
  }

  int_fast32_t get_l() const { return l_; }

  int_fast32_t get_k() const { return k_; }

  float get_bucket_width() const {return bucket_width_;}

  int_fast32_t get_id_width() const {return bucket_id_width_;}

  // encode multiple bits to a binary number
  static HashType compute_hash_single_table(const TransformedVectorType& v) {
    HashType res = 0;
    for (int_fast32_t jj = 0; jj < v.size(); ++jj) {
      res = res << bucket_id_width_;
      res = res | hash_round(v(jj), bucket_id_width_);
    }
    return res;
  }

  

  void add_table() { throw LSHFunctionError("not implemented"); }

 protected:
  GaussianHashBase(int dim, int_fast32_t k, int_fast32_t l, float w, int_fast_32_t id_width,
                     uint_fast64_t seed)
      : dim_(dim), k_(k), l_(l), seed_(seed), gen_(seed), bucket_width_(w), bucket_id_width_(id_width){
    if (dim_ < 1) {
      throw LSHFunctionError("Dimension must be at least 1.");
    }

    if (k_ < 1) {
      throw LSHFunctionError(
          "Number of hash functions must be"
          "at least 1.");
    }

    if (k_ * bucket_id_width_ > 8 * static_cast<int_fast32_t>(sizeof(HashType))) {
      throw LSHFunctionError(
          "More hash functions than supported by the "
          "hash type.");
    }

    if (l_ < 1) {
      throw LSHFunctionError("Number of hash tables must be at least 1.");
    }
  }

  // dimension of data points
  int dim_;
  // number of hash functions per table
  int_fast32_t k_;
  // number of hash tables
  int_fast32_t l_;
  uint_fast64_t seed_;
  
  // Gaussian matrix A
  Eigen::Matrix<CoordinateType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
      hyperplanes_;
      // uniform vector b
  TransformedVectorType translation_;
  // denumerator w
  float bucket_width_;
  // log2 of number of buckets on each dimension / number of bits of each bucket id for each hash function
  int_fast32_t bucket_id_width_;


   private:



 private:
  friend Query;
  // Helper class for multiprobe LSH
  
  // round coor to an integer bucket id with 2^levels levels
};

// Hash function implementation for the dense hyperplane hash.
// The only functionality that has to be implemented here is the mapping from
// an input point / vector to the result of multiplying with the hyperplanes
// (i.e., generating the "tansformed vector").
template <typename CoordinateType = float, typename HashType = uint32_t>
class GaussianHashDense
    : public GaussianHashBase<
          GaussianHashDense<CoordinateType, HashType>,
          Eigen::Matrix<CoordinateType, Eigen::Dynamic, 1, Eigen::ColMajor>,
          CoordinateType, HashType> {
 public:
  typedef Eigen::Matrix<CoordinateType, Eigen::Dynamic, 1, Eigen::ColMajor>
      DerivedVectorT;

  const MatrixType& get_hyperplanes() const { return hyperplanes_; }

  const DerivedVectorT& get_translation() const {return translation_;}

  GaussianHashDense(int dim, int_fast32_t k, int_fast32_t l,
                      uint_fast64_t seed, float w, int_fast_32_t id_width)
      :GaussianHashBase<GaussianHashDense<CoordinateType, HashType>,
                           DerivedVectorT, CoordinateType, HashType>(dim, k, l, w, id_width,seed) {

    std::normal_distribution<CoordinateType> gauss(0.0, 1.0);
    std::uniform_real_distribution<CoordinateType> uniform_dist(0.0, bucket_width_);
// dim: data dimension
    hyperplanes_.resize(this->k_ * this->l_, this->dim_);
    translation_.resize(this->k_ * this->l_);

    for (int ii = 0; ii < this->dim_; ++ii) {
      for (int jj = 0; jj < this->k_ * this->l_; ++jj) {
        hyperplanes_(jj, ii) = gauss(gen_);
      }
    }

    for (int jj = 0; jj < k_ * l_; ++jj){
      translation_(jj) = uniform_dist(gen_);
    }
  }
// (Ax+b)/w
  void get_multiplied_vector_all_tables(const DerivedVectorT& point,
                                        DerivedVectorT* res) const {
    *res = (hyperplanes_ * point + translation_) / this->bucket_width_;
  }

  void get_multiplied_vector_single_table(const DerivedVectorT& point,
                                          int_fast32_t l,
                                          DerivedVectorT* res) const {
    // TODO: check whether middleRows is as fast as building a memory map
    // manually.
    *res = (hyperplanes_.middleRows(l * this->k_, this->k_) * point + 
       translation_.middleRows(l * this->k_, this->k_)) / this->bucket_width_;
  }

  void hash(const VectorType& point, std::vector<HashType>* result,
            DerivedVectorT* tmp_hash_vector = nullptr) const {
    bool allocated = false;
    if (tmp_hash_vector == nullptr) {
      allocated = true;
      tmp_hash_vector = new DerivedVectorT(k_ * l_);
    }

    get_multiplied_vector_all_tables(point, tmp_hash_vector);

    std::vector<HashType>& res = *result;
    if (res.size() != static_cast<size_t>(l_)) {
      res.resize(l_);
    }
    for (int_fast32_t ii = 0; ii < this->l_; ++ii) {
      res[ii] = 0;
      for (int_fast32_t jj = 0; jj < this->k_; ++jj) {
        res[ii] = res[ii] << this->bucket_id_width_;
        res[ii] = res[ii] | hash_round((*tmp_hash_vector)[ii * this->k_ + jj], this->bucket_id_width_);
      }
    }

    if (allocated) {
      delete tmp_hash_vector;
    }
  }

private:
  Eigen::Matrix<CoordinateType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
      hyperplanes_;
      // uniform vector b
  DerivedVectorT translation_;
  std::mt19937_64 gen_;
};

// Hash function implementation for the sparse hyperplane hash.
// The only functionality that has to be implemented here is the mapping from
// an input point / vector to the result of multiplying with the hyperplanes
// (i.e., generating the "tansformed vector").
/* not implemented 
template <typename CoordinateType = float, typename HashType = uint32_t,
          typename IndexType = int32_t>
class HyperplaneHashSparse
    : public HyperplaneHashBase<
          HyperplaneHashSparse<CoordinateType, HashType, IndexType>,
          std::vector<std::pair<IndexType, CoordinateType>>, CoordinateType,
          HashType> {
 public:
  typedef std::vector<std::pair<IndexType, CoordinateType>> DerivedVectorT;
  typedef Eigen::Matrix<CoordinateType, Eigen::Dynamic, 1, Eigen::ColMajor>
      DerivedTransformedVectorT;

  HyperplaneHashSparse(IndexType dim, int_fast32_t k, int_fast32_t l,
                       uint_fast64_t seed)
      : HyperplaneHashBase<
            HyperplaneHashSparse<CoordinateType, HashType, IndexType>,
            DerivedVectorT, CoordinateType, HashType>(dim, k, l, seed) {}

  void get_multiplied_vector_all_tables(const DerivedVectorT& point,
                                        DerivedTransformedVectorT* res) const {
    // TODO: would row-major be a better storage order for sparse vectors?
    res->setZero();
    for (IndexType ii = 0; ii < static_cast<IndexType>(point.size()); ++ii) {
      *res += point[ii].second * this->hyperplanes_.col(point[ii].first);
    }
  }

  void get_multiplied_vector_single_table(
      const DerivedVectorT& point, int_fast32_t l,
      DerivedTransformedVectorT* res) const {
    res->setZero();
    for (IndexType ii = 0; ii < static_cast<IndexType>(point.size()); ++ii) {
      *res += point[ii].second *
              this->hyperplanes_.col(point[ii].first)
                  .segment(l * this->k_, this->k_);
    }
  }
};*/

}  // namespace core
}  // namespace falconn

#endif
