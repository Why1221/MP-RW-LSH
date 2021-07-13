#ifndef __GAUSSIAN_HASH_H__
#define __GAUSSIAN_HASH_H__

#include <cstdint>
#include <ctime>
#include <iterator>
#include <memory>
#include <random>
#include <vector>
#include <limits>
#include <algorithm>

#include <Eigen/Dense>
#include <wyhash32.h>

#include "data_storage.h"
#include "lsh_query_new.h"
#include "multiprobe.h"

// #include <range_sum.h>

  // Generate the hash_vector for one seed and length is universe+1
  // Currently use float type for simplicity
  // Now only calculated even number steps results
  template <class CoordinateType>
  std::vector<CoordinateType> gen_hash(uint_fast64_t seed, int_fast32_t universe) {
  // Generate length equal universe random walk
    std::mt19937_64 mt(seed);
    std::vector<int> tmp_vector;
    std::uniform_int_distribution<int> dist(0,1);
    for (int i=0; i < universe; i++){
      int temp = dist(mt);
      tmp_vector.push_back((temp>0)? 1:-1);
    }
    // Then generate the sum of this random walk
    CoordinateType sum = 0.0;
    std::vector<CoordinateType> tmp_hash_vector;
    tmp_hash_vector.push_back(0);
    int half_universe = (int)(universe/2);
    for (int i =0; i< half_universe;i++) {
      sum += tmp_vector[2*i]*1.0;
      sum += tmp_vector[2*i+1]*1.0;
      tmp_hash_vector.push_back((CoordinateType)sum);
    }
    return tmp_hash_vector;
  }

namespace falconn {
namespace core {
// Set number of hash bits to 21
const int hash_bits = 22;
// template <typename CoordinateType>
// inline int_fast32_t hash_round(CoordinateType coor, int_fast32_t levels){
//   int_fast32_t half_bucket_num = 1 << (levels-1);
//   int_fast32_t bucket_id = static_cast<int_fast32_t>(std::floor(coor)) + half_bucket_num;
//   if (bucket_id < 0){
//     bucket_id = 0;
//   } else if (bucket_id >= 2 * half_bucket_num){
//     bucket_id = 2 * half_bucket_num - 1;
//   }
//   return bucket_id;
// }

// Base class for both the dense and spare gaussian hash classes.
// The derived classes only have to define how to multiply an input vector
// with the gaussian weights. Everything else (next step in the hash computation,
// probing, etc.) is done in the base class.
// template parameters: Derived - the derived class that inherits this base class
// VectorT - type of data points  CoordinateType - type of gaussian weights
// HashT - type of hashed values (id of hash buckets)
template <typename Derived, typename VectorT, typename CoordinateT = float,
          typename HashT = uint32_t>
class GaussianHashBase {

 public:
  typedef VectorT VectorType;
  typedef HashT HashType;
  typedef CoordinateT CoordinateType;
  // vertical vector (x,y,...)T    store column by column
  typedef Eigen::Matrix<CoordinateType, Eigen::Dynamic, 1, Eigen::ColMajor>
      TransformedVectorType;
      // temporarily store transformation data
  typedef void* TransformationTmpData;

  class HashTransformation {
   public:
    HashTransformation(const Derived& parent) : parent_(parent) {}

    // apply y=(Ax+b)/w
    // This is hash transformation without quantization
    void apply(const VectorT& v, TransformedVectorType* result) const {
      parent_.get_multiplied_vector_all_tables(v, result);
    }

    void round(const TransformedVectorType& hash_vec, std::vector<HashType>& res){
      parent_.hash_to_bucket(hash_vec, res);
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

        (*res)[ii] = parent_.compute_hash_single_table(tmp_vector_,parent_.seed_hash2_);
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
  HashType compute_hash_single_table(const TransformedVectorType& v,uint_fast32_t seed) const {
    HashType res = 0;
    std::vector<int> res_rounded(v.size(), 0);
    for (int_fast32_t jj = 0; jj < v.size(); ++jj) {
      res_rounded[jj] =  static_cast<int_fast32_t>(std::floor(v[jj]));
    }
    res = wyhash32(res_rounded.data(), v.size() * sizeof(int), seed);
    res &= (1<<hash_bits)-1;
    return res;
  }

  void add_table() { throw LSHFunctionError("not implemented"); }

 protected:
  GaussianHashBase(int dim, int_fast32_t k, int_fast32_t l, float w, int_fast32_t id_width)
      : dim_(dim), k_(k), l_(l), bucket_width_(w), bucket_id_width_(id_width){
    if (dim_ < 1) {
      throw LSHFunctionError("Dimension must be at least 1.");
    }

    if (k_ < 1) {
      throw LSHFunctionError(
          "Number of hash functions must be"
          "at least 1.");
    }

    // if (k_ * bucket_id_width_ > 8 * static_cast<int_fast32_t>(sizeof(HashType))) {
    //   throw LSHFunctionError(
    //       "More hash functions than supported by the "
    //       "hash type.");
    // }

    if (l_ < 1) {
      throw LSHFunctionError("Number of hash tables must be at least 1.");
    }

  }
public:
  // dimension of data points
  int dim_;
  // number of hash functions per table
  int_fast32_t k_;
  // number of hash tables
  int_fast32_t l_;
  // denumerator w
  float bucket_width_;
  // log2 of number of buckets on each dimension / number of bits of each bucket id for each hash function
  int_fast32_t bucket_id_width_;

};

template <typename CoordinateType = float, typename HashType = uint32_t>
class GaussianHashDense
    : public GaussianHashBase<
          GaussianHashDense<CoordinateType, HashType>,
          Eigen::Matrix<CoordinateType, Eigen::Dynamic, 1, Eigen::ColMajor>,
          CoordinateType, HashType> {
 public:
  uint_fast32_t seed_hash2_;
  typedef Eigen::Matrix<CoordinateType, Eigen::Dynamic, 1, Eigen::ColMajor>
      DerivedVectorT;
  typedef Eigen::Matrix<CoordinateType, Eigen::Dynamic, Eigen::Dynamic,
                        Eigen::ColMajor>
      MatrixType;

  const MatrixType& get_hyperplanes() const { return hyperplanes_; }

  const DerivedVectorT& get_translation() const {return translation_;}

// For convinience we add universe here, though not needed
  GaussianHashDense(int dim, int_fast32_t k, int_fast32_t l, int_fast32_t universe,
                      uint_fast64_t seed, float w, int_fast32_t id_width)
      :GaussianHashBase<GaussianHashDense<CoordinateType, HashType>,
                           DerivedVectorT, CoordinateType, HashType>(dim, k, l, w, id_width), 
      seed_(seed),universe_(universe), gen1_(seed),gen2_(seed), seed_hash2_(seed) {

    std::cauchy_distribution<CoordinateType> cauchy(0.0, 1.0); 
    std::uniform_real_distribution<CoordinateType> uniform_dist(0.0, this->bucket_width_);
// dim: data dimension
    hyperplanes_.resize(this->k_ * this->l_, this->dim_);
    translation_.resize(this->k_ * this->l_);

for (int jj = 0; jj < this->k_ * this->l_; ++jj){
  for (int ii = 0; ii < this->dim_; ++ii) {
        hyperplanes_(jj, ii) = cauchy(gen1_);
      }
    }

    for (int jj = 0; jj < this->k_ * this->l_; ++jj){
      translation_(jj) = uniform_dist(gen2_);
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
        res_rounded[jj] = static_cast<int_fast32_t>(std::floor((*tmp_hash_vector)[ii * this->k_ + jj]));
      }
      res[ii] = wyhash32(res_rounded.data(), this->k_ * sizeof(int), seed_hash2_);
      res[ii] &= (1<<hash_bits)-1;
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
        res_rounded[jj] =  static_cast<int_fast32_t>(std::floor(hash_vec[ii * this->k_ + jj]));
      }
      res[ii] = wyhash32(res_rounded.data(), this->k_ * sizeof(int), seed_hash2_);
      res[ii] &= (1<<hash_bits) - 1;
    }
  }

  // void hash_to_bucket(const DerivedVectorT& hash_vec, std::vector<HashType>& res) const{
  //   if (res.size() != static_cast<size_t>(this->l_)) {
  //     res.resize(this->l_);
  //   }
  //   for (int_fast32_t ii = 0; ii < this->l_; ++ii) {
  //     res[ii] = 0;
  //     for (int_fast32_t jj = 0; jj < this->k_; ++jj) {
  //       res[ii] = res[ii] << this->bucket_id_width_;
  //       res[ii] = res[ii] | hash_round(hash_vec[ii * this->k_ + jj], this->bucket_id_width_);
  //     }
  //   }
  // }

private:
  // Gaussian Matrix A
  MatrixType hyperplanes_;
      // uniform vector b
  DerivedVectorT translation_;
  int_fast32_t universe_;
  std::mt19937_64 gen1_;  // For Matrix A
  std::mt19937_64 gen2_;  // For vector b
  uint_fast64_t seed_;
};

// Hash function implementation for the ToW l1 hash
// CoordianteType probably have to be float
template <typename CoordinateType = float, typename HashType = uint64_t>
class ToWHashDense
    : public GaussianHashBase<
          ToWHashDense<CoordinateType, HashType>,
          Eigen::Matrix<CoordinateType, Eigen::Dynamic, 1, Eigen::ColMajor>,
          CoordinateType, HashType> {
 public:
  uint_fast32_t seed_hash2_;
  typedef Eigen::Matrix<CoordinateType, Eigen::Dynamic, 1, Eigen::ColMajor>
      DerivedVectorT;
  typedef Eigen::Matrix<CoordinateType, Eigen::Dynamic, Eigen::Dynamic,
                        Eigen::ColMajor>
      MatrixType;

  const MatrixType& get_hyperplanes() const { return hyperplanes_; }

  const DerivedVectorT& get_translation() const {return translation_;}

// Still keep universe here, but not use it anymore
  ToWHashDense(int dim, int_fast32_t k, int_fast32_t l, int_fast32_t universe,
                      uint_fast64_t seed, float w, int_fast32_t id_width)
      :GaussianHashBase<ToWHashDense<CoordinateType, HashType>,
                           DerivedVectorT, CoordinateType, HashType>(dim, k, l, w, id_width), seed_(seed),universe_(universe), gen_(seed), seed_hash2_(seed) {

     //std::normal_distribution<CoordinateType> gauss(0.0, 1.0);
    std::uniform_real_distribution<CoordinateType> uniform_dist(0.0, this->bucket_width_); // Real distribution not support int.
// dim: data dimension
// Make hyperplanes_ a (universe/2)+1 columns, k*l*dim matrix
    hyperplanes_.resize(this->universe_/2 +1, this->dim_*this->k_ * this->l_);
    translation_.resize(this->k_ * this->l_);
    std::mt19937 generator(this->seed_);

for (int ii = 0; ii < this->dim_*this->k_ * this->l_; ++ii)  {
      // Generate the hash_vector for each universe
      uint_fast64_t new_seed = generator();
      std::vector<CoordinateType> hash_vector(gen_hash<CoordinateType>(new_seed,this->universe_));

      for (int jj = 0; jj <= this->universe_/2; ++jj) {
        hyperplanes_(jj, ii) = hash_vector[jj];
      }

    }

    for (int jj = 0; jj < this->k_ * this->l_; ++jj){
      translation_(jj) = uniform_dist(gen_);
    }
    }

// For ToW hash, it is (A(x)+b)/w
  void get_multiplied_vector_all_tables(const DerivedVectorT& point,
                                        DerivedVectorT* res) const {
    CoordinateType sum;
    for (int ii = 0; ii<this->k_ * this->l_; ++ii){
      sum = 0.0;
      for (int jj=0; jj < this->dim_; ++jj ) {
        int col = int(point(jj)/2);
        sum += hyperplanes_(col,ii*this->dim_+jj); // The k*l th hash function for current dimension
        }
      // sum is the hash value of k*l th hash function, then plus b and divide w
      (*res)(ii) = (sum+translation_(ii))/ this->bucket_width_;
    }
    }

  // For single table
  // Mark: Current problem.
  void get_multiplied_vector_single_table(const DerivedVectorT& point,
                                          int_fast32_t l,
                                          DerivedVectorT* res) const {
    for (int ii = 0; ii<this->k_ ; ++ii){
      CoordinateType sum = 0.0;
      int test = 0;
      for (int jj=0; jj<this->dim_; ++jj ) {
        int col = int(point(jj)/2);
        sum += hyperplanes_(col,l*this->dim_*this->k_+ii*this->dim_+jj); // The k_*l th hash function for current dimension
        test += col;
      }
      // sum is the hash value of k*l th hash function, then plus b and divide w
      (*res)(ii) = (sum+translation_(ii+l*this->k_))/ this->bucket_width_;
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
        res_rounded[jj] =  static_cast<int_fast32_t>(std::floor((*tmp_hash_vector)[ii * this->k_ + jj]));
      }
      res[ii] = wyhash32(res_rounded.data(), this->k_ * sizeof(int), seed_hash2_);
      res[ii] &= (1<<hash_bits)-1;
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
        res_rounded[jj] =  static_cast<int_fast32_t>(std::floor(hash_vec[ii * this->k_ + jj]));
      }
      res[ii] = wyhash32(res_rounded.data(), this->k_ * sizeof(int), seed_hash2_);
      res[ii] &= (1<<hash_bits)-1;
    }
  }

private:
  // The seed matrix for feigenbaum
  MatrixType hyperplanes_;
      // uniform vector b
  DerivedVectorT translation_;
  std::mt19937_64 gen_;
  uint_fast64_t seed_;
  int_fast32_t universe_;
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
