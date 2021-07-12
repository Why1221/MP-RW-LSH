
#ifndef _ANN_SP_HPP_
#define _ANN_SP_HPP_

#include <vector>
#include <bitset>
#include <chrono>
#include <random>

#include <boost/timer.hpp> // timer
#include <boost/progress.hpp>// progress bar

#include <spdlog/spdlog.h>// logging
#include <srs/SRSCoverTree.h>
#include <srs_utils.hpp>

#include <nanoflann/KDTreeVectorOfVectorsAdaptor.h>
#include <nanoflann/KDTreeFlatVectorAdaptor.h>

#include <io.hpp>
#include <distance_metric_sp.hpp>
#include <sketch_sp.hpp>

using namespace ss::ann::srs;

/// Timer and ProgressBar
/// If you want to avoid the use of boost library
/// You can simple change the following two lines to
/// the three directly after them that are commented
/// out (or you can create your own ones, but please
/// make sure they have at least the APIs provided by
/// the ss::gadget::SimpleTimer and ss::gadget::SimpleProgressDisplay,
/// respectively.
typedef boost::timer Timer;
typedef boost::progress_display ProgressBar;
// #include <gadget.hpp>
// typedef ss::gadget::SimpleProgressDisplay ProgressBar;
// typedef ss::gadget::SimpleTimer Timer;


namespace ss::ann {
/// flat vector version ANN schemes
/// As 2d vectors (vector of vectors) have very high overhead, especially when
/// the data dimension is not very high!!!
namespace flat_vector {
  /// flat vector representation of points in {0, 1}^d
  typedef std::vector<uint64_t > FlatVectorHamming64;
  ///
  /// \brief SRSCoverTreeHamming
  ///
  struct SRSCoverTreeHamming {
    ///
    /// \brief Constructor
    ///
    ///  construct an SRSCoverTreeHamming object
    ///
    /// \param raw_data                raw data (representing in flat vectors)
    /// \param enc_dim                 "dimension" after encoding (i.e., how many 64-bits, or the original dimension in
    ///                                 Hamming space is (roughly) word_size * enc_dim. For ease of implementation, we
    ///                                 assume that the word size is 64. The implementation can be easily extended to other
    ///                                 word size.
    ///
    ///                                 Remarks. For the cases where the dimension is not exact a multiple of the word size
    ///                                 the last few bits (less than a word size) is also viewed as a unsigned integer with
    ///                                 word-size bits (i.e., adding certain number of leading 0's to make it a word).
    ///
    /// \param projected_dim           dimension after projection (i.e., # of sketches for each data point)
    /// \param index_path              name for the path to storing the index (to avoid rebuilding index)
    /// \param seed                    random seed (optional). We add it to make the results reproducible.
    SRSCoverTreeHamming(const FlatVectorHamming64 &raw_data,
                        unsigned enc_dim,
                        unsigned projected_dim,
                        std::string index_path,
                        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count()) :
        _raw_data_ref(raw_data),
        _n(raw_data.size() / enc_dim),
        _d(enc_dim),
        _org_dim(_d * WORD_SIZE),
        _m(projected_dim),
        _seed(seed),
        _sketch(nullptr),
        _index(nullptr),
        _index_path(std::move(index_path)),
        _index_file_path(""),
        _para_file_path("") {
      /// make sure the "size" of the raw data is correct
      assert(raw_data.size() == enc_dim * _n);
      if (_index_path.back() != '/')
        _index_path.push_back('/');
      _index_file_path = _index_path + "index_srs_cover";
      _para_file_path = _index_path + "para_srs_cover.txt";
    }
    ///
    /// \brief build_index
    ///
    /// The index is built as follows. First, the raw data is projected to a lower dimensional space using
    /// the L2-stable distribution (i.e., normal distribution) based sketches (which we call the Gaussian
    /// sketch). Then we use cover tree to build index for the lower dimensional projecting data.
    ///
    /// We isolate this API mainly for the ease of users to measure the index construction time
    ///
    void build_index() {
      /// create sketch
      _sketch = std::make_unique<sketch::GaussianSketchHamming>(_m, _org_dim, _seed);

      auto *proj_data = new float[_n * _m];

      spdlog::info("Start to project the raw data onto space with lower dimension ...");
      Timer timer;
      timer.restart();
      {
        ProgressBar progress(_n);
        /// projecting raw data to lower dimensional space
        for (auto i = 0; i < _n; ++i) {
          /// project the ith data point
          _sketch->apply(&_raw_data_ref[i * _d], &proj_data[i * _m]);
          ++progress;/// update progress
        }
      }
      spdlog::info("Projecting finished! It takes {} seconds", timer.elapsed());
      /// build cover tree
      _index = std::make_unique<SRS_Cover_Tree>(_n, _m, new Proj_data(_n, _m, proj_data));
      /// write tree to file
      _index->write_to_disk_compressed(_index_file_path.c_str());
    }
    ///
    /// \brief 1nn query
    ///
    /// \param q     header pointer for the query point
    /// \param t     number of candidates to be checked (i.e., how many nearest neighbors need to be
    ///              checked in the lower dimensional space)
    /// \return      the approximate nearest neighbor
    res_pair_raw<unsigned> query(const uint64_t *q, size_t t) const {
      return query(q, t, -1);
    }

    ///
    /// \brief 1nn query with early stop
    ///
    /// \param q         header pointer for the query point
    /// \param t         number of candidates to be checked (i.e., how many nearest neighbors need to be
    ///                  checked in the lower dimensional space)
    /// \param thres     threshold
    /// \return          the approximate nearest neighbor
    res_pair_raw<unsigned> query(const uint64_t *q, size_t t, double thres) const {
      if (t > _n) {
        spdlog::warn("t can not exceed n, the number of points in the data base");
        return res_pair_raw<unsigned>{-1, 0};
      }
      auto *q_proj = new float[_m];
      _sketch->apply(q, q_proj);
      _index->init_search(q_proj);
      res_pair_raw<unsigned> res{};
      res.id = -1;
      res.dist = std::numeric_limits<unsigned>::max();
      int count = 0;
      while (count < t) {
        res_pair cover_tree_res = _index->increm_knn_search_compressed();
        ++count;
        if (thres > 0 && res.id >= 0 &&
            (cover_tree_res.dist * cover_tree_res.dist > res.dist * thres)) {
          _index->finish_search();/// 1st time test early-stop condition
          return res;
        }
        auto idx = cover_tree_res.id;
        auto dist = distance::hamming_distance(q, &_raw_data_ref[idx * _d], _d);
        bool changed = false;
        if (dist < res.dist) {
          res.id = idx;
          res.dist = dist;
          changed = true;
        }

        if (thres > 0 && changed && res.id >= 0
            && (cover_tree_res.dist * cover_tree_res.dist
                > res.dist * thres)) {  // 2nd time test early-stop condition
          _index->finish_search();
          return res;
        }
      }
      _index->finish_search();

      return res;
    }
    ///
    /// \brief knn query
    ///
    /// \param q         header pointer for the query point
    /// \param k         number of points (if possible) to be included in the result
    /// \param t         number of candidates to be checked
    /// \param heap      heap to store the results
    void query(const uint64_t *q, size_t k, size_t t, std::vector<res_pair_raw<unsigned> > &heap) const {
      query(q, k, t, -1, heap);
    }

    ///
    /// \brief knn query with early stop
    ///
    /// \param q         header pointer for the query point
    /// \param k         number of points (if possible) to be included in the result
    /// \param t         number of candidates to be checked
    /// \param thres     threshold
    /// \param heap      heap to store the results
    void query(const uint64_t *q, size_t k, size_t t, double thres, std::vector<res_pair_raw<unsigned> > &heap) const {
      if (k > t || t > _n) {
        spdlog::warn("k should be at most t, and t should be at most n, which is the number of points!\n");
        return;
      }
      auto *q_proj = new float[_m];
      _sketch->apply(q, q_proj);

      _index->init_search(q_proj);
      heap.clear();
      heap.reserve(k);
      int count = 0;
      while (count < t) {
        res_pair cover_tree_res = _index->increm_knn_search_compressed();
        ++ count;
        if (thres > 0 && heap.size() == k &&
            (cover_tree_res.dist * cover_tree_res.dist > heap.front().dist * thres)) {
          _index->finish_search();/// 1st time test early-stop condition
          return;
        }
        auto idx = cover_tree_res.id;
        res_pair_raw<unsigned> res =
            {idx, distance::hamming_distance(q, &_raw_data_ref[idx * _d], _d)};
        bool changed = false;
        if (heap.size() < k) {
          heap.push_back(res);
          std::push_heap(heap.begin(), heap.end());
          changed = true;
        } else if (res.dist < heap.front().dist) {  // update top-k heap
          std::pop_heap(heap.begin(), heap.end());
          heap.pop_back();
          heap.push_back(res);
          std::push_heap(heap.begin(), heap.end());
          changed = true;
        }
        if (thres > 0 && changed && heap.size() == k
            && (cover_tree_res.dist * cover_tree_res.dist
                > heap.front().dist * thres)) {  // 2nd time test early-stop condition
          _index->finish_search();
          return;
        }
      }
      _index->finish_search();
    }

    size_t index_size() const {
      return _index->usedMemory();
    }

   private:
    const FlatVectorHamming64 &_raw_data_ref;
    size_t _n;
    unsigned _d;
    unsigned _org_dim;
    unsigned _m;
    unsigned _seed;
    std::unique_ptr<sketch::GaussianSketchHamming> _sketch;
    std::unique_ptr<SRS_Cover_Tree> _index;
    // file paths
    std::string _index_path;
    std::string _index_file_path;
    std::string _para_file_path;

    static constexpr unsigned WORD_SIZE = 64u;
  };

  /// \brief TOWCoverTreeHamming
  struct TOWCoverTreeHamming {
    ///
    /// \brief constructor for TOWCoverTreeHamming
    ///
    /// \sa All parameters have the same meaning as those for SRSCoverTreeHamming()
    ///
    TOWCoverTreeHamming(const FlatVectorHamming64 &raw_data,
                        unsigned enc_dim,
                        unsigned projected_dim,
                        std::string index_path,
                        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count()) :
        _raw_data_ref(raw_data),
        _n(raw_data.size() / enc_dim),
        _d(enc_dim),
        _org_dim(enc_dim * WORD_SIZE),
        _m(projected_dim),
        _seed(seed),
        _sketch(nullptr),
        _index(nullptr),
        _index_path(std::move(index_path)),
        _index_file_path(""),
        _para_file_path("") {
      if (_index_path.back() != '/')
        _index_path.push_back('/');
      _index_file_path = _index_path + "index_tow_cover";
      _para_file_path = _index_path + "para_tow.txt";
    }

    /// \brief build_index
    ///
    /// \sa Same as SRSCoverTreeHamming::build_index(). The only difference is that this API uses tug of
    /// war sketch
    void build_index() {
      /// create sketch
      _sketch = std::make_unique<sketch::TugOfWarSketchHamming>(_m, _org_dim, _seed);
      auto *proj_data = new float[_n * _m];

      spdlog::info("Start to project the raw data onto space with lower dimension ...");
      Timer timer;
      timer.restart();
      {
        ProgressBar progress(_n);
        /// projecting raw data to lower dimensional space
        for (auto i = 0; i < _n; ++i) {
          /// project the ith data point
          _sketch->apply(&_raw_data_ref[i * _d], &proj_data[i * _m]);
          ++progress;
        }
      }
      spdlog::info("Projecting finished! It takes {} seconds", timer.elapsed());
      /// build cover tree
      _index = std::make_unique<SRS_Cover_Tree>(_n, _m, new Proj_data(_n, _m, proj_data));
      /// write tree to file
      _index->write_to_disk_compressed(_index_file_path.c_str());
    }

    /// \brief 1nn query
    ///
    /// \sa SRSCoverTreeHamming::query(uint64_t, size_t)
    res_pair_raw<unsigned> query(const uint64_t *q, size_t t) const {
      return query(q, t, -1);
    }

    /// \brief 1nn query with early stop
    ///
    /// \sa SRSCoverTreeHamming::query(uint64_t, size_t, double)
    res_pair_raw<unsigned> query(const uint64_t *q, size_t t, double thres) const {
      auto *q_proj = new float[_m];
      _sketch->apply(q, q_proj);

      _index->init_search(q_proj);
      res_pair_raw<unsigned> res{};
      res.id = -1;
      res.dist = std::numeric_limits<unsigned>::max();
      int count = 0;
      while (count < t) {
        res_pair cover_tree_res = _index->increm_knn_search_compressed();
        ++ count;
        if (thres > 0 && res.id >= 0 &&
            (cover_tree_res.dist * cover_tree_res.dist > res.dist * thres)) {
          _index->finish_search();/// 1st time test early-stop condition
          return res;
        }
        auto idx = cover_tree_res.id;
        if (idx == -1) {
          spdlog::warn("Can not find another one (count = {}/{})", count, t);
          continue;
        }
        auto dist = distance::hamming_distance(q, &_raw_data_ref[idx * _d], _d);
        bool changed = false;
        if (dist < res.dist) {
          res.id = idx;
          res.dist = dist;
          changed = true;
        }

        if (thres > 0 && changed && res.id >= 0
            && (cover_tree_res.dist * cover_tree_res.dist
                > res.dist * thres)) {  // 2nd time test early-stop condition
          _index->finish_search();
          return res;
        }
      }
      _index->finish_search();

      return res;
    }

    /// \brief knn query
    ///
    /// \sa SRSCoverTreeHamming::query(uint64_t, size_t, std::vector<res_pair_raw<unsigned> > &)
    void query(const uint64_t *q, size_t k, size_t t, std::vector<res_pair_raw<unsigned> > &heap) const {
      query(q, k, t, -1, heap);
    }

    /// \brief knn query with early stoo
    ///
    /// \sa SRSCoverTreeHamming::query(uint64_t, size_t, double, std::vector<res_pair_raw<unsigned> > &)
    void query(const uint64_t *q, size_t k, size_t t, double thres, std::vector<res_pair_raw<unsigned> > &heap) const {
      auto *q_proj = new float[_m];
      _sketch->apply(q, q_proj);
      _index->init_search(q_proj);
      heap.clear();
      heap.reserve(k);
      int count = 0;
      while (count < t) {
        res_pair cover_tree_res = _index->increm_knn_search_compressed();
        ++ count;
        if (thres > 0 && heap.size() == k &&
            (cover_tree_res.dist * cover_tree_res.dist > heap.front().dist * thres)) {
          _index->finish_search();/// 1st time test early-stop condition
          return;
        }
        auto idx = cover_tree_res.id;
        res_pair_raw<unsigned> res =
            {idx, distance::hamming_distance(q, &_raw_data_ref[idx * _d], _d)};
        bool changed = false;
        if (heap.size() < k) {
          heap.push_back(res);
          std::push_heap(heap.begin(), heap.end());
          changed = true;
        } else if (res.dist < heap.front().dist) {  // update top-k heap
          std::pop_heap(heap.begin(), heap.end());
          heap.pop_back();
          heap.push_back(res);
          std::push_heap(heap.begin(), heap.end());
          changed = true;
        }
        if (thres > 0 && changed && heap.size() == k
            && (cover_tree_res.dist * cover_tree_res.dist
                > heap.front().dist * thres)) {  // 2nd time test early-stop condition
          _index->finish_search();
          return;
        }
      }
      _index->finish_search();
    }

    /// \brief project_to
    ///
    /// This API is created for test purpose!
    /// Note that you should allocate enough memory for #res before passing it into
    /// this API. That is, this API won't handle any memory allocation or memory bound
    /// checks.
    ///
    /// \param data    data point in the original space
    /// \param res     data after projecting
    void project_to(const uint64_t* data, float *res) const {
      _sketch->apply(data, res);
    }

    size_t index_size() const {
      return _index->usedMemory();
    }

   private:
    const FlatVectorHamming64 &_raw_data_ref;
    size_t _n;
    unsigned _d;
    unsigned _org_dim;
    unsigned _m;
    unsigned _seed;
    std::unique_ptr<sketch::TugOfWarSketchHamming> _sketch;
    std::unique_ptr<SRS_Cover_Tree> _index;
    // file paths
    std::string _index_path;
    std::string _index_file_path;
    std::string _para_file_path;

    static constexpr unsigned WORD_SIZE = 64u;
  };
  /// types for different KD trees
  typedef KDTreeFlatVectorAdaptor<std::vector<float>, float> KDTreeFloat;
  typedef KDTreeFlatVectorAdaptor<std::vector<int16_t>, int16_t, -1, float > KDTreeInt16;
  /// maximum number of nodes in the leaf
  constexpr int MAX_LEAF = 10;

  /// \brief SRSKDTreeHamming
  struct SRSKDTreeHamming {
    ///
    /// constructor for SRSKDTreeHamming
    ///
    /// \sa SRSCoverTreeHamming()
    SRSKDTreeHamming(const FlatVectorHamming64 &raw_data,
                     unsigned enc_dim,
                     unsigned projected_dim,
                     std::string index_path,
                     unsigned seed = std::chrono::system_clock::now().time_since_epoch().count()) :
        _raw_data_ref(raw_data),
        _n(raw_data.size() / enc_dim),
        _d(enc_dim),
        _org_dim(enc_dim * WORD_SIZE),
        _m(projected_dim),
        _seed(seed),
        _sketch(nullptr),
        _proj_data{},
        _index(nullptr),
        _index_path(std::move(index_path)),
        _index_file_path(""),
        _para_file_path("") {
      if (_index_path.back() != '/')
        _index_path.push_back('/');
      _index_file_path = _index_path + "index_srs_kd";
      _para_file_path = _index_path + "para_srs_kd.txt";
    }

    /// \brief build_index
    ///
    /// \sa SRSCoverTreeHamming::build_index()
    void build_index() {
      /// create sketches
      _sketch = std::make_unique<sketch::GaussianSketchHamming>(_m, _org_dim, _seed);
      /// (re)allocate memory to projected data
      _proj_data.resize(_n * _m, 0);

      spdlog::info("Start to project the raw data onto space with lower dimension ...");
      Timer timer;
      timer.restart();
      {
        ProgressBar progress(_n);
        /// projecting raw data to lower dimensional space
        /// TODO: the efficiency of this for loop can be increased by using loop unrolling
        for (auto i = 0; i < _n; ++i) {
          /// project the ith data point
          _sketch->apply(&_raw_data_ref[i * _d], &_proj_data[i * _m]);
          ++progress;
        }
      }
      spdlog::info("Projecting finished! It takes {} seconds", timer.elapsed());
      /// build kd tree
      /// The saveIndex API for KD tree is not very-well designed, the index is HUGE when saving to file!!
      /// so we choose to write the projected data and random bits into file
      _index = std::make_unique<KDTreeFloat>(_m, _proj_data, MAX_LEAF);
      /// write tree to file
//      FILE *fp = fopen(_index_file_path.c_str(), "wb");
//      std::ofstream FILE(_index_file_path + ".projected_data", std::ios::out | std::ofstream::binary);
//      if (FILE.bad()) throw std::runtime_error("Error writing index file!");
//      std::copy(_proj_data.begin(), _proj_data.end(), std::ostreambuf_iterator<char>(FILE));
//      if (!fp)
//        throw std::runtime_error("Error writing index file!");
//      _index->index->saveIndex(fp);

//      fclose(fp);
    }

    /// \brief 1nn query
    ///
    /// \sa SRSCoverTreeHamming::query(const uint64_t*, size_t)
    res_pair_raw<unsigned> query(const uint64_t *q, size_t t) const {
      if (t > _n || t == 0) {
        spdlog::warn("t should be at least 1 and at most n, the number of points in the data base");
        return res_pair_raw<unsigned>{-1, 0};
      }
      std::vector<float> q_proj(_m, 0);
      _sketch->apply(q, &q_proj[0]);
      res_pair_raw<unsigned> res{};
      res.id = -1;
      res.dist = std::numeric_limits<unsigned>::max();

      std::vector<size_t> ret_indexes(t);
      std::vector<float> out_dists_sqr(t);

      _index->query(&q_proj[0], t, &ret_indexes[0], &out_dists_sqr[0]);

      for (int i = 0; i < t; ++i) {
        int idx = ret_indexes[i];
        unsigned dist = distance::hamming_distance(q, &_raw_data_ref[idx * _d], _d);
        if (dist < res.dist) {
          res.id = idx;
          res.dist = dist;
        }
      }

      return res;
    }
    /// \brief knn query
    ///
    /// \sa SRSCoverTreeHamming::query(const uint64_t*, size_t, size_t, std::vector<res_pair_raw<unsigned> > &)
    void query(const uint64_t *q, size_t k, size_t t, std::vector<res_pair_raw<unsigned> > &heap) const {
      if (t > _n || t < k || k == 0) {
        spdlog::warn("0 < k <= t <= n, where n is the number of points in the data base");
        return ;
      }

      std::vector<float> q_proj(_m, 0);
      _sketch->apply(q, &q_proj[0]);
      heap.clear();
      heap.reserve(k);

      std::vector<size_t> ret_indexes(t);
      std::vector<float> out_dists_sqr(t);
      _index->query(&q_proj[0], t, &ret_indexes[0], &out_dists_sqr[0]);

      res_pair_raw<unsigned> res{};
      /// find top-k
      for (size_t i = 0; i < t; ++i) {
        res.id = ret_indexes[i];
        res.dist = distance::hamming_distance(q, &_raw_data_ref[res.id * _d], _d);

        if (heap.size() < k) {
          heap.push_back(res);
          std::push_heap(heap.begin(), heap.end());
        } else if (res.dist < heap.front().dist) {
          std::pop_heap(heap.begin(), heap.end());
          heap.pop_back();
          heap.push_back(res);
          std::push_heap(heap.begin(), heap.end());
        }
      }
    }

    size_t index_size() const {
      return _index->index->usedMemory(*(_index->index));
    }

   private:
    const FlatVectorHamming64 &_raw_data_ref;
    size_t _n;
    unsigned _d;
    unsigned _org_dim;
    unsigned _m;
    unsigned _seed;
    std::unique_ptr<sketch::GaussianSketchHamming> _sketch;
    std::vector<float> _proj_data;
    std::unique_ptr<KDTreeFloat> _index;
    // file paths
    std::string _index_path;
    std::string _index_file_path;
    std::string _para_file_path;

    static constexpr unsigned WORD_SIZE = 64u;
  };

  /// \brief TOWKDTreeHamming
  struct TOWKDTreeHamming {
    /// \brief constructor for TOWKDTreeHamming
    ///
    /// \sa SRSCoverTreeHamming()
    TOWKDTreeHamming(const FlatVectorHamming64 &raw_data,
                     unsigned enc_dim,
                     unsigned projected_dim,
                     std::string index_path,
                     unsigned seed = std::chrono::system_clock::now().time_since_epoch().count()) :
        _raw_data_ref(raw_data),
        _n(raw_data.size() / enc_dim),
        _d(enc_dim),
        _org_dim(_d * WORD_SIZE),
        _m(projected_dim),
        _seed(seed),
        _sketch(nullptr),
        _proj_data{},
        _index(nullptr),
        _index_path(std::move(index_path)),
        _index_file_path(""),
        _para_file_path("") {
      if (_index_path.back() != '/')
        _index_path.push_back('/');
      _index_file_path = _index_path + "index_tow_kd";
      _para_file_path = _index_path + "para_tow_kd.txt";
    }

    /// \brief build_index
    ///
    /// \sa SRSCoverTree::build_index()
    void build_index() {
      /// create sketches
      _sketch = std::make_unique<sketch::TugOfWarSketchHamming>(_m, _org_dim, _seed);
      _proj_data.resize(_n * _m, 0);

      spdlog::info("Start to project the raw data onto space with lower dimension ...");
      Timer timer;
      timer.restart();
      {
        ProgressBar progress(_n);
        /// projecting raw data to lower dimensional space
        for (auto i = 0; i < _n; ++i) {
          /// project the ith data point
          _sketch->apply(&_raw_data_ref[i * _d], &_proj_data[i * _m]);
          ++progress;
        }
      }
      spdlog::info("Projecting finished! It takes {} seconds", timer.elapsed());
      /// build kd tree
      _index = std::make_unique<KDTreeInt16>(_m, _proj_data, MAX_LEAF);
      /// write tree to file
//      FILE *fp = fopen(_index_file_path.c_str(), "wb");
//      if (!fp)
//        throw std::runtime_error("Error writing index file!");
//      _index->index->saveIndex(fp);
//      fclose(fp);
    }
    ///
    /// \brief 1nn query
    ///
    /// \sa SRSCoverTree::query(const uint64_t *, size_t)
    res_pair_raw<unsigned> query(const uint64_t *q, size_t t) const {
      std::vector<int16_t> q_proj(_m, 0);
      _sketch->apply(q, &q_proj[0]);
      res_pair_raw<unsigned> res{};
      res.id = -1;
      res.dist = std::numeric_limits<unsigned>::max();

      std::vector<size_t> ret_indexes(t, 0);
      //std::vector<int16_t> out_dists_sqr(t, 0);
      std::vector<float> out_dists_sqr(t, 0);
      _index->query(&q_proj[0], t, &ret_indexes[0], &out_dists_sqr[0]);

      for (int i = 0; i < t; ++i) {
        int idx = ret_indexes[i];
        unsigned dist = distance::hamming_distance(q, &_raw_data_ref[idx * _d], _d);
        if (dist < res.dist) {
          res.id = idx;
          res.dist = dist;
        }
      }

      return res;
    }
    ///
    /// \brief knn query
    ///
    /// \sa SRSCoverTree::query(const uint64_t *, size_t, size_t, std::vector<res_pair_raw<unsigned> > &)
    void query(const uint64_t *q, size_t k, size_t t, std::vector<res_pair_raw<unsigned> > &heap) const {
      std::vector<int16_t> q_proj(_m, 0);
      _sketch->apply(q, &q_proj[0]);
      heap.clear();
      heap.reserve(k);

      std::vector<size_t> ret_indexes(t);
      //std::vector<int16_t> out_dists_sqr(t);
      std::vector<float> out_dists_sqr(t);
      _index->query(&q_proj[0], t, &ret_indexes[0], &out_dists_sqr[0]);

      res_pair_raw<unsigned> res{};

      for (size_t i = 0; i < t; ++i) {
        res.id = ret_indexes[i];
        res.dist = distance::hamming_distance(q, &_raw_data_ref[res.id * _d], _d);

        if (heap.size() < k) {
          heap.push_back(res);
          std::push_heap(heap.begin(), heap.end());
        } else if (res.dist < heap.front().dist) {
          std::pop_heap(heap.begin(), heap.end());
          heap.pop_back();
          heap.push_back(res);
          std::push_heap(heap.begin(), heap.end());
        }
      }

    }

    ///
    /// \brief project_to
    ///
    /// \sa SRSKDTreeHamming::project_to()
    void project_to(const uint64_t* data, float *res) const {
      _sketch->apply(data, res);
    }

    size_t index_size() const {
      return _index->index->usedMemory(*(_index->index));
    }

   private:
    const FlatVectorHamming64 &_raw_data_ref;
    size_t _n;
    unsigned _d;
    unsigned _org_dim;
    unsigned _m;
    unsigned _seed;
    std::unique_ptr<sketch::TugOfWarSketchHamming> _sketch;
    std::vector<int16_t > _proj_data;
    std::unique_ptr<KDTreeInt16> _index;
    // file paths
    std::string _index_path;
    std::string _index_file_path;
    std::string _para_file_path;

    static constexpr unsigned WORD_SIZE = 64u;
  };

}// end namespace flat vector

struct SRSCoverTreeHamming {
  SRSCoverTreeHamming(const std::vector<std::vector<uint64_t >> &raw_data,
                      unsigned projected_dim, std::string index_path,
                      unsigned seed = std::chrono::system_clock::now().time_since_epoch().count()) :
      _raw_data_ref(raw_data),
      _n(raw_data.size()),
      _d(raw_data.front().size()),
      _org_dim(_d * WORD_SIZE),
      _m(projected_dim),
      _seed(seed),
      _sketch(nullptr),
      _index(nullptr),
      _index_path(std::move(index_path)),
      _index_file_path(""),
      _para_file_path("") {
    if (_index_path.back() != '/')
      _index_path.push_back('/');
    _index_file_path = _index_path + "index_srs_cover";
    _para_file_path = _index_path + "para_srs_cover.txt";
  }

  void build_index() {
    _sketch = std::make_unique<sketch::GaussianSketchHamming>(_m, _org_dim, _seed);
    auto *proj_data = new float[_n * _m];
    /// projecting raw data to lower dimensional space
    typedef boost::timer Timer;
    typedef boost::progress_display ProgressBar;
    spdlog::info("Start to project the raw data onto space with lower dimension ...");
    Timer timer;
    timer.restart();
    {
      ProgressBar progress(_n);
      for (auto i = 0; i < _n; ++i) {
        /// project the ith data point
        _sketch->apply(&_raw_data_ref[i][0], &proj_data[i * _m]);
        ++progress;
      }
    }
    spdlog::info("Projecting finished! It takes {} seconds", timer.elapsed());
    /// build cover tree
    _index = std::make_unique<SRS_Cover_Tree>(_n, _m, new Proj_data(_n, _m, proj_data));
    /// write tree to file
    _index->write_to_disk_compressed(_index_file_path.c_str());
  }

  res_pair_raw<unsigned> query(const uint64_t *q, size_t t) const {
    return query(q, t, -1);
  }

  res_pair_raw<unsigned> query(const uint64_t *q, size_t t, double thres) const {
    auto *q_proj = new float[_m];
    _sketch->apply(q, q_proj);
    _index->init_search(q_proj);
    res_pair_raw<unsigned> res{};
    res.id = -1;
    res.dist = std::numeric_limits<unsigned>::max();
    int count = 0;
    while (count < t) {
      res_pair cover_tree_res = _index->increm_knn_search_compressed();
      ++count;
      if (thres > 0 && res.id >= 0 &&
          (cover_tree_res.dist * cover_tree_res.dist > res.dist * thres)) {
        _index->finish_search();/// 1st time test early-stop condition
        return res;
      }
      auto idx = cover_tree_res.id;
      auto dist = distance::hamming_distance(q, &_raw_data_ref[idx][0], _d);
      bool changed = false;
      if (dist < res.dist) {
        res.id = idx;
        res.dist = dist;
        changed = true;
      }

      if (thres > 0 && changed && res.id >= 0
          && (cover_tree_res.dist * cover_tree_res.dist
              > res.dist * thres)) {  // 2nd time test early-stop condition
        _index->finish_search();
        return res;
      }
    }
    _index->finish_search();

    return res;
  }

  void query(const uint64_t *q, size_t k, size_t t, std::vector<res_pair_raw<unsigned> > &heap) const {
    query(q, k, t, -1, heap);
  }

  void query(const uint64_t *q, size_t k, size_t t, double thres, std::vector<res_pair_raw<unsigned> > &heap) const {
    auto *q_proj = new float[_m];
    _sketch->apply(q, q_proj);
    _index->init_search(q_proj);
    heap.clear();
    heap.reserve(k);
    int count = 0;
    while (count < t) {
      res_pair cover_tree_res = _index->increm_knn_search_compressed();
      ++count;
      if (thres > 0 && heap.size() == k &&
          (cover_tree_res.dist * cover_tree_res.dist > heap.front().dist * thres)) {
        _index->finish_search();/// 1st time test early-stop condition
        return;
      }
      auto idx = cover_tree_res.id;
      res_pair_raw<unsigned> res =
          {idx, distance::hamming_distance(q, &_raw_data_ref[idx][0], _d)};
      bool changed = false;
      if (heap.size() < k) {
        heap.push_back(res);
        std::push_heap(heap.begin(), heap.end());
        changed = true;
      } else if (res.dist < heap.front().dist) {  // update top-k heap
        std::pop_heap(heap.begin(), heap.end());
        heap.pop_back();
        heap.push_back(res);
        std::push_heap(heap.begin(), heap.end());
        changed = true;
      }
      if (thres > 0 && changed && heap.size() == k
          && (cover_tree_res.dist * cover_tree_res.dist
              > heap.front().dist * thres)) {  // 2nd time test early-stop condition
        _index->finish_search();
        return;
      }
    }
    _index->finish_search();
  }

 private:
  const std::vector<std::vector<uint64_t >> &_raw_data_ref;
  size_t _n;
  unsigned _d;
  unsigned _org_dim;
  unsigned _m;
  unsigned _seed;
  std::unique_ptr<sketch::GaussianSketchHamming> _sketch;
  std::unique_ptr<SRS_Cover_Tree> _index;
  // file paths
  std::string _index_path;
  std::string _index_file_path;
  std::string _para_file_path;

  static constexpr unsigned WORD_SIZE = 64u;
};

struct TOWCoverTreeHamming {
  TOWCoverTreeHamming(const std::vector<std::vector<uint64_t >> &raw_data,
                      unsigned projected_dim, std::string index_path,
                      unsigned seed = std::chrono::system_clock::now().time_since_epoch().count()) :
      _raw_data_ref(raw_data),
      _n(raw_data.size()),
      _d(raw_data.front().size()),
      _org_dim(_d * WORD_SIZE),
      _m(projected_dim),
      _seed(seed),
      _sketch(nullptr),
      _index(nullptr),
      _index_path(std::move(index_path)),
      _index_file_path(""),
      _para_file_path("") {
    if (_index_path.back() != '/')
      _index_path.push_back('/');
    _index_file_path = _index_path + "index_tow_cover";
    _para_file_path = _index_path + "para_tow.txt";
  }

  void build_index() {
    _sketch = std::make_unique<sketch::TugOfWarSketchHamming>(_m, _org_dim, _seed);
    auto *proj_data = new float[_n * _m];
    /// projecting raw data to lower dimensional space
    typedef boost::timer Timer;
    typedef boost::progress_display ProgressBar;
    spdlog::info("Start to project the raw data onto space with lower dimension ...");
    Timer timer;
    timer.restart();
    {
      ProgressBar progress(_n);
      for (auto i = 0; i < _n; ++i) {
        /// project the ith data point
        _sketch->apply(&_raw_data_ref[i][0], &proj_data[i * _m]);
#ifdef DEBUG_TOW_COVER_TREE
        spdlog::debug("id: {}, data: {}, projected: {}", i, ss::io::stringtify(_raw_data_ref[i].begin(), _raw_data_ref[i].end(), "{:064b}"),
            ss::io::stringtify(&proj_data[i * _m], &proj_data[(i + 1) * _m]));
#endif
        ++progress;
      }
    }
    spdlog::info("Projecting finished! It takes {} seconds", timer.elapsed());
    /// build cover tree
    _index = std::make_unique<SRS_Cover_Tree>(_n, _m, new Proj_data(_n, _m, proj_data));
    /// write tree to file
    _index->write_to_disk_compressed(_index_file_path.c_str());
  }

  res_pair_raw<unsigned> query(const uint64_t *q, size_t t) const {
    return query(q, t, -1);
  }

  res_pair_raw<unsigned> query(const uint64_t *q, size_t t, double thres) const {
    auto *q_proj = new float[_m];
    _sketch->apply(q, q_proj);
#ifdef DEBUG_TOW_COVER_TREE
    spdlog::debug("query -- data: {}, projected: {}", ss::io::stringtify(q, q + _d, "{:064b}"),
                  ss::io::stringtify(q_proj, q_proj + _m));
#endif
    _index->init_search(q_proj);
    res_pair_raw<unsigned> res{};
    res.id = -1;
    res.dist = std::numeric_limits<unsigned>::max();
    int count = 0;
    while (count < t) {
      res_pair cover_tree_res = _index->increm_knn_search_compressed();
      ++count;
      if (thres > 0 && res.id >= 0 &&
          (cover_tree_res.dist * cover_tree_res.dist > res.dist * thres)) {
        _index->finish_search();/// 1st time test early-stop condition
        return res;
      }
      auto idx = cover_tree_res.id;
      auto dist = distance::hamming_distance(q, &_raw_data_ref[idx][0], _d);
      bool changed = false;
      if (dist < res.dist) {
        res.id = idx;
        res.dist = dist;
        changed = true;
      }

      if (thres > 0 && changed && res.id >= 0
          && (cover_tree_res.dist * cover_tree_res.dist
              > res.dist * thres)) {  // 2nd time test early-stop condition
        _index->finish_search();
        return res;
      }
    }
    _index->finish_search();

    return res;
  }

  void query(const uint64_t *q, size_t k, size_t t, std::vector<res_pair_raw<unsigned> > &heap) const {
    query(q, k, t, -1, heap);
  }

  void query(const uint64_t *q, size_t k, size_t t, double thres, std::vector<res_pair_raw<unsigned> > &heap) const {
    auto *q_proj = new float[_m];
    _sketch->apply(q, q_proj);
    _index->init_search(q_proj);
    heap.clear();
    heap.reserve(k);
    int count = 0;
    while (count < t) {
      res_pair cover_tree_res = _index->increm_knn_search_compressed();
      ++count;
      if (thres > 0 && heap.size() == k &&
          (cover_tree_res.dist * cover_tree_res.dist > heap.front().dist * thres)) {
        _index->finish_search();/// 1st time test early-stop condition
        return;
      }
      auto idx = cover_tree_res.id;
      res_pair_raw<unsigned> res =
          {idx, distance::hamming_distance(q, &_raw_data_ref[idx][0], _d)};
      bool changed = false;
      if (heap.size() < k) {
        heap.push_back(res);
        std::push_heap(heap.begin(), heap.end());
        changed = true;
      } else if (res.dist < heap.front().dist) {  // update top-k heap
        std::pop_heap(heap.begin(), heap.end());
        heap.pop_back();
        heap.push_back(res);
        std::push_heap(heap.begin(), heap.end());
        changed = true;
      }
      if (thres > 0 && changed && heap.size() == k
          && (cover_tree_res.dist * cover_tree_res.dist
              > heap.front().dist * thres)) {  // 2nd time test early-stop condition
        _index->finish_search();
        return;
      }
    }
    _index->finish_search();
  }

  void project_to(const uint64_t* data, float *res) const {
    _sketch->apply(data, res);
  }

 private:
  const std::vector<std::vector<uint64_t >> &_raw_data_ref;
  size_t _n;
  unsigned _d;
  unsigned _org_dim;
  unsigned _m;
  unsigned _seed;
  std::unique_ptr<sketch::TugOfWarSketchHamming> _sketch;
  std::unique_ptr<SRS_Cover_Tree> _index;
  // file paths
  std::string _index_path;
  std::string _index_file_path;
  std::string _para_file_path;

  static constexpr unsigned WORD_SIZE = 64u;
};

typedef KDTreeVectorOfVectorsAdaptor<std::vector<std::vector<float>>, float> KDTreeFloat;
typedef KDTreeVectorOfVectorsAdaptor<std::vector<std::vector<int16_t >>, int16_t> KDTreeInt16;
constexpr int MAX_LEAF = 10;

struct SRSKDTreeHamming {
  SRSKDTreeHamming(const std::vector<std::vector<uint64_t >> &raw_data,
                   unsigned projected_dim, std::string index_path,
                   unsigned seed = std::chrono::system_clock::now().time_since_epoch().count()) :
      _raw_data_ref(raw_data),
      _n(raw_data.size()),
      _d(raw_data.front().size()),
      _org_dim(_d * WORD_SIZE),
      _m(projected_dim),
      _seed(seed),
      _sketch(nullptr),
      _proj_data{},
      _index(nullptr),
      _index_path(std::move(index_path)),
      _index_file_path(""),
      _para_file_path("") {
    if (_index_path.back() != '/')
      _index_path.push_back('/');
    _index_file_path = _index_path + "index_srs_kd";
    _para_file_path = _index_path + "para_srs_kd.txt";
  }

  void build_index() {
    _sketch = std::make_unique<sketch::GaussianSketchHamming>(_m, _org_dim, _seed);
    _proj_data.resize(_n, std::vector<float>(_m, 0));
    /// projecting raw data to lower dimensional space
    typedef boost::timer Timer;
    typedef boost::progress_display ProgressBar;
    spdlog::info("Start to project the raw data onto space with lower dimension ...");
    Timer timer;
    timer.restart();
    {
      ProgressBar progress(_n);
      for (auto i = 0; i < _n; ++i) {
        /// project the ith data point
        _sketch->apply(&_raw_data_ref[i][0], &_proj_data[i][0]);
        ++progress;
      }
    }
    spdlog::info("Projecting finished! It takes {} seconds", timer.elapsed());
    /// build cover tree
    _index = std::make_unique<KDTreeFloat>(_m, _proj_data, MAX_LEAF);
    /// write tree to file
    FILE *fp = fopen(_index_file_path.c_str(), "wb");
    if (!fp)
      throw std::runtime_error("Error writing index file!");
    _index->index->saveIndex(fp);
    fclose(fp);
  }

  res_pair_raw<unsigned> query(const uint64_t *q, size_t t) const {
    std::vector<float> q_proj(_m, 0);
    _sketch->apply(q, &q_proj[0]);
    res_pair_raw<unsigned> res{};
    res.id = -1;
    res.dist = std::numeric_limits<unsigned>::max();

    nanoflann::SearchParams params;
    std::vector<size_t> ret_indexes(t);
    std::vector<float> out_dists_sqr(t);

    _index->query(&q_proj[0], t, &ret_indexes[0], &out_dists_sqr[0]);

    for (int i = 0; i < t; ++i) {
      int idx = ret_indexes[i];
      unsigned dist = distance::hamming_distance(q, &_raw_data_ref[idx][0], _d);
      if (dist < res.dist) {
        res.id = idx;
        res.dist = dist;
      }
    }

    return res;
  }

  void query(const uint64_t *q, size_t k, size_t t, std::vector<res_pair_raw<unsigned> > &heap) const {
    query(q, k, t, -1, heap);
  }

  void query(const uint64_t *q, size_t k, size_t t, double thres, std::vector<res_pair_raw<unsigned> > &heap) const {
    std::vector<float> q_proj(_m, 0);
    _sketch->apply(q, &q_proj[0]);
    heap.clear();
    heap.reserve(k);

    nanoflann::SearchParams params;
    std::vector<size_t> ret_indexes(t);
    std::vector<float> out_dists_sqr(t);
    _index->query(&q_proj[0], t, &ret_indexes[0], &out_dists_sqr[0]);
    for (size_t i = 0; i < t; ++i) {
      int idx = ret_indexes[i];
      unsigned dist = distance::hamming_distance(q, &_raw_data_ref[idx][0], _d);
      res_pair_raw<unsigned> res = {idx, dist};
      if (heap.size() < k) {
        heap.push_back(res);
        std::push_heap(heap.begin(), heap.end());
      } else if (res.dist < heap.front().dist) {
        std::pop_heap(heap.begin(), heap.end());
        heap.pop_back();
        heap.push_back(res);
        std::push_heap(heap.begin(), heap.end());
      }
    }
  }

 private:
  const std::vector<std::vector<uint64_t >> &_raw_data_ref;
  size_t _n;
  unsigned _d;
  unsigned _org_dim;
  unsigned _m;
  unsigned _seed;
  std::unique_ptr<sketch::GaussianSketchHamming> _sketch;
  std::vector<std::vector<float>> _proj_data;
  std::unique_ptr<KDTreeFloat> _index;
  // file paths
  std::string _index_path;
  std::string _index_file_path;
  std::string _para_file_path;

  static constexpr unsigned WORD_SIZE = 64u;
};

struct TOWKDTreeHamming {
  TOWKDTreeHamming(const std::vector<std::vector<uint64_t >> &raw_data,
                   unsigned projected_dim, std::string index_path,
                   unsigned seed = std::chrono::system_clock::now().time_since_epoch().count()) :
      _raw_data_ref(raw_data),
      _n(raw_data.size()),
      _d(raw_data.front().size()),
      _org_dim(_d * WORD_SIZE),
      _m(projected_dim),
      _seed(seed),
      _sketch(nullptr),
      _proj_data{},
      _index(nullptr),
      _index_path(std::move(index_path)),
      _index_file_path(""),
      _para_file_path("") {
    if (_index_path.back() != '/')
      _index_path.push_back('/');
    _index_file_path = _index_path + "index_tow_kd";
    _para_file_path = _index_path + "para_tow_kd.txt";
  }

  void build_index() {
    _sketch = std::make_unique<sketch::TugOfWarSketchHamming>(_m, _org_dim, _seed);
    _proj_data.resize(_n, std::vector<int16_t>(_m, 0));
    /// projecting raw data to lower dimensional space
    typedef boost::timer Timer;
    typedef boost::progress_display ProgressBar;
    spdlog::info("Start to project the raw data onto space with lower dimension ...");
    Timer timer;
    timer.restart();
    {
      ProgressBar progress(_n);
      for (auto i = 0; i < _n; ++i) {
        /// project the ith data point
        _sketch->apply(&_raw_data_ref[i][0], &_proj_data[i][0]);
        ++progress;
      }
    }
    spdlog::info("Projecting finished! It takes {} seconds", timer.elapsed());
    /// build cover tree
    _index = std::make_unique<KDTreeInt16>(_m, _proj_data, MAX_LEAF);
    /// write tree to file
    FILE *fp = fopen(_index_file_path.c_str(), "wb");
    if (!fp)
      throw std::runtime_error("Error writing index file!");
    _index->index->saveIndex(fp);
    fclose(fp);
  }

  res_pair_raw<unsigned> query(const uint64_t *q, size_t t) const {
    std::vector<int16_t> q_proj(_m, 0);
    _sketch->apply(q, &q_proj[0]);
    res_pair_raw<unsigned> res{};
    res.id = -1;
    res.dist = std::numeric_limits<unsigned>::max();

    nanoflann::SearchParams params;
    std::vector<size_t> ret_indexes(t, 0);
    std::vector<int16_t> out_dists_sqr(t, 0);
    _index->query(&q_proj[0], t, &ret_indexes[0], &out_dists_sqr[0]);

    for (int i = 0; i < t; ++i) {
      int idx = ret_indexes[i];
      unsigned dist = distance::hamming_distance(q, &_raw_data_ref[idx][0], _d);
      if (dist < res.dist) {
        res.id = idx;
        res.dist = dist;
      }
    }

    return res;
  }

  void query(const uint64_t *q, size_t k, size_t t, std::vector<res_pair_raw<unsigned> > &heap) const {
    query(q, k, t, -1, heap);
  }

  void query(const uint64_t *q, size_t k, size_t t, double thres, std::vector<res_pair_raw<unsigned> > &heap) const {
    std::vector<int16_t> q_proj(_m, 0);
    _sketch->apply(q, &q_proj[0]);
    heap.clear();
    heap.reserve(k);

    nanoflann::SearchParams params;
    std::vector<size_t> ret_indexes(t);
    std::vector<int16_t> out_dists_sqr(t);
    _index->query(&q_proj[0], t, &ret_indexes[0], &out_dists_sqr[0]);

    for (size_t i = 0; i < t; ++i) {
      int idx = ret_indexes[i];
      unsigned dist = distance::hamming_distance(q, &_raw_data_ref[idx][0], _d);
      res_pair_raw<unsigned> res = {idx, dist};
      if (heap.size() < k) {
        heap.push_back(res);
        std::push_heap(heap.begin(), heap.end());
      } else if (res.dist < heap.front().dist) {
        std::pop_heap(heap.begin(), heap.end());
        heap.pop_back();
        heap.push_back(res);
        std::push_heap(heap.begin(), heap.end());
      }
    }
  }

  void project_to(const uint64_t* data, float *res) const {
    _sketch->apply(data, res);
  }

 private:
  const std::vector<std::vector<uint64_t >> &_raw_data_ref;
  size_t _n;
  unsigned _d;
  unsigned _org_dim;
  unsigned _m;
  unsigned _seed;
  std::unique_ptr<sketch::TugOfWarSketchHamming> _sketch;
  std::vector<std::vector<int16_t >> _proj_data;
  std::unique_ptr<KDTreeInt16> _index;
  // file paths
  std::string _index_path;
  std::string _index_file_path;
  std::string _para_file_path;

  static constexpr unsigned WORD_SIZE = 64u;
};

}// end namespace ss::ann



#endif //_ANN_SP_HPP_
