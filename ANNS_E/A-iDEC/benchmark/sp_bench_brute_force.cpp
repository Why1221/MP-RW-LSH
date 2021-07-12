#include <boost/timer.hpp>
#include <boost/filesystem.hpp>
#include <boost/progress.hpp>
//#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include <highfive/H5File.hpp>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>

#include <cassert>
#include <vector>
#include <cstring>
#include "benchmark_config.h"

#include <io.hpp>
#include <srs_utils.hpp>
#include <brute_force_sp.hpp>
#include <fmt/format.h>


using namespace ss::io;
using namespace ss::ann::srs;
using namespace HighFive;

namespace fs = boost::filesystem;
namespace bch = bench_config::hamming;

constexpr char SS_SEP[] = "---------------------------------------------------";

template<typename T>
inline constexpr std::string_view alg_name() {
  typedef typename std::remove_cv_t<T> Tnocv;
  typedef ss::ann::brute_force::flat_vector::BruteForceHamming BF;
  if constexpr (std::is_same_v<Tnocv, BF>)
    return "brute-force";
  else
    return "unknown";
}
/// \brief head (for Hamming dataset)
///
/// Show the first #line data points in #data. Similar to the head API of pandas.DataFrame.
///
/// \param data         dataset (representing in a flat array)
/// \param enc_dim      dimension after encoding
/// \param line         how many lines to show
void head(const std::vector<uint64_t> &data,
          unsigned enc_dim,
          unsigned line = 5u) {
  unsigned n = std::min(line, (unsigned) data.size() / enc_dim);
  assert(data.size() != n * enc_dim);
  spdlog::info(SS_SEP);
  for (unsigned i = 0; i < n; ++i) {
    spdlog::info("{} : {}", i, stringtify(&data[i * enc_dim], &data[(i + 1) * enc_dim], "{:064b}"));
  }
  spdlog::info(SS_SEP);
}

/// \brief head (for dataset)
///
/// Show the first #line data points in #data. Similar to the head API of pandas.DataFrame.
///
/// \param data         dataset (representing in a flat array)
/// \param dim          dimension of the data
/// \param line         how many lines to show
template <typename T>
void head(const std::vector<T> &data,
          unsigned dim,
          unsigned line = 5u) {
  unsigned n = std::min(line, (unsigned) data.size() / dim);
  assert(data.size() != n * dim);
  spdlog::info(SS_SEP);
  for (unsigned i = 0; i < n; ++i) {
    spdlog::info("{} : {}", i, stringtify(&data[i * dim], &data[(i + 1) * dim]));
  }
  spdlog::info(SS_SEP);
}

/// \brief head (for dataset)
///
/// Show the first #line data points in #data. Similar to the head API of pandas.DataFrame.
///
/// \param data         dataset (representing in a 2d array)
/// \param dim          dimension of the data
/// \param line         how many lines to show
template <typename T>
void head(const std::vector<std::vector<T>> &data,
          unsigned line = 5u) {
  unsigned n = std::min(line, (unsigned) data.size());
  spdlog::info(SS_SEP);
  for (unsigned i = 0; i < n; ++i) {
    spdlog::info("{} : {}", i, stringtify(data[i].begin(), data[i].end()));
  }
  spdlog::info(SS_SEP);
}

/// \brief load
///
/// load Hamming compact dataset
///
/// \param ds_fname      (hdf5) data set file name
/// \param enc_dim        dimension after encoding
/// \param raw_data      data to be stored
/// \param n_blocks      number of blocks to load
void load(const std::string &ds_fname,
          unsigned &enc_dim,
          std::vector<uint64_t> &raw_data,
          int n_blocks) {

  std::vector<uint64_t> block;
  uint64_t tot_blocks = 0;
  uint64_t block_size = 0;
  unsigned point_cnt;

  enc_dim = 0;
  File file(ds_fname, File::ReadOnly);
  {
    // we get the dataset
    DataSet dataset = file.getDataSet(bch::billion_level::RAW_DATASET_NBLOCKS_NAME);
    dataset.read(tot_blocks);
    // default: reading 100%
    if (n_blocks == -1) {
      n_blocks = tot_blocks ;/// 10;
    }
  }

  {
    // we get the dataset
    DataSet dataset = file.getDataSet(bch::billion_level::RAW_DATASET_BLOCKSIZE_NAME);
    dataset.read(block_size);
  }

  {
    auto nb = std::min(uint64_t(n_blocks), tot_blocks);
    raw_data.reserve(block_size * nb * enc_dim);/// allocate space


    {
      spdlog::info("Start to load raw data ...");
      //boost::progress_display progress(nb);
      for (int i = 0; i < nb; ++i) {
        // we get the dataset
        const std::string ds_name = fmt::format("{}{}", bch::billion_level::RAW_DATASET_PREFIX_NAME, i + 1);
        spdlog::info("Reading {} which is {}/{} blocks", ds_name, i + 1, nb);
        DataSet dataset = file.getDataSet(ds_name);
        auto dims = dataset.getDimensions();
        /// make sure the data is 2d
        assert(dims.size() == 2);
        /// number of points
        point_cnt = dims.front();
        assert(point_cnt <= block_size);
        if (enc_dim == 0) enc_dim = dims.back();
        else
          assert(enc_dim == dims.back());// make sure dimensions consistent
        // we convert the hdf5 dataset to a single dimension vector
        // Here we allocate just enough memory!
        block.resize(point_cnt * enc_dim);// allocate space
        dataset.read(&block[0]);
        raw_data.insert(raw_data.end(), block.begin(), block.end());
        block.clear();
        //++ progress;
      }
    }
  }
}

/// \brief  PrepareDirectories
///
/// !!Make sure all directories (for million-level data sets) do exist!!
///
void PrepareDirectories() {
  fs::path dp(bch::DATA_PATH);
  assert(exists(dp) && is_directory(dp));
  fs::path rp(bch::RESULT_PATH);
  if (!fs::exists(rp)) {
    fs::create_directories(rp);
  } else {
    assert(fs::is_directory(rp));
  }
  fs::path ip(std::string(bch::RESULT_PATH) + bch::INDEX_PATH_REL);
  if (!fs::exists(ip)) {
    fs::create_directories(ip);
  } else {
    assert(fs::is_directory(ip));
  }
}

/// \brief PrepareBLDirectories
///
/// !!Make sure all directories (for billion-level data sets) do exist!!
///
void PrepareBLDirectories() {
  fs::path dp(bch::billion_level::DATA_PATH);
  assert(exists(dp) && is_directory(dp));
  fs::path rp(bch::billion_level::RESULT_PATH);
  if (!fs::exists(rp)) {
    fs::create_directories(rp);
  } else {
    assert(fs::is_directory(rp));
  }
  fs::path ip(std::string(bch::billion_level::RESULT_PATH) + bch::billion_level::INDEX_PATH_REL);
  if (!fs::exists(ip)) {
    fs::create_directories(ip);
  } else {
    assert(fs::is_directory(ip));
  }
}

/// \brief DataSetFullnames
///
/// Get the data set file names (storing in a std::vector<std::string> where
/// the order is the same as the user specificied)
///
std::vector<std::string> DataSetFullnames() {
  char *pch;
  char str[1024];
  strcpy(str, bch::DATA_SETS);

  std::string dp(bch::DATA_PATH);
  std::vector<std::string> data_sets;
  pch = strtok(str, ";");
  while (pch != nullptr) {
    data_sets.push_back(dp + pch);
    fs::path tmp_p(data_sets.back());
    assert(fs::exists(tmp_p));
    pch = strtok(nullptr, ";");
  }

  return data_sets;
}

/// \brief DataSetFullnamesBL
///
/// Get the billion-level data set file names (storing in a std::vector<std::string> where
/// the order is the same as the user specificied)
///
std::vector<std::string> DataSetFullnamesBL() {
  char *pch;
  char str[1024];
  strcpy(str, bch::billion_level::DATA_SETS);

  std::string dp(bch::billion_level::DATA_PATH);
  std::vector<std::string> data_sets;
  pch = strtok(str, ";");
  while (pch != nullptr) {
    data_sets.push_back(dp + pch);
    fs::path tmp_p(data_sets.back());
    assert(fs::exists(tmp_p));
    pch = strtok(nullptr, ";");
  }

  return data_sets;
}

/// save results
void save_results_bl(const std::vector<std::vector<res_pair_raw<unsigned >>> &gnd,
                     std::string ds,
                     unsigned k) {
  fs::path dp(ds);
  std::string ofname(bch::billion_level::RESULT_PATH);
  std::string ofname_i(bch::billion_level::RESULT_PATH);
  ofname += dp.stem().c_str();
  ofname_i += dp.stem().c_str();
  ofname += "_ground_truth_dist.txt";
  ofname_i += "_ground_truth_ind.txt";
  spdlog::info("Create results file: {} and {}", ofname, ofname_i);
  std::ofstream ofp(ofname, std::ios::out);
  std::ofstream ofp_i(ofname_i, std::ios::out);
  if (ofp.bad()) spdlog::error("Create {} failed", ofname);
  if (ofp_i.bad()) spdlog::error("Create {} failed", ofname_i);
  ofp << fmt::format("K: {}, ds: {}\n", k, ds);
  ofp_i << fmt::format("K: {}, ds: {}\n", k, ds);
  for (const auto &gr : gnd) {
    for (const auto &ge : gr) {
      ofp << ge.dist << " ";
      ofp_i << ge.id << " ";
    }
    ofp << "\n";
    ofp_i << "\n";
  }
  ofp.close();
  ofp_i.close();
}

using namespace ss::ann::brute_force::flat_vector;
///
/// \brief benchmark_brute_force
///
/// \param raw_data                     raw data (or data base)
/// \param queries                      query points
/// \param gnd                          ground truth
/// \param enc_dim                      encoded dimension
/// \param os                           output stream
/// \param k                            number of results required
/// \param perform_correctness          whether or not to perform correctness check
/// \param save_results                 whether or not to save the (ground truth) results
/// \param ds                           data set name (only required when saving results is enabled)
void benchmark_brute_force(const std::vector<uint64_t> &raw_data,
                           const std::vector<uint64_t> &queries,
                           std::vector<std::vector<unsigned>> &gnd,
                           unsigned enc_dim,
                           std::ostream &os,
                           unsigned k = 1,
                           bool perform_correctness = true,
                           bool save_results = false,
                           std::string ds = "") {

  spdlog::info("{} starting ...", __func__);

  BruteForceHamming bf(enc_dim, raw_data);
  boost::timer timer;

  unsigned q = queries.size() / enc_dim;

  double exe_time = -1;
  unsigned idx = 0;
  spdlog::info("Query starts (with k = {})...", k);

  if (k == 1) {
    /// 1nn query
    std::vector<res_pair_raw<unsigned>> results(q);
    timer.restart();
    {
      boost::progress_display progress(q);
      for (int qi = 0; qi < q; ++qi) {
        results[idx] = bf.query(&queries[qi * enc_dim]);
        ++idx;
        ++progress;
      }
    }
    exe_time = timer.elapsed();
    spdlog::info("\n\nQuery finished, elapsed time: {} s", exe_time);
    os << fmt::format("{},{},{:.6f}\n", alg_name<decltype(bf)>(), k, exe_time);
    /// correctness check
    if (perform_correctness) {
      spdlog::info("\nPerform correctness check ...");
      // correctness check
      for (idx = 0; idx < q; ++idx) {
        if (gnd[idx][0] != results[idx].dist) {
          spdlog::error("query {}/{} got wrong answer: Expected : {}, Got : {}",
                        idx + 1,
                        k,
                        gnd[idx][0],
                        results[idx].dist);
        }
      }
    }
  } else {
    /// knn query
    std::vector<std::vector<res_pair_raw<unsigned>>> results(q);
    idx = 0;
    timer.restart();
    {
      boost::progress_display progress(q);
      for (int qi = 0; qi < q; ++qi) {
        bf.query(&queries[qi * enc_dim], k, results[idx]);
        ++idx;
        ++progress;
      }
    }
    exe_time = timer.elapsed();
    spdlog::info("\n\nQuery finished, elapsed time: {} s", exe_time);
    os << fmt::format("{},{},{:.6f}\n", alg_name<decltype(bf)>(), k, exe_time);

    /// sort results
    for (auto &res_: results) std::sort(res_.begin(), res_.end());

    /// correctness check
    if (perform_correctness) {
      spdlog::info("\nPerform correctness check ...");
      // correctness check
      for (idx = 0; idx < q; ++idx) {
        for (int i = 0; i < k; ++i) {
          if (gnd[idx][i] != results[idx][i].dist) {
            spdlog::error("query {}/{} got wrong answer: Expected : {}, Got : {}",
                          idx,
                          i,
                          gnd[idx][i],
                          results[idx][i].dist);
          }
        }
      }
    }

    /// save results
    if (save_results) {
      assert(!ds.empty());
      save_results_bl(results, ds, k);
      gnd.resize(q);
      unsigned qi = 0;
      for (auto &res_: results) {
        gnd[qi].resize(k, 0);
        for(int i = 0;i < k;++ i) gnd[qi][i] = res_[i].dist;
      }
    }
  }
}

/// \brief million_level_datasets_bf
///
/// !! benchmark brute force for million-scale datasets !!
///
void million_level_datasets_bf() {
  {
    // prepare directories
    PrepareDirectories();
    // get data set files
    auto data_sets = DataSetFullnames();

    spdlog::info("data sets: {}", stringtify(data_sets.begin(), data_sets.end()));

    std::vector<uint64_t> raw_data;
    std::vector<uint64_t> queries;
    std::vector<std::vector<unsigned>> ground_truth;

    unsigned K;
    unsigned enc_dim;
    unsigned point_cnt;

    for (const auto &ds : data_sets) {
      spdlog::info("Processing data set: {}", ds);

      // we create a new hdf5 (read-only) file object
      File file(ds, File::ReadOnly);
      {
        // we get the dataset
        DataSet dataset = file.getDataSet(bch::RAW_DATASET_NAME);

        auto dims = dataset.getDimensions();
        assert(dims.size() == 2); // make sure it is 2d data
        point_cnt = dims.front();
        enc_dim = dims.back();
        spdlog::info("{}'s info: #points = {}, #dim={}", ds, point_cnt, enc_dim * (sizeof(uint64_t) << 3u));
        raw_data.resize(point_cnt * enc_dim, 0u);
        // we convert the hdf5 dataset to a single dimension vector
        dataset.read(&raw_data[0]);
      }

      {
        // we get the dataset
        DataSet dataset = file.getDataSet(bch::QUERY_DATASET_NAME);

        auto dims = dataset.getDimensions();
        assert(dims.size() == 2); // make sure it is 2d data
        point_cnt = dims.front();
        assert(dims.back() == enc_dim); // make sure query data matches raw data's dimension
        queries.resize(point_cnt * enc_dim);// MUST allocate memory first
        // we convert the hdf5 dataset to a single dimension vector
        dataset.read(&queries[0]);
      }

      {
        // we get the dataset
        DataSet dataset = file.getDataSet(bch::GND_DIST_DATASET_NAME);
        dataset.read(ground_truth);
      }

      {
        unsigned word_size = 0;
        // we get the dataset
        DataSet dataset = file.getDataSet(bch::COMPACT_WORDSIZE_NAME);
        // we convert the hdf5 dataset to a single dimension vector
        dataset.read(word_size);
        assert(word_size == (sizeof(uint64_t) << 3u));
      }

      {
        // we get the dataset
        DataSet dataset = file.getDataSet(bch::KNN_K_NAME);
        // we convert the hdf5 dataset to a single dimension vector
        dataset.read(K);

        assert(K == ground_truth.front().size());
      }

      {// show some info of the data set
        spdlog::info("Raw data sets");
        head(raw_data, enc_dim, 5);
        spdlog::info("Query data sets");
        head(queries, enc_dim, 5);
        spdlog::info("Ground truth");
        head(ground_truth, 5);
        spdlog::info("K = {}", K);
      }

      fs::path dp(ds);
      std::string ofname(bch::RESULT_PATH);
      ofname += dp.stem().c_str();
      ofname += ".csv";
      spdlog::info("Create results file: {}", ofname);
      std::ofstream ofp(ofname, std::ios::out);
      if (ofp.bad()) spdlog::error("Create {} failed", ofname);

      /// write header
      ofp << "algorithm,k,query_time\n";
      {// 1NN
        unsigned k = 1;
        /// bench 1nn
        benchmark_brute_force(raw_data, queries, ground_truth, enc_dim, ofp, k);
      }

      {// 10NN
        unsigned k = 10;
        /// bench 10nn
        benchmark_brute_force(raw_data, queries, ground_truth, enc_dim, ofp, k);
      }

      {// 100NN
        unsigned k = K;
        /// bench 100nn
        benchmark_brute_force(raw_data, queries, ground_truth, enc_dim, ofp, k);
      }

      ofp.close();
    }
  }
}

/// \brief billion_level_datasets_bf
///
/// !! benchmark brute force for billion-scale datasets !!
///
void billion_level_datasets_bf() {
  {
    spdlog::info("Start billion-level experiments ...");
    PrepareBLDirectories();
    auto data_sets = DataSetFullnamesBL();

    spdlog::info("Data sets: {}", stringtify(data_sets.begin(), data_sets.end()));

    std::vector<uint64_t> raw_data;
    std::vector<uint64_t> queries;
    std::vector<std::vector<unsigned>> ground_truth;
    unsigned K = 100;
    unsigned enc_dim;
    unsigned  point_cnt;

    for (const auto &ds : data_sets) {
      spdlog::info("Processing data set: {}", ds);


      // we create a new hdf5 file
      File file(ds, File::ReadOnly);
      {
        if (ds.find("sift") == std::string::npos) load(ds, enc_dim, raw_data, -1);// gist
        else load(ds, enc_dim, raw_data, 10);
      }

      {
        // we get the dataset
        DataSet dataset = file.getDataSet(bch::billion_level::QUERY_DATASET_NAME);
        auto dims = dataset.getDimensions();
        point_cnt = dims.front();
        assert(enc_dim == dims.back());
        queries.resize(enc_dim * point_cnt, 0u);
        // we convert the hdf5 dataset to a single dimension vector
        dataset.read(&queries[0]);
      }


      {// show some info of the data set
        head(raw_data, enc_dim, 5);/// fixed
        head(queries, enc_dim, 5);
      }

      fs::path dp(ds);
      std::string ofname(bch::billion_level::RESULT_PATH);
      ofname += dp.stem().c_str();
      ofname += ".csv";
      spdlog::info("Create results file: {}", ofname);
      std::ofstream ofp(ofname, std::ios::out);
      if (ofp.bad()) spdlog::error("Create {} failed", ofname);

      /// write header
      ofp << "algorithm,k,query_time\n";
      {// 100NN
        unsigned k = K;
        benchmark_brute_force(raw_data, queries, ground_truth, enc_dim, ofp, k, false, true, ds);
      }
      ofp.close();
    }
  }
}

/// main
int main() {
  //million_level_datasets_bf();
  billion_level_datasets_bf();
}

