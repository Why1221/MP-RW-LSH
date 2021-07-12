#include <boost/timer.hpp>
#include <boost/filesystem.hpp>
#include <boost/progress.hpp>
#include <fmt/format.h>
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

using namespace ss::io;
using namespace ss::ann::srs;
using namespace HighFive;

namespace fs = boost::filesystem;
namespace bch = bench_config::hamming;


void head_hamming(const std::vector<std::vector<uint64_t>> &data, unsigned line = 5u) {
  unsigned n = std::min(line, (unsigned) data.size());
  std::string sep(80, '-');
  spdlog::info(sep);
  for (unsigned i = 0; i < n; ++i) {
    spdlog::info("{} : {}", i, stringtify(data[i].begin(), data[i].end(), "{:064b}"));
  }
}

void head(const std::vector<std::vector<unsigned>> &data, unsigned line = 5u) {
  unsigned n = std::min(line, (unsigned) data.size());
  std::string sep(80, '-');
  spdlog::info(sep);
  for (unsigned i = 0; i < n; ++i) {
    spdlog::info("{} : {}", i, stringtify(data[i].begin(), data[i].end()));
  }
}

void load_from_billion_level_datasets(const std::string &ds_fname,
                                      std::vector<std::vector<uint64_t>> &raw_data,
                                      int n_blocks = -1) {

  std::vector<std::vector<uint64_t>> block;

  uint64_t tot_blocks = 0;
  uint64_t block_size = 0;

  File file(ds_fname, File::ReadOnly);

  {
    // we get the dataset
    DataSet dataset = file.getDataSet(bch::billion_level::RAW_DATASET_NBLOCKS_NAME);
    dataset.read(tot_blocks);
    if (n_blocks == -1) {
      // default reading 1/10
      n_blocks = tot_blocks / 10;
//      n_blocks = tot_blocks;
    }
  }

  {
    // we get the dataset
    DataSet dataset = file.getDataSet(bch::billion_level::RAW_DATASET_BLOCKSIZE_NAME);
    dataset.read(block_size);
  }

  {
    auto nb = std::min(uint64_t(n_blocks), tot_blocks);
    raw_data.reserve(block_size * nb);
    for (int i = 0; i < nb; ++i) {
      // we get the dataset
      DataSet dataset = file.getDataSet(fmt::format("{}{}", bch::billion_level::RAW_DATASET_PREFIX_NAME, i + 1));
      // we convert the hdf5 dataset to a single dimension vector
      dataset.read(block);
      raw_data.insert(raw_data.end(), block.begin(), block.end());
      block.clear();
    }
  }
}

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

void save_results_bl(const std::vector<std::vector<res_pair_raw<unsigned >>> &gnd,
                     std::string ds, unsigned k) {
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
using namespace ss::ann::brute_force;
void benchmark_brute_force(const std::vector<std::vector<uint64_t>> &raw_data,
                           const std::vector<std::vector<uint64_t>> &queries,
                           const std::vector<std::vector<unsigned>> &gnd,
                           std::ostream &os,
                           unsigned k = 1,
                           bool perform_correctness = true,
                           bool save_results = false,
                           std::string ds = "") {

  spdlog::info("{} starting ...", __func__);

  os << "k,query_time\n";

  BruteForceHamming bf(raw_data);
  boost::timer timer;

  unsigned q = queries.size();

  unsigned idx = 0;
  spdlog::info("Query starts (with k = {})...", k);

  if (k == 1) {
    std::vector<res_pair_raw<unsigned>> results(q);
    timer.restart();
    {
      boost::progress_display progress(q);
      for (const auto &query: queries) {
        results[idx] = bf.query(&query[0]);
        ++idx;
        ++progress;
      }
    }
    spdlog::info("\n\nQuery finished, elapsed time: {} s", timer.elapsed());
    os << fmt::format("{}, {:.6f}\n", k, timer.elapsed());

    if (perform_correctness) {
      spdlog::info("\nPerform correctness check ...");
      // correctness check
      for (idx = 0; idx < q; ++idx) {
        if (gnd[idx][0] != results[idx].dist) {
          spdlog::error("query {}/{} got wrong answer: Expected : {}, Got : {}",
                        idx,
                        k,
                        gnd[idx][0],
                        results[idx].dist);
        }
      }
    }

  } else {
    std::vector<std::vector<res_pair_raw<unsigned>>> results(q);
    idx = 0;
    timer.restart();
    {
      boost::progress_display progress(q);
      for (const auto &query: queries) {
        bf.query(&query[0], k, results[idx]);
        ++idx;
        ++progress;
      }
    }
    spdlog::info("\n\nQuery finished, elapsed time: {} s", timer.elapsed());
    os << fmt::format("{}, {:.6f}\n", k, timer.elapsed());

    for (auto &res_: results) std::sort(res_.begin(), res_.end());

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

    if (save_results) {
      assert(!ds.empty());
      save_results_bl(results, ds, k);
    }
  }
}

void million_level_datasets_bf() {
  {
    //
    PrepareDirectories();
    //
    auto data_sets = DataSetFullnames();
    for (const auto &ds : data_sets) {
      spdlog::info("{}", ds);
    }

    std::vector<std::vector<uint64_t >> raw_data;
    std::vector<std::vector<uint64_t >> queries;
    std::vector<std::vector<unsigned>> ground_truth;
    unsigned K;

    for (const auto &ds : data_sets) {
      spdlog::info("Processing data set: {}", ds);
      // we create a new hdf5 file
      File file(ds, File::ReadOnly);
      {
        // we get the dataset
        DataSet dataset = file.getDataSet(bch::RAW_DATASET_NAME);
        // we convert the hdf5 dataset to a single dimension vector
        dataset.read(raw_data);
      }

      {
        // we get the dataset
        DataSet dataset = file.getDataSet(bch::QUERY_DATASET_NAME);
        // we convert the hdf5 dataset to a single dimension vector
        dataset.read(queries);
      }

      {
        // we get the dataset
        DataSet dataset = file.getDataSet(bch::GND_DIST_DATASET_NAME);
        // we convert the hdf5 dataset to a single dimension vector
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
      }

      {// show some info of the data set
        spdlog::info("Raw data sets");
        head_hamming(raw_data);
        spdlog::info("Query data sets");
        head_hamming(queries);
        spdlog::info("Ground truth");
        head(ground_truth);

        spdlog::info("K = {}", K);
      }

      fs::path dp(ds);
      std::string ofname(bch::RESULT_PATH);
      ofname += dp.stem().c_str();
      ofname += ".csv";
      spdlog::info("Create results file: {}", ofname);
      std::ofstream ofp(ofname, std::ios::out);
      if (ofp.bad()) spdlog::error("Create {} failed", ofname);

      {// 1NN
        unsigned k = 1;
        benchmark_brute_force(raw_data, queries, ground_truth, ofp, k);
      }

      {// 10NN
        unsigned k = 10;
        benchmark_brute_force(raw_data, queries, ground_truth, ofp, k);
      }

      {// 100NN
        unsigned k = K;
        benchmark_brute_force(raw_data, queries, ground_truth, ofp, k);
      }
      ofp.close();
    }
  }
}

void billion_level_datasets_bf() {
  {
    spdlog::info("Start billion-level experiments ...");
    //
    PrepareBLDirectories();
    //
    auto data_sets = DataSetFullnamesBL();
    for (const auto &ds : data_sets) {
      spdlog::info("{}", ds);
    }

    std::vector<std::vector<uint64_t >> raw_data;
    std::vector<std::vector<uint64_t >> queries;
    std::vector<std::vector<unsigned>> ground_truth;
    unsigned K = 100;

    for (const auto &ds : data_sets) {
      spdlog::info("Processing data set: {}", ds);
      // we create a new hdf5 file
      File file(ds, File::ReadOnly);
      {
        load_from_billion_level_datasets(ds, raw_data);
      }

      {
        // we get the dataset
        DataSet dataset = file.getDataSet(bch::billion_level::QUERY_DATASET_NAME);
        // we convert the hdf5 dataset to a single dimension vector
        dataset.read(queries);
      }

//      {
//        // we get the dataset
//        DataSet dataset = file.getDataSet(bch::KNN_K_NAME);
//        // we convert the hdf5 dataset to a single dimension vector
//        dataset.read(K);
//      }

      {// show some info of the data set
        spdlog::info("Raw data sets: {} X {} ({} X {})",
            raw_data.size(), raw_data.front().size(),
            raw_data.capacity(), raw_data.front().capacity());
        head_hamming(raw_data);
        spdlog::info("Query data sets: {} X {} ({} X {})",
            queries.size(), queries.front().size(),
            queries.capacity(), queries.front().capacity());
        head_hamming(queries);
//        spdlog::info("Ground truth");
//        head(ground_truth);
//
//        spdlog::info("K = {}", K);
      }

      fs::path dp(ds);
      std::string ofname(bch::billion_level::RESULT_PATH);
      ofname += dp.stem().c_str();
      ofname += ".csv";
      spdlog::info("Create results file: {}", ofname);
      std::ofstream ofp(ofname, std::ios::out);
      if (ofp.bad()) spdlog::error("Create {} failed", ofname);

//      {// 1NN
//        unsigned k = 1;
//        benchmark_brute_force(raw_data, queries, ground_truth, ofp, k);
//      }
//
//      {// 10NN
//        unsigned k = 10;
//        benchmark_brute_force(raw_data, queries, ground_truth, ofp, k);
//      }

      {// 100NN
        unsigned k = K;
        benchmark_brute_force(raw_data, queries, ground_truth, ofp, k, false, true, ds);
      }
      ofp.close();
    }
  }
}

int main() {
  billion_level_datasets_bf();
}