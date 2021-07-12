
#ifndef _SPBENCHMARKHELPER_H_
#define _SPBENCHMARKHELPER_H_

#include <fstream>
#include <fmt/format.h>
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

namespace ss::ann::bench_helper {
using namespace ss::io;
using namespace HighFive;

namespace fs = boost::filesystem;
namespace bch = bench_config::hamming;

constexpr char SS_SEP[] = "---------------------------------------------------------------------------";

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
template<typename T>
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
template<typename T>
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
      n_blocks = tot_blocks;/// 10;
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
template<typename ResPair>
void save_results_bl(const std::vector<std::vector<ResPair>> &gnd,
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

template<typename ResPair>
void save_results_bl(const std::vector<ResPair> &gnd,
                     std::string ds) {
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
  ofp << fmt::format("K: {}, ds: {}\n", 1, ds);
  ofp_i << fmt::format("K: {}, ds: {}\n", 1, ds);
  for (const auto &gr : gnd) {
//    for (const auto &ge : gr) {
    ofp << gr.dist << " ";
    ofp_i << gr.id << " ";
//    }
    ofp << "\n";
    ofp_i << "\n";
  }
  ofp.close();
  ofp_i.close();
}
}// end namespace ss::ann::ben_helper

#endif //_SPBENCHMARKHELPER_H_
