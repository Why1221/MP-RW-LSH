#include "SpBenchmarkHelper.h"

#include <srs_utils.hpp>
#include <brute_force_sp.hpp>
#include <ann_sp.hpp>

#include <thread>
using namespace ss::ann::flat_vector;
using namespace ss::ann::brute_force::flat_vector;
using namespace ss::ann::bench_helper;
using namespace std::chrono_literals;

template<typename T>
inline constexpr std::string_view alg_name() {
  typedef typename std::remove_cv_t<T> Tnocv;
  typedef ss::ann::brute_force::flat_vector::BruteForceHamming BF;
  if constexpr (std::is_same_v<Tnocv, BF>)
    return "brute-force";
  else if constexpr (std::is_same_v<Tnocv, SRSCoverTreeHamming>)
    return "SRS-CoverTree";
  else if constexpr (std::is_same_v<Tnocv, SRSKDTreeHamming>)
    return "SRS-KDTree";
  else if constexpr (std::is_same_v<Tnocv, TOWCoverTreeHamming>)
    return "TOW-CoverTree";
  else if constexpr (std::is_same_v<Tnocv, TOWKDTreeHamming>)
    return "TOW-KDTree";
  else
    return "unknown";
}

struct Result {
  std::string algorithm;
  int m{};
  long long t{};
  size_t index_size{};
  double query_time{}, construction_time{};
  std::vector<std::pair<float /* approximation factor */, float /* recall */>> recalls;
  Result() : algorithm(""),
             m(-1),
             t(-1),
             index_size(0),
             query_time(-1),
             construction_time{-1},
             recalls() {

  }
  static std::string header() {
    return "algorithm,m,t,index_size,construction_time,query_time,recalls";
  }
  std::string to_string() const {
    std::string s;
    for (size_t k = 0; k < recalls.size(); ++k) {
      s += fmt::format("{:.1f};{:.4f}", recalls[k].first, recalls[k].second);
      if (k != recalls.size() - 1)
        s += ";";
    }
    return fmt::format("{4},{0},{1},{5},{2:.6f},{3:.4f},", m, t, construction_time, query_time, algorithm, index_size) + s;
  }
};

///
/// \brief benchmark
///
/// benchmarks for various algorithms
///
/// \tparam AlgorithmType           type for the algorithm
///
/// \param raw_data                 raw data (1d vector)
/// \param queries                  query data
/// \param gnd                      ground truth
/// \param enc_dim                  dimenison after encoding
/// \param diff_m                   different m (projected dimension) to be benchmarked
/// \param seed                     random seed
/// \param index_path               index path (to store indices)
/// \param diff_c                   different c (approximation factor)
/// \param diff_t                   different t (# of candidates)
/// \param os                       output stream
/// \param time_out                 time out threshold (for early termination)
/// \param recall_thres             recall threshold (for early termination)
template<typename AlgorithmType>
void benchmark(const std::vector<uint64_t> &raw_data,
               const std::vector<uint64_t> &queries,
               const std::vector<std::vector<unsigned>> &gnd,
               const unsigned enc_dim,
               const std::vector<unsigned> &diff_m,
               const unsigned seed,
               const std::string &index_path,
               const std::vector<float> &diff_c,
               const std::vector<size_t> &diff_t,
               std::ostream &os,
               double time_out = -1,
               float recall_thres = -1) {

  spdlog::info("Benchmarking {}", alg_name<AlgorithmType>());
  boost::timer timer;

  unsigned n = raw_data.size() / enc_dim;
  unsigned nq = queries.size() / enc_dim;
  spdlog::info("data set size: {}, query size: {}, dimension: {}", n, nq, enc_dim * (sizeof(uint64_t) << 3u));

  assert(gnd.size() == nq);
  constexpr unsigned INIT_DIST = std::numeric_limits<unsigned>::max();
  std::vector<unsigned> result_distances(nq);

  Result bench_res;
  bench_res.recalls.resize(diff_c.size());

  bench_res.algorithm = alg_name<AlgorithmType>();
  for (const auto &m : diff_m) {
    spdlog::info("Perform benchmark for m = {}", m);
    bench_res.m = m;
    spdlog::info(SS_SEP);
    {
      // raw_data, enc_dim, m, index_path, seed
      /// create algorithm object
      AlgorithmType alg(raw_data, enc_dim, m, index_path, seed);

      spdlog::info("Start to construct index ...");
      timer.restart();
      {
        alg.build_index();
      }
      spdlog::info("Index construction took {:.6f} s", timer.elapsed());
      bench_res.construction_time = timer.elapsed();
      bench_res.index_size = alg.index_size();

      spdlog::info("Start to perform query ...");
      for (auto t : diff_t) {
        spdlog::info("RUNNING QUERIES (with #Candidates_{})...", t);
        std::fill(result_distances.begin(), result_distances.end(), INIT_DIST);
        bench_res.t = t;

        unsigned q_idx = 0;
        timer.restart();
        {
          boost::progress_display progress(nq);
          for (; q_idx < nq; ++q_idx) {
            auto res_q = alg.query(&queries[q_idx * enc_dim], t);
            result_distances[q_idx] = res_q.dist;
            ++progress;
          }
        }
        spdlog::info("QUERY TIME: {}", timer.elapsed());
        bench_res.query_time = timer.elapsed() / nq * 1000.0;/// convert to ms

        /// calculate recalls
        //// reset recalls
        for(auto& recall: bench_res.recalls) {
          recall.first = 0;
          recall.second = 0;
        }
        q_idx = 0;
        spdlog::info(SS_SEP);
        for (const auto &dist : result_distances) {
          for (size_t c_idx = 0; c_idx < diff_c.size(); ++c_idx) {
            auto c = diff_c[c_idx];
            if (dist <= c * gnd[q_idx].front()) {
              ++ bench_res.recalls[c_idx].second;
            }
          }
          spdlog::info("query_id: {0}, expected: {1}, got: {2} : ###{0},{1},{2}###", q_idx, gnd[q_idx].front(), dist);
          ++q_idx;
        }
        spdlog::info(SS_SEP);

        float min_recall = 2.0;
        for (size_t c_idx = 0; c_idx < diff_c.size(); ++c_idx) {
          auto c = diff_c[c_idx];
          bench_res.recalls[c_idx].first = c;
          bench_res.recalls[c_idx].second /= nq;
          if (bench_res.recalls[c_idx].second < min_recall) min_recall = bench_res.recalls[c_idx].second;
        }

        auto s = bench_res.to_string();
        os << s << "\n";
        spdlog::info(SS_SEP);
        spdlog::info(s);
        spdlog::info(SS_SEP);

        if (recall_thres > 0 && min_recall > recall_thres) {
          spdlog::info("Minimum recall ({}) is larger than the threshold ({}). Trigger early stop!",
                       min_recall, recall_thres);
          break;
        }
        if (time_out > 0 && bench_res.query_time > time_out) {
          spdlog::info("Query time ({}) exceeds time out threshold ({}). Trigger early stop!",
                       bench_res.query_time, time_out);
          break;
        }
      }
    }
    spdlog::info(SS_SEP);
    spdlog::info("Sleep for 1 minute before next benchmark");
    std::this_thread::sleep_for(60s);
  }

}
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
///
/// \return excution time (in seconds)
double benchmark_brute_force(const std::vector<uint64_t> &raw_data,
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
    /// save results
    if (save_results) {
      assert(!ds.empty());
      save_results_bl(results, ds);
      gnd.resize(q);
      unsigned qi = 0;
      for (auto &res_: results) {
        gnd[qi].resize(k, 0);
        gnd[qi][0] = res_.dist;
        ++qi;/// pay attenttion
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
    os << fmt::format("{},{},{:.6f}\n", alg_name<decltype(bf)>(), k, exe_time / q * 1000.0);

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
        for (int i = 0; i < k; ++i) gnd[qi][i] = res_[i].dist;
        ++ qi;/// pay attenttion
      }
    }
  }

  return exe_time;
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

    unsigned K = 0;
    unsigned enc_dim;
    unsigned point_cnt;
    unsigned nq;
    unsigned seed = 20190430u;//std::chrono::system_clock::now().time_since_epoch().count();

    spdlog::info("Random seed : {}", seed);

    constexpr float recall_thres = 0.99;
    const std::vector<float> diff_c{1.5, 2.0, 3.0};
    const std::vector<size_t> diff_t
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90, 100, 200, 300, 400,
         500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000};
    const std::vector<unsigned> diff_m{8, 10, 12, 6, 16, 4};

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
        nq = point_cnt;
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
      std::string ofname_bf(bch::RESULT_PATH);
      ofname += dp.stem().c_str();
      ofname_bf += dp.stem().c_str();
      ofname += ".csv";
      ofname_bf += "_bf.csv";
      spdlog::info("Create results file: {} and {}", ofname, ofname_bf);
      std::ofstream ofp(ofname, std::ios::out);
      std::ofstream ofp_bf(ofname_bf, std::ios::out);
      if (ofp.bad()) spdlog::error("Create {} failed", ofname);
      if (ofp_bf.bad()) spdlog::error("Create {} failed", ofname_bf);
      /// write header
      ofp_bf << "algorithm,k,query_time\n";
      ofp << Result::header() << "\n";

      {// 1NN
        unsigned k = 1;
        /// bench 1nn
        double time_out = 0;
        time_out = benchmark_brute_force(raw_data, queries, ground_truth, enc_dim, ofp_bf, k);
        ofp_bf.close();
        time_out = time_out / nq * 1000.0;

        spdlog::info(SS_SEP);
        spdlog::info("time out thres: {:.6f}, recall thres: {:.4f}", time_out, recall_thres);
        std::string
            index_path = std::string(bench_config::hamming::RESULT_PATH) + bench_config::hamming::INDEX_PATH_REL;

        spdlog::info("Index will be stored in {}", index_path);
        std::this_thread::sleep_for(10s);
        spdlog::info(SS_SEP);

        benchmark<TOWCoverTreeHamming>(raw_data, queries, ground_truth, enc_dim, diff_m, seed, index_path,
                                       diff_c, diff_t, ofp, time_out, recall_thres);
        spdlog::info(SS_SEP);
        std::this_thread::sleep_for(10s);

        benchmark<SRSKDTreeHamming>(raw_data, queries, ground_truth, enc_dim, diff_m, seed, index_path,
                                    diff_c, diff_t, ofp, time_out, recall_thres);
        spdlog::info(SS_SEP);
        std::this_thread::sleep_for(10s);
        benchmark<TOWKDTreeHamming>(raw_data, queries, ground_truth, enc_dim, diff_m, seed, index_path,
                                    diff_c, diff_t, ofp, time_out, recall_thres);
        spdlog::info(SS_SEP);
        std::this_thread::sleep_for(10s);
        benchmark<SRSCoverTreeHamming>(raw_data, queries, ground_truth, enc_dim, diff_m, seed, index_path,
                                       diff_c, diff_t, ofp, time_out, recall_thres);

      }
      ofp.close();
    }
  }
}

/// \brief billion_level_datasets_bf
///
/// !! benchmark brute force for billion-scale datasets !!
///
//void billion_level_datasets_bf(int nb = -1, bool ignore_gist = false) {
void billion_level_datasets_bf(unsigned m, int nb = -1, bool ignore_gist = false) {
  {
    spdlog::info("Start billion-level experiments (# of blocks: {})...", nb);
    PrepareBLDirectories();
    auto data_sets = DataSetFullnamesBL();

    spdlog::info("Data sets: {}", stringtify(data_sets.begin(), data_sets.end()));

    std::vector<uint64_t> raw_data;
    std::vector<uint64_t> queries;
    std::vector<std::vector<unsigned>> ground_truth;
    unsigned K = 100;
    unsigned enc_dim = 0;
    unsigned point_cnt;
    unsigned nq;
    unsigned seed = 20190431u;//std::chrono::system_clock::now().time_since_epoch().count();

    spdlog::info("Random seed : {}", seed);

    constexpr float recall_thres = 0.99;
    const std::vector<float> diff_c{1.5, 2.0, 3.0};
    const std::vector<size_t> diff_t
        {1, 2, 4, 8, 16, 32, 64, 128, 256, 400, 512, 800, 1024, 1500, 2048, 3000, 4096,
         6000, 8192, 10000, 12000, 14000, 16000, 18000, 20000, 24000, 26000, 28000, 30000,
         40000};
    //const std::vector<unsigned> diff_m{8, 10, 12, 6, 16, 4};
    const std::vector<unsigned> diff_m{m};

    for (const auto &ds : data_sets) {
      spdlog::info("Processing data set: {}", ds);

      // we create a new hdf5 file
      File file(ds, File::ReadOnly);
      {
        if (ds.find("sift") == std::string::npos) {
          if (ignore_gist) continue;//
          load(ds, enc_dim, raw_data, -1);// gist
        } else load(ds, enc_dim, raw_data, nb);
      }

      {
        // we get the dataset
        DataSet dataset = file.getDataSet(bch::billion_level::QUERY_DATASET_NAME);
        auto dims = dataset.getDimensions();
        point_cnt = dims.front();
        nq = point_cnt;
        assert(enc_dim == dims.back());
        queries.resize(enc_dim * point_cnt, 0u);
        // we convert the hdf5 dataset to a single dimension vector
        dataset.read(&queries[0]);
      }

      {// show some info of the data set
        head(raw_data, enc_dim, 5);
        head(queries, enc_dim, 5);
      }

      fs::path dp(ds);
      std::string ofname(bch::billion_level::RESULT_PATH);
      std::string ofname_bf(bch::billion_level::RESULT_PATH);
      std::string ds_ = dp.stem().c_str();
      if (ds.find("sift") != std::string::npos) ds_ += fmt::format("_bs{}", nb);
      ofname += ds_;
      ofname_bf += ds_;
      ofname += ".csv";
      ofname_bf += "_bf.csv";
      spdlog::info("Create results file: {} and {}", ofname, ofname_bf);
      std::ofstream ofp(ofname, std::ios::out);
      std::ofstream ofp_bf(ofname_bf, std::ios::out);
      if (ofp.bad()) spdlog::error("Create {} failed", ofname);
      if (ofp_bf.bad()) spdlog::error("Create {} failed", ofname_bf);
      /// write header
      ofp_bf << "algorithm,k,query_time\n";
      ofp << Result::header() << "\n";

      {// 100NN

        double time_out = 0;
        unsigned k = 1;
        time_out = benchmark_brute_force(raw_data, queries, ground_truth, enc_dim, ofp_bf, k, false, true, ds_);
        time_out = time_out / nq * 1000.0;
        ofp_bf.close();

        spdlog::info(SS_SEP);
        spdlog::info("time out thres: {:.6f}, recall thres: {:.4f}", time_out, recall_thres);

        std::string
            index_path = std::string(bench_config::hamming::RESULT_PATH) + bench_config::hamming::INDEX_PATH_REL;
        spdlog::info("Index will be stored in {}", index_path);
        spdlog::info(SS_SEP);

        std::this_thread::sleep_for(10s);
        benchmark<SRSKDTreeHamming>(raw_data, queries, ground_truth, enc_dim, diff_m, seed, index_path,
                                    diff_c, diff_t, ofp, time_out, recall_thres);
        spdlog::info(SS_SEP);
        std::this_thread::sleep_for(10s);
        benchmark<TOWKDTreeHamming>(raw_data, queries, ground_truth, enc_dim, diff_m, seed, index_path,
                                    diff_c, diff_t, ofp, time_out, recall_thres);
        spdlog::info(SS_SEP);
        std::this_thread::sleep_for(10s);
        benchmark<SRSCoverTreeHamming>(raw_data, queries, ground_truth, enc_dim, diff_m, seed, index_path,
                                       diff_c, diff_t, ofp, time_out, recall_thres);
        spdlog::info(SS_SEP);
        std::this_thread::sleep_for(10s);
        benchmark<TOWCoverTreeHamming>(raw_data, queries, ground_truth, enc_dim, diff_m, seed, index_path,
                                       diff_c, diff_t, ofp, time_out, recall_thres);

      }
      ofp.close();
    }
  }
}

int main() {
  //million_level_datasets_bf();
  billion_level_datasets_bf(8, 10);
  billion_level_datasets_bf(8, 20, true);
  billion_level_datasets_bf(8, 30, true);
  billion_level_datasets_bf(8, 40, true);
  billion_level_datasets_bf(8, 50, true);
  billion_level_datasets_bf(8, 60, true);
  billion_level_datasets_bf(8, 70, true);
  billion_level_datasets_bf(8, 80, true);
//  billion_level_datasets_bf(20, true);
//  billion_level_datasets_bf(30, true);
//  billion_level_datasets_bf(40, true);
//  billion_level_datasets_bf(50, true);
}
