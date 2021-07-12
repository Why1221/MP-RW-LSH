#include <benchmark/benchmark.h>
#include <string_embed_sp.hpp>
using namespace ss::ann::embed;

static void BM_NaiveImpl(benchmark::State& state) {

  const std::string alphabet("ACGT");
  std::string sa("CGTAATAAGGTTCATTGAGCGCAAATGGTGACGTCTTAATAAACGTGGAGATAAACCGACAATATTGATGCTCGCTGCGAAGTTTTTCCGCCGCCCGGGC");
  CGKEmbed emb(alphabet.c_str(), alphabet.size(), sa.size() * 3, 1u);
  std::vector<std::string> res_naive(1, "");

  // Perform setup here
  for (auto _ : state) {
    // This code gets timed
    emb.apply_naive(sa.c_str(), sa.size(), &res_naive[0]);
  }
}
static void BM_EJImpl(benchmark::State& state) {

  const std::string alphabet("ACGT");
  std::string sa("CGTAATAAGGTTCATTGAGCGCAAATGGTGACGTCTTAATAAACGTGGAGATAAACCGACAATATTGATGCTCGCTGCGAAGTTTTTCCGCCGCCCGGGC");
  CGKEmbed emb(alphabet.c_str(), alphabet.size(), sa.size() * 3, 1u);
  std::vector<std::string> res_naive(1, "");

  // Perform setup here
  for (auto _ : state) {
    // This code gets timed
    emb.apply(sa.c_str(), sa.size(), &res_naive[0]);
  }
}

// Register the function as a benchmark
BENCHMARK(BM_NaiveImpl);
BENCHMARK(BM_EJImpl);

// Run the benchmark
BENCHMARK_MAIN();

/*
 * -------------------------------------------------------
 * Results on my VM
 * -------------------------------------------------------

  2019-04-27 01:45:35
  Running bench_cgk_impl
      Run on (4 X 3600 MHz CPU s)
  CPU Caches:
  L1 Data 32K (x4)
  L1 Instruction 32K (x4)
  L2 Unified 256K (x4)
  L3 Unified 8192K (x4)
  Load Average: 0.76, 1.22, 1.50
  -------------------------------------------------------
  Benchmark             Time             CPU   Iterations
  -------------------------------------------------------
  BM_NaiveImpl     595918 ns       587845 ns        10049
  BM_EJImpl         17725 ns        17696 ns        38040

  Process finished with exit code 0
 */
