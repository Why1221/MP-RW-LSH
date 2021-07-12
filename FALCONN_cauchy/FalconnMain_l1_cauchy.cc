#include <falconn/lsh_nn_table.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <AnnResultWriter.hpp>
#include <Exception.h>
#include <StringUtils.hpp>
#include <Timer.hpp>
#include <cstdio>

using namespace StringUtils;
const size_t MAX_MEM = 1e10; // 10 GB

using std::cerr;
using std::cout;
using std::endl;
using std::exception;
using std::make_pair;
using std::max;
using std::mt19937_64;
using std::pair;
using std::runtime_error;
using std::string;
using std::uniform_int_distribution;
using std::unique_ptr;
using std::vector;

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;

using falconn::compute_number_of_hash_functions;
using falconn::construct_table;
using falconn::DenseVector;
using falconn::DistanceFunction;
using falconn::get_default_parameters;
using falconn::GaussianFunctionType;
using falconn::MultiProbeType;
using falconn::LSHConstructionParameters;
using falconn::LSHFamily;
using falconn::LSHNearestNeighborQuery;
using falconn::LSHNearestNeighborTable;
using falconn::QueryStatistics;
using falconn::StorageHashTable;

typedef DenseVector<float> Point;

const int NUM_QUERIES = 1000;
const int SEED = 4057218;
const int NUM_HASH_TABLES = 50;
const int NUM_HASH_BITS = 18;
const int NUM_ROTATIONS = 1;
const int BUCKET_ID_WIDTH = 10; // Not used

/*
 * An auxiliary function that reads a point from a binary file that is produced
 * by a script 'prepare-dataset.sh'
 */
bool read_point(FILE *file, Point *point) {
  int d;
  if (fread(&d, sizeof(int), 1, file) != 1) {
    return false;
  }
  float *buf = new float[d];
  if (fread(buf, sizeof(float), d, file) != (size_t)d) {
    throw runtime_error("can't read a point");
  }
  point->resize(d);
  for (int i = 0; i < d; ++i) {
    (*point)[i] = buf[i];
  }
  delete[] buf;
  return true;
}

/*
 * An auxiliary function that reads a dataset from a binary file that is
 * produced by a script 'prepare-dataset.sh'
 */
void read_dataset(string file_name, vector<Point> *dataset) {
  FILE *file = fopen(file_name.c_str(), "rb");
  if (!file) {
    throw runtime_error("can't open the file with the dataset");
  }
  Point p;
  dataset->clear();
  while (read_point(file, &p)) {
    dataset->push_back(p);
  }
  if (fclose(file)) {
    throw runtime_error("fclose() error");
  }
}

/*
 * Author:  David Robert Nadeau
 * Site:    http://NadeauSoftware.com/
 * License: Creative Commons Attribution 3.0 Unported License
 *          http://creativecommons.org/licenses/by/3.0/deed.en_US
 */

#if defined(_WIN32)
#include <psapi.h>
#include <windows.h>

#elif defined(__unix__) || defined(__unix) || defined(unix) ||                 \
    (defined(__APPLE__) && defined(__MACH__))

#include <sys/resource.h>
#include <unistd.h>

#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>

#elif (defined(_AIX) || defined(__TOS__AIX__)) ||                              \
    (defined(__sun__) || defined(__sun) ||                                     \
     defined(sun) && (defined(__SVR4) || defined(__svr4__)))
#include <fcntl.h>
#include <procfs.h>

#elif defined(__linux__) || defined(__linux) || defined(linux) ||              \
    defined(__gnu_linux__)

#endif

#else
#error "Cannot define getPeakRSS( ) or getCurrentRSS( ) for an unknown OS."
#endif

/**
 * Returns the peak (maximum so far) resident set size (physical
 * memory use) measured in bytes, or zero if the value cannot be
 * determined on this OS.
 */
static size_t getPeakRSS() {
#if defined(_WIN32)
  /* Windows -------------------------------------------------- */
  PROCESS_MEMORY_COUNTERS info;
  GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
  return (size_t)info.PeakWorkingSetSize;

#elif (defined(_AIX) || defined(__TOS__AIX__)) ||                              \
    (defined(__sun__) || defined(__sun) ||                                     \
     defined(sun) && (defined(__SVR4) || defined(__svr4__)))
  /* AIX and Solaris ------------------------------------------ */
  struct psinfo psinfo;
  int fd = -1;
  if ((fd = open("/proc/self/psinfo", O_RDONLY)) == -1)
    return (size_t)0L; /* Can't open? */
  if (read(fd, &psinfo, sizeof(psinfo)) != sizeof(psinfo)) {
    close(fd);
    return (size_t)0L; /* Can't read? */
  }
  close(fd);
  return (size_t)(psinfo.pr_rssize * 1024L);

#elif defined(__unix__) || defined(__unix) || defined(unix) ||                 \
    (defined(__APPLE__) && defined(__MACH__))
  /* BSD, Linux, and OSX -------------------------------------- */
  struct rusage rusage;
  getrusage(RUSAGE_SELF, &rusage);
#if defined(__APPLE__) && defined(__MACH__)
  return (size_t)rusage.ru_maxrss;
#else
  return (size_t)(rusage.ru_maxrss * 1024L);
#endif

#else
  /* Unknown OS ----------------------------------------------- */
  return (size_t)0L; /* Unsupported. */
#endif
}

/**
 * Returns the current resident set size (physical memory use) measured
 * in bytes, or zero if the value cannot be determined on this OS.
 */
static size_t getCurrentRSS() {
#if defined(_WIN32)
  /* Windows -------------------------------------------------- */
  PROCESS_MEMORY_COUNTERS info;
  GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
  return (size_t)info.WorkingSetSize;

#elif defined(__APPLE__) && defined(__MACH__)
  /* OSX ------------------------------------------------------ */
  struct mach_task_basic_info info;
  mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info,
                &infoCount) != KERN_SUCCESS)
    return (size_t)0L; /* Can't access? */
  return (size_t)info.resident_size;

#elif defined(__linux__) || defined(__linux) || defined(linux) ||              \
    defined(__gnu_linux__)
  /* Linux ---------------------------------------------------- */
  long rss = 0L;
  FILE *fp = NULL;
  if ((fp = fopen("/proc/self/statm", "r")) == NULL)
    return (size_t)0L; /* Can't open? */
  if (fscanf(fp, "%*s%ld", &rss) != 1) {
    fclose(fp);
    return (size_t)0L; /* Can't read? */
  }
  fclose(fp);
  return (size_t)rss * (size_t)sysconf(_SC_PAGESIZE);

#else
  /* AIX, BSD, Solaris, and Unknown OS ------------------------ */
  return (size_t)0L; /* Unsupported. */
#endif
}

// Error message
#define NPP_ERROR_MSG(M)                                                       \
  do {                                                                         \
    fprintf(stderr, "%s:%d: " M, __FILE__, __LINE__);                          \
  } while (false)

// print parameters to stdout
void show_params(const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);

  while (*fmt != '\0') {
    char *name = va_arg(args, char *);
    if (*fmt == 'i') {
      int val = va_arg(args, int);
      printf("%s: %d\n", name, val);
    } else if (*fmt == 'c') {
      int val = va_arg(args, int);
      printf("%s: \'%c\'\n", name, val);
    } else if (*fmt == 'f') {
      double val = va_arg(args, double);
      printf("%s: %f\n", name, val);
    } else if (*fmt == 's') {
      char *val = va_arg(args, char *);
      printf("%s: \"%s\"\n", name, val);
    } else {
      NPP_ERROR_MSG("Unsupported format");
    }
    ++fmt;
  }

  va_end(args);
}

void usage() {
  printf("Falconn\n");
  printf("Options\n");
  printf("-d {value}     \trequired \tdimensionality\n");
  printf("-ds {string}   \trequired \tdataset file\n");
  printf("-n {value}     \trequired \tcardinality\n");
  printf("-l {value}     \trequired \tparameter l (# of hash tables)\n");
  printf("-m {value}     \trequired \tparameter M (# of hash "
         "functions)\n");
  printf("-w {value}     \trequired \tparameter W (bucket_width) \n");
  printf("-u {value}     \trequired \tparameter U (dataset universe)\n");
  printf("-t {value}     \trequired \tparameter t (the number "
         "of probes)\n");
  printf("-k {value}     \toptional \tnumber of neighbors "
         "(default: 1)\n");
  printf("-gt {string}   \trequired  \tfile of exact results\n");
  printf("-qs {string}   \trequired \tfile of query set\n");
  printf("-qn {string}   \trequired \tnumber of queries\n");
  printf("-rf {string}   \trequired \tresult file\n");
  printf("-if {string}   \trequired \tindex folder\n");

  printf("\n");
  printf("Run falconn (indexing and querying)\n");
  printf("-d -n -ds -l -t -m -u -w -gt -qs -qn -rf -if [-k]\n");

  printf("\n");
}

std::unique_ptr<LSHNearestNeighborTable<Point>>
indexing(const char *ds_filename, // database filename
         const char *i_pathname,  // database filename
         std::vector<Point> &train,
         unsigned num,         // # of points
         unsigned dim,         // dimension
         unsigned num_hashes,  // parameter l (# of hash tables)
         unsigned num_hashfuncs, // parameter m (hash functions)
         unsigned bucket_width, // parameter w (bucket width)
         unsigned universe //parameter u (dataset universe)
) {
  read_dataset(ds_filename, &train);

  // setting parameters and constructing the table
  LSHConstructionParameters params;
  params.dimension = dim;
  params.lsh_family = LSHFamily::Gaussian;
  params.k = num_hashfuncs;
  params.l = num_hashes;
  params.distance_function = DistanceFunction::L1Norm;
  // compute_number_of_hash_functions<Point>(num_hashbits, &params);
  params.multi_probe = MultiProbeType::Precomputed;
  params.gauss_type = GaussianFunctionType::Cauchy;
  params.num_setup_threads = 1;
  params.storage_hash_table = StorageHashTable::FlatHashTable;
  params.bucket_id_width = BUCKET_ID_WIDTH;
  params.bucket_width = bucket_width*1.0;
  params.universe = universe;

  std::string perf_filename = join(
      {"falconn-cauchy-indexing", "n" + std::to_string(num), "d" + std::to_string(dim),
       "l" + std::to_string(num_hashes), "m" + std::to_string(num_hashfuncs)},
      "-");
  perf_filename += ".txt";

  AnnResultWriter writer(std::string(i_pathname) + "/" + perf_filename);
  writer.writeRow(
      "s",
      "dsname,#n,#dim,#hashes,#functions,index_size(bytes),construction_time(us)");
  const char *fmt = "siiiiif";

  HighResolutionTimer timer;
  timer.restart();
  auto table = construct_table<Point>(train, params);
  auto e = timer.elapsed();

  auto isz = getPeakRSS();

  writer.writeRow(fmt, ds_filename, num, dim, num_hashes, num_hashfuncs, isz, e);

  return table;
}

void knn(LSHNearestNeighborTable<Point> *table, const std::vector<Point> &train,
         const char *q_filename,  // query filename
         const char *gt_filename, // ground truth filename
         const char *r_filename,  // result filename
         unsigned num,            // # of points in database
         unsigned dim,            // dimensionality
         unsigned qn,             // # of queries
         unsigned K,              // # of NNs
         unsigned num_probes) {
  NPP_ENFORCE(table != nullptr);
  std::vector<Point> test;
  read_dataset(q_filename, &test);
  NPP_ENFORCE(test.size() == qn);
  NPP_ENFORCE(test.front().size() == dim);

  unsigned r_qn, r_maxk;
  FILE *fp = fopen(gt_filename, "r");
  NPP_ENFORCE(fp != NULL);
  NPP_ENFORCE(fscanf(fp, "%d %d\n", &r_qn, &r_maxk) >= 0);
  NPP_ENFORCE(r_qn >= qn && r_maxk >= K);

  std::vector<float> gt(qn * r_maxk, -1.0f);

  for (unsigned i = 0; i < qn; ++i) {
    unsigned j;
    NPP_ENFORCE(fscanf(fp, "%d", &j) >= 0);
    NPP_ENFORCE(j == i);
    for (j = 0; j < r_maxk; ++j) {
      NPP_ENFORCE(fscanf(fp, " %f", &gt[i * r_maxk + j]) >= 0);
    }
    NPP_ENFORCE(fscanf(fp, "\n") >= 0);
  }
  NPP_ENFORCE(fclose(fp) == 0);

#ifdef DEBUG
  printf("Reading gt finished\n");
#endif
  HighResolutionTimer timer;
  AnnResultWriter writer(r_filename);

  writer.writeRow("s", AnnResults::_DEFAULT_HEADER_I_);
  unique_ptr<LSHNearestNeighborQuery<Point>> query_object =
      table->construct_query_object(num_probes);

  for (unsigned i = 0; i < qn; i++) {
    std::vector<int32_t> res;
#ifdef DEBUG
    printf("Query startes for %d\n", i);
#endif
    timer.restart();
    query_object->reset_query_statistics();
#ifdef DEBUG
    printf("Perform query for %d\n", i);
#endif
    query_object->find_k_nearest_neighbors(test[i], K, &res);

    auto query_time = timer.elapsed();

#ifdef DEBUG
    printf("Query finished for %d\n", i);
#endif
    for (unsigned j = 0; j < K; ++j) {
      float dist = 0.0f;
      if (res.size() == K) {
        for (unsigned d = 0; d < dim; ++d) {
          auto temp = (train[res[j]][d] - test[i][d]);
          dist += std::abs(temp);
        }
        int gdist = std::round(gt[i * r_maxk + j]);
        writer.writeRow(AnnResults::_DEFAULT_FMT_I_, i, j, res[j], (int)dist,
                        gdist, 1.0*dist/gdist, query_time);
      }
      else if (!res.empty()) {
        int temp_size = res.size();
        if (j<res.size()) {
        for (unsigned d = 0; d < dim; ++d) {
          auto temp = (train[res[j]][d] - test[i][d]);
          dist += std::abs(temp);
        }
        int gdist = std::round(gt[i * r_maxk + j]);
        writer.writeRow(AnnResults::_DEFAULT_FMT_I_, i, j, res[j], (int)dist,
                        gdist, 1.0*dist/gdist, query_time);
        }
        else {
        int gdist = std::round(gt[i * r_maxk + j]);
        writer.writeRow(AnnResults::_DEFAULT_FMT_I_, i, j, -1, -1, gdist, -1.0,
                        query_time);
        }
      } 
      else {
        int gdist = std::round(gt[i * r_maxk + j]);
        writer.writeRow(AnnResults::_DEFAULT_FMT_I_, i, j, -1, -1, gdist, -1.0,
                        query_time);
      }
    }
  }
}

/*
 * Get the index of next unblank char from a string.
 */
int GetNextChar(char *str) {
  int rtn = 0;

  // Jump over all blanks
  while (str[rtn] == ' ') {
    rtn++;
  }

  return rtn;
}

/*
 * Get next word from a string.
 */
void GetNextWord(char *str, char *word) {
  // Jump over all blanks
  while (*str == ' ') {
    str++;
  }

  while (*str != ' ' && *str != '\0') {
    *word = *str;
    str++;
    word++;
  }

  *word = '\0';
}

int main(int argc, char **argv) {
  // These two are global variables
  unsigned nPoints = 0;         // the number of points
  unsigned pointsDimension = 0; // the dimensionality of points

  int qn = -1; // the number of queries
  int k = 1;   // the k of k-NN

  char ds[200] = ""; // the file path of dataset
  char qs[200] = ""; // the file path of query set
  char gt[200] = ""; // the file path of ground truth

  char rf[200] = "";      // the folder path of results
  char indexf[200] = "."; // the folder path of results

  unsigned num_hashes = 0, num_hashfuncs = 0, num_probes = 0, universe = 0;

  int bucket_width = -1;

  int cnt = 1;
  bool failed = false;
  char *arg;
  int i;
  char para[10];

  std::string err_msg;
  while (cnt < argc && !failed) {
    arg = argv[cnt++];
    if (cnt == argc) {
      failed = true;
      break;
    }

    i = GetNextChar(arg);
    if (arg[i] != '-') {
      failed = true;
      err_msg = "Wrong format!";
      break;
    }

    GetNextWord(arg + i + 1, para);

    arg = argv[cnt++];

    if (strcmp(para, "n") == 0) {
      nPoints = atoi(arg);
      if (nPoints <= 0) {
        failed = true;
        err_msg = "n should a positive integer!";
        break;
      }
    } else if (strcmp(para, "d") == 0) {
      pointsDimension = atoi(arg);
      if (pointsDimension <= 0) {
        failed = true;
        err_msg = "d should a positive integer!";
        break;
      }
    } else if (strcmp(para, "u") == 0) {
      universe = atoi(arg);
      if (universe <= 0) {
        failed = true;
        err_msg = "Universe should a positive integer!";
        break;
      }
    } else if (strcmp(para, "qn") == 0) {
      qn = atoi(arg);
      if (qn <= 0) {
        failed = true;
        err_msg = "qn should a positive integer!";
        break;
      }
    } else if (strcmp(para, "k") == 0) {
      k = atoi(arg);
      if (k <= 0) {
        failed = true;
        err_msg = "k should a positive integer!";
        break;
      }
    } else if (strcmp(para, "l") == 0) {
      num_hashes = atoi(arg);
      if (num_hashes <= 0) {
        failed = true;
        err_msg = "num_hash_tables should a positive integer!";
        break;
      }
    } else if (strcmp(para, "w") == 0) {
      bucket_width = atoi(arg);
      if (bucket_width <= 0) {
        failed = true;
        err_msg = "bukcet_width should a positive integer!";
        break;
      }
    } else if (strcmp(para, "m") == 0) {
      num_hashfuncs = atoi(arg);
      if (num_hashfuncs <= 0) {
        failed = true;
        err_msg = "m should a positive integer!";
        break;
      }
    } else if (strcmp(para, "t") == 0) {
      num_probes = atoi(arg);
      if (num_probes <= 0) {
        failed = true;
        err_msg = "t should a positive integer!";
        break;
      }
    } else if (strcmp(para, "ds") == 0) {
      GetNextWord(arg, ds);

    } else if (strcmp(para, "qs") == 0) {
      GetNextWord(arg, qs);

    } else if (strcmp(para, "gt") == 0) {
      GetNextWord(arg, gt);

    } else if (strcmp(para, "rf") == 0) {
      GetNextWord(arg, rf);
    } else if (strcmp(para, "if") == 0) {
      GetNextWord(arg, indexf);
    } else {
      failed = true;
      fprintf(stderr, "Unknown option -%s!\n\n", para);
    }
  }

  if (failed) {
    fprintf(stderr, "%s:%d: %s\n\n", __FILE__, __LINE__, err_msg.c_str());
    usage();
    return EXIT_FAILURE;
  }

  int nargs = (cnt - 1) / 2;

  if (!(nargs == 12 || nargs == 13 || nargs == 14)) {
    fprintf(stderr, "%s:%d: %s\n\n", __FILE__, __LINE__,
            "Wrong number of arguements!");
    usage();
    return EXIT_FAILURE;
  }

#ifndef DISABLE_VERBOSE
  printf("=====================================================\n");
  show_params("iiiiiiiiissss", "# of points", nPoints, "dimension",
              pointsDimension, "# of queries", qn, "# of hash tables", num_hashes,
              "# of hash functions", num_hashfuncs, "# of probes", num_probes, "k", k,
              "dataset universe",universe,"bucket width",bucket_width, 
              "dataset filename", ds, "index folder", indexf, "result filename",
              rf, "ground truth filename", gt);
  printf("=====================================================\n");
#endif

  std::vector<Point> train;
  try {
    NPP_ENFORCE(strlen(ds) != 0);
    NPP_ENFORCE(strlen(qs) != 0);
    NPP_ENFORCE(strlen(rf) != 0);
    NPP_ENFORCE(num_hashes > 0 && num_probes > 0 && num_hashfuncs > 0 &&
                nPoints > 0 && universe>0 && bucket_width >0 && pointsDimension > 0);
    auto table = indexing(ds, indexf, train, nPoints, pointsDimension,
                          num_hashes, num_hashfuncs,bucket_width,universe);

    knn(&*table, train, qs, gt, rf, nPoints, pointsDimension, qn, k,
        num_probes);

  } catch (const npp::Exception &e) {
    std::cerr << e.what() << std::endl;
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
  }

  return EXIT_SUCCESS;
}