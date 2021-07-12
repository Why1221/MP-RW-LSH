namespace bench_config::hamming{
    // million-level data sets
    constexpr char DATA_PATH[]              = "/media/gtnetuser/Dell/StringSimilarity/data/Hamming/compact/"; // path where data is storing
    constexpr char DATA_SETS[]              = "word2bits-800-hamming.hdf5;sift-256-hamming.hdf5"; // names of data set files (separating by ";")
    constexpr char RESULT_PATH[]            = "/media/gtnetuser/Dell/StringSimilarity/results/2021-02-11/million-level/"; // result path
    constexpr char INDEX_PATH_REL[]         = "temporary"; // index path (relative to result path)
    constexpr char RAW_DATASET_NAME[]       = "base_compact"; //
    constexpr char QUERY_DATASET_NAME[]     = "query_compact";
    constexpr char GND_DIST_DATASET_NAME[]  = "ground_truth_dist";
    constexpr char COMPACT_WORDSIZE_NAME[]  = "word_size";
    constexpr char KNN_K_NAME[]             = "K";



    // billion-level data sets
    namespace billion_level {
            constexpr char DATA_PATH[]                      = "/Data/Dropbox/HUGE-DATA/"; // path where data is storing
            constexpr char DATA_SETS[]                      = "Hamming_mih_128_sift_1B.hdf5;Hamming_mih_128_gist_80M.hdf5"; // names of data set files (separating by ";")
            constexpr char RESULT_PATH[]                    = "/media/gtnetuser/Dell/StringSimilarity/results/2021-02-11/billion-level/"; // result path
            constexpr char INDEX_PATH_REL[]                 = "temporary"; // index path (relative to result path)
            constexpr char RAW_DATASET_PREFIX_NAME[]        = "base/BLK_"; //
            constexpr char RAW_DATASET_BLOCKSIZE_NAME[]     = "size_block"; //
            constexpr char RAW_DATASET_NBLOCKS_NAME[]       = "number_blocks"; //
            constexpr char QUERY_DATASET_NAME[]             = "query";
            constexpr char COMPACT_WORDSIZE_NAME[]          = "";
            constexpr char KNN_K_NAME[]                     = "";
    }
}
