#include "range_sum.h"
#include <random>
#include <iostream>
#include <fstream>
#include "dyatree.h"

using namespace std;
using namespace falconn::core;

int main(int argc, char* argv[]){
    random_device rd;
    mt19937 rng(rd());
    uniform_int_distribution<int> dist(0, 1);
    normal_distribution<float> gau(0, 1000);
    ofstream fout(argv[1]);

    int sum = 0;
    float var = 0;
    for (int i = 0; i < 1000000; ++i){
        //int x = EH3Interval(1000000, rng(), rng());
        //fout << x << endl;
        //DyaSimTree<> dst(31, 10000000u, rng);
        //fout << dst.range_sum(1000000) << endl;
        fout << gau(rng) << endl;
    }
    return 0;
}