#include "dyatree.h"
#include <vector>
#include <random>
#include <cmath>

using namespace std;
using namespace falconn::core;



int main(){
    random_device rd;
    mt19937 rng(rd());
    vector<DyaSimTree<unsigned>> dsts;
    for (int i = 0; i < 100000; ++i){
        dsts.emplace_back(30, 1500000, rng);
    }

    uniform_int_distribution<int> dist(0, 1500000);
    for (int i = 0; i < 10; ++i){
        int start = dist(rng), end = dist(rng);
        float gt = abs((float)(start - end));
        float esti = 0, var = 0;
        for (int j = 0; j < 100000; ++j){
            float diff = dsts[j].range_sum(start) - dsts[j].range_sum(end);
            //cout << diff << endl;
            esti += diff * diff;
            
            var +=  (gt-diff * diff) * (gt-diff * diff);
        }
        cout << abs(start - end) << "\t" << (esti/ 100000)<< endl;
        cout << "V\t" << (gt * gt * 2) << "\t" << (var/ 100000)<< endl;

    }
    return 0;
}