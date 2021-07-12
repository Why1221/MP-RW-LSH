# #!/usr/bin/env bash
# make clean
# make falconn-l1
# Stop on error
set -e
# -l <num-hash-tables> -t <num-probes> -c <hash-bits>
# M=14
# W=204071
# L=(100 200 400 600 800 1000 1200 1400 1600)

# ./falconn-l1-cauchy -d 960 -n 1000000 -ds ../EXPERIMENTS/GIST/GIST-l1-960-train.fvecs -l $L -t $L -m $M -u 2954 -k 50 -w $W -gt ../EXPERIMENTS/GIST/gnd.txt -qs ../EXPERIMENTS/GIST/GIST-l1-960-test.fvecs -qn 1000 -if ./index-tow -rf ./test_results/test_cauchy_gist-$L-$M-$W-$L.txt
# ./falconn-l1-cauchy -d 100 -n 1192514 -ds ../EXPERIMENTS/glove/glove-l1-100-train.fvecs -l $L -t $L -m $M -u 24972 -k 50 -w $W -gt ../EXPERIMENTS/glove/gnd.txt -qs ../EXPERIMENTS/glove/glove-l1-100-test.fvecs -qn 200 -if ./index-tow -rf ./test_results/test_cauchy_glove-$L-$M-$W-$L.txt
M=10
W=3000000

for L in 100 200 400 800 1200 1600 2000
do
./falconn-l1-cauchy -d 192 -n 53387 -ds ../EXPERIMENTS/audio/audio-l1-192-train.fvecs -l $L -t $L -m $M -u 200000 -k 50 -w $W -gt ../EXPERIMENTS/audio/gnd.txt -qs ../EXPERIMENTS/audio/audio-l1-192-test.fvecs -qn 200 -if ./index-tow -rf ./test_results/test_cauchy_audio-$L-$M-$W.txt

done
echo "Done"