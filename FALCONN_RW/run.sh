# #!/usr/bin/env bash
# make clean
# make falconn-l1
# Stop on error
set -e
# -l <num-hash-tables> -t <num-probes> -c <hash-bits>
# L=10
# T=1000
# M=31
# W=818
# Parameter for audio
M=12
W=3200

# for L in 100
# do
# T=$((100*L))
L=4
T=400
./falconn-l1 -d 192 -n 53387 -ds ../EXPERIMENTS/audio/audio-l1-192-train.fvecs -l $L -t $T -m $M -u 200000 -k 50 -w $W -gt ../EXPERIMENTS/audio/gnd.txt -qs ../EXPERIMENTS/audio/audio-l1-192-test.fvecs -qn 200 -if ./index-tow -rf ./test_results/test_tow_audio-$L-$M-$W-$T.txt

# done
echo "Done"