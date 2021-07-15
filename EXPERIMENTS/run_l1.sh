#!/usr/bin/env bash

# Bash script for running ANN algorithms (beautified by `beautysh`)

# Fail on error
set -e

# Echo on
# set -x

function usage {
    printf '\nUsage: %s [-ahr]\n\n' $(basename $0) >&2

    printf 'This script attempts to run in-memory ANN experiments\n' >&2

    printf 'options:\n' >&2
    printf -- ' -a: (default) run (A)ll algorithms - good luck!\n' >&2
    printf -- ' -c: clean all!\n' >&2
    printf -- ' -h: print this (H)elp message\n' >&2
    printf -- ' -r <alg>: only run <alg>, where <alg>=LinearScan|FALCONN|FALCONN_cauchy|FALCONN_RW\n' >&2
    exit 2
}

# set DEBUG to 1 if you only want to debug this script
DEBUG=0

# Global variables
WORKING_DIR=$(pwd)
LINEARSCAN_DIR=../LinearScan
DATASETS=( $( cat dataset_info.txt ) )


# Parameters for algorithms
MAX_K=50
APP_RATIO=2

# # SRS parameters
SRS_M=10
SRS_T=(0.65)
SRS_K=(2000)
# SRS_T=(0.000010 0.000011 0.000013 0.000014 0.000016 0.000018 0.000020 0.000023 0.000026 0.000029 0.000032 0.000036 0.000041 0.000046 0.000052 0.000058 0.000065 0.000073 0.000082 0.000092 0.000104 0.000117 0.000131 0.000148 0.000166 0.000187 0.000210 0.000236 0.000265 0.000298 0.000335 0.000377 0.000424 0.000476 0.000536 0.000602 0.000677 0.000761 0.000855 0.000962 0.001081 0.001215 0.001366 0.001536 0.001727 0.001941 0.002183 0.002454 0.002759 0.003101 0.003486 0.003919 0.004406 0.004954 0.005569 0.006261 0.007038 0.007912 0.008895 0.010000)
# A special SRS T for high dimensional dataset
# SRS_T=(0.023 0.025 0.03 0.035)
# SRS_T=(0.011 0.012 0.013 0.014 0.015 0.020 0.025 0.030 0.035)


# A special iDEC T for high dimensional dataset

# # iDEC parameters
iDEC_M=6
# iDEC_T=(0.0000000006 0.0000000012 0.000000025 0.000000050 0.000000060 0.000000072 0.000000086 0.000000104 0.000000125 0.000000150 0.000000180 0.000000216 0.000000259 0.000000311 0.000000373 0.000000448 0.000000538 0.000000645 0.000000775 0.000000930 0.000001116 0.000001340 0.000001609 0.000001931 0.000002319 0.000002783 0.000003341 0.000004011 0.000004815 0.000005780 0.000006939 0.000008330 0.000010000 0.000020000)
iDEC_T=(0.0000000006)
# iDEC_T=(0.000040 0.000057 0.000082 0.000097 0.000116 0.000139 0.000166 0.000199 0.000237 0.000284 0.000339 0.000405 0.000484 0.000579 0.000691 0.000826 0.000987 0.001180 0.001410 0.001684 0.002013 0.002405 0.002874 0.003435 0.004104 0.004904 0.005861 0.007003 0.008368 0.010000)
# A more density t for plotting figure
# iDEC_T=(0.000010 0.000011 0.000013 0.000014 0.000016 0.000018 0.000020 0.000023 0.000026 0.000029 0.000032 0.000036 0.000041 0.000046 0.000052 0.000058 0.000065 0.000073 0.000082 0.000092 0.000104 0.000117 0.000131 0.000148 0.000166 0.000187 0.000210 0.000236 0.000265 0.000298 0.000335 0.000377 0.000424 0.000476 0.000536 0.000602 0.000677 0.000761 0.000855 0.000962 0.001081 0.001215 0.001366 0.001536 0.001727 0.001941 0.002183 0.002454 0.002759 0.003101 0.003486 0.003919 0.004406 0.004954 0.005569 0.006261 0.007038 0.007912 0.008895 0.010000)
# Some extra iDEC points
# iDEC_T=(0.011 0.012 0.013 0.014)


# A special iDEC T for high dimensional dataset
# iDEC_T=(0.025 0.03 0.035 0.040 0.045)

# # FALCONN (MP LSH and Cauchy LSH) parameter
FALCONN_L_cauchy=(100 120)
FALCONN_L_ToW=(8)

# # OPQ parameter
OPQ_MULTIPLICITY=(2)
# MUST enable rerank
OPQ_NNC=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
# not used
OPQ_SCCS=(4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536)
# An error exit function
function error_exit
{
    echo "$1" 1>&2
    exit 1
}

# Compiling LinearScan
function CompileLinearScan {
    cd ${WORKING_DIR}
    cd ${LINEARSCAN_DIR}
    echo "Compile LinearScan_L1 ..."

    make clean
    make linear-scan-l1
    cp linear-scan-l1 ${WORKING_DIR}

    cd ${WORKING_DIR}
    chmod +x ./linear-scan-l1
    echo "Done"
}


# Run LinearScan
function RunLinearScan {
    cd ${WORKING_DIR}
    if [ ! -f ./linear-scan-l1 ]; then
        error_exit "Executable file ${WORKING_DIR}/linear-scan-l1 does not exist, please compile it and copy it to ${WORKING_DIR} first ..."
    fi

    dsname=$1
    dsh5=$2
    n=$3
    qn=$4
    dim=$5

    printf "Run linear scan for L_1 on dataset %s ...\n" "${dsname}"

    if [ -f ${dsname}/gnd.txt ]; then
        error_exit "Result file ${dsname}/gnd.txt already exists (indicating that you have already run linear scan on dataset ${dsname}) ..."
    fi

    mkdir -p ${dsname}

    echo "./linear-scan-l1 -k ${MAX_K} -ds ./datasets/${dsh5} -rf ${dsname}/gnd.txt -ri ${dsname}/gni.txt -n ${n} -qn ${qn} -d ${dim} -pf ${dsname}/linear-scan.txt"
    if [ $DEBUG == 0 ]
    then
        ./linear-scan-l1 -k ${MAX_K} -ds ./datasets/${dsh5} -rf ${dsname}/gnd.txt -ri ${dsname}/gni.txt -n ${n} -qn ${qn} -d ${dim} -pf ${dsname}/linear-scan.txt
    fi

    echo "Done\n"
}

# Clean LinearScan
function CleanLinearScan {
    echo "Clean LinearScan ..."
    cd ${WORKING_DIR}
    cd ${LINEARSCAN_DIR}
    make clean
    echo "Done"
}

# Compile FALCONN (MP LSH)
function CompileFALCONN {
    cd ${WORKING_DIR}
    echo "Compile FALCONN ..."
    cd ../FALCONN_RW
    make clean
    make falconn-l1
    chmod +x ./falconn-l1
    cp ./falconn-l1 ${WORKING_DIR}
    cd ${WORKING_DIR}
    echo "Done."
}

# Run FALCONN (MP LSH)
function RunFALCONN {
    cd ${WORKING_DIR}
    if [ ! -f ./falconn-l1 ]; then
        error_exit "Executable file ${WORKING_DIR}/falconn-l1 does not exist, please compile it and copy it to ${WORKING_DIR} first ..."
    fi

    dsname=$1
    dstrainb=$2
    dstestb=$3
    n=$4
    qn=$5
    dim=$6
    universe=$7
    FALCONN_M_ToW=$8
    FALCONN_W_ToW=$9
    printf "Run FALCONN-l1 on dataset %s ...\n" "${dsname}"
    #mkdir -p FALCONN/${dsname}/index
    # mkdir -p FALCONN_L1_feigenbaum/${dsname}/results
    # cp ./falconn-l1 FALCONN_L1_feigenbaum/${dsname}
    # cd FALCONN_L1_feigenbaum/${dsname}
    mkdir -p FALCONN_L1/${dsname}/results
    cp ./falconn-l1 FALCONN_L1/${dsname}
    cd FALCONN_L1/${dsname}
    for L in "${FALCONN_L_ToW[@]}"
    do
        mkdir -p index-${FALCONN_M_ToW}-${FALCONN_W_ToW}-$L
        # T=$(($L*60+$L))
        # echo "./falconn-l1 -d ${dim} -n ${n} -ds ../../${dsname}/${dstrainb} -l $L -t $T -m ${FALCONN_M_ToW} -u ${universe} -k 50 -w ${FALCONN_W_ToW} -gt ../../${dsname}/gnd.txt -qs ../../${dsname}/${dstestb} -qn ${qn} -if ./index-${FALCONN_M_ToW}-${FALCONN_W_ToW}-$L -rf ./results/falconn_l1-$L-${FALCONN_M_ToW}-${FALCONN_W_ToW}-$T.txt"
        # if [ $DEBUG == 0 ]
        # then
        #     ./falconn-l1 -d ${dim} -n ${n} -ds ../../${dsname}/${dstrainb} -l $L -t $T -m ${FALCONN_M_ToW} -u ${universe} -k 50 -w ${FALCONN_W_ToW} -gt ../../${dsname}/gnd.txt -qs ../../${dsname}/${dstestb} -qn ${qn} -if ./index-${FALCONN_M_ToW}-${FALCONN_W_ToW}-$L -rf ./results/falconn_l1-$L-${FALCONN_M_ToW}-${FALCONN_W_ToW}-$T.txt
        # fi
        T=$(($L*100+$L))
        cmd="./falconn-l1 -a precompute -d ${dim} -n ${n} -ds ../../${dsname}/${dstrainb} -l $L -t $T -m ${FALCONN_M_ToW} -u ${universe} -k ${MAX_K} -w ${FALCONN_W_ToW} -gt ../../${dsname}/gnd.txt -qs ../../${dsname}/${dstestb} -qn ${qn} -if ./index-${FALCONN_M_ToW}-${FALCONN_W_ToW}-$L -rf ./results/falconn_l1-$L-${FALCONN_M_ToW}-${FALCONN_W_ToW}-$T.txt -li false -lp ./indices/falconn_index-$L-${FALCONN_M_ToW}-${FALCONN_W_ToW}-$T.txt"
        echo $cmd
        if [ $DEBUG == 0 ]
        then
           # ./falconn-l1 -a dyasim -d ${dim} -n ${n} -ds ../../${dsname}/${dstrainb} -l $L -t $T -m ${FALCONN_M_ToW} -u ${universe} -k 50 -w ${FALCONN_W_ToW} -gt ../../${dsname}/gnd.txt -qs ../../${dsname}/${dstestb} -qn ${qn} -if ./index-${FALCONN_M_ToW}-${FALCONN_W_ToW}-$L -rf ./results/falconn_l1-$L-${FALCONN_M_ToW}-${FALCONN_W_ToW}-$T.txt
           ./falconn-l1 -a precompute -d ${dim} -n ${n} -ds ../../${dsname}/${dstrainb} -l $L -t $T -m ${FALCONN_M_ToW} -u ${universe} -k ${MAX_K} -w ${FALCONN_W_ToW} -gt ../../${dsname}/gnd.txt -qs ../../${dsname}/${dstestb} -qn ${qn} -if ./index-${FALCONN_M_ToW}-${FALCONN_W_ToW}-$L -rf ./results/falconn_l1-$L-${FALCONN_M_ToW}-${FALCONN_W_ToW}-$T.txt -li false -lp ./indices/falconn_index-$L-${FALCONN_M_ToW}-${FALCONN_W_ToW}-$T.txt
        fi
    done
    echo "Done"
}

# Clean FALCONN (MP LSH)
function CleanFALCONN {
    echo "Clean FALCONN ..."
    cd ${WORKING_DIR}
    cd ../FALCONN_RW
    make clean
    echo "Done"
}

# Compile FALCONN (No MP LSH)
function CompileFALCONN_RW {
    cd ${WORKING_DIR}
    echo "Compile FALCONN ..."
    cd ../FALCONN_cauchy
    make clean
    make falconn-l1-rw
    chmod +x ./falconn-l1-rw
    cp ./falconn-l1-rw ${WORKING_DIR}
    cd ${WORKING_DIR}
    echo "Done."
}

# Run FALCONN (MP LSH)
function RunFALCONN_RW {
    cd ${WORKING_DIR}
    if [ ! -f ./falconn-l1-rw ]; then
        error_exit "Executable file ${WORKING_DIR}/falconn-l1 does not exist, please compile it and copy it to ${WORKING_DIR} first ..."
    fi

    dsname=$1
    dstrainb=$2
    dstestb=$3
    n=$4
    qn=$5
    dim=$6
    universe=$7
    FALCONN_M_ToW=$8
    FALCONN_W_ToW=$9

    printf "Run FALCONN-l1 on dataset %s ...\n" "${dsname}"
    #mkdir -p FALCONN/${dsname}/index
    mkdir -p FALCONN_L1_RW/${dsname}/results
    cp ./falconn-l1-rw FALCONN_L1_RW/${dsname}
    cd FALCONN_L1_RW/${dsname}
    for L in "${FALCONN_L_ToW[@]}"
    do
        T=$L
        mkdir -p index-${FALCONN_M_ToW}-${FALCONN_W_ToW}-$L
        echo "./falconn-l1-rw -d ${dim} -n ${n} -ds ../../${dsname}/${dstrainb} -l $L -t $L -m ${FALCONN_M_ToW} -u ${universe} -k ${MAX_K} -w ${FALCONN_W_ToW} -gt ../../${dsname}/gnd.txt -qs ../../${dsname}/${dstestb} -qn ${qn} -if ./index-${FALCONN_M_ToW}-${FALCONN_W_ToW}-$L -rf ./results/falconn_l1_ToW-$L-${FALCONN_M_ToW}-${FALCONN_W_ToW}-$T.txt "
        if [ $DEBUG == 0 ]
        then
            ./falconn-l1-rw -d ${dim} -n ${n} -ds ../../${dsname}/${dstrainb} -l $L -t $L -m ${FALCONN_M_ToW} -u ${universe} -k ${MAX_K} -w ${FALCONN_W_ToW} -gt ../../${dsname}/gnd.txt -qs ../../${dsname}/${dstestb} -qn ${qn} -if ./index-${FALCONN_M_ToW}-${FALCONN_W_ToW}-$L -rf ./results/falconn_l1_ToW-$L-${FALCONN_M_ToW}-${FALCONN_W_ToW}-$T.txt 
        fi
    done
    echo "Done"
}

# Clean FALCONN (MP LSH)
function CleanFALCONN_RW {
    echo "Clean FALCONN ..."
    cd ${WORKING_DIR}
    cd ../FALCONN_cauchy
    make clean
    echo "Done"
}


# Compile FALCONN-cauchy (No MP LSH)
function CompileFALCONN_cauchy {
    cd ${WORKING_DIR}
    echo "Compile FALCONN_cauchy ..."
    cd ../FALCONN_cauchy
    make clean
    make falconn-l1-cauchy
    chmod +x ./falconn-l1-cauchy
    cp ./falconn-l1-cauchy ${WORKING_DIR}
    cd ${WORKING_DIR}
    echo "Done."
}

# Run FALCONN-cauchy (No MP LSH)
function RunFALCONN_cauchy {
    cd ${WORKING_DIR}
    if [ ! -f ./falconn-l1-cauchy ]; then
        error_exit "Executable file ${WORKING_DIR}/falconn-l1-cauchy does not exist, please compile it and copy it to ${WORKING_DIR} first ..."
    fi

    dsname=$1
    dstrainb=$2
    dstestb=$3
    n=$4
    qn=$5
    dim=$6
    universe=$7
    FALCONN_M_Cauchy=$8
    FALCONN_W_Cauchy=$9

    printf "Run FALCONN-l1-cauchy on dataset %s ...\n" "${dsname}"

    #mkdir -p FALCONN/${dsname}/index
    mkdir -p FALCONN_L1_Cauchy/${dsname}/results
    cp ./falconn-l1-cauchy FALCONN_L1_Cauchy/${dsname}
    cd FALCONN_L1_Cauchy/${dsname}
    for L in "${FALCONN_L_cauchy[@]}"
    do
        echo $L
        T=$L
        mkdir -p index-${FALCONN_M_Cauchy}-${FALCONN_W_Cauchy}-$L
        echo "./falconn-l1-cauchy -d ${dim} -n ${n} -ds ../../${dsname}/${dstrainb} -l $L -t $L -m ${FALCONN_M_Cauchy} -u ${universe} -k {MAX_K} -w ${FALCONN_W_Cauchy} -gt ../../${dsname}/gnd.txt -qs ../../${dsname}/${dstestb} -qn ${qn} -if ./index-${FALCONN_M_Cauchy}-${FALCONN_W_Cauchy}-$L -rf ./results/falconn_l1-$L-${FALCONN_M_Cauchy}-${FALCONN_W_Cauchy}.txt"
        if [ $DEBUG == 0 ]
        then
            ./falconn-l1-cauchy -d ${dim} -n ${n} -ds ../../${dsname}/${dstrainb} -l $L -t $L -m ${FALCONN_M_Cauchy} -u ${universe} -k {MAX_K} -w ${FALCONN_W_Cauchy} -gt ../../${dsname}/gnd.txt -qs ../../${dsname}/${dstestb} -qn ${qn} -if ./index-${FALCONN_M_Cauchy}-${FALCONN_W_Cauchy}-$L -rf ./results/falconn_l1-$L-${FALCONN_M_Cauchy}-${FALCONN_W_Cauchy}.txt
        fi
    done
    echo "Done"
}

# Clean FALCONN (MP LSH)
function CleanFALCONN_cauchy {
    echo "Clean FALCONN_cauchy ..."
    cd ${WORKING_DIR}
    cd ../FALCONN_cauchy
    make clean
    echo "Done"
}

# Clean all
function clean_all
{
    # CleanLinearScan
    CleanFALCONN
    CleanFALCONN_cauchy
    CleanFALCONN_RW
}

# Run all algorithms
function all
{

    CompileFALCONN
    CompileFALCONN_RW
    CompileFALCONN_cauchy

    # BuildKNNGraphs
    for ds in "${DATASETS[@]}"
    do
        # read dataset information
        IFS='%'
        read -ra ds_info <<< "$ds" # str is read into an array as tokens separated by IFS

        # TODO: change accordingly
        dsname=${ds_info[0]}
        dstrainfvecs=${ds_info[1]}
        dstestfvecs=${ds_info[2]}
        dstraintxt=${ds_info[3]}
        dstesttxt=${ds_info[4]}
        n=${ds_info[5]}
        dim=${ds_info[6]}
        qn=${ds_info[7]}
        dsh5=${ds_info[8]}

        # RunLinearScan $dsname $dsh5 $n $qn $dim
        
        RunFALCONN $dsname $dstrainfvecs $dstestfvecs $n $qn $dim $universe $M_tow $W_tow 
        
        RunFALCONN_cauchy $dsname $dstrainfvecs $dstestfvecs $n $qn $dim $universe $M_cauchy $W_cauchy

        RunFALCONN_RW $dsname $dstrainfvecs $dstestfvecs $n $qn $dim $universe $M_tow $W_tow 

    done

    clean_all
}

# Compile a single algorithm
function compile_single
{
    case "$1" in
        LinearScan)
            CompileLinearScan
            ;;
        FALCONN)
            CompileFALCONN
            ;;
        FALCONN_RW)
            CompileFALCONN_RW
            ;;
        FALCONN_cauchy)
            CompileFALCONN_cauchy
            ;;
        *)
            echo "Unknown option $1."
            usage
            exit 1
    esac
}

function single
{
    # CompileLinearScan
    compile_single $1

    for ds in "${DATASETS[@]}"
    do
        # read dataset information
        IFS='%'
        # TODO: change accordingly
        read -ra ds_info <<< "$ds" # str is read into an array as tokens separated by IFS
        dsname=${ds_info[0]}
        dstrainfvecs=${ds_info[1]}
        dstestfvecs=${ds_info[2]}
        dstraintxt=${ds_info[3]}
        dstesttxt=${ds_info[4]}
        n=${ds_info[5]}
        dim=${ds_info[6]}
        qn=${ds_info[7]}
        dsh5=${ds_info[8]}
        universe=${ds_info[9]}
        M_tow=${ds_info[10]}
        W_tow=${ds_info[11]}
        M_cauchy=${ds_info[12]}
        W_cauchy=${ds_info[13]}


        cd ${WORKING_DIR}

        # if [ ! -f ./$dsname/gnd.txt ]; then
        #     RunLinearScan $dsname $dsh5 $n $qn $dim
        # fi

        case "$1" in
            LinearScan)
                RunLinearScan $dsname $dsh5 $n $qn $dim
                ;;
            FALCONN)
                RunFALCONN $dsname $dstrainfvecs $dstestfvecs $n $qn $dim $universe $M_tow $W_tow 
                ;;
            FALCONN_cauchy)
                RunFALCONN_cauchy $dsname $dstrainfvecs $dstestfvecs $n $qn $dim $universe $M_cauchy $W_cauchy 
                ;;
            FALCONN_RW)
                RunFALCONN_RW $dsname $dstrainfvecs $dstestfvecs $n $qn $dim $universe $M_tow $W_tow 
                ;;
            *)
                echo "Unknown option $1."
                usage
        esac
    done
    clean_all
}

# ---
if [ $# -eq 0 ]
then
    all
else
    while getopts 'achr:' OPTION
    do
        case $OPTION in
            a)    all ;;
            c)    clean_all ;;
            h)    usage ;;
            r)    single $OPTARG ;;
            ?)    usage ;;
        esac
    done
    shift $(($OPTIND - 1))
fi