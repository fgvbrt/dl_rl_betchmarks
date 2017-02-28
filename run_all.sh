#!/usr/bin/env bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd SCRIPT_DIR

NUM_THREADS=(0 -1 1)
MODELS=("fc" "cnn_dqn" "cnn_openai" "rnn_fc" "rnn_openai" "rnn_dqn")
RNN_TYPES=("LSTM")
BATCH_SIZES=(1 64)
SEQ_LENS=(20)
BACKENDS=("th" "tf")

# remove old data
RESULTS_FILE="$SCRIPT_DIR/results.txt"
if [ -f $RESULTS_FILE ]; then
   rm $RESULTS_FILE
fi

# run all experiments
for num_threads in "${NUM_THREADS[@]}"; do
    for batch_size in "${BATCH_SIZES[@]}"; do
        for model in "${MODELS[@]}"; do
            # rnn model
            if [[ $model == *"rnn"* ]]; then
                for seq_len in "${SEQ_LENS[@]}"; do
                    for rnn_type in "${RNN_TYPES[@]}"; do
                        for backend in "${BACKENDS[@]}"; do
                            python run_benchmark.py --model $model --batch_size $batch_size --seq_len $seq_len \
                             --rnn_type $rnn_type --backend $backend --num_threads $num_threads --fname $RESULTS_FILE
                        done
                    done
                done
            # fc model
            else
                for backend in "${BACKENDS[@]}"; do
                    python run_benchmark.py --model $model --batch_size $batch_size --backend $backend \
                     --num_threads $num_threads --fname $RESULTS_FILE
                done
            fi
            done
    done
done