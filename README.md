# Deep Reinforcement learning benchmarks

Comparison of theano and tensorflow for deep RL.

Main idea is test arcitectures suitable for deep RL purposes i.e. not so deep CNN/FC networks optionally with LSTM layer.

## Operations to compare:
  * forward pass with batch size one for communicating with environment
  * forward + backward pass with various batch sized (for training)

Test script runs all networks on GPU as well as CPU (1 thread and all available threads).

## Architectures to compare:

1. FC network
2. DQN CNN aritecture
3. openai-universe-starter(openai) agent CNN arcitecture
4. LSTM network
5. LSTM with DQN convolution
6. LSTM with openai convolution

## Running benchmark

To run benchmarks just use script:

    $ ./run_all.sh

Optionally a separate experiment can be run with command

    $ python run_benchmark.py
    
Use flag --help to see options for experiment.

## Results

Raw results are in file [results.txt](results.txt). Preprocessed result can be found [here](results.md).
All results were obtained with GeForce GTX 980 and Intel(R) Core(TM) i7-4770K CPU @ 3.50GHz.

