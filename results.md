

 GPU batch_size 1

|Model|Theano predict, ms|Tensorflow predict, ms|Theano train, ms|Tensorflow train, ms|
| :---|:---:|:---:|:---:|---:|
fc|0.84|1.44|0.15|0.44
cnn_openai|2.21|3.03|0.81|1.01
cnn_dqn|3.02|4.59|0.81|1.38
LSTM_fc|7.02|15.99|2.24|3.86
LSTM_openai|10.24|19.75|3.17|4.8
LSTM_dqn|15.78|27.18|4.9|6.16


 GPU batch_size 64

|Model|Theano predict, ms|Tensorflow predict, ms|Theano train, ms|Tensorflow train, ms|
| :---|:---:|:---:|:---:|---:|
fc|0.92|1.5|0.2|0.48
cnn_openai|5.41|5.49|2.78|2.57
cnn_dqn|9.66|11.05|4.68|6.36
LSTM_fc|15.23|17.88|4.2|4.95
LSTM_openai|41.21|46.43|15.32|17.15
LSTM_dqn|87.02|91.3|33.85|43.59


 CPU(1 thread) batch_size 1

|Model|Theano predict, ms|Tensorflow predict, ms|Theano train, ms|Tensorflow train, ms|
| :---|:---:|:---:|:---:|---:|
fc|0.18|0.62|0.04|0.24
cnn_openai|1.7|1.96|0.53|0.66
cnn_dqn|17.17|28.95|2.63|1.95
LSTM_fc|18.78|22.93|5.16|3.4
LSTM_openai|35.43|35.01|11.25|8.4
LSTM_dqn|80.07|146.91|19.3|27.43


 CPU(1 thread) batch_size 64

|Model|Theano predict, ms|Tensorflow predict, ms|Theano train, ms|Tensorflow train, ms|
| :---|:---:|:---:|:---:|---:|
fc|0.37|1.01|0.09|0.35
cnn_openai|61.47|53.49|24.31|20.21
cnn_dqn|133.39|205.6|46.73|56.73
LSTM_fc|135.17|213.51|41.8|76.49
LSTM_openai|1083.4|919.22|374.78|378.69
LSTM_dqn|2279.12|3400.8|711.54|1070.82
