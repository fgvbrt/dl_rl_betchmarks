import pandas as pd


def generate_markdown_results(df):
    header = '|Model|Theano predict, ms|Tensorflow predict, ms|Theano train, ms|Tensorflow train, ms|\n' \
             '| :---|:---:|:---:|:---:|---:|'

    def get_ms(value_sec):
        return round(value_sec * 10 ** 3, 2)

    def get_times(filt):
        train_time = get_ms(df[filt]['train_time'].iloc[0])
        predict_time = get_ms(df[filt]['predict_time'].iloc[0])
        return predict_time, train_time

    to_write = ''
    for num_threads in ['gpu', '1']:
        for batch_size in [1, 64]:
            device = 'GPU' if num_threads == 'gpu' else 'CPU'
            to_write += "\n\n {} batch_size {}\n\n".format(device, batch_size)
            to_write += header + '\n'

            for model in ['fc', 'cnn_openai', 'cnn_dqn', 'LSTM_fc', 'LSTM_openai', 'LSTM_dqn']:
                filt = (df['num_threads'] == num_threads) & \
                       (df['batch_size'] == batch_size) & \
                       (df['model'] == model)

                th_filt = filt & (df['backend'] == 'th')
                tf_filt = filt & (df['backend'] == 'tf')

                th_train, th_predict = map(str, get_times(th_filt))
                tf_train, tf_predict = map(str, get_times(tf_filt))

                to_write += '|'.join([model, th_predict, tf_predict, th_train, tf_train])
                to_write += '\n'

    return to_write


if __name__ == '__main__':
    df = pd.read_csv('results.txt')
    to_write = generate_markdown_results(df)
    with open('results.md', 'wb')as f:
        f.write(to_write)
