import os
import argparse
import numpy as np
from time import time

parser = argparse.ArgumentParser(description="Run commands",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', type=str, default='fc',
                    help='name of model to build as in file models.py without prefix "build_"'
                         'i.e to build cf rnn "rnn_fc should be provided')
parser.add_argument('--rnn_type', type=str, default='LSTM', choices=['LSTM', 'GRU'])
parser.add_argument('--backend', type=str, default='th', choices=['th', 'tf'])
parser.add_argument('--num_threads', type=int, default=0, help="number of cpu threads to use. If zero use gpu. If negative use all cpus")
parser.add_argument('--batch_size', type=int, default=1, help="batch size for training (back and forward pass)")
parser.add_argument('--seq_len', type=int, default=20, help="sequence length for rnn")
parser.add_argument('--unroll_rnn', action='store_true', help="Unroll rnn for theano")
parser.add_argument('--fname', type=str, default=None, help='fname where to save experiment results')
args = parser.parse_args()


if __name__ == '__main__':
    # set seed before any keras input
    np.random.seed(0)

    if args.num_threads != 0:
        # set theano flags
        os.environ['THEANO_FLAGS'] = 'device=cpu'
        # set tensorflow flags
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        consume_less = 'cpu'
        num_threads = 'all'
        if args.num_threads > 0:
            num_threads = str(args.num_threads)
            os.environ['OMP_NUM_THREADS'] = num_threads
    else:
        consume_less = num_threads ='gpu'

    # choose backend
    if args.backend == 'th':
        os.environ['KERAS_BACKEND'] = 'theano'
        import keras.backend as K
        K.set_image_dim_ordering('th')
    else:
        os.environ['KERAS_BACKEND'] = 'tensorflow'
        import keras.backend as K
        K.set_image_dim_ordering('tf')

        if args.num_threads > 0:
            import tensorflow as tf
            config = tf.ConfigProto(intra_op_parallelism_threads=args.num_threads,
                                    inter_op_parallelism_threads=args.num_threads)
            sess = tf.Session(config=config)
            K.set_session(sess)

    import models

    build_fn = getattr(models, 'build_' + args.model)

    if 'rnn' in args.model and args.unroll_rnn:
        model = build_fn(seq_len=args.seq_len, consume_less=consume_less, rnn_type=args.rnn_type)
        model_str = args.rnn_type + args.model[3:]
    elif 'rnn' in args.model:
        model = build_fn(consume_less=consume_less, rnn_type=args.rnn_type)
        model_str = args.rnn_type + args.model[3:]
    else:
        model = build_fn()
        model_str = args.model

    print 'running experiment with model:'
    model.summary()

    input_shape = list(model.input_shape)
    output_shape = list(model.output_shape)
    input_shape[0] = output_shape[0] = args.batch_size

    if 'rnn' in args.model:
        input_shape[1] = output_shape[1] = args.seq_len
    else:
        args.unroll_rnn = args.seq_len = 'None'

    # get training data
    x = np.random.rand(*input_shape)
    y = np.random.rand(*output_shape) > 0.5
    # print input_shape, output_shape

    # force model to compile
    print 'compiling...'
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.predict_on_batch(x[:1])
    model.train_on_batch(x[:1], y[:1])

    print 'running tests'
    times_predict = []
    times_train = []
    for _ in xrange(100):
        t0 = time()
        model.train_on_batch(x, y)
        times_train.append(time() - t0)

        t0 = time()
        model.predict_on_batch(x)
        times_predict.append(time() - t0)

    mean_time_predict = np.mean(times_predict)
    mean_time_train = np.mean(times_train)
    print 'predict', mean_time_predict
    print 'train', mean_time_train

    #for w in model.get_weights():
    #    print w.shape, w.mean()

    # save results
    if args.fname:
        header = None
        if not os.path.exists(args.fname):
            header = 'backend,num_threads,model,batch_size,train_time,predict_time,unroll_rnn,seq_len'

        with open(args.fname, 'ab') as f:
            if header:
                f.write(header+'\n')

            line = [
                args.backend, num_threads, model_str, args.batch_size,
                mean_time_train, mean_time_predict, args.unroll_rnn, args.seq_len
            ]
            line_str = ','.join(map(str, line))
            f.write(line_str+'\n')
