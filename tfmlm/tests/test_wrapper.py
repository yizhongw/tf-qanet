import tensorflow as tf
from bilm.wrapper import BilmWrapper
import argparse

def main(args):
    max_batch_size = 4
    wrapper = BilmWrapper(args.option_file, args.model_file, max_batch_size=max_batch_size)
    print ('num of layers: %s' % wrapper.num_layers())
    print ('dim per layers: %s' % wrapper.dim_per_layer())
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sentences = [
        'I have a dream .'.split(),
        'So be it'.split()
    ]
    # this will output the token embedding together with all the transformer layers.
    # the shape should be [batch_size, wrapper.num_layers(), seq_length, wrapper.dim_per_layer()].
    lm_embedding0 = wrapper.run(sess, sentences)
    print(lm_embedding0.shape)
    sentences = [
        'I have a dream .'.split(),
        'So be it'.split()
    ]
    lm_embedding1 = wrapper.run(sess, sentences)
    print(lm_embedding1.shape)
    with open('../embedding.output.txt', 'w') as f:
        f.write(str(lm_embedding0) + '\n')
        f.write(str(lm_embedding1) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', help='Model file in hdf5 format')
    parser.add_argument('--option_file', help='Option file in json format')

    args = parser.parse_args()
    main(args)
