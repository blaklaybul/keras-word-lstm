import numpy as np
import argparse
import os

from model import Model

def load_data(data_dir):

    input_file = os.path.join(data_dir, "input.txt")
    vocab_fil = os.path.join(data_dir, "vocab.pkl")

def train(args):
    data = load_data(args.data_dir)
    model = Model(args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data",
                        help="data directory containing input.txt")
    parser.add_argument("--input_encoding", type=str, default=None,
                        help="character encoding of input.txt")
    parser.add_argument("--save_dir", type=str, default="save",
                        help="directory to store checkpointed models.")
    parser.add_argument("--rnn_size", type=int, default=256,
                        help="size of RNN hidden state")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="number of layers in the RNN")
    parser.add_argument("--model", type=str, default='lstm',
                        help="rnn, gru, or lstm")
    parser.add_argument("--seq_length", type=int, default=25,
                        help="rnn sequence length")
    parser.add_argument("--batch_size", type=int, default=50,
                        help="minibatch size")
    parser.add_argument("--save_every", type=int, default=1000,
                        help="save frequency")
    parser.add_argument("--grad_clip", type=float, default=5.,
                        help="clip gradients at this value")
    parser.add_argument("--learning_rate", type=float, default=0.002,
                        help="learning rate")
    parser.add_argument("--decay_rate", type=float, default=0.97,
                        help="decay rate for rmsprop")
    parser.add_argument("--gpu_mem", type=float, default=0.6667,
                        help="%% of gpu memory to be allocated, default is 66.67%%")
    parser.add_argument("--init_from", type=str, default=None,
                        help="location of saved model as h5 file")

    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()