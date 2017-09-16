import numpy as np
import os
import sys
# import model
import keras
from keras import layers
import re
import random
import sys

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def prepare_data(path):
    text = open(path).read()
    text = open(path).read().lower()
    print("Corpus length ", len(text))
    text = re.sub('([-.,!()])', r' \1 ', text)
    text = re.sub('\s{2,}', ' ', text)
    toks = text.split()

    max_len = 25
    step = 5

    sentences = []
    next_toks = []

    for i in range(0, len(toks)-max_len, step):
        sentences.append(" ".join(toks[i : i + max_len]))
        next_toks.append(toks[i+max_len])
    print("Number of Sequences:", len(sentences))

    print(sentences[0:10])
    print(next_toks[0:10])

    # list unique tokens in the corpus
    tokens = sorted(list(set(toks)))
    print("Unique tokens:", len(tokens))
    # dictionary mapping unique tokens to indices in tokens
    tok_indices = dict((token, tokens.index(token)) for token in tokens)

    # now encode the tokens into binary arrays
    print("One Hot........")
    x = np.zeros((len(sentences), max_len, len(tokens)), dtype=np.bool)
    y = np.zeros((len(sentences), len(tokens)), dtype=np.bool)

    for i, sentence in enumerate(sentences):
        for t, token in enumerate(sentence.split()):
            x[i, t, tok_indices[token]] = 1
        y[i, tok_indices[next_toks[i]]] = 1

    model = keras.models.Sequential()
    model.add(layers.LSTM(256, return_sequences=True, input_shape=(max_len, len(tokens))))
    model.add(layers.Dropout(0.2))
    model.add(layers.LSTM(256, return_sequences=False))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(len(tokens), activation='softmax'))

    optimizer = keras.optimizers.RMSprop(lr=0.02)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    for epoch in range(1,50):
        print('epoch', epoch)

        model.fit(x, y, batch_size= 50, epochs=1)

        start_index = random.randint(0, len(toks) - max_len - 1)


        for temperature in [0.2, 0.5, 1.0, 1.2]:
            # generated_text  = ["The", "woman", "looked", "like"]
            generated_text = toks[start_index: start_index + max_len]
            print('--temp:', temperature)
            sys.stdout.write(" ".join(generated_text))

            # we generate 10 tokens
            for i in range(25):
                sampled = np.zeros((1, max_len, len(tokens)))
                for t, token in enumerate(generated_text):
                    sampled[0, t, tok_indices.get(token,0)] = 1

                preds = model.predict(sampled, verbose=0)[0]
                next_index = sample(preds, temperature)
                next_token = tokens[next_index]

                generated_text.append(next_token)
                generated_text = generated_text[1:]

                sys.stdout.write(" " + next_token)
                sys.stdout.flush()
            print()


def train(data_dir):
    path = os.path.join(data_dir, 'input.txt')
    prepare_data(path)

if __name__ == "__main__":
    data_dir = sys.argv[1]
    train(data_dir)