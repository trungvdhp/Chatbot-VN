""" A neural chatbot using sequence to sequence model with
attentional decoder. 

This is based on Google Translate Tensorflow model 
https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/

Sequence to sequence model by Cho et al.(2014)

Created by Chip Huyen as the starter code for assignment 3,
class CS 20SI: "TensorFlow for Deep Learning Research"
cs20si.stanford.edu

This file contains the code to do the pre-processing for the
Cornell Movie-Dialogs Corpus.

See readme.md for instruction on how to run the starter code.
"""
from __future__ import print_function

import os
import random
import re
import codecs

import numpy as np

import config

def get_lines():
    id2line = {}
    file_path = os.path.join(config.DATA_PATH, config.LINE_FILE)
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.split(' +++$+++ ')
            if len(parts) == 5:
                if parts[4][-1] == '\n':
                    parts[4] = parts[4][:-1]
                id2line[parts[0]] = parts[4]
    return id2line

def get_convos():
    """ Get conversations from the raw data """
    filenames = []
    
    for i in range(1, config.NUM_INPUT_FILES):
        filenames.append(str(i) + '.txt')
    output_path = os.path.join(config.DATA_PATH, config.CONVO_FILE)
    f = open(output_path, encoding='utf-8', mode='w')
    
    for filename in filenames:
        file = codecs.open(os.path.join(config.DATA_PATH, filename), encoding='utf-8', mode='r')
        for line in file.readlines():
            if line[0] == '-':
                line = ' '.join(line.split())
                line = line.split('-')[1][1:] + '\n'
                f.write(line)
        file.close()
    f.close()

def question_answers(id2line, convos):
    """ Divide the dataset into two sets: questions and answers. """
    questions, answers = [], []
    for convo in convos:
        for index, line in enumerate(convo[:-1]):
            questions.append(id2line[convo[index]])
            answers.append(id2line[convo[index + 1]])
    assert len(questions) == len(answers)
    return questions, answers

def get_question_answers():
    """ Divide the dataset into sets: questions and answers. """
    file_path = os.path.join(config.DATA_PATH, config.CONVO_FILE)
    convos = []
    with codecs.open(file_path, encoding='utf-8', mode='r') as f:
        max_length = config.BUCKETS[-1][1]
        print('Max length=' + str(max_length))
        convo = []
    
        question = f.readline()
        convo.append(question)

        for line in f.readlines():
            convo.append(line)
            convos.append(convo)
            convo = []
            convo.append(line)
            if len(convos) >= config.MAX_CONVOS_SIZE:
                break
    return convos
    
def prepare_dataset(convos):
    # create path to store all the train & test encoder & decoder
    make_dir(config.PROCESSED_PATH)
    
    # random convos to create the test set
    test_ids = random.sample([i for i in range(len(convos))],config.TESTSET_SIZE)
    
    filenames = ['train.enc', 'train.dec', 'test.enc', 'test.dec']
    files = []
    
    for filename in filenames:
        files.append(codecs.open(os.path.join(config.PROCESSED_PATH, filename), encoding='utf-8', mode='w'))

    for i in range(len(convos)):
        if i in test_ids:
            files[2].write(convos[i][0])
            files[3].write(convos[i][1])
        else:
            files[0].write(convos[i][0])
            files[1].write(convos[i][1])

    for file in files:
        file.close()

def analyse_dataset():
    # create path to store all the train & test encoder & decoder
    make_dir(config.PROCESSED_PATH)
    filenames = ['train.enc', 'train.dec', 'test.enc', 'test.dec']
    files = []
    
    for filename in filenames:
        files.append(codecs.open(os.path.join(config.PROCESSED_PATH, filename), encoding='utf-8', mode='r'))
    bucket_count = int(input("Input number of buckets: "))
    for i in range(0, 4):
        print(filenames[i])
        wordcounts = {}
        line_count = 0
        for line in files[i].readlines():
            line_count += 1
            wordcount = len(basic_tokenizer(line))
            if not wordcount in wordcounts:
                wordcounts[wordcount] = 0
            wordcounts[wordcount] += 1
        avg = line_count/bucket_count
        start = 1
        end = 1
        size = 0
        rs = ""
        span = avg/(bucket_count*2)
        for k, v in sorted(wordcounts.items()):
            if size + v < avg + span:
                size += v
                end = k
                if end==len(wordcounts):
                     rs += '(' + str(start) + ',' + str(end) + '):' + str(size) + ', '
            else:
                if k==len(wordcounts):
                    size += v
                rs += '(' + str(start) + ',' + str(end) + '):' + str(size) + ', '
                start = k
                end = k
                size = v
        print('Total: ' + str(line_count))
        print('Average ' + str(avg) + ' per bucket')
        print(' '.join('(' + str(k) + ',' + str(v) + ')' for k, v in sorted(wordcounts.items())))
        print(rs)
        files[i].close()

def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass

def basic_tokenizer(line, normalize_digits=True):
    """ A basic tokenizer to tokenize text into tokens.
    Feel free to change this to suit your need. """
    line = re.sub('<u>', '', line)
    line = re.sub('</u>', '', line)
    line = re.sub('\[', '', line)
    line = re.sub('\]', '', line)
    words = []
    _WORD_SPLIT = re.compile("([.,!?\"'-<>:;)(])")
    _DIGIT_RE = re.compile(r"\d")
    
    for fragment in line.strip().lower().split():
        for token in re.split(_WORD_SPLIT, fragment):
            if not token:
                continue
            if normalize_digits:
                token = re.sub(_DIGIT_RE, '#', token)
            words.append(token)
    return words

def build_vocab(filename, normalize_digits=True):
    in_path = os.path.join(config.PROCESSED_PATH, filename)
    out_path = os.path.join(config.PROCESSED_PATH, 'vocab.{}'.format(filename[-3:]))

    vocab = {}
    with codecs.open(in_path, encoding='utf-8', mode='r') as f:
        for line in f.readlines():
            for token in basic_tokenizer(line):
                if not token in vocab:
                    vocab[token] = 0
                vocab[token] += 1

    sorted_vocab = sorted(vocab, key=vocab.get, reverse=True)
    with codecs.open(out_path, encoding='utf-8', mode='w') as f:
        f.write('<pad>' + '\n')
        f.write('<unk>' + '\n')
        f.write('<s>' + '\n')
        f.write('<\s>' + '\n') 
        index = 4
        for word in sorted_vocab:
            if vocab[word] < config.THRESHOLD:
                #print(word + '-' + str(index))
                with open(config.CONFIG_PATH, 'a') as cf:
                    if filename[-3:] == 'enc':
                        cf.write('ENC_VOCAB = ' + str(index) + '\n')
                    else:
                        cf.write('DEC_VOCAB = ' + str(index) + '\n')
                break
            f.write(word + '\n')
            index += 1

def load_vocab(vocab_path):
    with codecs.open(vocab_path, encoding='utf-8', mode='r') as f:
        words = f.read().splitlines()
    return words, {words[i]: i for i in range(len(words))}

def sentence2id(vocab, line):
    return [vocab.get(token, vocab['<unk>']) for token in basic_tokenizer(line)]

def token2id(data, mode):
    """ Convert all the tokens in the data into their corresponding
    index in the vocabulary. """
    vocab_path = 'vocab.' + mode
    in_path = data + '.' + mode
    out_path = data + '_ids.' + mode

    _, vocab = load_vocab(os.path.join(config.PROCESSED_PATH, vocab_path))
    in_file = codecs.open(os.path.join(config.PROCESSED_PATH, in_path), encoding='utf-8', mode='r')
    out_file = codecs.open(os.path.join(config.PROCESSED_PATH, out_path), encoding='utf-8', mode= 'w')
    
    lines = in_file.read().splitlines()
    for line in lines:
        if mode == 'dec': # we only care about '<s>' and </s> in encoder
            ids = [vocab['<s>']]
        else:
            ids = []
        ids.extend(sentence2id(vocab, line))
        # ids.extend([vocab.get(token, vocab['<unk>']) for token in basic_tokenizer(line)])
        if mode == 'dec':
            ids.append(vocab['<\s>'])
        out_file.write(' '.join(str(id_) for id_ in ids) + '\n')

def prepare_raw_data():
    print('Preparing raw data into train set and test set ...')
    #id2line = get_lines()
    #convos = get_convos()
    convos = get_question_answers()
    prepare_dataset(convos)

def process_data():
    print('Preparing data to be model-ready ...')
    build_vocab('train.enc')
    build_vocab('train.dec')
    token2id('train', 'enc')
    token2id('train', 'dec')
    token2id('test', 'enc')
    token2id('test', 'dec')

def load_data(enc_filename, dec_filename, max_training_size=None):
    encode_file = codecs.open(os.path.join(config.PROCESSED_PATH, enc_filename), encoding='utf-8', mode='r')
    decode_file = codecs.open(os.path.join(config.PROCESSED_PATH, dec_filename), encoding='utf-8', mode='r')
    encode, decode = encode_file.readline(), decode_file.readline()
    data_buckets = [[] for _ in config.BUCKETS]
    i = 0
    while encode and decode:
        if (i + 1) % 10000 == 0:
            print("Bucketing conversation number", i)
        encode_ids = [int(id_) for id_ in encode.split()]
        decode_ids = [int(id_) for id_ in decode.split()]
        for bucket_id, (encode_max_size, decode_max_size) in enumerate(config.BUCKETS):
            if len(encode_ids) <= encode_max_size and len(decode_ids) <= decode_max_size:
                data_buckets[bucket_id].append([encode_ids, decode_ids])
                break
        encode, decode = encode_file.readline(), decode_file.readline()
        i += 1
    return data_buckets

def _pad_input(input_, size):
    return input_ + [config.PAD_ID] * (size - len(input_))

def _reshape_batch(inputs, size, batch_size):
    """ Create batch-major inputs. Batch inputs are just re-indexed inputs
    """
    batch_inputs = []
    for length_id in range(size):
        batch_inputs.append(np.array([inputs[batch_id][length_id]
                                    for batch_id in range(batch_size)], dtype=np.int32))
    return batch_inputs


def get_batch(data_bucket, bucket_id, batch_size=1):
    """ Return one batch to feed into the model """
    # only pad to the max length of the bucket
    encoder_size, decoder_size = config.BUCKETS[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    for _ in range(batch_size):
        encoder_input, decoder_input = random.choice(data_bucket)
        # pad both encoder and decoder, reverse the encoder
        encoder_inputs.append(list(reversed(_pad_input(encoder_input, encoder_size))))
        decoder_inputs.append(_pad_input(decoder_input, decoder_size))

    # now we create batch-major vectors from the data selected above.
    batch_encoder_inputs = _reshape_batch(encoder_inputs, encoder_size, batch_size)
    batch_decoder_inputs = _reshape_batch(decoder_inputs, decoder_size, batch_size)

    # create decoder_masks to be 0 for decoders that are padding.
    batch_masks = []
    for length_id in range(decoder_size):
        batch_mask = np.ones(batch_size, dtype=np.float32)
        for batch_id in range(batch_size):
            # we set mask to 0 if the corresponding target is a PAD symbol.
            # the corresponding decoder is decoder_input shifted by 1 forward.
            if length_id < decoder_size - 1:
                target = decoder_inputs[batch_id][length_id + 1]
            if length_id == decoder_size - 1 or target == config.PAD_ID:
                batch_mask[batch_id] = 0.0
        batch_masks.append(batch_mask)
    return batch_encoder_inputs, batch_decoder_inputs, batch_masks

if __name__ == '__main__':
    prepare_raw_data()
    process_data()
    #get_convos()