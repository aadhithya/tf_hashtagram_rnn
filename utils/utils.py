import numpy as np
import pickle
import re
import json


def load_data(filepath):
    captions = []
    tags=[]
    with open(filepath, 'r+') as file:
        doc = file.read()

    doc = json.loads(doc)
    for obj in doc:
        for post in doc[obj]:
            hashtags = doc[obj][post]['tags']
            if len(hashtags)>0:
                captions += [''.join(doc[obj][post]['caption'])]
                tags += [hashtags]
    return captions,tags

def write_pickle(data, filepath):
    with open(filepath,'wb') as handle:
       pickle.dump(data,handle)
    return

def get_batch_idx(x,batch_size):
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    ix = 0

    while ix+batch_size < len(indices):
        batch_idx = indices[ix:ix+batch_size]
        ix += batch_size
        yield batch_idx

def word_idx_mappings(doc, tags=True):

    '''
    Creates word2indexand index2word mappings for the given document.

    Input:
    doc: a string containing the document

    Returns:
    word2idx: dictionary with word2index mapping
    idx2word: dictionary with index2word mapping
    vocab_size: size of the vocabulary
    '''
    word2idx = {}
    word2idx['_PAD_']=0
    idx=1

    for word in doc.lower().split():
        if word not in word2idx:
            word2idx[word]=idx
            idx += 1
    idx2word = {index:word for word,index in word2idx.items()}

    vocab_size = len(idx2word)

    return word2idx,idx2word,vocab_size

def generate_data(text_idx):
    for line in text_idx:
        _input = np.array(line[:-1],dtype=np.int32)
        _output = np.array(line[1:],dtype=np.int32)
        yield (_input,_output)

def make_input_output_pairs(data):
    inputs=[]
    outputs=[]
    for line in data:
        inputs += [np.array(line[:-1],dtype=np.int32)]
        outputs += [np.array(line[1:],dtype=np.int32)]
    return inputs, outputs

def text2idx(doc,word2idx, dowrite=False, write_path=None, tags=False):


    lines = doc.split('\n')
    doc_lines = []
    lengths = [] 
    for line in lines:
        if len(line)<3:
            #skip empty lines.
            continue
        words = line.lower().split()
        if not tags:
            line_idx = [word2idx[word] for word in words]
        else:
            line_idx = [word2idx[word] for word in words]
        doc_lines +=[line_idx]
        lengths += [len(line_idx)]
    
    if dowrite:
        if write_path is None:
            raise Exception('No File Path specified! Not writing anything')
            return doc_lines
        else:
            write_pickle(doc_lines,write_path)
    return doc_lines, lengths

def pad_data(data, lengths):
    max_len = lengths[np.argmax(lengths)]
    padded_data=[]
    for line in data:
        line += [0]*(max_len-len(line))
        padded_data += [line]
    return padded_data

def generate_train_test_split(inputs,outputs, lengths,split=0.8):
    
    lens = len(inputs)
    indices = np.arange(lens, dtype=np.int32)
    np.random.shuffle(indices)
    end_idx = int(lens * split)
    train_idx = indices[:end_idx]
    test_idx = indices[end_idx:]
    train_inputs = inputs[train_idx]
    train_outputs = outputs[train_idx]
    test_inputs = inputs[test_idx]
    test_outputs = inputs[test_idx]
    train_lengths = lengths[train_idx]
    test_lengths = lengths[test_idx]
    return train_inputs,train_outputs,train_lengths, test_inputs, test_outputs, test_lengths


def make_one_hot(x,vocab_size, word2idx):
    one_hots=[]
    for line in x:
        mat = np.zeros(shape=(len(line),vocab_size),dtype=np.int32)
        for i,idx in enumerate(line):
            mat[i,idx]=1
        one_hots +=[mat]
    return np.array(one_hots)
            

def generate_sample_data(orig_doc, timesteps=15):

    data_set=[]

    doc = orig_doc.split()

    for ix in range(len(doc)-timesteps):
        sample = ' '.join(doc[ix:ix+timesteps])
        data_set += [sample]
    return data_set
    
    