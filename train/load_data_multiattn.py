import csv
import numpy as np
import pandas as pd
from dataprocess.load_descriptions import load_description
from collections import defaultdict
import os


'''
load_description: return {code: code_des}
'''
def gen_batch(data_dir,dicts,max_len=2500,batch_size=16,des_embed=False):
    print('Load data ... ')
    c2ind = dicts['c2ind']#{code:ind_vec}
    ind2c = dicts['ind2c']
    w2ind = dicts['w2ind']
    des_inds_dict = dicts['des_inds_dict'] #{code:dex_ind}
    with open(data_dir,'r') as data:
        rows = csv.reader(data)
        next(rows)
        x_batch = []
        y_batch = []
        des_batch = []
        length = 0
        for row in rows:
            des = []
            if len(x_batch) == batch_size:
                x_batch = np.array(pad_text(x_batch,length))
                yield x_batch,np.array(y_batch),np.array(des_batch)
                x_batch = []
                y_batch = []
                des_batch = []
            l_inds,code_set,labelled = process_label(row[3],c2ind,labelled=False)
            if not labelled:
                continue
            y_batch.append(l_inds)
            xi = [int(w2ind[w]) if w in w2ind.keys() else len(w2ind)+1 for w in row[2].split()]
            if len(xi) > max_len:
                xi = xi[:max_len]
            x_batch.append(xi)
            length = min(int(row[4]),max_len)
            if des_embed:
                for ind in code_set:
                    c = ind2c[ind]
                    if c in des_inds_dict.keys():
                        des.append(des_inds_dict[c][:])
                    else:
                        des.append([len(w2ind)+1])
                des_batch.append(pad_des(des))

        x_batch = np.array(pad_text(x_batch,length))
        yield x_batch,np.array(y_batch),np.array(des_batch)



def process_label(y,c2ind,labelled=False):
    num_labels = len(c2ind)
    y_matrix = np.zeros(num_labels)
    code_set = set()
    labels = y.split(';')

    for l in labels:
        if l in c2ind.keys():
            y_matrix[int(c2ind[l])] = 1
            code_set.add(int(c2ind[l])) #inds of codes
            labelled = True

    return y_matrix,code_set,labelled



def batch_iter(df_set,dicts,batch_size=16):
    '''

    :param train_dir:  os.getcwd()/dev_data/train_dataset.csv
    training set
    vocab_dir: vocab in training set:VOCAB_TRAIN.csv
    codes_dir: code_filtered_dir
    :param batch_size:16
    :return:
    '''
    #dftrain_set = pd.read_csv(train_dir,dtype={"LABELS": str})
    c2ind = dicts['c2ind']#{code:ind_vec}
    ind2c = dicts['ind2c']
    w2ind = dicts['w2ind']
    des_inds_dict = dicts['des_inds_dict'] #{code:dex_ind}

    #df_set = clean_unlabel_data(dftrain_set,c2ind)

    data_len = df_set.shape[0]
    num_batch = int((data_len - 1)/batch_size) + 1

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        rows = df_set.iloc[start_id:end_id] #dataframe
        x = list(rows['TEXT'])
        y = list(rows['LABELS']) #code_set: list of tuple
        # the unique code inds in each instance
        x_ = process_note(x,w2ind)
        y_,code_set = multi_hot_label(y,c2ind) #batch_size,num_labels np.array
        des = process_des(des_inds_dict,w2ind,ind2c,code_set)
        yield x_, y_, des

def clean_unlabel_data(train_dir,dicts):
    df = pd.read_csv(train_dir,dtype={"LABELS": str})
    c2ind = dicts['c2ind']
    inds_del = []
    for i in range(df.shape[0]):
        y = df.iloc[i]['LABELS']
        labels = str(y).split(';')
        inds = [c2ind[c] for c in labels if c in c2ind.keys()]
        if len(inds) == 0:
            inds_del.append(i)
    clean_df = df.drop(df.index[inds_del])

    return clean_df



def process_note(x,w2ind,max_length=2500):
    '''
    icd9_codes = all icd9_codes in dataset -- list
    :return:
    '''

    x_ = []
    length = 0
    for xi in x:
        text = [int(w2ind[w]) if w in w2ind.keys() else len(w2ind)+1 for w in xi.strip().split()]
        length = len(text)
        if length >= max_length:
            text = text[0:max_length]
        x_.append(text)

    if length < max_length:
        max_length = length
    notes = pad_text(x_,max_length)

    return notes


def pad_text(notes,max_length):

    for note in notes:
        length = len(note)
        note.extend([0 for i in range(max_length-length)])
    return notes


def multi_hot_label(y,c2ind):
    batch_size = len(y)
    num_labels = len(c2ind)

    y_matrix = np.zeros((batch_size,num_labels))
    code_set = []

    for i in range(batch_size):
        codes = set()
        labels = str(y[i]).split(';')
        #label = str(y[i]).split(';')
        for l in labels:
            if l in c2ind.keys():
                c = int(c2ind[l])
                y_matrix[i,c] = 1
                #codes.add(l)
                codes.add(c) #inds of codes
        code_set.append(codes)
    return y_matrix,code_set


def process_des(des_inds_dict,w2ind,ind2c,code_set):
    '''

    :param y:
    :param icd_di_dic:
    :param w2ind:
    :param code_set: list[(code inds),(),...]
    :return:
    '''

    des = []
    for inds in code_set:
        codes = [ind2c[i] for i in inds]
        ind_vec = [des_inds_dict[c][:] if c in des_inds_dict.keys() else [len(w2ind)+1] for c in codes]#！！！
        des_padded = pad_des(ind_vec)
        des.append(des_padded)
    #des_padded [batch_size,max_label,max_seqlen]

    des_arr = np.array(des)

    return des_arr

def pad_des(des):

    max_len = max([len(v) for v in des])
    pad_vecs = []
    for v in des:
        if len(v) < max_len:
            v.extend([0 for i in range(max_len-len(v))])
        pad_vecs.append(v)
    return pad_vecs


def load_all_dict(label_set='full'):
    dicts = {}
    c2ind,ind2c = all_codes_list(label_set)
    w2ind,ind2w = all_vocab_list()
    des_inds_dict = load_des_vector()
    dicts['c2ind'] = c2ind
    dicts['ind2c'] = ind2c
    dicts['w2ind'] = w2ind
    dicts['ind2w'] = ind2w
    dicts['des_inds_dict'] = des_inds_dict
    return dicts


def all_vocab_list():
    vocabs_dir = os.path.join(os.pardir,'mimicdata/mimic3/vocab.csv')
    vocabs = set()
    '''
    with open(vocabs_dir,'r') as f:
        reader = csv.reader(f)
        for row in reader:
            vocabs.add(row[0])
    '''
    with open(vocabs_dir,'r') as f:
        for i, row in enumerate(f):
            row = row.rstrip()
            if row != '':
                vocabs.add(row.strip())

    ind2w = {i+1:w for i,w in enumerate(sorted(vocabs))}
    w2ind = {w:i for i,w in ind2w.items()}
    #ind2w = {i+1:w for i,w in enumerate(sorted(vocabs))}
    #w2ind = {w:i+1 for i,w in enumerate(sorted(vocabs))}

    return w2ind,ind2w


def all_codes_list(label_set='full'):
    codes = set()
    if label_set == 'full':
        for s in ['train','dev','test']:
            code_dir = os.path.join(os.pardir,'mimicdata/mimic3/%s_full.csv' %s)
            with open(code_dir) as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    cs = row[3].split(';')
                    for c in cs:
                        if c != '':
                            codes.add(c)
        '''
        code_dir = os.path.join(os.pardir,'mimicdata/mimic3/ALL_CODES_filtered.csv')
        with open(code_dir) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if row[2] != '':
                    codes.add(row[2])
        '''
    else:
        code_dir =  os.path.join(os.pardir,'mimicdata/mimic3/TOP_50_CODES.csv')
        with open(code_dir) as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] != '':
                    codes.add(row[2])
    ind2c = defaultdict(str, {i:c for i,c in enumerate(sorted(codes))})
    c2ind = {c:i for i,c in ind2c.items()}
    #c2ind = {c:i for i,c in enumerate(sorted(codes))}
    #ind2c = {i:c for i,c in enumerate(sorted(codes))}

    return c2ind,ind2c


def load_embedding_vector():

    vocabs_matrix_dir = os.path.join(os.pardir,'mimicdata/mimic3/processed_full.embed')#vocab_matrix.w2v')##
    embed_matrix = []
    with open(vocabs_matrix_dir) as f:
        for l in f:
            wv = l.rstrip().split()[1:]
            vec = np.array(wv).astype(np.float)
            vec = vec / (np.linalg.norm(vec) + 1e-6)
            embed_matrix.append(vec)
        vec = np.random.randn(len(embed_matrix[-1]))
        vec = vec / (np.linalg.norm(vec) + 1e-6)
        embed_matrix.append(vec)
    embed_matrix = np.array(embed_matrix)
    return embed_matrix


def load_des_vector():
    ''':return {code: des_inds}'''

    w2ind,ind2w = all_vocab_list()

    icd_des_dic=load_description() #get {code, [des_words_list]}
    des_inds_dict = {}

    for code, des in icd_des_dic.items():
        des_inds = [w2ind[i] if i in w2ind.keys() else len(w2ind)+1 for i in des]
        des_inds_dict[code] = des_inds
    return des_inds_dict


