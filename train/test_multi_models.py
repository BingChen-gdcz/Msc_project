
import sys
sys.path.append('../')

import train.models_multiattn_torch as models
import train.load_data_multiattn as load_data_tools
import train.evaluation as evaluation

import torch
from torch.autograd import Variable
import numpy as np
import os
from tqdm import tqdm
import csv


def test(save_path,test_dir,dicts,n):

    #model = torch.load(save_path)
    model.load_state_dict(torch.load(save_path))
    model.cuda()

    print('Test on test dataset ... ')

    model.eval()
    testdf = load_data_tools.clean_unlabel_data(test_dir,dicts)
    test_batches = load_data_tools.gen_batch(test_dir,dicts,max_len=2500,batch_size=1,des_embed=des_embed)
    #test_batches = load_data_tools.batch_iter(testdf,dicts,32)
    losses_test = []
    ys_hat_test = []
    ys_true_test = []

    if model.des_embed:
        code_vecs,code_inds = unseen_code_vecs(dicts,label_set)
        Z_unseen = model.des_model(code_vecs)[0]
        #code_inds = Variable(torch.LongTensor(code_inds)).cuda()
        #inds = torch.LongTensor(code_inds).cuda()
        for i,ind in enumerate(code_inds):
            model.fc2.weight[ind,:] = Z_unseen.data[i]
            model.fc2.bias[ind] = 0

    for test_batch in tqdm(test_batches):
        x_test, y_test,des = test_batch
        x_test = Variable(torch.LongTensor(x_test), requires_grad=False).cuda()
        y_test = Variable(torch.FloatTensor(y_test)).cuda()
        with torch.no_grad():
            model.zero_grad()
            y_hat_test, test_loss,alpha = model(x_test,y_test,des) ##(batch,seq_length,#labels)
        losses_test.append(test_loss.data[0])
        ys_hat_test.append(y_hat_test.data.cpu().numpy())
        ys_true_test.append(y_test.data.cpu().numpy())
    yh_test = np.concatenate(ys_hat_test)
    yt_test = np.concatenate(ys_true_test)

    meatures = evaluation.eval(yh_test,yt_test,n=n)
    evaluation.print_results(meatures)
    print('The loss of dev: ' + str(np.mean(losses_test)))

def unseen_code_inds(dicts,label_set):
    MIMIC3_DIR = os.path.join(os.path.pardir,'mimicdata/mimic3')
    train_dir = '%s/train_%s.csv' % (MIMIC3_DIR,str(label_set))

    #for des_embed = true
    c2ind = dicts['c2ind']
    unseen_codes = set(c2ind.keys())

    with open(train_dir,'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            codes = set(str(row[3]).split(';'))
            unseen_codes = unseen_codes.difference(codes)
    inds = set([c2ind[c] for c in unseen_codes if c != ''])
    return inds

def unseen_code_vecs(dicts,label_set):

    code_inds = unseen_code_inds(dicts,label_set)
    ind2c = dicts['ind2c']
    w2ind = dicts['w2ind']
    des_inds_dict = dicts['des_inds_dict'] #get {code, [des_words_inds}
    code_vecs = load_data_tools.process_des(des_inds_dict,w2ind,ind2c,[code_inds])
    '''
    code_vecs = []
    for ind in code_inds:
        code = ind2c[ind]
        if code in des_inds_dict:
            v = des_inds_dict[code][:]
        else:
            v = [len(w2ind)+1]
        code_vecs.append(v)

    codes = [ind2c[ind] for ind in code_inds]
    code_vecs = [des_inds_dict[c][:] if c in des_inds_dict.keys() else [len(ind2w)+1] for c in codes]
    code_vecs = load_data_tools.pad_des(code_vecs)
    '''
    return code_vecs,code_inds

if __name__ == '__main__':
    #vocab_matrix_dir = FLAGS.vocab_matrix_dir

    des_embed = False #T or F
    label_set = 'full' #full or 50
    model_name = 'CAML' #CAML or CNN
    num_attn = 3

    MIMIC3_DIR = os.path.join(os.path.pardir,'mimicdata/mimic3')
    test_dir = '%s/test_full.csv' % MIMIC3_DIR

    if des_embed:
        model_name_ = 'DR_%s_%s_attn_%s' % (model_name,str(label_set),str(num_attn))
    else:
        model_name_ = '%s_%s_attn_%s' % (model_name,str(label_set),str(num_attn))
    save_path = 'best_%s.pt' % model_name_

    if label_set == 'full':
        n = [8,15]
    else:
        n = [5]

    print('Configuring ... ')
    dicts = load_data_tools.load_all_dict(label_set=label_set)

    num_labels = len(dicts['c2ind'])
    vocab_size = len(dicts['w2ind'])

    config = models.CNNConfig(num_labels,label_set=label_set,batch_size=1)
    model = models.CAML_model(config,num_attn=num_attn,des_embed=des_embed) #true:drcaml/false:drcaml


    if model_name == 'CNN':
        des_embed = False
        config = models.CNNConfig(num_labels,vocab_size,filter_width=4,num_filters=500,learning_rate=0.003,label_set=label_set)
        model = models.CNN_model(config)
    test(save_path,test_dir,dicts,n)
