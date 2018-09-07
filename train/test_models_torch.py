
import sys
sys.path.append('../')

import train.models_torch as models
import train.load_data_tools_pytorch as load_data_tools
import train.evaluation as evaluation
import train.extraction as extraction

import torch
from torch.autograd import Variable
import numpy as np
import os
from tqdm import tqdm
import csv


def test(save_path,test_dir,dicts,n,label_set,version='mimic3'):

    #model = torch.load(save_path)
    model.load_state_dict(torch.load(save_path))
    model.cuda()

    print('Test on test dataset ... ')

    tp_file = open('%s_tp.txt'%save_path,'w')
    fp_file = open('%s_fp.txt'%save_path,'w')
    window_size = model.conv.weight.data.size()[2]


    test_batches = load_data_tools.gen_batch(test_dir,dicts,max_len=2500,batch_size=1,des_embed=des_embed)
    #testdf = load_data_tools.clean_unlabel_data(test_dir,dicts)
    #test_batches = load_data_tools.batch_iter(testdf,dicts,1)
    losses_test = []
    ys_hat_test = []
    ys_true_test = []
    code_vecs,code_inds = unseen_code_vecs(dicts,label_set,version=version)

    if model.des_embed and len(code_inds)>0:
        print('Unseen code process ... ')
        Z_unseen = model.des_model(code_vecs)[0]
        #code_inds = Variable(torch.LongTensor(code_inds)).cuda()
        #inds = torch.LongTensor(code_inds).cuda()
        for i,ind in enumerate(code_inds):
            model.fc2.weight.data[ind,:] = Z_unseen.data[i]
            model.fc2.bias.data[ind] = 0

    model.eval()
    for i,test_batch in enumerate(tqdm(test_batches)):
        x_test, y_test,des = test_batch
        x_test = Variable(torch.LongTensor(x_test), volatile=True,requires_grad=False).cuda()
        y_test = Variable(torch.FloatTensor(y_test)).cuda()
        #with torch.no_grad():
        model.zero_grad()
        if i % 300 == 0:
            y_hat_test, test_loss,alpha = model(x_test,y_test,des,get_attention=True)
        else:
            y_hat_test, test_loss,alpha = model(x_test,y_test,des) ##(batch,seq_length,#labels)
        losses_test.append(test_loss.data[0])
        ys_hat_test.append(y_hat_test.data.cpu().numpy())
        ys_true_test.append(y_test.data.cpu().numpy())
        y_test_data = y_test.data.cpu().numpy()
        #x_test_data = x_test.data.cpu().numpy()
        if i % 300 == 0:
            extraction.info_snippets(alpha,x_test,y_hat_test,y_test_data,dicts,filter_width=window_size,k=4)
            #extraction.info_snippets(alpha,x_test,y_hat_test,ind2w,dicts)

    yh_test = np.concatenate(ys_hat_test)
    yt_test = np.concatenate(ys_true_test)


    meatures = evaluation.eval(yh_test,yt_test,n=n)
    evaluation.print_results(meatures)
    print('The loss of dev: ' + str(np.mean(losses_test)))

def unseen_code_inds(dicts,label_set,version):
    if version == 'mimic2':
        MIMIC_DIR = os.path.join(os.path.pardir,'mimicdata/mimic2')
        train_dir = '%s/train.csv' % (MIMIC_DIR)
    else:
        MIMIC_DIR = os.path.join(os.path.pardir,'mimicdata/mimic3')
        train_dir = '%s/train_%s.csv' % (MIMIC_DIR,str(label_set))

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

def unseen_code_vecs(dicts,label_set,version):

    code_inds = unseen_code_inds(dicts,label_set,version=version)
    ind2c = dicts['ind2c']
    w2ind = dicts['w2ind']
    des_inds_dict = dicts['des_inds_dict'] #get {code, [des_words_inds}
    #print(code_inds)
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

def pick_model(model_name,num_labels,version):
    print(model_name)
        #CNN
    if model_name == 'CNN':
        config = models.CNNConfig(num_labels,filter_width=4,num_filters=500,learning_rate=0.003,label_set=label_set,version=version)
        model = models.CNN_model(config)

    if model_name == 'self_attn':
        config = models.CNNConfig(num_labels,label_set=label_set,version=version)
        model = models.self_Attn(config) #true:drcaml/false:drcaml
        batch_size = 16
    #CAML
    if model_name == 'CAML':
        if version == 'mimic2':
            lr = 0.0003
            reglambda = 0.1
        else:
            lr = 0.0001
            reglambda = 0.01
        config = models.CNNConfig(num_labels,reg_lambda=reglambda,learning_rate=lr,label_set=label_set,version=version)
        model = models.CAML_model(config,des_embed=des_embed) #true:drcaml/false:drcaml
    return config, model


if __name__ == '__main__':
    #vocab_matrix_dir = FLAGS.vocab_matrix_dir

    des_embed = False #T or F
    label_set = 'full' #full or 50
    model_name = 'CNN' #CAML or CNN
    version = 'mimic3'

    MIMIC3_DIR = os.path.join(os.path.pardir,'mimicdata/mimic3')
    MIMIC_DIR = os.path.join(os.path.pardir,'mimicdata/'+version)
    test_dir = '%s/test_full.csv' % MIMIC3_DIR
    if version == 'mimic2':
        test_dir = '%s/test.csv' % (MIMIC_DIR)

    if des_embed:
        model_name_ = 'DR_%s_%s' % (model_name,str(label_set))
    else:
        model_name_ = '%s_%s' % (model_name,str(label_set))
    save_path = 'best_%s.pt' % model_name_
    '''

    if des_embed:
        saved_model_name_ = 'DR_%s_%s_%s' % (model_name,str(label_set),version)
    else:
        saved_model_name_ = '%s_%s_%s' % (model_name,str(label_set),version)
    save_path = 'best_%s.pt' % saved_model_name_
    print(saved_model_name_)
    '''

    if label_set == 'full':
        n = [8,15]
    else:
        n = [5]

    print('Configuring ... ')
    dicts = load_data_tools.load_all_dict(label_set=label_set,version=version)

    num_labels = len(dicts['c2ind'])
    vocab_size = len(dicts['w2ind'])
    ind2w = dicts['ind2w']
    config, model = pick_model(model_name,num_labels,version=version)
    #config = models.CNNConfig(num_labels,filter_width=4,num_filters=500,label_set=label_set)
    #config = models.CNNConfig(num_labels,vocab_size,label_set=label_set)
    #model = models.CAML_model(config,des_embed=des_embed) #true:drcaml/false:drcaml
    test(save_path,test_dir,dicts,n,label_set,version=version)
