import sys
sys.path.append('../')

import train.models_torch as models
import train.load_data_tools_pytorch as load_data_tools
import train.evaluation as evaluation

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
from tqdm import tqdm
import csv


def train(label_set,des_embed,model_name,version='mimic3',gpu=True):


    csv.field_size_limit(sys.maxsize)
    print('training ... ')

    print('Configuring ... ')
    dicts = load_data_tools.load_all_dict(label_set=label_set,version=version)
    num_labels = len(dicts['c2ind'])
    config, model = pick_model(model_name,num_labels,version=version)
    if gpu:
        model.cuda()
    optimizer = optim.Adam(model.parameters(),lr=config.learning_rate,weight_decay=0)
    print(model)


    MIMIC_DIR = os.path.join(os.path.pardir,'mimicdata/'+version)
    MODEL_DIR = os.path.join(os.path.pardir,'train/'+version)
    train_dir = '%s/train_%s.csv' % (MIMIC_DIR,str(config.label_set))
    dev_dir = '%s/dev_%s.csv' % (MIMIC_DIR,str(config.label_set))
    test_dir = '%s/test_%s.csv' % (MIMIC_DIR,str(config.label_set))
    if version == 'mimic2':
        train_dir = '%s/train.csv' % (MIMIC_DIR)
        dev_dir = '%s/test.csv' % (MIMIC_DIR)
        test_dir = '%s/test.csv' % (MIMIC_DIR)

    save_path = os.path.join(MODEL_DIR,'torch_models/best_CAML_model_torch')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if model.des_embed:
        saved_model_name_ = 'DR_%s_%s_%s' % (model_name,str(config.label_set),version)
    else:
        saved_model_name_ = '%s_%s_%s' % (model_name,str(config.label_set),version)

    print('Model: ' + saved_model_name_)

    if config.label_set != 'full':
        pn = [5]
    else:
        pn = [8,15]

    criterion = []
    best_p8 = 0.0
    datadf = load_data_tools.clean_unlabel_data(train_dir,dicts)

    for epoch in range(config.num_epochs):
        losses = []
        model.train()
        #batches = load_data_tools.gen_batch(train_dir,dicts,max_len=2500,batch_size=16,des_embed=des_embed)
        batches = load_data_tools.batch_iter(datadf,dicts,batch_size=16) #config.batch_size
        for i, batch in tqdm(enumerate(batches)):
            x_batch, y_batch,des = batch
            x_batch = Variable(torch.LongTensor(x_batch))
            y_batch = Variable(torch.FloatTensor(y_batch))
            if gpu:
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()


            optimizer.zero_grad()

            y_hat,loss,_ = model(x_batch,y_batch,des)
            loss.backward()
            optimizer.step()
            losses.append(loss.data[0])

            if i % config.print_per_batch == 0:
                print(str(epoch) + ' | Total_batch:' + str(i) +' | train loss: '+ str(np.mean(losses[-10:]))
                      + ' | seq_length: ' + str(x_batch.size(1)))
                #print('Saving model ...')
                #save_path = 'best_%s.pt' % model_name_

                #torch.save(model.cpu().state_dict(),save_path)
                #model.cuda()
                #dev_test(model,test_dir,dicts)
                #test(model,test_dir,dicts)
                #model.cuda()


        p_at_8,f1,dev_l = dev(model,dev_dir,dicts,des_embed,pn,gpu=gpu)
        criterion.append(p_at_8)
        #if config.label_set == 'full':
        #    criterion.append(p_at_8)
        #else:
        #    criterion.append(dev_l)

        #if p_at_8 > best_p8:
        #    best_p8 = p_at_8
        if p_at_8 > best_p8:
            best_p8 = p_at_8
            print('Saving model ...')
            save_path = 'best_%s.pt' % saved_model_name_
            torch.save(model.cpu().state_dict(),save_path)
            if gpu:
                model.cuda()

        if early_stop(criterion,config.label_set,patience):
            #print(criterion)

            print('Early stop ... ')
            print('Test on test set: ')
            test(save_path,test_dir,des_embed,pn,model_name,label_set,gpu=gpu,version=version)
            #dev_test(model,test_dir,dicts,pn)

            break


def dev(model,dev_dir,dicts,des_embed,pn,gpu):

    model.eval()
    print('Test on dev set: ')
    #devdf = load_data_tools.clean_unlabel_data(dev_dir,dicts)
    #dev_batches = load_data_tools.batch_iter(devdf,dicts,16)
    ys_hat_dev = []
    ys_true_dev = []
    losses_dev = []
    dev_batches = load_data_tools.gen_batch(dev_dir,dicts,max_len=2500,batch_size=1,des_embed=des_embed)

    for dev_batch in tqdm(dev_batches):

        x_dev, y_dev,des = dev_batch
        x_dev = Variable(torch.LongTensor(x_dev), requires_grad=False)
        y_dev = Variable(torch.FloatTensor(y_dev))
        if gpu:
            x_dev = x_dev.cuda()
            y_dev = y_dev.cuda()
        model.zero_grad()
        y_hat_dev, dev_loss,_ = model(x_dev,y_dev,des)
        losses_dev.append(dev_loss.data[0])
        ys_hat_dev.append(y_hat_dev.data.cpu().numpy())
        ys_true_dev.append(y_dev.data.cpu().numpy())
    yh_dev = np.concatenate(ys_hat_dev,axis=0)
    yt_dev = np.concatenate(ys_true_dev,axis=0)

    meatures = evaluation.eval(yh_dev,yt_dev,n=pn)
    p_at_8 = meatures[4][0]
    macro_f1 = meatures[0][-1]
    #criterion.append(p_at_8)
    evaluation.print_results(meatures)
    dev_l = np.mean(losses_dev)
    print('The loss of dev: ' + str(dev_l))

    return p_at_8,macro_f1,dev_l


def test(save_path,test_dir,des_embed,pn,model_name,label_set,gpu,version='mimic3'):
    print('Test Configuring ... ')
    dicts = load_data_tools.load_all_dict(label_set=label_set,version=version)

    num_labels = len(dicts['c2ind'])
    vocab_size = len(dicts['w2ind'])

    config,model = pick_model(model_name,num_labels,version=version)

    #model = torch.load(save_path)
    model.load_state_dict(torch.load(save_path))
    if gpu:
       model.cuda()

    print('Test on test dataset ... ')

    model.eval()
    #testdf = load_data_tools.clean_unlabel_data(test_dir,dicts)
    #test_batches = load_data_tools.batch_iter(testdf,dicts,16)
    losses_test = []
    ys_hat_test = []
    ys_true_test = []

    if model.des_embed and label_set=='full':
        code_vecs,code_inds = unseen_code_vecs(dicts,label_set)
        Z_unseen = model.des_model(code_vecs)[0]
        for i, ind in enumerate(code_inds):
            model.fc2.weight[ind,:] = Z_unseen.data[i]
            model.fc2.bias[ind] = 0

    test_batches = load_data_tools.gen_batch(test_dir,dicts,max_len=2500,batch_size=1,des_embed=des_embed)


    for test_batch in tqdm(test_batches):
        x_test, y_test,des = test_batch
        x_test = Variable(torch.LongTensor(x_test), requires_grad=False)
        y_test = Variable(torch.FloatTensor(y_test))
        if gpu:
            x_test = x_test.cuda()
            y_test = y_test.cuda()
        model.zero_grad()
        y_hat_test, test_loss,_ = model(x_test,y_test,des)
        losses_test.append(test_loss.data[0])
        ys_hat_test.append(y_hat_test.data.cpu().numpy())
        ys_true_test.append(y_test.data.cpu().numpy())
    yh_test = np.concatenate(ys_hat_test,axis=0)
    yt_test = np.concatenate(ys_true_test,axis=0)

    meatures = evaluation.eval(yh_test,yt_test,n=pn)

    evaluation.print_results(meatures)
    print('The loss of test: ' + str(np.mean(losses_test)))


def early_stop(criterion,label_set,patience):
    if not np.all(np.isnan(criterion)) :#when criterion all = nan, return false,else: true
        #if label_set == 'full':
        return np.nanargmax(criterion) < len(criterion) - 10#patience
        #else:
        #    return np.nanargmin(criterion) < len(criterion) - 10 ###???
    else:
        return False

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

def update_weight(beta,z,inds):
    z = np.squeeze(z).transpose()
    num_label = len(inds)
    for i in range(num_label):
        ind = inds[i]
        beta[:,ind] = z[:,i]
    return beta

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
        #config = models.CNNConfig(num_labels,reg_lambda=reglambda,learning_rate=lr,label_set=label_set,version=version)
        config = models.CNNConfig(num_labels,filter_width=4,num_filters=500,reg_lambda=reglambda,learning_rate=lr,label_set=label_set,version=version)
        model = models.CAML_model(config,des_embed=des_embed) #true:drcaml/false:drcaml
    return config, model

if __name__ == '__main__':
    #vocab_matrix_dir = FLAGS.vocab_matrix_dir
    #parameters:
    des_embed = False #T or F
    label_set = 'full' #full or 50
    model_name = 'self_attn' #CAML or CNN self_attn
    version = 'mimic2' #mimic3/mimic2
    patience = 10
    batch_size = 16
    gpu = True


    MIMIC_DIR = os.path.join(os.path.pardir,'mimicdata')
    DATA_DIR = '%s/%s'%(MIMIC_DIR, version)
    test_dir = '%s/test_full.csv' % DATA_DIR

    train(label_set,des_embed,model_name,version=version,gpu=gpu)

