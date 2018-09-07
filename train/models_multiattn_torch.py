import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch .nn.init import xavier_uniform
from math import floor

import train.load_data_tools_pytorch as load_data_tools

class CNNConfig(object):
    #from models.cnn_model import CNNModel

    '''
    config of CNN: de, dc, k,learning_rate, dropout_prob
    num_labels, vocab_size
    '''

    embedding_dim = 100 #de
    seq_len = 2500 #doc_length
    num_epochs = 200

    save_per_batch = 300
    print_per_batch = 50

    def __init__(self,num_labels,reg_lambda=0.01,filter_width=10,num_filters=50,
                 dropout_prob=0.2,learning_rate=0.0001,label_set='full',batch_size=16):
        self.num_labels = num_labels
        self.reg_lambda = reg_lambda
        self.filter_width = filter_width #k
        self.num_filters = num_filters #dc
        self.dropout_prob = dropout_prob
        self.learning_rate = learning_rate
        self.label_set = label_set
        self.batch_size = batch_size

class CAML_model(nn.Module):

    def __init__(self,config,num_attn=1,des_embed=False):
        super(CAML_model, self).__init__()
        torch.manual_seed(1337)
        self.config = config
        self.num_attn = num_attn
        self.des_embed = des_embed

        embed_matrix = torch.Tensor(load_data_tools.load_embedding_vector())
        self.word_embeddings = nn.Embedding(embed_matrix.size(0),embed_matrix.size(1))
        self.word_embeddings.weight.data = embed_matrix.clone()

        self.dropout = nn.Dropout(config.dropout_prob)
        self.conv = nn.Conv1d(config.embedding_dim,config.num_filters,config.filter_width,stride=1,
                              padding=floor(config.filter_width/2))
        xavier_uniform(self.conv.weight)

        self.multiattn = nn.ModuleList([nn.Linear(config.num_filters,config.num_labels) for i in range(self.num_attn)])
        for fc in self.multiattn:
            xavier_uniform(fc.weight)

        #self.fc1 = nn.Linear(config.num_filters,config.num_labels)
        #xavier_uniform(self.fc1.weight)

        self.fc2 = nn.Linear(config.num_filters * self.num_attn,config.num_labels) #beta([8921, 50])
        xavier_uniform(self.fc2.weight)

        if des_embed:
            self.des_embedding = nn.Embedding(embed_matrix.size(0),embed_matrix.size(1))
            self.des_embedding.weight.data = embed_matrix.clone()

            self.conv_des = nn.Conv1d(config.embedding_dim,config.num_filters,config.filter_width,stride=1,
                              padding=floor(config.filter_width/2))
            xavier_uniform(self.conv_des.weight)

            self.fc_des = nn.Linear(config.num_filters,config.num_filters)
            xavier_uniform(self.fc_des.weight)

    def des_model(self,des):

        #diffs = []
        Z = []
        for d in des:
            if len(d) > 0:
                dtensor = Variable(torch.cuda.LongTensor(d))
                d_vecs = self.des_embedding(dtensor)
                d_vecs = torch.transpose(d_vecs,1,2)
                d_conv = F.tanh(self.conv_des(d_vecs))
                d_mp = F.max_pool1d(d_conv,kernel_size=d_conv.size()[2])
                z = self.fc_des(torch.squeeze(d_mp))
                Z.append(z)
        return Z

    def get_diff(self,y,Z):
        diffs = []
        for i, z in enumerate(Z):
            inds = torch.nonzero(y[i].data)
            inds = torch.squeeze(inds)
            b = self.fc2.weight[inds,:]
            diff = torch.mul(z-b,z-b).mean()#sum()
            ny = z.size()[0]
            diffs.append(diff*self.config.reg_lambda*ny)
        reg_term = torch.stack(diffs).mean()

        return reg_term

    def forward(self,x,y,des):
        x = self.word_embeddings(x) #(N,seq_length,100)
        x = self.dropout(x)
        'conv input:(N,Cin,Lin); output:(N,Cout,Lout)'
        x = torch.transpose(x,1,2) #(batch,100,seq_length)
        '''

        H = F.tanh(self.conv(x).transpose(1,2))
        alpha = F.softmax(self.fc1.weight.matmul(H.transpose(1,2)),dim=2)
        V = alpha.matmul(H)
        z = self.fc2.weight.mul(V).sum(dim=2).add(self.fc2.bias)
        '''

        H = F.tanh(self.conv(x).transpose(1,2)) #(batch,50,seq_length)
        'fc: input(N,*,in_features) output(N,*,out_features)'
        #H = torch.transpose(H,1,2)
        #multi_attn
        V = []
        alpha = []
        for fc in self.multiattn:
            a = F.softmax(fc.weight.matmul(H.transpose(1,2)),dim=2)
            v = a.matmul(H)
            V.append(v)
            alpha.append(a)

        '''
        for i,fc in enumerate(self.multiattn):
            atten_op = fc(torch.transpose(H,1,2))
            alpha = F.softmax(atten_op,dim=1)
            v = torch.matmul(H,alpha)
            v = torch.transpose(v,1,2)
            V.append(v)
        '''

        V = torch.cat(V,2) #1 ? #[batch,#labels,50 * num_attn]
        #print(V[0].size())
        z = self.fc2.weight.mul(V).sum(dim=2).add(self.fc2.bias)

        #atten_op = [fc(H) for fc in self.multiattn] #(batch,seq_length,#labels) * num_attn
        #print(atten_op.size())
        #alpha = [F.softmax(fc_op,dim=1) for fc_op in atten_op] #(batch,seq_length,#labels) * num_attn
        #print(alpha[0].size())
        #H = torch.transpose(H,1,2)
        #V = [torch.matmul(H,a) for a in alpha]#[batch,50,#labels] * num_attn
        #print(V[0].size())
        #V = [torch.transpose(v,1,2) for v in V]#[batch,#labels,50] * num_attn
        #print(V[0].size())

        #op_fc1 = self.fc1(torch.transpose(H,1,2))#(batch,seq_length,#labels)
        #alpha = F.softmax(op_fc1,dim=1)#(batch,seq_length,#labels)
        #V = torch.matmul(H,alpha) #[batch,50,#labels]
        #V = torch.transpose(V,1,2)#[batch,#labels,50]

        #weight: [num_labels,num_filters * num_attn]
        #z = torch.sum(torch.mul(V,self.fc2.weight),dim=2).add(self.fc2.bias) #fc2.weight[#labels,#filters]
        #print(z.size())
        self.y_hat = F.sigmoid(z)
        self.loss = F.binary_cross_entropy(self.y_hat,y)

        '''embedding the descriptions
        if self.des_embed:
            print('Start : ... ')
            diffs = []
            for i,d in enumerate(des):
                if len(d) > 0:
                    print(len(d))
                    dtensor = Variable(torch.cuda.LongTensor(d))
                    print(dtensor.size())
                    d_vecs = self.des_embedding(dtensor) #[n_labels,des_len,100]
                    d_vecs = torch.transpose(d_vecs,1,2)
                    d_conv = F.tanh(self.conv_des(d_vecs))#[n_labels,50,des_len]
                    d_mp = F.max_pool1d(d_conv,kernel_size=d_conv.size()[2])#[n_labels,50,1]
                    print(d_mp.size())
                    Z = self.fc_des(torch.squeeze(d_mp))#[n_labels,50]
                    inds = torch.nonzero(y[i].data)
                    inds = torch.squeeze(inds)
                    diff = []

                    for ix, ind in enumerate(inds):
                        b = self.fc2.weight[ind]
                        z = Z[ix]
                        diff.append(torch.mul(z-b,z-b).sum())
                    diff_sum = torch.stack(diff).sum()
                    ny = inds.size()[0]

                    diffs.append(diff_sum*self.config.reg_lambda*(1/ny))

            reg_term = torch.stack(diffs).mean()
            self.loss = self.loss + reg_term
        '''
        if self.des_embed:
            #b_batch = self.embed_descriptions(des, True)
            #diffs = self._compare_label_embeddings(y, b_batch, des)

            Z = self.des_model(des)
            reg_term = self.get_diff(y,Z)
            self.loss = self.loss + reg_term

        return self.y_hat,self.loss,alpha





