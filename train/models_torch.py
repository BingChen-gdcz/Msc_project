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

    batch_size = 16
    num_epochs = 200

    save_per_batch = 300
    print_per_batch = 25

    def __init__(self,num_labels,reg_lambda=0.01,filter_width=10,num_filters=50,
                 dropout_prob=0.2,learning_rate=0.0001,label_set='full',version='mimic3'):
        self.num_labels = num_labels
        #self.vocab_size = vocab_size + 2# ind2embedding.shape
        self.reg_lambda = reg_lambda
        self.filter_width = filter_width #k
        self.num_filters = num_filters #dc
        self.dropout_prob = dropout_prob
        self.learning_rate = learning_rate
        self.label_set = label_set
        self.version = version

class CAML_model(nn.Module):

    def __init__(self,config,des_embed=False):
        super(CAML_model, self).__init__()
        torch.manual_seed(1337)
        self.config = config
        self.des_embed = des_embed

        embed_matrix = torch.Tensor(load_data_tools.load_embedding_vector(version=config.version))
        self.word_embeddings = nn.Embedding(embed_matrix.size(0),embed_matrix.size(1))
        self.word_embeddings.weight.data = embed_matrix.clone()

        #self.word_embeddings = nn.Embedding(config.vocab_size+1,config.embedding_dim)
        #embed_matrix = load_data_tools.load_embedding_vector()
        #self.word_embeddings.weight.data.copy_(torch.from_numpy(embed_matrix))

        self.dropout = nn.Dropout(config.dropout_prob)
        self.conv = nn.Conv1d(config.embedding_dim,config.num_filters,kernel_size=config.filter_width,
                              padding=floor(config.filter_width/2))
        xavier_uniform(self.conv.weight)

        self.fc1 = nn.Linear(config.num_filters,config.num_labels)
        xavier_uniform(self.fc1.weight) #U: num_labels * num_filters

        self.fc2 = nn.Linear(config.num_filters,config.num_labels) #beta([8921, 50])
        xavier_uniform(self.fc2.weight) #beta: num_labels * num_filters

        if des_embed:
            embed_matrix = self.word_embeddings.weight.data
            self.des_embedding = nn.Embedding(embed_matrix.size(0),embed_matrix.size(1))
            self.des_embedding.weight.data = embed_matrix.clone()

            #self.des_embedding = nn.Embedding(config.vocab_size+1,config.embedding_dim)
            #self.des_embedding.weight.data.copy_(torch.from_numpy(embed_matrix))
            self.conv_des = nn.Conv1d(config.embedding_dim,config.num_filters,kernel_size=config.filter_width,
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
                #inds = torch.nonzero(y[i].data)
                #inds = torch.squeeze(inds)
                #b = self.fc2.weight[inds,:]
                #diff = torch.mul(z-b,z-b).sum()
                #ny = z.size()[0]
                #diffs.append(diff*self.config.reg_lambda*(1/ny))
                Z.append(z)
            else:
                Z.append([])

        #reg_term = torch.stack(diffs).mean()

        return Z

    def get_diff(self,y,Z):
        diffs = []
        for i, z in enumerate(Z):
            #z = Z[i]
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


        H = F.tanh(self.conv(x).transpose(1,2))
        alpha = F.softmax(self.fc1.weight.matmul(H.transpose(1,2)),dim=2)
        V = alpha.matmul(H)
        z = self.fc2.weight.mul(V).sum(dim=2).add(self.fc2.bias)
        '''

        H = F.tanh(self.conv(x)) #(batch,50,seq_length)
        'fc: input(N,*,in_features) output(N,*,out_features)'

        op_fc1 = self.fc1(torch.transpose(H,1,2))#(batch,seq_length,#labels)

        alpha = F.softmax(op_fc1,dim=1)#(batch,seq_length,#labels)

        V = torch.matmul(H,alpha) #[batch,50,#labels]
        V = torch.transpose(V,1,2)#[batch,#labels,50]

        z = torch.sum(torch.mul(V,self.fc2.weight),dim=2).add(self.fc2.bias) #fc2.weight[#labels,#filters]
        '''


        self.y_hat = F.sigmoid(z)
        self.loss = F.binary_cross_entropy(self.y_hat,y)

        if self.des_embed:
            #b_batch = self.embed_descriptions(des, True)
            #diffs = self._compare_label_embeddings(y, b_batch, des)

            Z = self.des_model(des)
            reg_term = self.get_diff(y,Z)
            self.loss = self.loss + reg_term

        return self.y_hat,self.loss,alpha

class self_Attn(nn.Module):

    def __init__(self,config,des_embed=False):
        super(self_Attn, self).__init__()
        torch.manual_seed(1337)
        self.config = config
        self.des_embed=False

        embed_matrix = torch.Tensor(load_data_tools.load_embedding_vector(version=config.version))
        self.word_embeddings = nn.Embedding(embed_matrix.size(0),embed_matrix.size(1))
        self.word_embeddings.weight.data = embed_matrix.clone()

        #self.word_embeddings = nn.Embedding(config.vocab_size+1,config.embedding_dim)
        #embed_matrix = load_data_tools.load_embedding_vector()
        #self.word_embeddings.weight.data.copy_(torch.from_numpy(embed_matrix))

        self.dropout = nn.Dropout(config.dropout_prob)
        self.conv = nn.Conv1d(config.embedding_dim,config.num_filters,kernel_size=config.filter_width,
                              padding=floor(config.filter_width/2))
        xavier_uniform(self.conv.weight)

        self.W_1 = nn.Linear(config.num_filters,5000,bias=False)
        xavier_uniform(self.W_1.weight) #U: 3000 * num_filters

        self.W_2 = nn.Linear(5000,config.num_labels,bias=False)
        xavier_uniform(self.W_2.weight) #W_2 [8921,3000]

        self.fc = nn.Linear(config.num_filters,config.num_labels) #beta([8921, 50])
        xavier_uniform(self.fc.weight) #beta: num_labels * num_filters
        #self.I = torch.eye(config.num_labels).cuda()

    def forward(self,x,y,des):
        x = self.word_embeddings(x) #(N,seq_length,100)
        x = self.dropout(x)
        'conv input:(N,Cin,Lin); output:(N,Cout,Lout)'
        x = torch.transpose(x,1,2) #(batch,100,seq_length)


        H = F.tanh(self.conv(x).transpose(1,2)) #[batch,N, dc]
        #print(H.size())
        E = F.tanh(self.W_1.weight.matmul(H.transpose(1,2))) #E[batch,3000,N]
        #print(E.size())
        alpha = F.softmax(self.W_2.weight.matmul(E),dim=2) #A[batch,dl,N]
        #print(alpha.size())
        V = alpha.matmul(H)
        #print(V.size())
        z = self.fc.weight.mul(V).sum(dim=2).add(self.fc.bias)
        #print(z.size())

        #reg = alpha.matmul(alpha.transpose(1,2)) - self.I #AAt = [batch,dl,dl]
        #for_reg = reg.mul(reg).sum()


        self.y_hat = F.sigmoid(z)
        self.loss = F.binary_cross_entropy(self.y_hat,y) #+ 0.001 * 1/self.y_hat.size(0) * for_reg.mean()

        return self.y_hat,self.loss,alpha


class CNN_model(nn.Module):

    def __init__(self,config,des_embed=False):
        super(CNN_model, self).__init__()
        self.config = config
        self.des_embed = des_embed

        embed_matrix = torch.Tensor(load_data_tools.load_embedding_vector(version=config.version))
        self.word_embeddings = nn.Embedding(embed_matrix.size(0),embed_matrix.size(1))
        self.word_embeddings.weight.data = embed_matrix.clone()

        self.dropout = nn.Dropout(config.dropout_prob)
        self.conv = nn.Conv1d(config.embedding_dim,config.num_filters,config.filter_width,stride=1,
                              padding=floor(config.filter_width/2))

        self.fc = nn.Linear(config.num_filters,config.num_labels)

    def forward(self,x,y,des,get_attention=False):
        #print('CNN model ... ')
        x = self.word_embeddings(x) #(N,seq_length,100)
        x = self.dropout(x)
        'conv input:(N,Cin,Lin); output:(N,Cout,Lout)'
        x = torch.transpose(x,1,2) #(batch,100,seq_length)

        H = self.conv(x) #(batch,50,seq_length)

        if get_attention:
            #get argmax vector too
            v_mp, argmax = F.max_pool1d(F.tanh(H), kernel_size=H.size()[2], return_indices=True)
            alpha = self.construct_attention(argmax, H.size()[2])
        else:
            v_mp = F.max_pool1d(F.tanh(H), kernel_size=H.size()[2])
            alpha = None

        'fc: input(N,*,in_features) output(N,*,out_features)'
        #v_mp = F.max_pool1d(H,kernel_size=H.size(2))#(batch,50]
        v_mp = torch.squeeze(v_mp,dim=2)#(batch,50]

        op_fc = self.fc(v_mp)#(batch,#labels)
        #print(op_fc.size())
        self.y_hat = F.sigmoid(op_fc)
        self.loss = F.binary_cross_entropy(self.y_hat,y)

        return self.y_hat,self.loss,alpha

    def construct_attention(self, argmax, num_windows):
        attn_batches = []
        for argmax_i in argmax:
            attns = []
            for i in range(num_windows):
                mask = (argmax_i == i).repeat(1,self.config.num_labels).t()
                weights = self.fc.weight[mask].view(-1,self.config.num_labels)
                print(weights.size())
                if len(weights.size()) > 1:
                    window_attns = weights.sum(dim=0)
                    attns.append(window_attns)
                else:
                    attns.append(Variable(torch.zeros(self.config.num_labels)).cuda())
            attn = torch.stack(attns)
            attn_batches.append(attn)
        attn_full = torch.stack(attn_batches)
        attn_full = attn_full.transpose(1,2)
        return attn_full





