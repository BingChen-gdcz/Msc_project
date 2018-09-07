from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve,auc
import numpy as np
from tabulate import tabulate
import time




def eval(y_hat,y_true,n=[8,15]):

    '''

    :param y_pred:
    :param y_true:
    :return:
    '''
    #===================================================================================================================

    y_pred = np.round(y_hat)
    tp = np.logical_and(y_pred, y_true).sum(axis=0).astype(float)
    p = np.logical_or(y_pred, y_true).sum(axis=0).astype(float)
    #tp = np.sum(np.logical_and(y_pred,y_true),axis=0,dtype=np.float32) #true positive
    #p = np.sum(np.logical_or(y_pred,y_true),axis=0,dtype=np.float32) #tp+fp

    'Macro: acc, precision,recall,f1'
    macro_acc = np.mean(tp/(p + 1e-10))
    macro_prec = np.mean(tp/(np.sum(y_pred,axis=0)+1e-10)) #y_pred.sum: tp + fp
    macro_recall = np.mean(tp/(np.sum(y_true,axis=0)+1e-10)) #y.sum: tp + fn
    if macro_prec + macro_recall == 0:
        macro_f1 = 0
    else:
        macro_f1 = 2 * (macro_prec*macro_recall)/(macro_prec+macro_recall)


    'Micro: acc, precision,recall,f1'
    y_pred_s = np.ravel(y_pred)
    y_hat_s = np.ravel(y_hat)
    y_tr_s = np.ravel(y_true)
    tp_ = np.sum(np.logical_and(y_pred_s,y_tr_s),axis=0,dtype=np.float32)
    p_ = np.sum(np.logical_or(y_pred_s,y_tr_s),axis=0,dtype=np.float32)
    micro_acc = np.mean(tp_/(p_ + 1e-10))
    micro_prec = np.mean(tp_/(np.sum(y_pred_s,axis=0)+1e-10))
    micro_recall = np.mean(tp_/(np.sum(y_tr_s,axis=0)+1e-10))
    if micro_prec + micro_recall == 0:
        micro_f1 = 0
    else:
        micro_f1 = 2 * (micro_prec*micro_recall)/(micro_prec+micro_recall)

    auc = []
    for i in range(y_hat.shape[1]):
        if np.sum(y_true[:,i]) > 1:
            auc.append(roc_auc_score(y_true[:,i],y_hat[:,i]))
    macro_auc = np.mean(auc)
    micro_auc = roc_auc_score(y_tr_s,y_hat_s)
    AUC = [macro_auc,micro_auc]
    macro = ['Macro_label_avg',macro_acc,macro_prec,macro_recall,macro_f1]
    micro = ['Micro_label_avg',micro_acc,micro_prec,micro_recall,micro_f1]

    label_avg = [macro,micro]

    #===================================================================================================================
    #instance average
    tp_inst = np.sum(np.logical_and(y_pred,y_true),axis=1,dtype=np.float32) #tp: sum over labels
    p_inst = np.sum(np.logical_or(y_pred,y_true),axis=1,dtype=np.float32) #tp+fp: sum over labels

    'Macro: acc, precision,recall,f1'
    macro_acc_inst = np.mean(tp_inst/(p_inst + 1e-10))
    macro_prec_inst = np.mean(tp_inst/(np.sum(y_pred,axis=1)+1e-10)) #y_pred.sum: tp + fp
    macro_recall_inst = np.mean(tp_inst/(np.sum(y_true,axis=1)+1e-10)) #y.sum: tp + fn
    if macro_prec_inst + macro_recall_inst == 0:
        macro_f1_inst = 0
    else:
        macro_f1_inst = 2 * (macro_prec_inst*macro_recall_inst)/(macro_prec_inst+macro_recall_inst)

    '''
    'Micro: acc, precision,recall,f1'
    y_inst = np.ravel(y_pred)
    y_tr_inst = np.ravel(y_true)
    tp_insts = np.sum(np.logical_and(y_inst,y_tr_inst),axis=1,dtype=np.float32)
    p_insts = np.sum(np.logical_or(y_inst,y_tr_inst),axis=1,dtype=np.float32)
    micro_acc_inst = np.mean(tp_insts/(p_insts + 1e-10))
    micro_prec_inst = np.mean(tp_insts/(np.sum(y_inst,axis=1)+1e-10))
    micro_recall_inst = np.mean(tp_insts/(np.sum(y_tr_inst,axis=1)+1e-10))

    if micro_prec_inst + micro_recall_inst == 0:
        micro_f1_inst = 0
    else:
        micro_f1_inst = 2 * (micro_prec_inst*micro_recall_inst)/(micro_prec_inst+micro_recall_inst)
    micro_inst = ['Micro_inst_avg',micro_acc_inst,micro_prec_inst,micro_recall_inst,micro_f1_inst]
    '''
    macro_inst = ['Macro_inst_avg',macro_acc_inst,macro_prec_inst,macro_recall_inst,macro_f1_inst]


    r_at_k = []
    p_at_k = []
    #inst_avg = [macro_inst,micro_inst]
    for ni in n:
        r_at_k.append(r_at_n(y_true,y_hat,ni))
        p_at_k.append(p_at_n(y_true,y_hat,ni))

    '''P@n:n=8&n=15'''

    #pn8 = p_at_n(y_true,y_hat,n)
    #pn15 = p_at_n(y_true,y_hat,n=15)
    #pn = [pn8,pn15]

    return macro,micro,macro_inst,AUC,p_at_k,r_at_k

def print_results(results):

    macro,micro,macro_inst,AUC,pn,r_at_k = results

    print(time.strftime('%b_%d %H:%M', time.localtime()))
    print(tabulate([macro,micro,macro_inst],headers=[' ','Accuracy','Precision','recall','F1']))
    print('AUC_Macro: %.4f ' % AUC['auc_macro'])
    print('AUC_Micro: %.4f ' % AUC['auc_micro'])
    if len(pn) > 1:
        print('P@8: ' + str(pn[0]))
        print('P@15: ' + str(pn[1]))
    else:
        print('P@5: ' + str(pn[0]))
    if len(r_at_k) > 1:
        print('Recall@8: ' + str(r_at_k[0]))
        print('Recall@15: ' + str(r_at_k[1]))
    else:
        print('Recall@5: ' + str(r_at_k[0]))



def p_at_n(y,y_hat,n):
    y_sorted_ind = np.argsort(y_hat) #descend order
    y_n_inds = y_sorted_ind[:,-n:] #top n[num_y,n]
    pn = []
    for i,inds in enumerate(y_n_inds):
        if len(inds) > 0:
            num_y = np.sum(y[i,inds]) #number of correct predictions
            pn.append(num_y/float(len(inds)))

    return np.mean(pn)


def r_at_n(y,y_hat,n):
    y_sorted_ind = np.argsort(y_hat) #descend order
    y_n_inds = y_sorted_ind[:,-n:] #top n[num_y,n]
    rn = []
    for i,inds in enumerate(y_n_inds):
        if len(inds) > 0:
            num_y = np.sum(y[i,inds])
            rn.append(num_y/float(np.sum(y[i,:])))

    return np.mean(rn)
