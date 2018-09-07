
import numpy as np

def info_snippets(alpha,x,y_hat,y_true,dicts,filter_width=10,k=4):
    #(batch,seq_length,#labels)'
    ind2c = dicts['ind2c']
    ind2w = dicts['ind2w']
    print(alpha.size())
    print(y_hat.size())
    y_hati = y_hat[0]
    y_truei = y_true[0]
    #y_hati = y_hat.data[0].cpu().numpy()
    #y_true = y_true.data[0].cpu().numpy()
    yh_inds = np.array([i for i,yh in enumerate(np.round(y_hati)) if yh==1])
    yt_inds = np.array([i for i,yt in enumerate(y_truei) if yt==1])
    labels_pred = [ind2c[y] for y in yh_inds]
    labels_true = [ind2c[y] for y in yt_inds]
    print('Predict ICD: ' + ';'.join(labels_pred))
    print('True ICD: ' + ';'.join(labels_true))
    a = alpha.data[0,yh_inds,:].cpu().numpy() #,#labels x seq_length
    high_score_inds = np.argsort(a,axis=1)[:,-10:]
    #sorted_score_inds = np.sort(high_score_inds,axis=0)
    #sorted_score_inds = np.transpose(sorted_score_inds)
    text_inds = x.data[0].cpu().numpy()
    text = [ind2w[wi] if wi in ind2w.keys() else 'UNK' for wi in text_inds]
    print('Text: '+ ' '.join(text))


    for ix in range(high_score_inds.shape[0]):
        icd_ind = yh_inds[ix]
        print(str(ind2c[icd_ind]))
        conf = y_hati[i]
        print('Confidence: '+str(conf))

        inds = high_score_inds[ix]
        ai = a[ix,:]
        for i in range(3):
            ind = inds[-i]
            if ind + filter_width < text_inds.shape[0]:
                t_ins = text_inds[ind:ind+filter_width]
            else:
                t_ins = text_inds[ind:]
            sp = [ind2w[wi] if wi in ind2w.keys() else 'UNK' for wi in t_ins]
            sp_text = ' '.join(sp)
            print('Snippets with score'+ str(ai[ind]) + ': ' + sp_text)
