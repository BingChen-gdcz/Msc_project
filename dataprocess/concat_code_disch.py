import os
import pandas as pd
from tqdm import tqdm
import csv

def get_code_disch(dficd_codes_filtered,df_discharge):
    outfile = os.path.join(os.getcwd(),'dev_data/ALL_CODES_NOTES.csv')
    df_discharge = df_discharge.sort_values(['SUBJECT_ID', 'HADM_ID'])
    dficd_codes_filtered = dficd_codes_filtered.sort_values(['SUBJECT_ID', 'HADM_ID'])
    df = dficd_codes_filtered[['SUBJECT_ID', 'HADM_ID']].copy()
    #get the unique set of ['SUBJECT_ID', 'HADM_ID']
    df = df.drop_duplicates(keep='first')
    df_ = df_discharge[['SUBJECT_ID', 'HADM_ID']].copy()
    df_ = df_.drop_duplicates(keep='first')
    print(str(df.shape)+' & '+str(df_.shape))

    with open(outfile,'w') as of:
        of.write(','.join(['SUBJECT_ID', 'HADM_ID','TEXT','LABELS'])+'\n')
        for i in tqdm(range(df.shape[0])):
            row = df.iloc[i]
            dftemp = dficd_codes_filtered.loc[(dficd_codes_filtered['SUBJECT_ID']== row[0])
                                      & (dficd_codes_filtered['HADM_ID']== row[1])]
            labels = []
            for j in range(dftemp.shape[0]):
                labels.append(str(dftemp.iloc[j][2]))
            label = (';').join(labels)
            dftemp_ = df_discharge.loc[(df_discharge['SUBJECT_ID']== row[0])
                                      & (df_discharge['HADM_ID']== row[1])]
            note = ''
            for n in range(dftemp_.shape[0]):
                note = note + ' ' + dftemp_.iloc[n][3]
            of.write(','.join([str(row[0]),str(row[1]),note,label])+'\n')
    return outfile

def split_dataset(df):
    split_data_dir = os.path.join(os.getcwd(),'split_data')
    split_out_dir = os.path.join(os.getcwd(),'dev_data')
    sets = ['train','test','dev']
    outfile = set()

    for st in sets:
        print('Processing '+st+' dataset ... ')
        set_dir = os.path.join(split_data_dir,'%s_full_hadm_ids.csv'%st)
        out_dir = os.path.join(split_out_dir,'%s_dataset.csv'%st)
        outfile.add(out_dir)
        hadm_ids = pd.read_csv(set_dir,names=['HADM_ID'])
        with open(out_dir,'w') as of:
            of.write(','.join(['SUBJECT_ID', 'HADM_ID','TEXT','LABELS'])+'\n')
            for i in tqdm(range(hadm_ids.shape[0])):
                hadm_id = hadm_ids.iloc[i][0]
                row = df.loc[df['HADM_ID']==hadm_id]
                r = row.iloc[0]
                of.write(','.join([str(r[0]),str(r[1]),r[2],str(r[3])])+'\n')
    return outfile



