import os
import pandas as pd
from tqdm import tqdm


'''
deal with pro_icd and diag_icd：
1. reformat icd9 codes: reformat_code
2. concat two table: all_codes


'''
def diag_pro_codes(pro_dir,diag_dir):
    '''
    reformat ICD code and concat procedure and diagnoise code
    Input: PROCEDURES_ICD.csv & DIAGNOSES_ICD.csv
    Output: ALL_ICD.csv
    '''
    outfile = os.path.join(os.getcwd(),'dev_data/ALL_ICD.csv')
    pro_data = pd.read_csv(pro_dir)
    diag_data = pd.read_csv(diag_dir)
    pro_code = [reformat_icdcode(str(c),False) for c in pro_data['ICD9_CODE']]
    diag_code = [reformat_icdcode(str(c)) for c in diag_data['ICD9_CODE']]
    dfpro = pro_data[['ROW_ID','SUBJECT_ID','HADM_ID','SEQ_NUM']].copy()
    dfpro['ICD9_formatted'] = pro_code
    dfdiag = diag_data[['ROW_ID','SUBJECT_ID','HADM_ID','SEQ_NUM']].copy()
    dfdiag['ICD9_formatted'] = diag_code
    dfpro = dfpro.fillna('Miss')
    dfdiag = dfdiag.fillna('Miss')
    df_all = pd.concat([dfpro,dfdiag],ignore_index=True)
    df_all.to_csv(outfile,columns = ['ROW_ID','SUBJECT_ID','HADM_ID','SEQ_NUM','ICD9_formatted'],index=False)


def code_filter(dficd_codes,df_discharge):

    outfile = os.path.join(os.getcwd(),'dev_data/ALL_CODES_FILTERED.csv')
    #dficd_codes = dficd_codes.sort_values(['SUBJECT_ID', 'HADM_ID'])
    #df_discharge = df_discharge.sort_values(['SUBJECT_ID', 'HADM_ID'])

    #filter the discharge file
    hadm_ids = df_discharge['HADM_ID'].unique().copy()
    print(len(hadm_ids))

    with open(outfile,'w') as of:
        of.write(','.join(['SUBJECT_ID','HADM_ID','ICD9_CODE','ADMITTIME','DISCHTIME'])+'\n')
        for i in tqdm(range(dficd_codes.shape[0])):
            row = dficd_codes.iloc[i]
            if int(row[2]) in hadm_ids:
                of.write(','.join([str(row[1]),str(row[2]),str(row[4]),'',''])+'\n')


def reformat_code(code,is_diag=True):
    new_code = code
    if is_diag:
        if len(code) > 4:
            if code[0] == 'E':
                new_code = code[:4]+'.'+code[4:]
            else:
                new_code = code[:3]+'.'+code[3:]
        if len(code) > 3:
            new_code = code[:3]+'.'+code[3:]

    else:
        new_code = code[:2]+'.'+code[2:]
    return new_code


def reformat_icdcode(code,is_diag=True):
    new_code = ''.join(code.strip().split('.'))
    if is_diag:
        if code[0] == 'E':
            if len(code) > 4:
                new_code = code[:4]+'.'+code[4:]
        elif len(code) > 3:
            new_code = code[:3]+'.'+code[3:]

    else:
        new_code = code[:2]+'.'+code[2:]
    return new_code


