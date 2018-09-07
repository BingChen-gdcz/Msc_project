import sys
sys.path.append('../')
import os
import csv
import pandas as pd
#from tqdm import tqdm
from dataprocess.all_codes import reformat_icdcode
from dataprocess.discharge_summary import token

def load_description():

    data_dir = os.path.join(os.getcwd(),'data')
    diag_des = os.path.join(data_dir,'D_ICD_DIAGNOSES.csv')
    pro_des = os.path.join(data_dir,'D_ICD_PROCEDURES.csv')
    icd_des = os.path.join(data_dir, 'ICD9_descriptions.txt')
    icd_des_dic = {}
    with open(diag_des,'r') as df:
        reader = csv.reader(df)
        next(reader) #header
        for row in reader:
            code = reformat_icdcode(str(row[1]))
            if code not in icd_des_dic.keys():
                des = [t.lower() for t in token(row[3]) if not t.isnumeric()]
                icd_des_dic[code] = ' '.join(des)

    with open(pro_des,'r') as pf:
        reader = csv.reader(pf)
        next(reader)
        for row in reader:
            code = reformat_icdcode(str(row[1]),False)
            if code not in icd_des_dic.keys():
                des = [t.lower() for t in token(row[3]) if not t.isnumeric()]
                icd_des_dic[code] = ' '.join(des)

    with open(icd_des,'r') as f:
        f.readline() #header
        all_lines = f.readlines()

        for line in all_lines:
            row = line.split()
            code = row[0]

            if len(str(''.join(code.strip().split('.')))) < 4 and code[0] != 'V': #for diag code: longer than 3
                code = ''.join(code.strip().split('.'))
                code = code[:2]+'.'+code[2:]
            if len(str(''.join(code.strip().split('.')))) > 3 and code[0] != 'E':
                code = ''.join(code.strip().split('.'))
                code = code[:3]+'.'+code[3:]
            if code in ['360.1', '360.5', '360.2',  '58.5']:
                code = ''.join(code.strip().split('.'))
                if len(code) > 3:
                    code = code[:2] + '.' + code[2:]

            if code not in icd_des_dic.keys():
                des = ' '.join(row[1:])
                des = [t.lower() for t in token(des) if not t.isnumeric()]
                icd_des_dic[code] = ' '.join(des)
            icd_des_dic['17.'] = 'Other miscellaneous procedures'

    icd_des_dic['Miss'] = '<PAD>'
    return icd_des_dic

def load_all_codes():

    code_filtered_dir = os.path.join(os.getcwd(),'dev_data/ALL_CODES_FILTERED.csv')
    dficd_codes_filtered = pd.read_csv(code_filtered_dir,dtype={"ICD9_CODE": str})
    dficd_codes_filtered = dficd_codes_filtered.fillna('Miss')
    icd9_codes = dficd_codes_filtered['ICD9_CODE'].unique()

    return icd9_codes


