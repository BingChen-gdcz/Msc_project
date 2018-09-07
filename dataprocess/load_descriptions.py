import sys
sys.path.append('../')
import os
import csv
import pandas as pd
#from tqdm import tqdm
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

def load_description(version='mimic3'):

    icd_des_dic = defaultdict(str)
    data_dir = os.path.join(os.pardir,'mimicdata')
    if version == 'mimic2':
        with open('%s/%s/MIMIC_ICD9_mapping'%(data_dir,version),'r') as f:
            r = csv.reader(f)
            for row in r:
                row = row[0].split(';')
                des = [t.lower() for t in tokenizer.tokenize(row[2]) if not t.isnumeric()]
                icd_des_dic[str(row[1])] = des
    else:
        diag_des = os.path.join(data_dir,'D_ICD_DIAGNOSES.csv')
        pro_des = os.path.join(data_dir,'D_ICD_PROCEDURES.csv')
        icd_des = os.path.join(data_dir, 'ICD9_descriptions')
        with open(diag_des,'r') as df:
            reader = csv.reader(df)
            next(reader) #header
            for row in reader:
                code = reformat(row[1],True)
                if code not in icd_des_dic.keys():
                    des = [t.lower() for t in tokenizer.tokenize(row[3]) if not t.isnumeric()]
                    icd_des_dic[code] = des
                    #icd_des_dic[code] = ' '.join(des)

        with open(pro_des,'r') as pf:
            reader = csv.reader(pf)
            next(reader)
            for row in reader:
                code = reformat(row[1],False)
                if code not in icd_des_dic.keys():
                    #des = row[-1]

                    des = [t.lower() for t in tokenizer.tokenize(row[3]) if not t.isnumeric()]
                    icd_des_dic[code] = des
                    #icd_des_dic[code] = ' '.join(des)

        with open(icd_des,'r') as f:
            f.readline() #header
            all_lines = f.readlines()

            for line in all_lines:
                row = line.strip().split()
                code = row[0]
                if code not in icd_des_dic.keys():
                    des = ' '.join(row[1:])
                    des = [t.lower() for t in tokenizer.tokenize(des) if not t.isnumeric()]
                    icd_des_dic[code] = des
                    #icd_des_dic[code] = ' '.join(des)


    return icd_des_dic


def reformat(code, is_diag):

    code = ''.join(code.split('.'))
    if is_diag:
        if code.startswith('E'):
            if len(code) > 4:
                code = code[:4] + '.' + code[4:]
        else:
            if len(code) > 3:
                code = code[:3] + '.' + code[3:]
    else:
        code = code[:2] + '.' + code[2:]
    return code




