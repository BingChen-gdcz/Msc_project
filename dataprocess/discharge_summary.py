import os
import csv
import re
#from tqdm import tqdm

'''
to create the discharge summary
filter out discharge summary from NOTEEVENTS.csv data
'''
def get_disch(note_file):


    outfile = os.path.join(os.getcwd(),'dev_data/DISCH_SUM.csv')
    with open(note_file) as f:
        with open(outfile,'w') as outf:
            reader = csv.reader(f)
            #print(str(next(reader)))
            #header
            next(reader)
            outf.write(','.join(['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'TEXT']) + '\n')
            for row in reader: #tqdm(reader):

                if row[6] == 'Discharge summary':
                    text = [t.strip().lower() for t in token(row[10]) if not t.isnumeric()]
                    disch = ' '.join(text)
                    outf.write(','.join([row[1],row[2],row[4],disch]) + '\n')

def clean_data(string):
    string = re.sub(r"\\","",string)
    string = re.sub(r"\'","",string)
    string = re.sub(r"\"","",string)
    return string.strip().lower()

def token(txt):
    #txt = clean_data(txt)
    txt = txt.strip().lower()
    pattern = re.compile(r'(?u)\b\w+\b')
    words = re.findall(pattern, txt)
    return words
