import os
import sys

from torch.functional import split
sys.path.append('../')
import numpy as np
import pandas as pd
import csv
from config import config

ct_prefix = ['.mhd', '.nrrd']

def split_cross_val_test(img_path, split_path, folds):
    '''
    split annotations into 10 folds
    '''
    for i in range(10):
        train_csv_path = split_path + '/{}_train.csv'.format(i)
        train_ct_list = []
        val_csv_path = split_path + '/{}_val.csv'.format(i)
        val_ct_list = []
        for j in range(10):
            if (i == j):
                subset = img_path + '/subset{}'.format(j)
                val_ct_files= os.listdir(subset) 
                for filename in val_ct_files:
                    prefix = os.path.splitext(filename)[1]
                    if prefix in ct_prefix:
                        val_ct_list.append(filename[:-4])
            else:
                subset = img_path + '/subset{}'.format(j)
                train_ct_files= os.listdir(subset) 
                for filename in train_ct_files:
                    prefix = os.path.splitext(filename)[1]
                    if prefix in ct_prefix:
                        train_ct_list.append(filename[:-4])
                        
        
        train_csv = open(train_csv_path, "w")
        try:
            writer = csv.writer(train_csv)
            for i in range(len(train_ct_list)):
                writer.writerow([train_ct_list[i]])
        finally:
            train_csv.close()
            
        val_csv = open(val_csv_path, "w")
        try:
            writer = csv.writer(val_csv)
            for i in range(len(val_ct_list)):
                writer.writerow([val_ct_list[i]])
        finally:
            val_csv.close()
                
       
            
        

if __name__ == '__main__':
    img_path = config['data_dir']
    seriesuids_path = config['seriesuids_dir']
    split_path = config['split_save_dir']

    test_num = 88
    folds = 10
    
    split_cross_val_test(img_path, split_path, folds)