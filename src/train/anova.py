import csv
import random
from sys import argv
import warnings  
from xgb_utils_binary import *
import scipy.stats as stats
from multiprocessing import set_start_method
from sklearn.decomposition import PCA
import pickle as pk
from fs_module.feature_selector import FeatureSelector
import pandas as pd
import scipy.stats as stats

#participant_list = np.array([6,7,8,13,14,18,26,31,32,40,42,43])
participant_list = np.array([1,2,5,6,7,8,9,13,14,15,16,17])
fold = len(participant_list)

def feature_clean(data, feat_name, part_name):
    for column in range(data.shape[1],0,-1):
        for n, element in enumerate(data[:,column-1]):
            if str(element) == 'nan' or str(element)=='inf':
                data = np.delete(data,column-1,1)
                feat_name = np.delete(feat_name,column-1)
                break
    data = data.astype('float')
    for column in range(data.shape[1],0,-1):
        for participant in participant_list:
            where = np.where(part_name==participant)[0]
            if np.std(data[where[0]:where[-1]+1, column-1])<= 1e-15:
                data = np.delete(data,column-1,1)
                feat_name = np.delete(feat_name,column-1)
                break
    return data, feat_name

def readfile(filename,task):
    file = open(filename,'r')
    name_all = []
    label_all_list = []
    data_all_list = []
    data_all_dict = {}
    label_all_dict = {}
    alllist = list(csv.reader(file))
    for n, row in enumerate(alllist[1:]):
        name = int(row[0].split('_')[0])
        if name in participant_list:
            name_all.append(name)
            if float(row[-1])==2:
                label_all_list.append(float(row[-1])-2)
            else:
                label_all_list.append(float(row[-1]))
            data_all_list.append(row[1:-1])
            #print(float(row[-1]),label_all_list[-1])
        #input()
    file.close()
    feat_name = np.array(alllist[0][1:-1])
    label_all_list = np.array(label_all_list).astype('int')
    data_all_list = np.array(data_all_list).astype('float')
    data_all_list1, feat_name1 = feature_clean(data_all_list[:,task[0][0]:task[0][1]], feat_name[task[0][0]:task[0][1]], name_all)
    data_all_list = data_all_list1
    feat_name = feat_name1
    print(data_all_list.shape, feat_name.shape, np.unique(label_all_list))
    #input()
    for i, name in enumerate(name_all):
        if name not in data_all_dict:
            data_all_dict[name] = [data_all_list[i]]
            label_all_dict[name] = [label_all_list[i]]
        else:
            data_all_dict[name].append(data_all_list[i])
            label_all_dict[name].append(label_all_list[i])
        
    for i in sorted(list(data_all_dict.keys())):
        mean = np.mean(np.array(data_all_dict[i]),axis=0)
        std = np.std(np.array(data_all_dict[i]),axis=0)
        data_all_dict[i] = (data_all_dict[i] - mean)/std

    for num, part in enumerate(participant_list):
        if num == 0:
            X_train = data_all_dict[part].tolist()
            Y_train = label_all_dict[part]
        else: 
            X_train = X_train + data_all_dict[part].tolist()
            Y_train = Y_train + label_all_dict[part]
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    #X_train = (X_train-np.mean(X_train,axis=0))/np.std(X_train,axis=0)

    pd_data = pd.DataFrame(data=X_train,columns=feat_name)
    pd_label = pd.DataFrame(data=Y_train,columns=['label'])
    fs = FeatureSelector(data = pd_data, labels = pd_label)
    fs.identify_missing(missing_threshold=0.6)
    fs.identify_single_unique()
    fs.identify_collinear(correlation_threshold=0.9)
    correlated_features = fs.ops['collinear']
    #fs.plot_collinear(plot_all=True)
    #plt.show()
    pd_data_new = fs.remove(methods=['collinear'])
    #new_data = pd_data_new.values
    #new_name = pd_data_new.columns.values

    new_data = pd_data.values
    new_name = pd_data.columns.values
    #print(new_data.shape, new_name.shape)
    data_all_dict = {}
    for i, name in enumerate(name_all):
        if name not in data_all_dict:
            data_all_dict[name] = [new_data[i]]
        else:
            data_all_dict[name].append(new_data[i])
    return new_name, label_all_dict ,data_all_dict


def main():
    outfile = 'anova.csv'
    name, label_all, data_all = readfile(infile,[[0,196]])
    "======cross fold validation======"
    for num, val in enumerate(participant_list):
        print('val subject ', val)
        if num == 0:
            X = np.array(data_all[val]).tolist()
            Y = np.array(label_all[val]).tolist()
        else:
            X += np.array(data_all[val]).tolist()
            Y += np.array(label_all[val]).tolist()
    X = np.array(X)
    Y = np.array(Y)
    print(X.shape)
    print(Y.shape)
    zero = []
    one = []

    for n, data in enumerate(X):
        if Y[n] == 0:
            zero.append(data)
        elif Y[n] == 1:
            one.append(data)
    zero = np.array(zero)
    one = np.array(one)

    p_value = []
    for i in range(len(name)):
        statistic, p = stats.f_oneway(zero[:,i], one[:,i])
        p_value.append(p)

    file = open('p_value.csv','a')
    for i, n in enumerate(name):
        file.write(','+str(n))
    file.write('\n')
    exp_type = infile.split('_')[2]+'_'+infile.split('_')[3].split('.')[0]
    file.write(exp_type)
    for i, n in enumerate(p_value):
        file.write(','+str(n))
    file.write('\n')
    file.close()

if __name__ == '__main__':
    warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)  
    infile = argv[1]
    main()
