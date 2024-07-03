
import os
import warnings

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import KFold
from feature_process import get_features,feature_plot,feature_selection

from sklearn.ensemble import (AdaBoostRegressor, BaggingRegressor,
                              GradientBoostingRegressor, RandomForestRegressor)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.linear_model import (BayesianRidge, ElasticNet, Lasso,
                                  LinearRegression, Ridge)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

'''
def split_strings_to_vectors(string_list):
    vectors = []
    for s in string_list:
        vector = s.split(',')
        vectors.append(vector)
    return vectors
def load_data(file_name):
    name = []
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            name.append(line)
        name_list = split_strings_to_vectors(name)
        name_list1 = [row[1:] for row in name_list]
        for row in name_list1[1:]:  # 从第二行开始遍历（索引1）
            for i, item in enumerate(row):
                row[i] = float(item)  # 转换每个元素为浮点数
    return name_list1
'''
def mape(y_true, y_pred):
    return np.mean(np.abs((np.array(y_pred) - np.array(y_true)) / np.array(y_true))) * 100

file_path = 'power_data/cpu_dynamic_power.csv'
gp_kernel = DotProduct() + WhiteKernel()
model_all = [LinearRegression(),Lasso(max_iter=100000),Ridge(),ElasticNet(),BayesianRidge(),GaussianProcessRegressor(kernel=gp_kernel,alpha=1e-10),KNeighborsRegressor(),
             LinearSVR(max_iter=1000000),SVR(kernel='poly'),SVR(kernel='rbf'),DecisionTreeRegressor(),RandomForestRegressor(),AdaBoostRegressor(),
             GradientBoostingRegressor(),BaggingRegressor(),XGBRegressor(max_depth=5,min_child_weigh=3,max_iter = 1000000)]

class Csv_data():
    def __init__(
            self, file_path
    ):
        self.csv_data = pd.read_csv(file_path)
        self.last_column_name = self.csv_data.columns[-1]
        self.power_targets = np.array(self.csv_data[self.last_column_name]).astype(np.float64)
        self.feature_data = self.csv_data.drop(self.csv_data.columns[len(self.csv_data.columns) - 1], axis=1)
        self.feature_num = len(self.feature_data.columns)
        '''
        self.feature_data_a = np.array(self.feature_data)
        self.features_name = []
        self.feature_num = len(self.feature_data.columns) - 1
        self.feature = np.hstack(self.feature_data_a[:, 1:self.feature_num])
        self.features_name = (self.feature_data.columns.values[1:self.feature_num].tolist())
        self.feature = self.feature.astype(np.float)
        '''
    def feature_processing(self):
       self.feature, self.feature_name = get_features(self.csv_data,self.feature_num)

csv_data = Csv_data(file_path)
csv_data.feature_processing()


#除以cycle，算出每个特征每cycle的数据
feature_first_column = csv_data.feature[:, 0][:]#特征的所有列都除以cycle即第一列
for i in range(csv_data.feature.shape[0]):
    csv_data.feature[i, 1:] = csv_data.feature[i, 1:] / feature_first_column[i]
csv_data.feature = np.delete(csv_data.feature,0,axis=1)

#特征选取
my_mape = metrics.make_scorer(mape, greater_is_better=False)
feature_index , csv_data.feature = feature_selection(csv_data.feature,csv_data.power_targets,14,my_mape)

'''
打印选取特征的名称
feature_index.reshape(-1,1)
for i in range(len(feature_index)):
    print(csv_data.feature_name[0][feature_index[i]])
'''

kf = KFold(n_splits=5,shuffle=True)
'''
for model_index in range(len(model_all)):
    MAPE_ALL = []
    for train_index , test_index in kf.split(csv_data.feature):  # 调用split方法切分数据
        #print('train_index:%s , test_index: %s ' %(train_index,test_index))
        X_train, X_test = csv_data.feature[train_index], csv_data.feature[test_index]
        y_train, y_test = csv_data.power_targets[train_index], csv_data.power_targets[test_index]
        model = model_all[model_index]
        #model = XGBRegressor()
        model.fit(X_train, y_train.ravel())
        power_pred = model.predict(X_test)
        MAPE = mape(csv_data.power_targets[test_index], power_pred)
        MAPE_ALL.append(MAPE)
        #print(MAPE)
    print(np.mean(MAPE_ALL))
'''

MAPE_ALL = []
for train_index, test_index in kf.split(csv_data.feature):  # 调用split方法切分数据
    # print('train_index:%s , test_index: %s ' %(train_index,test_index))
    X_train, X_test = csv_data.feature[train_index], csv_data.feature[test_index]
    y_train, y_test = csv_data.power_targets[train_index], csv_data.power_targets[test_index]
    model = model_all[15]
    #model = XGBRegressor()
    model.fit(X_train, y_train.ravel())
    power_pred = model.predict(X_test)
    MAPE = mape(csv_data.power_targets[test_index], power_pred)
    MAPE_ALL.append(MAPE)
    print(power_pred)
    print(MAPE)
print(np.mean(MAPE_ALL))

'''
#测试用
file_num = csv_data.feature.shape[0]-1#文件数量减一找到最后一个索引
train_index = range(0,file_num-13)
test_index = range(file_num-12,file_num)
X_train, X_test = csv_data.feature[train_index], csv_data.feature[test_index]
y_train, y_test = csv_data.power_targets[train_index], csv_data.power_targets[test_index]
#model = LinearRegression()
model = XGBRegressor()
model.fit(X_train, y_train.ravel())
#power_pred = model.predict(X_test.reshape(1,-1))
power_pred = model.predict(X_test)
print(power_pred)
MAPE = mape(csv_data.power_targets[test_index], power_pred)
'''
#print(power_pred)
#print(MAPE)
#print(np.mean(MAPE_ALL))
print(csv_data.feature.shape[0])
