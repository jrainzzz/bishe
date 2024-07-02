import os
from select import select
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from scipy.stats import pearsonr
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Ridge
from statsmodels.stats.outliers_influence import variance_inflation_factor


#取出除最后一列（功耗）的其他列数据，即特征；并把文件名和特征名做处理
def get_features(feature_data_pd,feature_num):

    feature_data = np.array(feature_data_pd)
    features_name = []
    feature = feature_data[:, 1:feature_num]
    #feature = np.hstack(feature_data[:, 1:feature_num])
    features_name.append(feature_data_pd.columns.values[1:feature_num].tolist())
    return feature.astype(np.float), features_name


def feature_plot(feature, features_name, power_targets):
    r_list = [pearsonr(power_targets, feature[:, i])[0] for i in range(0, feature.shape[1])]
    VIF_list = [variance_inflation_factor(feature, i) for i in range(feature.shape[1])]
    VIF_list = np.array(VIF_list).reshape(feature.shape[1], )
    VIF_list[np.where(VIF_list > 100)] = 100

    index = np.arange(len(features_name))
    bar_width = 0.5
    feature_num = len(features_name)
    feature_index = range(0,feature_num)


    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(18, 5))

    ax1 = ax[0]
    ax1.set_ylabel("Correlation", fontsize=12)
    ax1.bar(index[feature_index], np.abs(r_list)[feature_index], width=bar_width, label='feture')
    ax1.legend(fontsize=7)

    ax2 = ax[1]
    ax2.set_ylabel("VIF", fontsize=12)
    ax2.bar(index[feature_index], np.abs(VIF_list)[feature_index], width=bar_width, label='feture')
    ax2.legend(fontsize=7)
    '''
    selectd_features_name = ['Core.Leakage', 'Core.Dynamic', 'Free_List.Dynamic', 'MMU.Dynamic',
                             'ROB.Dynamic', 'iew.exec_stores', 'int_alu_accesses', 'FU_FpMemRead',
                             'FU_IntDiv', 'FU_FpMult', 'FU_FpDiv', 'mem.conflictStores', 'rename.Maps',
                             'mem_ctrls.reads', 'icache.mshr_hits', 'dcache.accesses', 'dcache.mshr_hits']
    '''
    # selectd_idx = [x for x in range(0, len(dynamic_features_name)) if dynamic_features_name[x] in selectd_features_name]
    # other_idx = [x for x in index if x not in selectd_idx]

    # for idx in range(0, len(features_name)):
    #     if features_name[idx] in selectd_features_name:
    #         features_name[idx] = features_name[idx].replace('_', '\_')
    #         features_name[idx] = r"$\bf{" + str(features_name[idx]) + "}$"

    plt.xticks(index, labels=features_name, rotation=270, fontsize=6.2)
    plt.subplots_adjust(bottom=0.32)
    #plt.savefig(os.path.join(dirname, "feature_corr_vif.pdf"), bbox_inches='tight')
    #plt.close()

def feature_selection(total_features, targets, k_features, scoring, var_threshold=0):
    ### Feature Filtering
    var_sel = VarianceThreshold(threshold=var_threshold)
    var_sel.fit(total_features)
    slected_idx = var_sel.get_support()
    filter_features = total_features[:, slected_idx]

    ### Sequential Feature Selection
    ridge_model = Ridge(alpha=0.001)
    ridge_sfs = SFS(estimator=ridge_model, k_features=k_features, forward=True, scoring=scoring, cv=10)
    ridge_sfs.fit(filter_features, targets)

    slected_idx = [i for i, j in enumerate(slected_idx) if j == True]
    slected_idx = np.array(slected_idx)[list(ridge_sfs.k_feature_idx_)]
    select_fatures = filter_features[:, ridge_sfs.k_feature_idx_]

    return slected_idx, select_fatures