import numpy as np

dir_path = '/wsdm_cup/final'
model_list = [
    ('yll/finetune_train_result2.csv',
     'yll/finetune_dev_result2.csv',
     'yll/test_result2.csv',
     1),
    ('yll/finetune_train_result3.csv',
     'yll/finetune_dev_result3.csv',
     'yll/test_result3.csv',
     1),
    ('yll/finetune_train_result4.csv',
     'yll/finetune_dev_result4.csv',
     'yll/test_result4.csv',
     1),
    ('3L_dla_neg10_jmtitle_max_bs5_14000/best_3l_dla_9_82.csv',
     '3L_dla_neg10_jmtitle_max_bs5_14000/best_3l_dla_9_82_dev.csv',
     '3L_dla_neg10_jmtitle_max_bs5_14000/best_3l_dla_9_82_test2.csv',
     1),
    ('12l_dla_combine_neg10_replace_list_freq/best_train_9_742.csv',
     '12l_dla_combine_neg10_replace_list_freq/best_dev_10_34.csv',
     '12l_dla_combine_neg10_replace_list_freq/best_test2.csv',
     1),
    ('12l_dla_combine_neg20_replace_list/best_train_9_7689.csv',
     '12l_dla_combine_neg20_replace_list/best_dev_10_38.csv',
     '12l_dla_combine_neg20_replace_list/best_test2.csv',
     1),
    ('12l_dla_combine_neg10_replace_list/best_train_9_88.csv',
     '12l_dla_combine_neg10_replace_list/best_dev_10_56.csv',
     '12l_dla_combine_neg10_replace_list/best_test2.csv',
     1),
    ('yll/finetune_train_result5.csv',
     'yll/finetune_dev_result5.csv',
     'yll/test_result5.csv',
     1),
    ('yll/finetune_train_result6.csv',
     'yll/finetune_dev_result6.csv',
     'yll/test_result6.csv',
     1),
    ('yll/finetune_train_result7.csv',
     'yll/finetune_dev_result7.csv',
     'yll/test_result7.csv',
     1),
    ('yll/finetune_train_result8.csv',
     'yll/finetune_dev_result8.csv',
     'yll/test_result8.csv',
     1),
    ('yll/finetune_train_result9_9.793.csv',
     'yll/finetune_dev_result9_10.537.csv',
     'yll/test_result9.csv',
     1),
    ('yll/finetune_train_result10_9.812.csv',
     'yll/finetune_dev_result10_10.6496.csv',
     'yll/test_result10.csv',
     1),
    ('3l_dla_feature_freq_k0_b0neg0_9810/3l_dla_9_810_neg0_k0b0train.csv',
     '3l_dla_feature_freq_k0_b0neg0_9810/3l_dla_9_810_neg0_k0b0dev.csv',
     '3l_dla_feature_freq_k0_b0neg0_9810/3l_dla_9_810_neg0_k0b0test2.csv',
     1),
    ('3l_dla_feature_freq_k1.5_b0.75neg0_816/3l_dla_9_816__neg0_k1.5b0.5train.csv',
     '3l_dla_feature_freq_k1.5_b0.75neg0_816/3l_dla_9_816__neg0_k1.5b0.5dev.csv',
     '3l_dla_feature_freq_k1.5_b0.75neg0_816/3l_dla_9_816__neg0_k1.5b0.5test2.csv',
     1),
    ('3L_dla_max_neg0_9_818_nofreq/best_3l_dla_9_81_train.csv',
     '3L_dla_max_neg0_9_818_nofreq/best_3l_dla_9_81_dev.csv',
     '3L_dla_max_neg0_9_818_nofreq/best_3l_dla_9_81_test2.csv',
     1),
    (
        '3l_dla_combine834_neg10_list_vote1_replacequerylen/best_train_9_792.csv',
        '3l_dla_combine834_neg10_list_vote1_replacequerylen/best_dev_10_59.csv',
        '3l_dla_combine834_neg10_list_vote1_replacequerylen/best_test2.csv',
        1
    ),
    (
        '12l_na_0110_best_9_785/12l_na_0110_best_9_785train.csv',
        '12l_na_0110_best_9_785/12l_na_0110_best_9_785dev.csv',
        '12l_na_0110_best_9_785/12l_na_0110_best_9_785test2.csv',
        1
    )
]

feat_num = 0

train_feature_path = 'data/features/finetine_train.txt'
dev_feature_path = 'data/features/finetune_dev.txt'
test_feature_path = 'data/features/wsdm_test_2_all.txt'
train_features = np.loadtxt(train_feature_path)
dev_features = np.loadtxt(dev_feature_path)
test_features = np.loadtxt(test_feature_path)

for train_file, dev_file, test_file, alpha in model_list:
    print(train_file)
    feat_num += 1
    train_feat = np.loadtxt(dir_path + train_file)
    dev_feat = np.loadtxt(dir_path + dev_file)
    test_feat = np.loadtxt(dir_path + test_file)

    train_features = np.concatenate([train_features, np.expand_dims(train_feat, axis=1)], axis=-1)
    dev_features = np.concatenate([dev_features, np.expand_dims(dev_feat, axis=1)], axis=-1)
    test_features = np.concatenate([test_features, np.expand_dims(test_feat, axis=1)], axis=-1)

print(train_features.shape)
print(dev_features.shape)
print(test_features.shape)
np.savetxt('new_train_features.txt', train_features, fmt='%f', delimiter=' ')
np.savetxt('new_dev_features.txt', dev_features, fmt='%f', delimiter=' ')
np.savetxt('new_test2_features.txt', test_features, fmt='%f', delimiter=' ')
# (359072, 43)
