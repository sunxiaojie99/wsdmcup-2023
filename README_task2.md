# WSDM Cup 2023: Pre-training for Web Search

## 1 Dependencies
This code requires the following:
- Python 3.6+
- paddlepaddle-gpu 2.3.1 + CUDA 10.2

## 2 Inference
> In this part, you can quickly reproduce the competition results without training the big model.
> 

1. Load the trained model to score the test data set
```shell
python -u eval.py \
--emb_dim 768 --nlayer 12 --nhead 12 \
--init_parameters 'output/base2-pair-bs352-lr2e-7-neg10/best_model.model' \
--test_annotate_path "data/annotate_data/wsdm_test_2_all.txt"  \
--eval_batch_size 2000 \
--result_path 'output/base2-pair-bs352-lr2e-7-neg10/wsdm_test_2_all.txt'
```

2. Combined with the model scoring under the final folder obtained from task1, perform lgb model training. For convenience, you can directly use our trained model to predict and get the final online submission file.
  - task2_final_result.csv

```shell
cd lgb/code
python lgb_predict_2.py
```

## 3 Training
> In this part, you will see how to train all the models we used in part 2.

### 3.1 Pretrain
> We used different settings to train the unbiased LTR model, and please follow the code for reproducing our result.
> 

We rewrite unbiased_learning.py from pytorch to paddle, and we did not use any other deep learning framework except paddlepaddle in task2.

The usage of unbiased_learning.py for model training: 

```
"""
Please change the file_path in the following code, i.e., 
--init_parameters
--train_datadir
--train_feature_path
--test_data_path
--test_feature_path
--output_name
--mean_std_file
--freq_path
  and other paths that you want to change, e.g. valid_data_path and valid_feature_path
"""
output_name='example'
neg_num=10 
task1_epoch=1
freq_k=1.5
freq_b=0.75 
# add fetaure：train_load_feature=1，combine=1
combine=1  # input features to the transformer
train_load_feature=1  # load training features

# add projection：projection=1, rank_feature_size=870
# do not add projection: projection=0, rank_feature_size=793
projection=1
rank_feature_size=834  # 768+66

# add corrections to dla,  change_label='max' or 'delta', delta=xx, temperature=xx
change_label='max'  # choose from 'no', 'delta' or 'max'
temp=0.1
delta=2.0
feature_type='JM_title'
vote=0  # use the vote_feature as label; when vote=1 the feature_type is voted feature instead of the assigned type

# pairwise: mode='pair'
# dla: mode='list'
# replace the documents which did not receive click and was behind the last clicked document to the random negative samples: mode='replace_list'
mode='list'

method_name='DLA'  # can be chosen from "NaiveAlgorithm" (different from the originial method, add a fixed propensity weight), "DLA"
# when mode='pair' , change feature_pairs in args
add_freq='True' # add_freq: please choose from "True" or "False"

python -u ./unbiased_learning.py \
    --init_parameters /wsdm_cup/data/pre_model/baidu_ultr_3l_12h_768e.model \
    --emb_dim 768 \
    --nlayer 3 \
    --nhead 12 \
    --dropout 0.1 \
    --train_datadir /wsdm_cup/data/train_data/ \
    --train_feature_path /wsdm_cup/data/new_train/features/norm_part-00001.txt \
    --valid_data_path '' \
    --valid_feature_path '' \
    --test_data_path /wsdm_cup/data/new_train/finetune_train.txt \
    --test_feature_path /wsdm_cup/data/new_train/features/finetune_train.txt \
    --num_candidates 10 \
    --method_name $method_name  \
    --buffer_size 20000 \
    --n_gpus 1 \
    --negative_num $neg_num \
    --mean_std_file /wsdm_cup/data/new_train/features/mean_std.txt \
    --feature_type $feature_type \
    --temperature $temp \
    --output_name 'output/'$output_name \
    --eval_step 500 \
    --save_step 5000 \
    --n_queries_for_each_gpu 11 \
    --change_label $change_label \
    --lr 2e-6 \
    --delta $delta \
    --projection $projection \
    --rank_feature_size $rank_feature_size \
    --train_load_feature $train_load_feature \
    --combine $combine \
    --mode $mode \
    --task1_epoch $task1_epoch \
    --vote $vote \
    --freq_path "/wsdm_cup/data/new_train/query_frequency.txt" \
    --add_freq $add_freq \
    --freq_k $freq_k \
    --freq_b $freq_b \
```

We used different strategies to train the unbiased model, and then got the final result by using ensemble models. The settings for these ensemble models are as follows:
- setting1: 
```shell
# 12L DLA combine neg20, replace_list

init_parameters=/wsdm_cup/data/pre_model/baidu_ultr_12l_12h_768e.model
neg_num=20, add_freq='False', change_label='max', combine=1, 
eval_batch_size=2000, eval_step=500
rank_feature_size=869, negative_num=20
feature_type='JM_title', method_name='DLA', mode='replace_list'
n_queries_for_each_gpu=11, nlayers=12
```
- setting2: 
```shell
# 12L DLA combine neg10, replace_list, add frequency correction

add_freq='True', change_label='max', combine=1, delta=2.0, 
init_parameters='/wsdm_cup/data/pre_model/baidu_ultr_12l_12h_768e.model',  
combine=1, rank_feature_size=869, negative_num=10, feature_type='JM_title',   
freq_b=0.75, freq_k=1.5, method_name='DLA', mode='replace_list', 
n_queries_for_each_gpu=11, negative_num=10, nlayers=12,  projection=1, 
rank_feature_size=869, train_batch_size=300, train_load_feature=1,  vote=0
```
- setting3: 
```shell
# 12L DLA combine neg10, replace_list

change_label='max', combine=1, delta=2.0, emb_dim=768, method_name='DLA', 
init_parameters='/wsdm_cup/data/pre_model/baidu_ultr_12l_12h_768e.model',
combine=1, rank_feature_size=869, negative_num=10, 
change_label='max', feature_type='JM_title', mode='replace_list'
negative_num=10,  nlayers=12, projection=1, rank_feature_size=869
```
- setting4: 
```shell
# 3L DLA neg10 jmtitle_max, bs5

change_label='max', combine=1, eval_step=1000,
method_name='DLA', n_gpus=1, 
combine=1, rank_feature_size=869, negative_num=10, change_label='max', feature_type='JM_title', mode='list', 
n_queries_for_each_gpu=5, negative_num=10, nhead=12, nlayers=3, projection=1, rank_feature_size=869
```
- setting5: 
```shell
# 3L naive_pmax2.5_list_max_delta2_tem0.1_neg10_834projection_combine_3l

negative_num=10, change_label='max', feature_type='JM_title', mode='list', 
projection=1, rank_feature_size=834, method_name='NaiveAlgorithm'

The propensity weight in NA should be set as (in baseline_model/learning_algorithm/naive.py, line 106):
propensity_weight = [1.        , 1.18538058, 1.44317997, 1.58171761, 1.89201605, 1.84581888,
         1.95202422, 2.11557865, 2.26336575, 2.50935149]

```
- setting6: 
```shell
# 3L naive_pmax2.5_list_max(JM_title)_neg10_combine_3l

projection=1, negative_num=10, change_label='max', feature_type='JM_title', mode='list', method_name='NaiveAlgorithm'
The propensity weight in NA should be set as:
[1.        , 1.18538058, 1.44317997, 1.58171761, 1.89201605, 1.84581888,
         1.95202422, 2.11557865, 2.26336575, 2.50935149]
```

We use the default settings if not specified above.  And we train the models which are listed below the path "wsdm2023/final". We only introduce the main model settings and omit the other settings due to the space limitation. 

To reproduce the results, you can find the settings in "wsdm2023/final/readme.md" and train.  

### 3.2 Finetune
In this stage, we first split the validation data into two parts (i.e. Train & valid)

```
"""
Please modify the corresponding paths in the scripts/make_data.py file, 
which are line 150, 181，197，200，208，221，228，242，249，262.
"""
python ./scripts/make_data.py
```

Then, we fintune the initial model in the training set. Please use the following code to finetune the pretrained model:

```shell
sh finetune_base.sh  # To get the init model in the next script
sh finetune.sh
```

### 3.3 Ensemble

> Then, we use the output of the pretrained model and several unbiased LTR models (which is written in paddle)  to generate the final score. The procedure is as follows: 
> 

Firstly, we combine the predictions of different pretrained models by using the code:
```
"""
Please modify the corresponding paths in the lgb/code/merge_feature.py file, 
which are in line 3, 85, 86 and 87
"""
cd lgb/code
python merge_feature.py
```

Then, we process the data into numpy files for faster learning in the latter steps. 

```
cd lgb/code
"""
Please modify the corresponding paths in the read_data2.py file, 
which are in line 107, 108, 109, 119, 120 and 121
"""
python read_data2.py

"""
Please modify the corresponding paths in the read_test.py file, 
which are in line 37, 38 and 39
"""
python read_test.py

"""
Please modify the corresponding paths in the get_group_train2.py file, 
which are in line 29 and 36
"""
python get_group_train2.py
```

Then train a GDBT model and make predictions:

```shell
cd lgb/code

python run_lgb_2.py
python lgb_predict_2.py
```

Finally, the zipped file can be obtained by using:

```shell
zip task2_final_result.csv.zip task2_final_result.csv
```