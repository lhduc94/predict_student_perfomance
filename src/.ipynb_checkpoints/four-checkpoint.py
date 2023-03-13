import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
from collections import Counter
from sklearn.model_selection import  GroupKFold
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import event_features, room_features, fqid_lists, name_features,text_lists, room_lists, LEVELS, level_groups, plot_feature_importance
np.random.seed(42)


class Config:
    TRAIN_PATH = 'D:\Workspace\predict-student-performance-from-game-play\inputs/train.csv'
    TRAIN_LABELS = 'D:\Workspace\predict-student-performance-from-game-play\inputs/train_labels.csv'

def q2l(x):
    if x <= 3:
        return '0-4'
    if x <= 13:
        return '5-12'
    return '13-22'




def groupby_apply(g):
    res = {}
    elasped_time = g['elapsed_time'].values / 1000
    level = g['level'].values
    res['duration'] = elasped_time.max() - elasped_time.min()
    for i in range(0, 23):
        t = elasped_time[level == i]
        if len(t) > 0:
            res[f'duration_level_{i}'] = t.max() - t.min()
        else:
            res[f'duration_level_{i}'] = 0
    res['text_fqid_null'] = pd.isnull(g['text_fqid']).sum()
    event_name_dict = Counter(g['event_name'].values)
    event_sequence = g['event_name'].values
    room_event_dict = Counter(g['room_event'].values)
    room_sequence = g['main_room'].values
    for col in event_features:
        res[f'{col}_sum'] = event_name_dict.get(col, 0)
        for col2 in room_features:
            res[f'{col}_{col2}_sum'] = room_event_dict.get(f'{col}_{col2}', 0)
    room_dict = Counter(g['main_room'].values)
    for col in room_features:
        res[f'{col}_sum'] = room_dict.get(col, 0)

    elapsed_time_diff_all = g['elapsed_time_diff'].values
    res['elapsed_time_diff_mean'] = np.mean(elapsed_time_diff_all)
    res['elapsed_time_diff_std'] = np.std(elapsed_time_diff_all)
    res['elapsed_time_diff_max'] = np.max(elapsed_time_diff_all)
    res['elapsed_time_diff_min'] = np.min(elapsed_time_diff_all)
    res['elapsed_time_diff_positive'] = len(elapsed_time_diff_all[elapsed_time_diff_all > 0])

    for col in event_features:
        elapsed_time_diff_event = elapsed_time_diff_all[event_sequence == col]
        elapsed_time_diff_event = elapsed_time_diff_event if len(elapsed_time_diff_event) > 0 else [0]
        res[f'elapsed_time_diff_{col}_mean'] = np.mean(elapsed_time_diff_event)
        res[f'elapsed_time_diff_{col}_max'] = np.max(elapsed_time_diff_event)
        res[f'elapsed_time_diff_{col}_min'] = np.min(elapsed_time_diff_event)
        res[f'elapsed_time_diff_{col}_std'] = np.std(elapsed_time_diff_event)
    fqid_sequence = g['fqid'].values
    for col in fqid_lists:
        elapsed_time_diff_fqid = elapsed_time_diff_all[fqid_sequence == col]
        elapsed_time_diff_fqid = elapsed_time_diff_fqid if len(elapsed_time_diff_fqid) > 0 else [0]
        res[f'elapsed_time_diff_{col}_mean'] = np.mean(elapsed_time_diff_fqid)
        res[f'elapsed_time_diff_{col}_max'] = np.max(elapsed_time_diff_fqid)
        res[f'elapsed_time_diff_{col}_min'] = np.min(elapsed_time_diff_fqid)
        res[f'elapsed_time_diff_{col}_std'] = np.std(elapsed_time_diff_fqid)

    text_sequence = g['text_fqid'].values
    for col in text_lists:
        elapsed_time_diff_text = elapsed_time_diff_all[text_sequence == col]
        elapsed_time_diff_text = elapsed_time_diff_text if len(elapsed_time_diff_text) > 0 else [0]
        res[f'elapsed_time_diff_{col}_mean'] = np.mean(elapsed_time_diff_text)
        res[f'elapsed_time_diff_{col}_max'] = np.max(elapsed_time_diff_text)
        res[f'elapsed_time_diff_{col}_min'] = np.min(elapsed_time_diff_text)
        res[f'elapsed_time_diff_{col}_std'] = np.std(elapsed_time_diff_text)
    return pd.Series(res)

def feature_engineering(df, meta):
    df['main_room'] = df['room_fqid'].str.split('.').str[1]
    df['room_event'] =  df['event_name']+'_' + df['main_room']
    X = df.groupby(['session','level_group']).apply(groupby_apply).reset_index()
    X = meta.merge(X,how='left', on=['session','level_group'])
    X['question'] = X['question'].astype('category')
    X['level_group'] = X['level_group'].astype('category')
    for i in range(1,19):
        X[f'q{i}'] = X['question'] == i
    return X

print("Read_data")
train_df = pd.read_csv(Config.TRAIN_PATH, usecols=lambda x: x not in ['fullscreen','hq','music'])
train_labels = pd.read_csv(Config.TRAIN_LABELS)
train_df.rename(columns={'session_id':'session'},inplace=True)
train_labels['question'] = train_labels['session_id'].str.split('q').str[-1].astype('int')
train_labels['session'] = train_labels['session_id'].str.split('_').str[0].astype('int64')
train_labels['level_group'] = train_labels['question'].apply(lambda x: q2l(x))

train_df['elapsed_time_diff'] = train_df.groupby(['session','level'])['elapsed_time'].diff()
train_df['elapsed_time_diff'].fillna(0,inplace=True)


def time_feature(train):
    train["year"] = train["session_id"].apply(lambda x: int(str(x)[:2])).astype(np.uint8)
    train["month"] = train["session_id"].apply(lambda x: int(str(x)[2:4])+1).astype(np.uint8)
    train["day"] = train["session_id"].apply(lambda x: int(str(x)[4:6])).astype(np.uint8)
    train["hour"] = train["session_id"].apply(lambda x: int(str(x)[6:8])).astype(np.uint8)
    train["minute"] = train["session_id"].apply(lambda x: int(str(x)[8:10])).astype(np.uint8)
    train["second"] = train["session_id"].apply(lambda x: int(str(x)[10:12])).astype(np.uint8)


    return train

print("Feature_Engineering")

X = feature_engineering(train_df, train_labels)
X = time_feature(X)

del train_df
gc.collect()

FEATURES = X.columns[5:]
len(FEATURES)
print("Training")

n_splits=5
gkf = GroupKFold(n_splits=n_splits)
oof = np.zeros(X.shape[0])
models = {}

# COMPUTE CV SCORE WITH 5 GROUP K FOLD
for i, (train_index, valid_index) in enumerate(gkf.split(X, groups=X['session'])):
    print('#'*25)
    print('### Fold',i+1)
    print('#'*25)

    xgb_params = {
        'objective' : 'binary:logistic',
        'eval_metric':'logloss',
        'learning_rate': 0.01,
        'max_depth': 5,
        'n_estimators': 3000,
        'early_stopping_rounds': 50,
        'subsample':0.8,
        'colsample_bytree': 0.8,
        'seed':42,
        'use_label_encoder' : False}

    X_train = X.iloc[train_index][FEATURES]
    X_valid = X.iloc[valid_index][FEATURES]
    y_train = X.iloc[train_index]['correct'].values
    y_valid = X.iloc[valid_index]['correct'].values
        # TRAIN MODEL
    clf =  XGBClassifier(**xgb_params)
    clf.fit(X_train.astype('float32'), y_train,
            eval_set=[(X_train.astype('float32'), y_train), (X_valid.astype('float32'),y_valid)],
            verbose=100)
    print(f'({clf.best_ntree_limit}), ',end='')

        # SAVE MODEL, PREDICT VALID OOF
    models[i] = clf
    oof[valid_index] = clf.predict_proba(X_valid)[:,1]

    print()

feat_imp = {}
for k, v in models.items():
    for x, y in zip(v.feature_importances_, v.feature_names_in_):
        if y not in feat_imp:
            feat_imp[y] = x
        else:
            feat_imp[y]+=x
plot_feature_importance(list(feat_imp.values()),list(feat_imp.keys()),'')
import matplotlib.pyplot as plt
scores = []; thresholds = []
best_score = 0; best_threshold = 0

for threshold in np.arange(0.4,0.9,0.01):
    print(f'{threshold:.02f}, ',end='')
    preds = (oof>threshold).astype('int')
    m = f1_score(X['correct'], preds, average='macro')
    scores.append(m)
    thresholds.append(threshold)
    if m>best_score:
        best_score = m
        best_threshold = threshold
# PLOT THRESHOLD VS. F1_SCORE
plt.figure(figsize=(20,5))
plt.plot(thresholds,scores,'-o',color='blue')
plt.scatter([best_threshold], [best_score], color='blue', s=300, alpha=1)
plt.xlabel('Threshold',size=14)
plt.ylabel('Validation F1 Score',size=14)
plt.title(f'Threshold vs. F1_Score with Best F1_Score = {best_score:.4f} at Best Threshold = {best_threshold:.3}',size=18)
plt.show()