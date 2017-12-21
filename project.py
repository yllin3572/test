
# coding: utf-8
# Yihao Lin, Qiang Fei, Audrey Lee
# 2017-12-21

# FM library
import pywFM
# XGBoost library
import xgboost as xgb
import numpy as np
import pandas as pd
import ml_metrics as metrics
from sklearn import cross_valiadation
from sklearn.ensemble import RandomForestClassifier
# set environment varibale in Ubuntu 
get_ipython().magic('env LIBFM_PATH=/home/yl3572/libfm/bin/')


# A simple FM example to test the FM library and show the data structure (not 
# this project since there are too many features to show)
features = np.matrix([
#     Users  |     Movies     |    Movie Ratings   | Time | Last Movies Rated
#    A  B  C | TI  NH  SW  ST | TI   NH   SW   ST  |      | TI  NH  SW  ST
    [1, 0, 0,  1,  0,  0,  0,   0.3, 0.3, 0.3, 0,     13,   0,  0,  0,  0 ],
    [1, 0, 0,  0,  1,  0,  0,   0.3, 0.3, 0.3, 0,     14,   1,  0,  0,  0 ],
    [1, 0, 0,  0,  0,  1,  0,   0.3, 0.3, 0.3, 0,     16,   0,  1,  0,  0 ],
    [0, 1, 0,  0,  0,  1,  0,   0,   0,   0.5, 0.5,   5,    0,  0,  0,  0 ],
    [0, 1, 0,  0,  0,  0,  1,   0,   0,   0.5, 0.5,   8,    0,  0,  1,  0 ],
    [0, 0, 1,  1,  0,  0,  0,   0.5, 0,   0.5, 0,     9,    0,  0,  0,  0 ],
    [0, 0, 1,  0,  0,  1,  0,   0.5, 0,   0.5, 0,     12,   1,  0,  0,  0 ]
])
target = [5, 3, 1, 4, 5, 1, 5]

fm = pywFM.FM(task='regression', num_iter=5)

# split features and target for train/test
# first 5 are train, last 2 are test
model = fm.run(features[:5], target[:5], features[5:], target[5:])
print(model.predictions)
# you can also get the model weights
print(model.weights)


# Data preprocessing
import pandas as pd
from sklearn.decomposition import PCA

# A small data test 
#n=20000
#sample = pd.read_csv('train.csv',iterator=True)
#sample = sample.get_chunk(n)
# All data
sample = pd.read_csv('train.csv')
sample["date_time"] = pd.to_datetime(sample["date_time"])
sample["year"] = sample["date_time"].dt.year
sample["month"] = sample["date_time"].dt.month

# Supplementary destinations data
destinations = pd.read_csv('destinations.csv')

# Apply PCA and select the first three components
pca = PCA(n_components=3)
dest_small = pca.fit_transform(destinations[["d{0}".format(i + 1) for i in range(149)]])
dest_small = pd.DataFrame(dest_small,columns=['PCA1','PCA1','PCA3'])
dest_small["srch_destination_id"] = destinations["srch_destination_id"]

# Generate features
def calc_fast_features(df):
    df["date_time"] = pd.to_datetime(df["date_time"])
    df["srch_ci"] = pd.to_datetime(df["srch_ci"], format='%Y-%m-%d', errors="coerce")
    df["srch_co"] = pd.to_datetime(df["srch_co"], format='%Y-%m-%d', errors="coerce")
    
    props = {}
    for prop in ["month", "dayofweek", "quarter"]:
        props[prop] = getattr(df["date_time"].dt, prop)
    
    carryover = [p for p in df.columns if p not in ["date_time", "srch_ci", "srch_co"]]
    for prop in carryover:
        props[prop] = df[prop]
    
    date_props = ["month", "dayofweek", "quarter"]
    for prop in date_props:
        props["ci_{0}".format(prop)] = getattr(df["srch_ci"].dt, prop)
        props["co_{0}".format(prop)] = getattr(df["srch_co"].dt, prop)
    props["stay_span"] = (df["srch_co"] - df["srch_ci"]).astype('timedelta64[D]')
        
    ret = pd.DataFrame(props)
    
    ret = ret.join(dest_small, on="srch_destination_id", how='left', rsuffix="dest")
    ret = ret.drop("srch_destination_iddest", axis=1)
    return ret

df = calc_fast_features(sample)
df.fillna(-1, inplace=True)

# Train data set
df_train = df[((df.year == 2013) | ((df.year == 2014) & (df.month < 9)))]
# Test data set
df_test = sample[((df.year == 2014) & (df.month >= 9))]
df_test = df_test[df_test.is_booking == True]



# Model I
# Random Forest model
def random_forest_model(train_data, test_data):
    predictors = [variable for variable in train_data.columns if variable not in ["hotel_cluster"]]
    clf = RandomForestClassifier(n_estimators=20)
    # use cross-validation
    scores = cross_validation.cross_val_score(clf, train_data[predictors], train_data['hotel_cluster'], cv=3)

    # get the train_model
    clf.fit(train_data[predictors], train_data['hotel_cluster'])

    # use the model to predict
    test_proba = clf.predict_proba(test_data[predictors])
    predictions = []
    # get the five highest probabilty hotel_clusters
    for element in test_proba:
        predictions.append(ng.argsort(x)[::-1][:5])

    return predictions

# Get the five most popular hotel_cluster
def most_popular(group, n_max=5):
    popularity = group['popularity'].values
    hotel_cluster = group['hotel_cluster'].values
    most_popular = hotel_cluster[np.argsort(popularity)[::-1]][:n_max]

    return np.array_str(most_popular)[1:-1]  # remove square brackets


def popularity_ranking(file_path):
    # define the dataframe
    data = pd.read_csv(file_path, dtype={'is_booking': bool, 'srch_distination': np.int32,
                                         'hotel_cluster': np.int32}, usecols=['srch_distination_id', 'is_booking',
                                                                              hotel_cluster], chunksize=1000000)

    # read the data in chunk
    aggs = []
    dataframe = []
    for chunk in data:
        agg = chunk.groupby(['srch_distination_id', 'hotel_cluster'])['is_booking'].agg(['sum', 'count'])
        agg.reset_index(inplace=True)
        aggs.append(agg)
        dataframe.append(chunk)

    aggs = pd.concat(aggs, axis=0)
    dataframe = pd.concat(dataframe, axis=0)
    print('All data read successfully')

    # build the popularity ranking model
    CLICK_WEIGHT = 0.05
    agg = aggs.groupby(['srch_distination_id', 'hotel_cluster']).sum().reset_index()
    agg['count'] -= agg['sum']
    agg = agg.rename(columns={'sum': 'bookings', 'count': 'clicks'})
    agg['popularity'] = agg['bookings'] + CLICK_WEIGHT * agg['clicks']

    most_pop = agg.groupby(['srch_destination_id']).apply(most_popular)
    most_pop = pd.DataFrame(most_pop).rename(columns={0: 'hotel_cluster'})

    return most_pop

# Combine the two models above
def model_1(srch_id_list, most_pop, predictions, thres_probability=0.345):
    # combine random_forest and popularity ranking
    predictions = []

    for i in range(len(predictions)):
        rf_list = list((np.argsort(predictions[i])[::-1][:len(np.where(predictions[i] > thres)[0])]))

        srch_id = srch_id_list[i]
        pr_list = []
        for s in str(most_pop.loc[srch_id_list[i]][0]).split():
            pr_list.append(int(s))

        if (len(rf_list) == 5):
            predictions.append(rf_list)
            continue
        else:
            for element in pr_list:
                if element in rf_list:
                    pass
                else:
                    rf_list.append(element)

        if (len(rf_list) <= 5):
            pass
        else:
            rf_list = rf_list[:5]

        predictions.append(rf_list)

    return predictions


def get_accuracy(predictions, test_data):
    target = [[hotel_cluster] for hotel_cluster in test_data['hotel_cluster']]
    accuracy = metrics.mapk(target, predictions, k=5)
    return accuracy

# Run Model I and get accuracy
rf_pred = random_forest_model(df_train,df_test)
# Your data file path
file_path = '../../..'
most_pop = popularity_ranking(file_path)
srch_id_list = df_test['srch_destination_id'].values
m1_pred = model_1(srch_id_list, most_pop, rf_pred)
get_accuracy(m1_pred, df_test)

# Model II
# FM model
# Create dummy variables for user id
col=list(df.columns)
col.remove('hotel_cluster')
df_train2 = df_train[col]
df_test2 = df_test[col]
X_train = pd.get_dummies(df_train2, columns=['user_id'], sparse=1)
X_train = X_train.fillna(0)
Y_train = [[1 if x==i else 0 for x in df_train['hotel_cluster']] for i in range(100)]
X_test = pd.get_dummies(df_test2, columns=['user_id'], sparse=1)
X_test = X_test.fillna(0)
Y_test = [[1 if x==i else 0 for x in df_test['hotel_cluster']] for i in range(100)]

# Set FM model
fm = pywFM.FM(task='classification', num_iter=5)
# Predicton scores
pre = []
for i in range(100):
    model = fm.run(X_train, Y_train[i], X_test, Y_test[i])
    pre.append(model.predictions)
    print(i)
# You can also get the model weights
#print(model.weights)



# Save the predictions scores 
import csv
with open('fm.csv', "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(pre)



# XGBoost model
test_fm_score = list(sum(zip(*pre), ()))
train_fm_score = [1 if i==x else 0 for i in range(100) for x in df_train['hotel_cluster']]



# Book/click ratio for each hotel cluster
book = list(df_train.groupby(['hotel_cluster'])['is_booking'].sum())
click = list(np.subtract(list(df_train.groupby(['hotel_cluster'])['is_booking'].count()),book))
ratio = np.array(book)/np.array(click)



# Convert each row into 100 rows
train_book = book*len(df_train)
test_book = book*len(df_test)
train_click = click*len(df_train)
test_click = click*len(df_test)
train_cluster = [y+ratio[i]*(1-y) if i==x else 0 for i in range(100) for x,y in 
                 np.array(df_train[['hotel_cluster','is_booking']])]
test_cluster = [y+ratio[i]*(1-y) if i==x else 0 for i in range(100) for x,y in 
                np.array(df_test[['hotel_cluster','is_booking']])]

# Train data and validation data for XGBoost
train = pd.DataFrame({'book':train_book,'click':train_click,'cluster':train_cluster,'fm':train_fm_score})
val = pd.DataFrame({'book':test_book,'click':test_click,'cluster':test_cluster,'fm':test_fm_score})



# Independent variables and dependent variable
y = train.cluster
X = train.drop(['cluster'],axis=1)
val_y = val.cluster
val_X = val.drop(['cluster'],axis=1)

# Convert to xgb matrix
xgb_val = xgb.DMatrix(val_X,label=val_y)
xgb_train = xgb.DMatrix(X, label=y)

# Set xgb model
num_rounds = 100
params={
'booster':'gbtree',
'objective': 'binary:logistic', 
'gamma':0.1,  
'max_depth':5, 
'lambda':3, 
'subsample':0.7, 
'colsample_bytree':0.7, 
'min_child_weight':3, 
'silent':0 ,
'eta': 0.01, 
'seed':1000,
'nthread':16,
}

model = xgb.train(params,xgb_train, num_rounds)
# All predictions 
preds = model.predict(xgb_val,ntree_limit=model.best_ntree_limit)

# We can plot feature importance
#xgb.plot_importance(model)



# Final results and accuracy
result = []
for i in range(n-nn):
    result.append([preds[i*100+j] for j in range(100)])

predictions = []
for x in result:
    predictions.append(np.argsort(x)[::-1][:5])

target = [[x] for x in df['hotel_cluster'][nn:]]
metrics.mapk(target, predictions, k=5)

