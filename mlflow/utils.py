import datetime
import numpy as np
import pandas as pd

def getData():
    from sklearn.datasets import load_boston
    boston = load_boston()

    data = pd.DataFrame(boston.data,columns=boston.feature_names)
    data['target'] = pd.Series(boston.target)
    return data


def getData():
    from sklearn.datasets import load_boston
    boston = load_boston()

    data = pd.DataFrame(boston.data,columns=boston.feature_names)
    data['target'] = pd.Series(boston.target)
    return data

def eval_metrics(actual, pred):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    metrics = dict()
    metrics['rmse'] = np.sqrt(mean_squared_error(actual, pred))
    metrics['mae'] = mean_absolute_error(actual, pred)
    metrics['r2'] = r2_score(actual, pred)
    metrics['rlog'] = np.sqrt(mean_squared_error(np.log1p(actual), np.log1p(pred)))
    return metrics

def plotPredictionGraph():
    plt.plot(val_pred_df.PredictedRevenue,val_pred_df.transactionRevenue)
    plt.xlabel('real', fontsize=12)
    plt.ylabel('predicted', fontsize=12)
    plt.show()


def plotFeatureImportances(clf,columns=None,type="sklearn"):
    try:
        path = "tmp/featureimportance"
        import matplotlib.pyplot as plt
        if type=="lgb":
            fig, ax = plt.subplots(figsize=(12,12))
            lgb.plot_importance(clf, max_num_features=50, height=0.8, ax=ax)
            ax.grid(False)
            plt.title("LightGBM - Feature Importance", fontsize=15)
        if type=="sklearn":
            import pandas as pd
            importances=pd.Series(data=clf.feature_importances_,index=columns)
            importances.sort_values(ascending=False,inplace=True)

            plt.figure()
            plt.title("Feature importances")
            w=importances.index
            plt.barh(range(len(w)),importances.values)
            plt.yticks(range(len(w)),w)
        plt.savefig(path)
        plt.show()
        return path
    except:
        return False

# -----

def custom_train_test_split(train_df,splitdate=datetime.datetime(2017,5,31)):
    # Split the train dataset into development and valid based on time 
    dev_df = train_df[train_df['date']<=splitdate]
    val_df = train_df[train_df['date']>splitdate]
    transformedData = dict()
    transformedData['dev_X'] = dev_df.drop(columns=["fullVisitorId","transactionRevenue","date"])
    transformedData['dev_y'] = np.log1p(dev_df["transactionRevenue"].astype('float').values)
    transformedData['dev_id'] = pd.DataFrame(dev_df["fullVisitorId"])
    transformedData['val_X'] = val_df.drop(columns=["fullVisitorId","transactionRevenue","date"])
    transformedData['val_y'] = np.log1p(val_df["transactionRevenue"].astype('float').values)
    transformedData['val_id'] = pd.DataFrame(val_df["fullVisitorId"])
    transformedData['train_id'] = train_df["fullVisitorId"].values
    #transformedData['test_id'] = test_df["fullVisitorId"].values
    return transformedData

def createValPredDF(df,val_df,pred_val):
    pred_val[pred_val<0] = 0
    #df = pd.DataFrame(train_df['fullVisitorId'].loc[val_X.index])
    df["transactionRevenue"] = val_df
    df["PredictedRevenue"] = np.expm1(pred_val)
    df = df.groupby("fullVisitorId")["transactionRevenue", "PredictedRevenue"].sum().reset_index()
    return df

def removeNans(x,y):
    lines2drop = x[x['pageviews'].isna()].index
    if len(lines2drop)>0:
        # Drop lines that have NAN value in pageviews
        y = pd.Series(y,index=x.index).drop(lines2drop,axis=0)
        x = x.drop(lines2drop,axis=0)
        print(x.shape)
        print(y.shape)
        # Drop columns that still have NAN values
        x = x.dropna(axis=1)
    return x, y

def createEvaluationDataset(test_id,pred_test):
    sub_df = pd.DataFrame({"fullVisitorId":test_id})
    pred_test[pred_test<0] = 0
    sub_df["PredictedLogRevenue"] = np.expm1(pred_test)
    sub_df = sub_df.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
    sub_df.columns = ["fullVisitorId", "PredictedLogRevenue"]
    sub_df["PredictedLogRevenue"] = np.log1p(sub_df["PredictedLogRevenue"])
    sub_df.to_csv("baseline_lgb.csv", index=False)
    return(sub_df)
