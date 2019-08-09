import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


def clean_columns(df):
    df.columns = df.columns.str.strip().str.lower()
    df.columns = ['_'.join(col.split()) for col in df.columns]
    return df


def over_sample_data(X, y):
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_sample(X,y)
    return X_resampled, y_resampled

def under_sample_data(X, y):
    rus = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = rus.fit_sample(X,y)
    return X_resampled, y_resampled

def test_logistic_regression_thresholds(model, threshold_list, X_test, y_test):
    pred_proba_df = pd.DataFrame(model.predict_proba(X_test))
    for i in threshold_list:
        print ('\n******** For i = {} ******'.format(i))
        Y_test_pred = pred_proba_df.applymap(lambda x: 1 if x>i else 0)
        test_accuracy = accuracy_score(y_test.as_matrix().reshape(y_test.as_matrix().size,1),
                                               Y_test_pred.iloc[:,1].as_matrix().reshape(Y_test_pred.iloc[:,1].as_matrix().size,1))
        print('Our testing accuracy is {}'.format(test_accuracy))

        test_precision = precision_score(y_test.as_matrix().reshape(y_test.as_matrix().size,1),
                                               Y_test_pred.iloc[:,1].as_matrix().reshape(Y_test_pred.iloc[:,1].as_matrix().size,1))
        print('Our testing precision is {}'.format(test_precision))

        test_recall = recall_score(y_test.as_matrix().reshape(y_test.as_matrix().size,1),
                                               Y_test_pred.iloc[:,1].as_matrix().reshape(Y_test_pred.iloc[:,1].as_matrix().size,1))
        print('Our testing recall is {}'.format(test_recall))

        print(confusion_matrix(y_test.as_matrix().reshape(y_test.as_matrix().size,1),
                               Y_test_pred.iloc[:,1].as_matrix().reshape(Y_test_pred.iloc[:,1].as_matrix().size,1)))


def get_accuracy_precision_recall_f1_roc_auc_score_and_curve(model, X_test, y_test):
    pred = model.predict(X_test)
    pred_proba = model.predict_proba(X_test)[:,1]

    print('accuracy: ', accuracy_score(y_test, pred))
    print('precision: ', precision_score(y_test, pred))
    print('recall: ', recall_score(y_test, pred))
    print('f1: ', f1_score(y_test, pred))
    print('roc_auc_score: ', roc_auc_score(y_test, pred_proba))
    fpr, tpr, thresholds = roc_curve(y_test, pred_proba)

    plt.plot(fpr, tpr,lw=2)
    plt.plot([0,1],[0,1],c='violet',ls='--')
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.figure();
    print('')

def get_metrics_and_roc_curve(model, model_name, X_test, y_test):
    pred = model.predict(X_test)
    pred_proba = model.predict_proba(X_test)[:,1]

    roc_auc = np.round(roc_auc_score(y_test, pred_proba), decimals=6)
    print('***** {} *****'.format(model_name))
    print('accuracy: ', accuracy_score(y_test, pred))
    print('precision: ', precision_score(y_test, pred))
    print('recall: ', recall_score(y_test, pred))
    print('f1: ', f1_score(y_test, pred))
    print('roc_auc_score: ', roc_auc)
    fpr, tpr, thresholds = roc_curve(y_test, pred_proba)

    plt.plot(fpr, tpr,lw=2)
    plt.plot([0,1],[0,1],ls='--') # c='violet'
    plt.plot(fpr,tpr,label='{}, auc={}'.format(model_name, roc_auc))
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(loc=0)
    # plt.figure();
    print('')

def get_precision_recall_curve(model, X_test, y_test):
    pred_y=model.predict(X_test)

    probs_y=model.predict_proba(X_test)

    precision, recall, thresholds = precision_recall_curve(y_test, probs_y[:,
    1])

    pr_auc = auc(recall, precision)

    plt.title("Precision-Recall vs Threshold Chart")
    plt.plot(thresholds, precision[: -1], "b--", label="Precision")
    plt.plot(thresholds, recall[: -1], "r--", label="Recall")
    plt.ylabel("Precision, Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="lower left")
    plt.ylim([0,1])
    plt.figure();


def calc_iv(df, feature, target, pr=False):
    """
    Set pr=True to enable printing of output.

    Output:
      * iv: float,
      * data: pandas.DataFrame
    """

    lst = []

    df[feature] = df[feature].fillna("NULL")

    for i in range(df[feature].nunique()):
        val = list(df[feature].unique())[i]
        lst.append([feature,                                                        # Variable
                    val,                                                            # Value
                    df[df[feature] == val].count()[feature],                        # All
                    df[(df[feature] == val) & (df[target] == 0)].count()[feature],  # Good (think: Fraud == 0)
                    df[(df[feature] == val) & (df[target] == 1)].count()[feature]]) # Bad (think: Fraud == 1)

    data = pd.DataFrame(lst, columns=['Variable', 'Value', 'All', 'Good', 'Bad'])

    data['Share'] = data['All'] / data['All'].sum()
    data['Bad Rate'] = data['Bad'] / data['All']
    data['Distribution Good'] = (data['All'] - data['Bad']) / (data['All'].sum() - data['Bad'].sum())
    data['Distribution Bad'] = data['Bad'] / data['Bad'].sum()
    data['WoE'] = np.log(data['Distribution Good'] / data['Distribution Bad'])

    data = data.replace({'WoE': {np.inf: 0, -np.inf: 0}})

    data['IV'] = data['WoE'] * (data['Distribution Good'] - data['Distribution Bad'])

    data = data.sort_values(by=['Variable', 'Value'], ascending=[True, True])
    data.index = range(len(data.index))

    if pr:
        print(data)
        print('IV = ', data['IV'].sum())


    iv = data['IV'].sum()
    # print(iv)

    return iv, data
