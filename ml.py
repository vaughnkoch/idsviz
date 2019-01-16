# ML
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# Dataviz
import scikitplot
import matplotlib.pyplot as plt
from sklearn import metrics
from viz import show_data, show_output_dataviz, print_notebook

# Notebook
from utils.notebook import in_ipynb
from IPython.display import Markdown, display

# The dataset should be formatted in the following way:
# One row per user
# All numerical values
# Code to prep the fields for e.g. timestamps is included
# You have to do aggregation yourself before the csv is generated (e.g. count the number of purchases user X made)
# One column 'target' should be a binary 0/1 indicating the Y-value to model (e.g. did rebook/didn't rebook)


def predict_ml():
  # Get the dataset and extract the target (Y)
  # df = pd.read_csv('path-to-your-dataset-of-selected-users')

  df = get_clean_titanic_data()

  Y = df['target']
  X = df.drop('target', axis=1)

  # Exploratory dataviz
  show_data(X, Y)


  # Split the initial dataset into random train and test samples
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


  # Train the engine
  scale_pos_weight = get_scale_pos_weight(Y_train)
  fit_params = get_fit_params(X_test, Y_test)
  clf = get_classifier(scale_pos_weight=scale_pos_weight)
  clf.fit(X_train, Y_train, **fit_params)

  # Do the actual predictions
  threshold = 0.5
  Y_probs = clf.predict_proba(X_test)  # Raw probabilities
  Y_predicted = (Y_probs[:,1] >= threshold).astype('int')  # Actual predictions given the threshold


  # Results
  report_results(clf, fit_params, X, X_train, X_test, Y, Y_train, Y_test, Y_predicted, Y_probs)



def get_classifier(scale_pos_weight=1):
  # https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

  # Defaults
  hyperparameters = {
    # This gradually increases ROCAUC but at the cost of many iterations (n_estimators)
    'learning_rate': 0.1,  # Alias: 'eta'.
    'min_child_weight': 1,
    'max_depth': 3,
    'gamma': 0,
    'subsample': 1,
    'colsample_by_tree': 0.5,
    'scale_pos_weight': scale_pos_weight,

    # Learning task
    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    # 'reg_alpha': 1,
    'n_estimators': 148,

    # This extra value is here so you can test different hyperparameters; the tradeoff is increased train/prediction times
    # 'n_estimators': 1000,
  }

  clf = XGBClassifier(seed=42, **hyperparameters)

  return clf


def report_results(clf, fit_params, X, X_train, X_test, Y, Y_train, Y_test, Y_predicted, Y_probs):
  print_notebook('Machine Learning Results', heading=True)

  # Dataviz
  if in_ipynb():
    print_notebook('Confusion matrix: https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/')

    scikitplot.metrics.plot_confusion_matrix(Y_test, Y_predicted, [0, 1])
    plt.show()

    print_notebook('Roc curves: https://www.dataschool.io/roc-curves-and-auc-explained/')

    scikitplot.metrics.plot_roc(Y_test, Y_probs)
    plt.show()


  # Results
  compute_rocauc(Y_test, Y_probs)
  log_classification_report(clf, Y_test, Y_predicted)
  log_classifier_details(clf, fit_params)

  show_output_dataviz(clf, X_test, Y_test, Y_probs)



# XGBoost param (might be in other classifiers too)
# http://xgboost.readthedocs.io/en/latest/parameter.html
# "A typical value to consider: sum(negative cases) / sum(positive cases)"
# Meaning, count(no) / count (yes)
def get_scale_pos_weight(Y):
  counts = get_class_counts_by_label(Y)
  return counts[0] / counts[1]


def get_fit_params(X_test, Y_test):
  fit_params = {
    'verbose': False,
    'eval_metric': 'auc',
    'eval_set': [(X_test, Y_test)],
    'early_stopping_rounds': 50,
  }

  return fit_params


def compute_rocauc(Y_test, Y_probs):
  # Get the positive column. For binary problems this will just be 1, but it can be used for multiclass rocauc as well.
  Y_probs_positive = Y_probs[:, 1]
  rocauc = metrics.roc_auc_score(Y_test, Y_probs_positive)
  print (f'Rocauc: {rocauc}')


def log_classifier_details(clf, fit_params):
  print_notebook('Classifier details', heading=True)
  print (f'Chosen classifier:\n{clf}')
  print_notebook('XGBoost: https://xgboost.readthedocs.io/en/latest/')

  # hp_str = pprint.pformat(clf.get_xgb_params())
  hp_str = str(clf.get_xgb_params())
  print(f'Hyperparameters:\n{hp_str}')

  fit_params_copy = fit_params.copy()
  del fit_params_copy['eval_set']
  print(f'fit_params: {fit_params_copy}')


def log_classification_report(clf, Y_test, Y_predicted):
  report = metrics.classification_report(Y_test, Y_predicted, labels=clf.classes_, target_names=['no', 'yes'])
  f1_score = metrics.f1_score(Y_test, Y_predicted, labels=clf.classes_, average='macro')
  print(f'Classification report:\n{report}')
  print (f'F1 score: {f1_score}')



def get_class_counts_by_label(Y):
  labels = np.unique(Y)

  counts = []
  for label in labels:
    Y_label = Y[Y == label]
    counts.append(Y_label.shape[0])

  return counts



def get_clean_titanic_data():
  df = pd.read_csv('titanic_train.csv')


  df.loc[df['Sex'] == 'female', 'is_male'] = 0
  df.loc[df['Sex'] == 'male', 'is_male'] = 1

  df = df.drop(['Name', 'Ticket', 'Embarked', 'Cabin', 'Sex', 'PassengerId'], axis='columns')
  df = df.rename(index=str, columns={ "Survived": "target" })

  df.fillna(0, inplace=True)
  return df


predict_ml()
