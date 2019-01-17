# ML
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Dataviz
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Notebook
from utils.notebook import in_ipynb
from IPython.display import Markdown, display


def print_notebook(msg, heading=False):
  header_str = '# ' if heading else ''

  if in_ipynb():
    display( Markdown(f'{header_str}{msg}') )
  else:
    if heading:
      print(f'----------- {msg} ----------')
    else:
      print(msg)


def show_data(X, Y):

  print_notebook('Exploratory Dataviz', heading=True)

  # Show raw data
  print (X.head(20))

  set_viz_defaults()
  show_outcome_bar_chart(Y, ['no', 'yes'])

  for column in X.columns:
    C = X[column]
    show_histogram_of_outcomes_from_numerical(C, Y, column, 'Outcomes', column)



def show_histogram_of_outcomes_from_numerical(X, Y, label_x, label_y, variable_name, threshold=None, threshold_label='threshold', x_range=None):
  # print (f'Params: {plt.rcParams["figure.figsize"]}')

  # Create ndarrays
  X = np.array(X)
  Y = np.array(Y)

  # Filter the list for outputs of a particular value.
  # This works because Y == 0/Y == 1 each output a boolean array with True/False corresponding to each element.
  # If you pass that array into X, numpy uses fancy indexing to only select the values that are True in the index.
  X_0 = X[Y == 0]
  X_1 = X[Y == 1]

  num_bins = 20

  # If we're passed an explicit range, use that first.
  if x_range:
    min_x, max_x = x_range
  else:
    min_x, max_x = X.min(), X.max()


  bins = np.linspace(min_x, max_x, num_bins)

  alpha = 1
  index = np.arange(num_bins)

  fig, axes = plt.subplots(1, 2, False, figsize=(22, 6.67) )
  fig.suptitle(f'{variable_name}')

  ax0 = axes[0]
  ax1 = axes[1]

  # Histogram
  ax0.hist([X_0, X_1], bins, color=['r', 'b'], alpha=alpha, label=['no', 'yes'])

  ax0.set_xlim(min_x, max_x)
  ax0.set_title(f'Histogram of {variable_name}')
  ax0.set_xlabel(variable_name)
  ax0.set_ylabel('Count of outcomes')
  ax0.legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.5, shadow='true')

  if threshold:
    y_max = ax0.get_ylim()[1]
    ax0.axvline(x=threshold, ymin=0, ymax=1, label='threshold', color='red', alpha=0.2)
    ax0.text(threshold * 1.02, y_max * 0.65, threshold_label, rotation=90, color='red', alpha=0.2)


  # KDE
  # https://www.quora.com/What-is-kernel-density-estimation
  try:
    sns.distplot(X_0, bins=20, kde=True, hist=False, color='r', ax=ax1, label='no')
    sns.distplot(X_1, bins=20, kde=True, hist=False, color='b', ax=ax1, label='yes')

    ax1.legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.5, shadow='true')
    ax1.set_xlabel(variable_name)
    ax1.set_ylabel('Probability density')
    ax1.set_title(f'Kernel density estimation (KDE) of {variable_name}')

    if threshold:
      y_max = ax1.get_ylim()[1]
      ax1.axvline(x=threshold, ymin=0, ymax=1, label='threshold', color='red', alpha=0.2)
      ax1.text(threshold * 1.05, y_max * 0.65, threshold_label, rotation=90, color='red', alpha=0.2)

  except np.linalg.LinAlgError as e:
    log.warning('Can\'t complete KDE plot since all the values are the same. e: {e}')

  plt.tight_layout()
  plt.show()




PLOT_STYLE = 'seaborn'
plt.style.use(PLOT_STYLE)

# https://codeyarns.com/2014/10/27/how-to-change-size-of-matplotlib-plot/
def set_viz_defaults():
  ## Seaborn
  set_seaborn_defaults()

  ## Matplotlib
  plt.rc('axes.spines', top=False, right=False)

  # # Get current size
  # fig_size = plt.rcParams['figure.figsize']

  # Increase the graph size
  plt.rcParams['figure.figsize'] = (10, 6.67)

  plt.rcParams['figure.titlesize'] = '16'
  plt.rcParams['figure.titleweight'] = 'bold'


def set_seaborn_defaults():
  sns.set(color_codes=True)
  sns.set_style('darkgrid', {
    # 'xtick.major.size': '2',
    # 'xtick.minor.size': '1'
  })

  # sns.set_style('ticks')


def show_outcome_bar_chart(Y, friendly_outcomes):
  Y = np.array(Y)


  positions = np.arange(len(friendly_outcomes))
  index = np.arange(len(friendly_outcomes))
  num_outcomes = len(friendly_outcomes)

  unique, counts = np.unique(Y, return_counts=True)

  # Reverse the order
  # unique = np.flip(unique, 0)
  # counts = np.flip(counts, 0)
  # friendly_outcomes.sort(reverse=True)

  margin = 0.05
  margins = margin * 2
  max_x = 1
  full_width = (max_x - (margins)) / num_outcomes
  bar_width = full_width - margins
  half_width = full_width * 0.5
  x_positions = (positions * full_width) + half_width + margin


  fig, ax = plt.subplots(figsize=(5, 4))
  fig.suptitle(f'Outcomes - full set')

  plt.xticks(x_positions, friendly_outcomes)

  bc = ax.bar(x_positions, counts, width=bar_width, align='center', color=['r', 'b'])

  ax.set_xlabel('Outcomes')
  ax.set_ylabel('Count')
  ax.set_xlim(0, max_x)
  ax.grid(color='white', linewidth=1)
  ax.legend(bc, friendly_outcomes, loc='best', frameon=True, fancybox=True, framealpha=0.5, shadow='true')
  y_max = ax.get_ylim()[0]

  total = Y.shape[0]
  y_max = max(counts)
  for x, count in zip(x_positions, counts):
    count_pct = count / total * 100
    ax.text(x - 0.07, y_max * 0.01, f'{count_pct:0.1f}%', color='0.8', fontsize=14, fontweight='bold')



  plt.tight_layout()
  plt.show()




# Note that SHAP values are sensitive to correlation. If one or more variables are correlated, their relative importance will drop compared to other variables,
# even though a single one of them might be the most predictive compared to all other variables.
# See https://medium.com/civis-analytics/demystifying-black-box-models-with-shap-value-analysis-3e20b536fc80
def show_shap_analysis(clf, X, Y_true, Y_probs, downsample=500, first_n=1):
  shap.initjs()

  downsample = min(downsample, X.shape[0])

  X_dummy, X_sample, Y_dummy, Y_true_sample, Y_probs_dummy, Y_probs_sample = train_test_split(
    X, Y_true, Y_probs,  # Initial sets
    test_size=downsample - 1, random_state=42)


  # Create XGBoost friendly matrix first
  df = X_sample
  # df = pd.DataFrame(X_sample)
  # df.columns = pmodel.feature_names

  df_outcomes = df.copy()
  df_outcomes['outcome'] = Y_true_sample
  df_outcomes['pyes'] = Y_probs_sample[:,1]

  booster = clf.get_booster()
  shap_values = shap.TreeExplainer(booster).shap_values(df)

  # print(f'Force plot of all instances')
  # display(shap.force_plot(shap_values, df))

  print_notebook('Outcomes', heading=True)

  show_histogram_of_outcomes_from_numerical(Y_probs[:,1], Y_true, 'P-values', 'Outcomes', 'P-values', threshold=0.5)


  print_notebook('Shapley Analysis', heading=True)

  print(f'Showing shap analyis for first {first_n} instances.')


  print_notebook('Shap values: https://github.com/slundberg/shap')


  # Using link=shap.LogitLink() will make force_plot output actual probabilities. A shap value of 0 corresponds to P
  # Raw shap values are log-odds, so high negative values correspond to a very low P value.
  # https://github.com/slundberg/shap/issues/29
  for i in range(0, first_n):
    # print(f'Shap log-odds analysis for instance {i}, true={df_outcomes.iloc[i]["outcome"]}, P(yes) = {df_outcomes.iloc[i]["pyes"]}')
    # display( shap.force_plot(shap_values[i,:], df.iloc[i,:]) )

    print(f'Shap P-value analysis for instance {i}, true={df_outcomes.iloc[i]["outcome"]}, P(yes) = {df_outcomes.iloc[i]["pyes"]}')
    display( shap.force_plot(shap_values[i,:], df.iloc[i,:], link=shap.LogitLink(), plot_cmap='GnPR'))

  ## Summary plots

  # Summary for all outcomes
  shap.summary_plot(shap_values, df)


  # Summary for binary outcomes
  # df_0 = df.loc[df_outcomes['outcome'] == 0]
  # df_1 = df.loc[df_outcomes['outcome'] == 1]

  # shap_values_0 = shap_values[df_outcomes['outcome'] == 0]
  # shap_values_1 = shap_values[df_outcomes['outcome'] == 1]


  # print('Showing shap values for Y = 0')
  # shap.summary_plot(shap_values_0, df_0)

  # print('Showing shap values for Y = 1')
  # shap.summary_plot(shap_values_1, df_1)



  ## Dependence plots

  # for i, feature_name in enumerate(pmodel.feature_names):
  #   display(shap.dependence_plot('CleaningType:Deep Clean', shap_values, df, interaction_index=i))

  # Creates a lot of plots.
  # for feature_name in feature_names:
  #   display(shap.dependence_plot(feature_name, shap_values, df))




def show_output_dataviz(clf, X_test, Y_test, Y_probs):
  if in_ipynb():
    show_shap_analysis(clf, X_test, Y_test, Y_probs)
