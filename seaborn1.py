# Seaborn example 1
import seaborn as sns
sns.set(style="ticks")

df = sns.load_dataset("iris")
pp = sns.pairplot(df, hue="species")

from IPython.display import display
display(pp)
