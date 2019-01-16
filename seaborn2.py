# Seaborn example 2
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="ticks", color_codes=True)
tips = sns.load_dataset("tips")

pp = sns.catplot(x="total_bill", y="day", hue="time",
            kind="violin", data=tips)

from IPython.display import display
display(pp)
