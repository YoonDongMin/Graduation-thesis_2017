import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('DT2.csv', header=None)
df.columns = ['DAY', 'MONTH', 'WEEK', 'TEMPO', 'RAIN', 'Real Sale', 'Real Stock', 'Order']

sns.set(style='whitegrid', context='notebook')
cols = ['DAY', 'MONTH', 'WEEK', 'TEMPO', 'RAIN', 'Real Sale', 'Real Stock', 'Order']
sns.pairplot(df[cols], size=2.5)
plt.show()
sns.reset_orig()

cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size':15}, yticklabels=cols, xticklabels=cols)
plt.show()