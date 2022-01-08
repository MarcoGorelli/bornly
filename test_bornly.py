import seaborn as sns
import matplotlib.pyplot as plt
import bornly as bns
import plotly.express as px
tips = sns.load_dataset("tips")
fmri = sns.load_dataset("fmri")


import plotly.express as px
import pandas as pd

fig = bns.lineplot(x=fmri['timepoint'], y=fmri['signal'], hue=fmri['event'], style=fmri['region'])
fig