import seaborn as sns
import bornly as bns
import plotly.express as px
fmri = sns.load_dataset("fmri")
fig = bns.lineplot(data=fmri, x='timepoint', y='signal', hue='event')