import seaborn as sns
import bornly as bns
import plotly.express as px
fmri = sns.load_dataset("fmri")
fig = bns.lineplot(data=fmri.drop_duplicates('timepoint'), x='timepoint', y='signal')