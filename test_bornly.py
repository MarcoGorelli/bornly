import seaborn as sns
import bornly as bns
import plotly.express as px
tips = sns.load_dataset("tips")
fmri = sns.load_dataset("fmri")
# bns.relplot(data=tips, x="total_bill", y="tip", hue="day", col='time')
bns.relplot(data=fmri, x="timepoint", y="signal", hue="event", row='region', kind='line')