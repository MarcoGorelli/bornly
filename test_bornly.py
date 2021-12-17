import seaborn as sns
import matplotlib.pyplot as plt
import bornly as bns
import plotly.express as px
tips = sns.load_dataset("tips")
fmri = sns.load_dataset("fmri")


import bornly as bns

data = bns.load_dataset('fmri')
bns.regplot(data=data, 
                    x='timepoint',
                    y='signal',
                    scatter=False,
                    color='red')

# fig = bns.lineplot(x=fmri['timepoint'], y=fmri['signal'], style=fmri['region'])
# fig
# bns.lineplot(
#     data=fmri,
#     x="timepoint", y="signal", hue="event", style="event",
#     dashes=True, markers=True,
# )
# bns.lineplot(data=fmri, x="timepoint", y="signal", hue="event")
# bns.relplot(data=tips, x="total_bill", y="tip", hue="day", col='time')
# bns.relplot(data=fmri, x="timepoint", y="signal", hue="event", row='region', kind='line')
# bns.relplot(
#     data=tips, x="total_bill", y="tip", col="time",
#     hue="time", size="size", style="sex",
#     palette=["b", "r"], sizes=(10, 100)
# )
# flights_wide = sns.load_dataset("flights").pivot("year", "month", "passengers")
# bns.relplot(data=flights_wide, kind="line")
# bns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips)

# fig, ax = bns.subplots()
# # fig = bns.scatterplot(x=fmri['timepoint'], y=fmri['signal'], color='orange', ax=ax)
# bns.lineplot(
#     data=fmri.query("region == 'frontal'"),
#     x="timepoint", y="signal", hue=fmri['event'].astype('category').cat.codes, units="subject",
#     estimator=None, lw=1,
# )
# bns.lineplot(data=fmri, x='timepoint', y='signal', color='orange', ax=ax)
# fig = sns.scatterplot(data=fmri, x='timepoint', y='signal', hue='event', ax=ax)
# fig = sns.barplot(data=fmri, x='timepoint', y='signal', hue='event', ax=ax)
# fig, ax = plt.subplots()
# bns._sns.lineplot(data=fmri, x='timepoint', y='signal', ax=ax, color='b')