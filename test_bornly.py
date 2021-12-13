import seaborn as sns
import bornly as bns
import plotly.express as px
tips = sns.load_dataset("tips")
fmri = sns.load_dataset("fmri")
# bns.relplot(data=tips, x="total_bill", y="tip", hue="day", col='time')
# bns.relplot(data=fmri, x="timepoint", y="signal", hue="event", row='region', kind='line')
# bns.relplot(
#     data=tips, x="total_bill", y="tip", col="time",
#     hue="time", size="size", style="sex",
#     palette=["b", "r"], sizes=(10, 100)
# )
# flights_wide = sns.load_dataset("flights").pivot("year", "month", "passengers")
# bns.relplot(data=flights_wide, kind="line")
bns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips)