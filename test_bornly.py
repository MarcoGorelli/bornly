import bornly as bns
import seaborn as sns

fig, ax = bns.subplots()

tips = sns.load_dataset("tips")

bns.barplot(x="day", y="total_bill", data=tips, hue='time')