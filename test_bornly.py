import bornly as bns
import seaborn as sns

fig, ax = bns.subplots()

import seaborn as sns; sns.set_theme(color_codes=True)

tips = sns.load_dataset("tips")

ax = bns.regplot(x="total_bill", y="tip", data=tips, truncate=False)
