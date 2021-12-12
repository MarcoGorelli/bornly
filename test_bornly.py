import bornly as bns
import seaborn as sns

fig, ax = bns.subplots()

import seaborn as sns; sns.set_theme(color_codes=True)
fmri = sns.load_dataset("fmri")
bns.barplot(x=fmri['subject'], y=fmri['timepoint'], hue=fmri['event'])