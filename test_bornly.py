import bornly as bns
import seaborn as sns

fmri = sns.load_dataset('fmri')

fig, ax = bns.subplots()

bns.lineplot(data=fmri, x='timepoint', y='signal', ax=ax)
bns.lineplot(data=fmri, x='timepoint', y='signal', hue='event', ax=ax)
bns.lineplot(data=fmri, x='timepoint', y='signal')
bns.lineplot(data=fmri, x='timepoint', y='signal', hue='event')
bns.scatterplot(data=fmri, x='timepoint', y='signal', ax=ax)
bns.scatterplot(data=fmri, x='timepoint', y='signal', hue='event', ax=ax)
bns.scatterplot(data=fmri, x='timepoint', y='signal')
bns.scatterplot(data=fmri, x='timepoint', y='signal', hue='event')
