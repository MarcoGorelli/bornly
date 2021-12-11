import bornly as bns
import seaborn as sns

fmri = sns.load_dataset('fmri')

fig, ax = bns.subplots()

# bns.lineplot(data=fmri, x='timepoint', y='signal', ax=ax)
# bns.lineplot(data=fmri, x='timepoint', y='signal', hue='event', ax=ax)
# bns.lineplot(data=fmri, x='timepoint', y='signal')
# bns.lineplot(data=fmri, x='timepoint', y='signal', hue='event')
# bns.scatterplot(data=fmri, x='timepoint', y='signal', ax=ax)
# bns.scatterplot(data=fmri, x='timepoint', y='signal', hue='event', ax=ax)
# bns.scatterplot(data=fmri, x='timepoint', y='signal')
# bns.scatterplot(data=fmri, x='timepoint', y='signal', hue='event')

# bns.kdeplot(data=fmri, x='timepoint')

# iris = sns.load_dataset("iris")
# bns.kdeplot(data=iris)
from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="white")

# Generate a large random dataset
rs = np.random.RandomState(33)
d = pd.DataFrame(data=rs.normal(size=(100, 26)),
                 columns=list(ascii_letters[26:]))

# Compute the correlation matrix
corr = d.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
# bns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})

bns.heatmap(corr, mask=mask, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
