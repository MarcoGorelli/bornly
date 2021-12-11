# Bornly

Seaborn-like API for plotly.

## Installation

Note: the `$` is not part of the command:

```console
$ pip install -U bornly
```

## Demo

```diff
-import seaborn as sns
+import bornly as bns

fmri = bns.load_dataset("fmri")
-sns.lineplot(data=fmri, x='timepoint', y='signal', hue='event')
+bns.lineplot(data=fmri, x='timepoint', y='signal', hue='event')
```

