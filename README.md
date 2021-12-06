# Bornly

Seaborn-like API, but with plotly under the hood.

Very much work in progress!

## Installation

Note: the `$` is not part of the command:

```console
$ pip install -U bornly
```

## Demo

```python
import bornly as bs
import seaborn as sns

fmri = sns.load_dataset("fmri")
sns.lineplot(data=fmri, x="timepoint", y="signal", hue="event")
bs.lineplot(data=fmri, x='timepoint', y='signal', hue='event')
```

![](demo.png)
