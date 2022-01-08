import seaborn as sns
import matplotlib.pyplot as plt
import bornly as bns
import plotly.express as px
tips = sns.load_dataset("tips")
fmri = sns.load_dataset("fmri")


import plotly.express as px
import pandas as pd
import bornly as bns

import numpy as np
x = np.linspace(-5, 5, 100)
y = np.sin(x)
import bornly as bns
bns.scatterplot(x=x, y=y)