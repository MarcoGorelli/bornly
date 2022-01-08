import seaborn as sns
import matplotlib.pyplot as plt
import bornly as bns
import plotly.express as px
tips = sns.load_dataset("tips")
fmri = sns.load_dataset("fmri")


import plotly.express as px
import pandas as pd
import bornly as bns

penguins = bns.load_dataset("penguins")

bns.histplot(data=penguins)