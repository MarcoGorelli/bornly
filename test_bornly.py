import seaborn as sns
import bornly as bns
import plotly.express as px
penguins = sns.load_dataset("penguins")
bns.histplot(data=penguins, x="flipper_length_mm", kde=True)