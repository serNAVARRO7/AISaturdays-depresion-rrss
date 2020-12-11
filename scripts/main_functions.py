import seaborn as sns

def printHistograms(data, pColor):
	sns.set_style("white")
	sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
	data.plot.hist(subplots=True, layout=(2, 4), figsize=(15, 8), sharey=True,colormap=pColor)
	sns.despine()