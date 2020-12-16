import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

def printHistograms(data, pColor):
	sns.set_style("white")
	sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
	data.plot.hist(subplots=True, layout=(2, 4), figsize=(15, 8), sharey=True,colormap=pColor)
	sns.despine()

def printMatrixDiagram(data):
	g = sns.PairGrid(data, height=1.5)
	g.map_offdiag(sns.scatterplot)
	g.map_offdiag(sns.regplot)
	g.map_diag(plt.hist)

def printPearsonCorrelations(data):
	corr = data.corr()
	plt.figure(figsize = (10,10))
	cmap = sns.diverging_palette(220, 10, as_cmap=True)
	sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, cmap=cmap, vmax=1, vmin=-1,center=0, square=True, 		linewidths=.5, cbar_kws={"shrink": .82})
	plt.title('Heatmap of Correlation Matrix - Medians by Participant')
	plt.show()
	print(corr)

def calculateRegression(data,label,resultsummary):
	X_train, X_test, y_train, y_test = train_test_split(data, label, test_size = 0.2, random_state = 50)
	reg = LinearRegression() # Create the Linear Regression estimator
	result=reg.fit (X_train, y_train)  # Perform the fitting

	# P-values para decidir qué variables x son importantes para explicar y.
	mod = sm.OLS(y_train,X_train)
	fitt = mod.fit()
	p_values = fitt.summary2().tables[1]['P>|t|']
	p_value_max = p_values.idxmax()
	RMSE_Training = np.sqrt(np.mean((reg.predict(X_train) - y_train)**2))
	RMSE_Testing = np.sqrt(np.mean((reg.predict(X_test) - y_test)**2))
	R2_Training = reg.score(X_train, y_train)
	R2_Testing = reg.score(X_test, y_test)
	iteration = resultsummary['iteration'].max()
	if(np.isnan(iteration)):
		iteration=0
	else:
		iteration = iteration+1
	print("iteration")
	print(iteration)
	print(p_values)
	if(p_values[p_value_max] > 0.1):
		data.drop(p_value_max, axis=1, inplace=True)
		iteration = resultsummary['iteration'].max()
		if(np.isnan(iteration)):
			iteration=0
		else:
			iteration = iteration+1
		newrow ={'iteration': iteration, 'intercept':reg.intercept_ , 'RMSE_Training': RMSE_Training,'RMSE_Testing':RMSE_Testing,
		'R2_Training':R2_Training,'R2_Testing':R2_Testing,'p_value_max':p_values[p_value_max],'removed_var':p_value_max}
		resultsummary = resultsummary.append(newrow,ignore_index=True)

		calculateRegression(data,label,resultsummary)
	else:
		iteration = resultsummary['iteration'].max()+1
		newrow ={'iteration': iteration, 'intercept':reg.intercept_ , 'RMSE_Training': RMSE_Training,'RMSE_Testing':RMSE_Testing,
		'R2_Training':R2_Training,'R2_Testing':R2_Testing,'p_value_max':p_values[p_value_max],'removed_var':'-'}
		resultsummary = resultsummary.append(newrow,ignore_index=True)
		resultsummary = resultsummary.round(3)
		print(resultsummary.head(20))
		print()
		print(reg.coef_,reg.intercept_)
