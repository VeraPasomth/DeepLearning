#Data Visualization

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt




dataset = pd.read_csv("Preprocessed_Data.csv", index_col=False)
dataFrame = pd.DataFrame(dataset, columns = ['center_of_buoyancy', 'prismatic_coefficient',
                             'length_displacement', 'beam_draught_ratio',
                             'length_beam_ratio','froude_number','resistance'])
print(dataFrame.head(7))



#Boxplots
plt.rcParams["figure.autolayout"] = True
plt.rcParams["figure.figsize"] = [7.50, 6.50]
dataFrame[['length_displacement', 'beam_draught_ratio','length_beam_ratio','froude_number']].plot(kind='box',
                                                                        title='Independent Variable Boxplots')
plt.show()

dataFrame['center_of_buoyancy'].plot(kind='box',
                                     title='Center Buoyancy Boxplot',
                                     meanline=True)
plt.show()

dataFrame['resistance'].plot(kind='box',
                             title='Dependent Variable Boxplot',
                             vert=False,meanline=True)
plt.show()



#ScatterPlot
plt.title('Scatter Plot')
sns.scatterplot(data=dataFrame, x='center_of_buoyancy',y='resistance',label='center_of_buoyancy',color='navy')
sns.scatterplot(data=dataFrame, x='prismatic_coefficient',y='resistance', label='prismatic_coefficient',color='slateblue')
sns.scatterplot(data=dataFrame, x='length_displacement', y='resistance', label='length_displacement', color='darkslateblue')
sns.scatterplot(data=dataFrame, x='beam_draught_ratio', y='resistance', label='beam_draught_ratio',color='lightsteelblue')
sns.scatterplot(data=dataFrame, x='length_beam_ratio', y='resistance', label='length_beam_ratio',color='midnightblue')
plt.show()

sns.scatterplot(x = "froude_number", y = "resistance", data = dataFrame, hue ="froude_number")
plt.show()

#Heatmap Correlation Coefficients
correlations = dataFrame.corr()
dataplot = sns.heatmap(correlations, cmap="PuBu", annot=True)
plt.show()

#Histogram
pd.DataFrame.hist(dataset)
plt.show()

print('Descriptive Statistics')
print(dataFrame.describe())
print(dataFrame[['length_displacement','beam_draught_ratio','length_beam_ratio']].describe())
