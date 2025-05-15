# Software Description
The zipped folder contains the dataset and python scripts used for data pre-processing, visualization 
and building 3 non-linear regression models with Keras. The steps to use the models are as 
follows: 
1. Please run the python script labeled Data_Preprocessing. 
This script prepares the data to be used by the neural networks by first labeling and dividing 
the data into separate columns and exporting it into a csv file. 
2. Optionally, the script labeled Data_Visualization can be run. 
This will display the data in different charts as well as the descriptive statistical 
measurements of each variable. 
3. The scripts for each of the 3 models can be run. 
By running these scripts, the training for the model will start and the training loss and 
validation loss will be displayed for each epoch. Once it is finished training, the summary of 
the model will be displayed and the evaluation metrics root mean square error and R-square 
