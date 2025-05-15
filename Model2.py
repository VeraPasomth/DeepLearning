#Assignment 2 Model2

import numpy as np
import pandas as pd
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.models import Sequential
from sklearn.metrics import r2_score
from keras.optimizers import adamax_v2
from keras.metrics import mean_squared_error
from keras.optimizers import gradient_descent_v2
from sklearn.model_selection import train_test_split




#Import Dataset
dataset = pd.read_csv("Preprocessed_Data.csv")

#Creating DataFrame
dataFrame = pd.DataFrame(dataset, columns = ['center_of_buoyancy', 'prismatic_coefficient',
                             'length_displacement', 'beam_draught_ratio',
                             'length_beam_ratio','froude_number','resistance'])

print(dataFrame.head(7))
#Get Independent Variable Columns
independent_var = dataFrame.iloc[0:,0:6].values
dependent_var = dataFrame.iloc[0:,6:].values

#Split the Dataset
indep_train, indep_test, dep_train, dep_test = train_test_split(independent_var,dependent_var,
                                                              test_size=0.2, random_state=20,shuffle=True)
indep_train, indep_val, dep_train, dep_val = train_test_split(indep_train,dep_train,
                                                              test_size=0.25,random_state=8) # 0.25 x 0.8 = 0.2

#Build Model
model = Sequential()
model.add(Dense(16, activation="relu", input_dim=6))
model.add(Dense(16, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1,))


model.compile(optimizer=adamax_v2.Adamax(lr=0.01, decay=0.01/200 ),
              loss='mae',metrics=['mae'])
#Train Model
history = model.fit(indep_train, dep_train,
                    validation_data=(indep_val, dep_val),
                    epochs=250, batch_size=128, verbose=2)
model.summary()

PredTrainSet = model.predict(indep_train)
PredValSet = model.predict(indep_val)
PredTestSet = model.predict(indep_test)

#RMSE
print('RMSE Score on Training Set = ',np.sqrt(mean_squared_error(dep_train,PredTrainSet)))
print('RMSE Score on Test Set = ', np.sqrt(mean_squared_error(dep_test,PredTestSet)))

#R-Square
print('r_squared score on training set =  ', r2_score(dep_train,PredTrainSet))
print('r_squared score on test set =  ', r2_score(dep_test,PredTestSet))

plt.subplot(2,1,2)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.plot(history.history['loss'],'b',label='loss')
plt.plot(history.history['val_loss'],'r',label='validation_loss')
plt.title('Model 2: Training and Validation loss')
plt.legend()
plt.show()

