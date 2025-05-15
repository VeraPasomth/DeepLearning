#Assignment 2 Model3
#Weerawan Pasomthong 644357

import numpy as np
import pandas as pd
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.optimizers import adam_v2
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split





#Read Dataset
dataset = pd.read_csv("Preprocessed_Data.csv")

#Create Pandas Dataframe
dataFrame = pd.DataFrame(dataset, columns = ['center_of_buoyancy', 'prismatic_coefficient',
                                            'length_displacement', 'beam_draught_ratio',
                                            'length_beam_ratio','froude_number','resistance'])


#Specify Dependent Variable Columns
independent_var = dataFrame.iloc[0:,0:6].values
dependent_var = dataFrame.iloc[0:,6:].values
indep_train, indep_test, dep_train, dep_test = train_test_split(independent_var,dependent_var,
                                                                test_size=0.2, random_state=8,
                                                                shuffle=True)

#Split Validation Set
indep_train, indep_val, dep_train, dep_val = train_test_split(indep_train,dep_train,
                                                              test_size=0.25,
                                                              random_state= 8) # 0.25 x 0.8 = 0.2


#Build Model
model = Sequential()
model.add(Dense(128, activation="relu", input_dim=6))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1,))

model.compile(loss='mean_squared_error', optimizer=adam_v2.Adam(lr=1e-3, decay=1e-3 / 200))

#Train Model
history = model.fit(indep_train, dep_train, validation_data=(indep_val, dep_val), epochs=500, batch_size=100, verbose=2)

#Test Prediction and Make Summary
PredTrainSet = model.predict(indep_train)
PredValSet = model.predict(indep_val)
Predtest = model.predict(indep_test)
model.summary()

#Save Results
np.savetxt("TrainResults.csv", PredTrainSet, delimiter=",")
np.savetxt("ValResults.csv", PredValSet, delimiter=",")


#Model Evaluation:

#RMSE
print('RMSE Score on Training Set = ',np.sqrt(mean_squared_error(dep_train,PredTrainSet)))
print('RMSE Score on Test Set = ', np.sqrt(mean_squared_error(dep_test,Predtest)))

#R Square
print('r_squared score on training set =  ', r2_score(dep_train, PredTrainSet))
print('r_squared score on test set =  ', r2_score(dep_val,PredValSet))


#Display Results
plt.subplot(2,1,2)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title('Model 3: Training and Validation Loss')
plt.plot(history.history['loss'],'b',label='loss')
plt.plot(history.history['val_loss'],'r',label='validation_loss')
plt.legend()
plt.show()



