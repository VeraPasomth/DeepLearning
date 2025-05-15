#Creating Columns and Labels

import pandas as pd

dataset = pd.read_csv("yacht_hydrodynamics.data")
df = pd.DataFrame(dataset)

col = ['center_of_buoyancy']
df.columns = col


#divide data into columns
df[['center_of_buoyancy', 'prismatic_coefficient']] = df['center_of_buoyancy'].str.split(' ', 1, expand=True)
df[['prismatic_coefficient', 'length_displacement']] = df['prismatic_coefficient'].str.split(None, 1, expand=True)
df[['length_displacement', 'beam_draught_ratio']] = df['length_displacement'].str.split(' ', 1, expand=True)
df[['beam_draught_ratio', 'length_beam_ratio']] = df['beam_draught_ratio'].str.split(' ', 1, expand=True)
df[['length_beam_ratio', 'froude_number']] = df['length_beam_ratio'].str.split(' ', 1, expand=True)
df[['froude_number', 'resistance']] = df['froude_number'].str.split(' ', 1, expand=True)

PandaFrame = pd.DataFrame(df)
PandaFrame.to_csv('Preprocessed_Data.csv', index=False)
print(PandaFrame.head(7))






