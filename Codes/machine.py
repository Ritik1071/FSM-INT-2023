import streamlit as st
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[7]:


# Load the dataset and perform preprocessing
df = pd.read_csv('train_FD001.txt', sep=" ", header=None)
columns_to_drop = [27, 26]
df = df.drop(df.columns[columns_to_drop], axis=1)


# In[8]:


df.columns = df.columns.astype(str)
column_names = df.columns


# In[9]:


new_column_names = {
    '0': 'Engine_Unit_Number',
    '1': 'Time_In_Cycles',
    '2': 'Operating_settings_1',
    '3': 'Operating_settings_2',
    '4': 'Operating_settings_3',
    '5': 'Sensor_readings_1',
    '6': 'Sensor_readings_2',
    '7': 'Sensor_readings_3',
    '8': 'Sensor_readings_4',
    '9': 'Sensor_readings_5',
    '10': 'Sensor_readings_6',
    '11': 'Sensor_readings_7',
    '12': 'Sensor_readings_8',
    '13': 'Sensor_readings_9',
    '14': 'Sensor_readings_10',
    '15': 'Sensor_readings_11',
    '16': 'Sensor_readings_12',
    '17': 'Sensor_readings_13',
    '18': 'Sensor_readings_14',
    '19': 'Sensor_readings_15',
    '20': 'Sensor_readings_16',
    '21': 'Sensor_readings_17',
    '22': 'Sensor_readings_18',
    '23': 'Sensor_readings_19',
    '24': 'Sensor_readings_20',
    '25': 'Sensor_readings_21'
}




# In[10]:


df = df.rename(columns=new_column_names)


# In[11]:


# Drop columns with low fluctuations and add Remaining Useful Life (RUL) column
drop_columns = ['Sensor_readings_18', 'Sensor_readings_19', 'Sensor_readings_10', 'Sensor_readings_1']
df = df.drop(drop_columns, axis=1)


# In[12]:


def add_remaining_useful_life(df):
    # Getting the total number of cycles for each unit
    grouped_by_unit = df.groupby(by="Engine_Unit_Number")
    max_cycle = grouped_by_unit["Time_In_Cycles"].max()

    # Merging the max cycle back into the original frame
    result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='Engine_Unit_Number', right_index=True)

    # Calculating remaining useful life for each row
    remaining_useful_life = result_frame["max_cycle"] - result_frame["Time_In_Cycles"]
    result_frame["RUL"] = remaining_useful_life

    # dropping max_cycle as it's no longer needed
    result_frame = result_frame.drop("max_cycle", axis=1)
    return result_frame

df = add_remaining_useful_life(df)


# In[13]:


# Normalize selected columns
columns_to_normalize = ['Operating_settings_1', 'Operating_settings_2', 'Operating_settings_3',
                        'Sensor_readings_2', 'Sensor_readings_3', 'Sensor_readings_4',
                        'Sensor_readings_7', 'Sensor_readings_8', 'Sensor_readings_9',
                        'Sensor_readings_11', 'Sensor_readings_12', 'Sensor_readings_13',
                        'Sensor_readings_14', 'Sensor_readings_15', 'Sensor_readings_17',
                        'Sensor_readings_20', 'Sensor_readings_21']

columns_data = df[columns_to_normalize]
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(columns_data)
df[columns_to_normalize] = normalized_data


# In[14]:


# Train the Linear Regression model
x_train_lr = df[['Operating_settings_1', 'Operating_settings_2', 'Operating_settings_3',
                'Sensor_readings_2', 'Sensor_readings_3', 'Sensor_readings_4',
                'Sensor_readings_7', 'Sensor_readings_8', 'Sensor_readings_9',
                'Sensor_readings_11', 'Sensor_readings_12', 'Sensor_readings_13',
                'Sensor_readings_14', 'Sensor_readings_15', 'Sensor_readings_17',
                'Sensor_readings_20', 'Sensor_readings_21']]
y_train_lr = df['RUL']



# In[15]:


reg = LinearRegression()
reg.fit(x_train_lr, y_train_lr)


# In[16]:


# Get the list of all input columns (operating settings and sensor readings)
input_columns = ['Operating_settings_1', 'Operating_settings_2', 'Operating_settings_3',
                 'Sensor_readings_2', 'Sensor_readings_3', 'Sensor_readings_4',
                 'Sensor_readings_7', 'Sensor_readings_8', 'Sensor_readings_9',
                 'Sensor_readings_11', 'Sensor_readings_12', 'Sensor_readings_13',
                 'Sensor_readings_14', 'Sensor_readings_15', 'Sensor_readings_17',
                 'Sensor_readings_20', 'Sensor_readings_21']

# Create the Streamlit app
st.title('Remaining Useful Life Prediction')
st.markdown(
    """
    <style>
    body {
        background-color: black; /* Set the background color to black */
        color: white; /* Set text color to contrast with the background */
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.write('Enter the operating settings and sensor readings to predict RUL:')

# Create input components for users (using text boxes)
user_input = {}
for column in input_columns:
    user_input[column] = st.text_input(column, value='0.5')  # You can set any default value you prefer

# Convert user input to DataFrame and normalize
user_input_df = pd.DataFrame([user_input], columns=input_columns)
user_input_normalized = scaler.transform(user_input_df)

# Use the trained model to predict RUL
predicted_rul = reg.predict(user_input_normalized)
predicted_rul[predicted_rul <= 0] = 0

# Display the results to the user
st.write(f'Predicted Remaining Useful Life (RUL): {predicted_rul[0]}')






