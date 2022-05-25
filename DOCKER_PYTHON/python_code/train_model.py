import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import pickle

# pylint: disable=E1101

def train():
    # importing the data
    input_path ='C:/Users/JF5191/Documents/DSTI/ML Python Labs/ML_PythonLabs/DOCKER_PYTHON/python_code/Churn_Modelling.csv'
    df = pd.read_csv(input_path)

    df['Gender'] = df['Gender'].map({'Female': 0, 'Male':1}) # mapping gender
    df = pd.concat([df,pd.get_dummies(df['Geography'], prefix = 'country',
               drop_first= True)], axis = 1) # One-hot encoding Geography feature

    df_X = df.drop(['Geography','Exited','RowNumber','CustomerId','Surname'],
             axis = 1)
    X = df_X.values # creating X dataset
    Y = df['Exited'].values # creating Y label

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,
                                                    random_state = 10)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test) # we use the scale set calculated from the
    # training set just above and we apply it to transform the test set

    model = tf.keras.models.Sequential()

    #add input layer  and first hidden layer
    model.add(tf.keras.layers.Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu')) #initializer=uniform means to all the weights will initialized with the same value

    #xxx.Dense means that every node is connected with the nodes next to himself

    #add 2nd hidden layer
    model.add(tf.keras.layers.Dense(units = 6, kernel_initializer='uniform', activation = 'relu'))

    # Add output layer
    model.add(tf.keras.layers.Dense(units = 1, kernel_initializer='uniform', activation='sigmoid')) # sigmoid for binary, Softmax for multiclass

    # compilation
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    # Training
    model.fit(X_train, Y_train, batch_size  = 10, epochs = 3, verbose = 2)

    # save the model and sc to the disk
    filename = './finalized_model.h5'
    model.save(filename)

    filename2 = './std_scaler.bin'
    with open(filename2,'wb') as f:
        pickle.dump(sc,f)

    return (f'Model was trained - path: {filename}')