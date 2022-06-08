import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

# pylint: disable=E1101

def train():

    """
    this function will be called in the my_first_api.py
    in order to train a model. It loads the datas set, train the model and save
    the model and the standardizer so that we can use it to make a future
    prediction
    """

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
                                                    random_state = 10, stratify = Y)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test) # we use the scale set calculated from the
    # training set just above and we apply it to transform the test set

    model = RandomForestClassifier(bootstrap= True, max_features= 0.3, n_estimators= 200)

    # Training
    model.fit(X_train, Y_train)

    # save the model and sc to the disk
    filename = './finalized_model.h5'
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

    # save the standardizer  in order to apply on the vector to predict
    filename2 = './std_scaler.bin'
    with open(filename2,'wb') as f:
        pickle.dump(sc,f)

    return (f'Model was trained - path: {filename}')