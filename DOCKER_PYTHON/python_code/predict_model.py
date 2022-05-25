import pickle
import tensorflow as tf

def predict(input_data):
    
    filename1 = './std_scaler.bin'
    with open(filename1, 'rb') as f:
        sc = pickle.load(f)

    input_data = sc.transform(input_data)

    my_trained_model = tf.keras.models.load_model('finalized_model.h5')
    new_prediction = my_trained_model.predict(input_data)
    new_prediction = (new_prediction > 0.5)

    return new_prediction[0,0]