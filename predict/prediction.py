import pickle
import sklearn

# load the model from disk
loaded_model = pickle.load(open('./model/finalized_model.sav', 'rb'))


def prediction(features):

    prediction = loaded_model.predict(features)

    return prediction
