import uvicorn
from fastapi import FastAPI
import train_model, predict_model

app = FastAPI()

@app.get('/')
async def index():
    return {'text': 'API running'}

@app.get('/items/{name}')
async def get_items(name):
    return {'user_name': name}

@app.get('/train')
async def trainer():
    res = train_model.train()
    return {'model status': res}

@app.get('/predict/{features}')
async def predict(features):
    features_list = [[el for el in features.split(',')]]
    res = predict_model.predict(features_list)
    return {'New customer CHURN prediction': str(res)}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port = 8000)