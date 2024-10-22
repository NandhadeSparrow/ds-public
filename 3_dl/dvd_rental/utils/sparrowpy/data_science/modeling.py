import pickle
import os
import joblib
import json
import dill
from tensorflow.keras.models import load_model


models_folder_name = 'models'
def save_model(
        model_name=None,
        target='target', 
        features=None, 
        result=None, 
        encoders=None, 
        model=None, 
        model_format=None,
        scaler=None, 
        prep=None
    ):

    if model_name is None:
        model_name = target
    os.makedirs(f"{models_folder_name}/{model_name}", exist_ok=True)
    model_folder_path = os.path.join(models_folder_name, model_name).replace('\\', '/')
    
    if model:
        if model_format in ['keras', 'h5']:
            model_path = f'{model_folder_path}/model.{model_format}'
            model.save(model_path)
        elif model_format == 'sklearn':
            joblib.dump(model, model_path+'.joblib')
    else: 
        model_path = ''

    if scaler: 
        scaler_path = f'{model_folder_path}/scaler.pkl'
        with open(scaler_path, 'wb') as file:
            pickle.dump(scaler, file)
    else: 
        scaler_path = ''

    if result: 
        result_path = f'{model_folder_path}/result.pkl'
        with open(result_path, 'wb') as file:
            dill.dump(result, file)
    else:  
        result_path = ''

    if encoders: 
        encoders_path = f'{model_folder_path}/encoders.pkl'
        with open(encoders_path, 'wb') as file:
            dill.dump(encoders, file)
    else: 
        encoders_path = ''


    if prep: 
        prep_path = f'{model_folder_path}/prep.pkl'
        with open(prep_path, 'wb') as file:
            pickle.dump(prep, file)
    else: 
        prep_path = ''

    model_details_path = f'{model_folder_path}/model_details.json'

    model_details = {
        'model_name': model_name,
        'target': target,
        'features': features,
        'model_path': model_path,
        'model_format': model_format,
        'scaler_path': scaler_path,
        'result_path': result_path,
        'encoders_path': encoders_path,
    }

    with open(model_details_path, 'w') as file:
        json.dump(model_details, file, indent=4)


def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)
    

def load_saved_model(model_name):
    model_details_path = f'models/{model_name}/model_details.json'
    model_dict = {
        'prep': None,
        'scaler': None,
        'model': None,
        'features': None,
        'target': None,
        'result': None,
        'encoders': None,
        'model_name': None
    }

    with open(model_details_path, 'r') as file:
        model_details = json.load(file)

    model_path = model_details.get('model_path')
    scaler_path = model_details.get('scaler_path')
    result_path = model_details.get('result_path')
    encoders_path = model_details.get('encoders_path')
    prep_path = model_details.get('prep_path')
    model_format = model_details.get('model_format')

    if model_path:
        if model_format in ['keras', 'h5']:
            model = load_model(model_path)
        elif model_format == 'sklearn':
            model = joblib.load(model_path)
        model_dict['model'] = model

    if scaler_path:
        scaler = load_pickle(scaler_path)
        model_dict['scaler'] = scaler

    if result_path:
        with open(result_path, 'rb') as file:
            result =  dill.load(file)
        model_dict['result'] = result

    if encoders_path:
        with open(encoders_path, 'rb') as file:
            encoders =  dill.load(file)
        model_dict['encoders'] = encoders
        
    if prep_path:
        prep = load_pickle(prep_path)
        model_dict['prep'] = prep
    
    model_dict['features'] = model_details.get('features')
    model_dict['model_name'] = model_details.get('model_name')
    model_dict['target'] = model_details.get('target')
    
    return model_dict

