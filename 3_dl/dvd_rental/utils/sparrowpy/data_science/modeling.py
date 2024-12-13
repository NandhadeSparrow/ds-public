import pickle
import os
import joblib
import json
import dill
import shutil

models_folder_name = 'models'


def get_sub_names(directory):
    return [f.name for f in os.scandir(directory)]


def get_model_names():
    return get_sub_names(models_folder_name)


def get_model_file_names(model_name):
    return get_sub_names(models_folder_name+'/'+model_name)


def get_model_format(model):
    # Check if it's a Keras model
    try:
        if hasattr(model, 'get_config'):
            return "keras"
    except Exception:
        pass
    
    # Check if it's a Scikit-learn model
    try:
        if hasattr(model, 'get_params'):
            return "sklearn"
    except Exception:
        pass
    
    # Check if it's a PyTorch model
    # try:
    #     if isinstance(model, torch.nn.Module):
    #         return "PyTorch Model"
    # except Exception:
    #     pass
    
    return None


def save_model(
        model_name,
        reset=True,
        target='target', 
        features=None, 
        encoders=None, 

        prep=None,        
        scaler=None, 
        model=None, 
        model_format=None, 
        result=None, 
    ):

    if reset:
        model_path = models_folder_name+'/'+model_name
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
    os.makedirs(f"{models_folder_name}/{model_name}", exist_ok=True)
    model_folder_path = os.path.join(models_folder_name, model_name).replace('\\', '/')
    
    details_path = f'{model_folder_path}/details.json'
    details = {
        'model_name': model_name,
    }
    if target: details['target'] = target
    if features: details['features'] = features
    if model_format: details['model_format'] = model_format

    if model:
        if not features:
            raise TypeError("Model found, features missing")
        if not model_format:
            model_format = get_model_format(model)
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

    with open(details_path, 'w') as file:
        json.dump(details, file, indent=4)


def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)
    

def load_saved_model(model_name):
    details_path = f'models/{model_name}/details.json'
    model_files_names = get_model_file_names(model_name)
    with open(details_path, 'r') as file:
        details = json.load(file)
    model_dict = {
        'model_name': details.get('model_name'),
        'features': details.get('features'),
        'target': details.get('target'),
    }

    for (sname, fname) in [(f.split('.')[0], f) for f in model_files_names]:
        print(fname)

        if fname.startswith('model.'):
            model_path = f'{models_folder_name}/{model_name}/{fname}'
            model_format = details.get('model_format')
            print('hi')
            if not model_format:
                model_format = fname.split('.')[-1]
            if model_format in ['keras', 'h5']:
                from tensorflow.keras.models import load_model
                model = load_model(model_path)
            elif model_format == 'sklearn':
                model = joblib.load(model_path)
            model_dict['model'] = model

        if fname == 'scaler.pkl':
            scaler_path = f'{models_folder_name}/{model_name}/{fname}'
            scaler = load_pickle(scaler_path)
            model_dict['scaler'] = scaler

        if fname == 'result.pkl':
            result_path = f'{models_folder_name}/{model_name}/{fname}'
            with open(result_path, 'rb') as file:
                result =  dill.load(file)
            model_dict['result'] = result

        if fname == 'encoders.pkl':
            encoders_path = f'{models_folder_name}/{model_name}/{fname}'
            with open(encoders_path, 'rb') as file:
                encoders =  dill.load(file)
            model_dict['encoders'] = encoders

        if fname == 'prep.pkl':
            prep_path = f'{models_folder_name}/{model_name}/{fname}'
            prep = load_pickle(prep_path)
            model_dict['prep'] = prep




        
    

    

    return model_dict

