
import streamlit as st
from ..data_science import modeling
import numpy as np


def read_markdown_file(markdown_file):
    with open(markdown_file, 'r') as file:
        return file.read()


def get_key_by_value(d, target_value):
    for key, value in d.items():
        if value == target_value:
            return key
    return None


def intro():
    markdown_content = read_markdown_file('README.md')
    st.markdown(markdown_content)


def app():
    intro_page_name = 'intro'
    models = modeling.get_model_names()

    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select a page", [intro_page_name]+[x+' prediction' for x in models])
    
    if page == intro_page_name:
        intro()
    elif page.split()[0] in models:
        model_name = page.split()[0]
        model_dict = modeling.load_saved_model(model_name)
        st.write(f'## {model_dict['model_name']} Prediction')

        model = model_dict.get('model')
        features = model_dict.get('features')
        prep = model_dict.get('prep')
        scaler = model_dict.get('scaler')
        result = model_dict.get('result')
        target = model_dict.get('target')
        encoders = model_dict.get('encoders')

        if model:
            X_features = {}
            if features:
                for f in features:
                    if f in encoders:
                        val = st.selectbox(f, encoders[f].keys())
                        val = encoders[f][val]
                    else:
                        val = st.number_input(f)
                    X_features[f] = val
            else:
                val = st.number_input('input')
                X_features['input'] = val
            st.write(X_features)
            
            if st.button('Predict'):
                X_features = np.array([list(X_features.values())])
                if prep: X_features = prep(X_features)
                if scaler: X_features = scaler.transform(X_features)

                prediction = model.predict(X_features)
                if result:
                    st.write(result(prediction, encoders.get(target)))
                else:
                    st.write(f'Prediction: {prediction}')


if __name__ == "__main__":
    app()

