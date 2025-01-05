import streamlit as st
import numpy as np

class Sidebar:

    def __init__(self, categorical_mappings):
        self.data = {}
        self.categorical_mappings = categorical_mappings

    def convert_data(self):
        for feature, mapping in self.categorical_mappings.items():
            if feature in self.data:
                self.data[feature] = mapping.get(self.data[feature], 0)

        input_array = np.array(list(self.data.values())).reshape(1, -1)
        return input_array

    def generate_sidebar(self, selected_features, dataset):
        for feature in selected_features:
            if feature in self.categorical_mappings:
                options = list(self.categorical_mappings[feature].keys())
                self.data[feature] = st.sidebar.selectbox(f"{feature}:", options)
            else:
                if feature == 'YearBuilt':
                    max_year = dataset[feature].max()
                    self.data[feature] = st.sidebar.number_input(f"{feature}:", min_value=1800, max_value=max_year, step=1, value=max_year)
                else:
                    self.data[feature] = st.sidebar.number_input(f"{feature}:", value=0.0)
        
        return self.data, Sidebar.convert_data(self)