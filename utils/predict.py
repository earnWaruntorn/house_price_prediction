import streamlit as st
from utils.plot import PlotData

class Predictor:

    def __init__(self, dataset, categorical_mappings, scaler, model):
        self.dataset = dataset
        self.categorical_mappings = categorical_mappings
        self.scaler = scaler
        self.model = model
        self.visualization = PlotData(self.dataset, self.categorical_mappings)

    
    def scale_input(self, input_data):
        try:
            scaled_data = self.scaler.transform(input_data)
            return scaled_data
        except AttributeError as e:
            st.error(f"Error applying scaler: {e}")
        st.stop()

    def predict_data(self, input_data, input_array, selected_features):
        scaled_data = Predictor.scale_input(self, input_array)

        if st.sidebar.button("Predict"):
            prediction = self.model.predict(scaled_data)[0]
            st.subheader("Predicted House Price")
            st.metric(label="Price ($)", value=f"{prediction:,.2f}")


            @st.fragment
            def app_section_number_1() -> None:
                feature_to_plot = st.selectbox("Select a feature to visualize:", selected_features)
                user_feature_value = input_data.get(feature_to_plot, None)
                self.visualization.visualize_feature_vs_prediction(feature_to_plot, prediction, user_feature_value)
            app_section_number_1()