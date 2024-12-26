import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
import matplotlib.pyplot as plt

# Paths to dataset and model files
dataset_file_path = r'C:\testProject\cleaned_data.csv'
selected_features_file_path = r'C:\testProject\selected_features.txt'

model_file_path = r'C:\testProject\xgb_model.pkl'
scaler_file_path = r'C:\testProject\scaler.pkl'

# Load pickled objects (model, scaler, etc.)
def load_pkl(fname):
    with open(fname, 'rb') as f:
        obj = pickle.load(f)
    return obj

# Read dataset and selected features
def get_data():
    try:
        dataset = pd.read_csv(dataset_file_path)
    except FileNotFoundError:
        st.error("Dataset file not found. Please check the file path and try again.")
        st.stop()

    try:
        with open(selected_features_file_path) as f:
            selected_str = f.read()
            selected_features = selected_str.split('\n')
            return dataset, selected_features
    except FileNotFoundError:
        st.error("Selected features file not found. Please check the file path and try again.")
        st.stop()

def visualize_feature_vs_prediction(dataset, feature, prediction, user_input):
    """
    Visualize a selected feature vs. predicted price.

    Parameters:
    - dataset (pd.DataFrame): The dataset containing feature values and target variable.
    - feature (str): The feature to plot against the predicted price.
    - prediction (float): The predicted price.
    - user_input: The user-provided input for the selected feature.
    """

    # Ensure the feature exists in the dataset
    if feature not in dataset.columns:
        st.error(f"The feature '{feature}' does not exist in the dataset.")
        return

    # Group by the selected feature and calculate average price
    avg_price_per_feature = dataset.groupby(feature)["SalePrice"].mean()

    # Create the plot
    fig, ax = plt.subplots()
    ax.plot(avg_price_per_feature.index, avg_price_per_feature.values, color="skyblue", label="Average Price")

    # Highlight the predicted price
    ax.axhline(y=prediction, color="red", linestyle="--", label=f"Predicted Price (${prediction:,.2f})")

    # Highlight the user-selected feature value
    if user_input is not None:
        ax.axvline(x=user_input, color="green", linestyle="--", label=f"Selected {feature} ({user_input})")

    # Labels and legend
    ax.set_xlabel(feature)
    ax.set_ylabel("Average Sale Price ($)")
    ax.set_title(f"{feature} vs. Average Price")
    ax.legend()

    # Show the plot in Streamlit
    st.pyplot(fig)

# Categorical to numerical mapping
categorical_mappings = {
    "OverallQual": {"Very Excellent": 10,"Excellent": 9,"Very Good": 8,"Good": 7,"Above Average": 6,"Average": 5,"Below Average": 4,"Fair": 3,"Poor": 2,"Very Poor": 1,},
    "BsmtQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0},
    "BsmtFinType1": {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "NA": 0},
    "KitchenQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1},
    "FireplaceQu": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0},
    "GarageFinish": {"Fin": 3, "RFn": 2, "Unf": 1, "NA": 0},
    "GarageQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0},
    "GarageCond": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0},
    "Functional_Min1": {"Min1": 1, "Min2": 0},
    "Functional_Typ": {"Typ": 1, "NonTyp": 0},
    "MSZoning_RM": {1: 1, 0: 0},
    "Condition1_PosN": {1: 1, 0: 0},
    "Exterior1st_BrkFace": {1: 1, 0: 0},
    "CentralAir_Y": {1: 1, 0: 0},
    "GarageType_Attchd": {1: 1, 0: 0},
    "GarageType_Detchd": {1: 1, 0: 0},
    "Neighborhood_Edwards": {1: 1, 0: 0},
}

# Main app
def main():
    model = load_pkl(model_file_path)
    scaler = load_pkl(scaler_file_path)
    dataset, selected_features = get_data()

    st.title("House Price Prediction")

    

    # Initialize user data dictionary
    user_data = {}

    # Generate input fields for each selected feature
    for feature in selected_features:
        if feature in categorical_mappings:
            options = list(categorical_mappings[feature].keys())
            user_data[feature] = st.sidebar.selectbox(f"{feature}:", options)
        else:
            if feature == 'YearBuilt':
                current_year = datetime.date.today().year
                user_data[feature] = st.sidebar.number_input(f"{feature}:", min_value=1800, max_value=current_year, step=1, value=current_year)
            else:
                user_data[feature] = st.sidebar.number_input(f"{feature}:", value=0.0)

    # Convert categorical inputs to numerical values
    for feature, mapping in categorical_mappings.items():
        if feature in user_data:
            user_data[feature] = mapping.get(user_data[feature], 0)

    # Convert user inputs to array format
    input_array = np.array(list(user_data.values())).reshape(1, -1)

    # Scale input data
    try:
        scaled_data = scaler.transform(input_array)
    except AttributeError as e:
        st.error(f"Error applying scaler: {e}")
        st.stop()

    # Predict house price
    if st.sidebar.button("Predict"):
        prediction = model.predict(scaled_data)[0]
        st.subheader("Predicted House Price")
        st.metric(label="Price ($)", value=f"{prediction:,.2f}")

        @st.fragment
        def app_section_number_1() -> None:

            feature_to_plot = st.selectbox("Select a feature to visualize:", selected_features)
            user_feature_value = user_data.get(feature_to_plot, None)
            st.write(feature_to_plot, user_feature_value)
            visualize_feature_vs_prediction(dataset, feature_to_plot, prediction, user_feature_value)
        app_section_number_1()

        

if __name__ == "__main__":
    main()
