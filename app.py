import streamlit as st
from models.model_loader import PKLLoader
from utils.data_loader import DataLoader
from utils.sidebar import Sidebar
from utils.predict import Predictor

dataset_file_path = r'C:\data_science_project\house_price_prediction\data\cleaned_data.csv'
selected_features_file_path = r'C:\data_science_project\house_price_prediction\data\selected_features.txt'

model_file_path = r'C:\data_science_project\house_price_prediction\models\xgb_model.pkl'
scaler_file_path = r'C:\data_science_project\house_price_prediction\models\scaler.pkl'

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

def main():
    pkl_loader = PKLLoader()
    model = pkl_loader.load_pkl(model_file_path)
    scaler = pkl_loader.load_pkl(scaler_file_path)

    data_loader = DataLoader()
    dataset = data_loader.get_csv(dataset_file_path)
    selected_features = data_loader.get_txt(selected_features_file_path)

    st.title("House Price Prediction")

    sidebar = Sidebar(categorical_mappings)
    input_data, input_array = sidebar.generate_sidebar(selected_features, dataset)

    predictor = Predictor(dataset, categorical_mappings, scaler, model)
    predictor.predict_data(input_data, input_array, selected_features)
        

if __name__ == "__main__":
    main()
