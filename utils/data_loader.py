import pandas as pd
import streamlit as st

class DataLoader:

    def get_csv(self, csv_path):
        try:
            dataset = pd.read_csv(csv_path)
            if dataset.empty:
                raise ValueError(f"The CSV file at {csv_path} is empty.")
            return dataset
        except FileNotFoundError:
            st.error(f"Dataset file not found at path: {csv_path}. Please check the file path.")
            st.stop()
        except Exception as e:
            st.error(f"An error occurred while loading the CSV file: {e}")
            st.stop()

    def get_txt(self, txt_path):
        try:
            with open(txt_path) as f:
                selected_str = f.read().strip()
                if not selected_str:
                    raise ValueError(f"The TXT file at {txt_path} is empty.")
                selected_features = selected_str.split('\n')
                return selected_features
        except FileNotFoundError:
            st.error(f"Selected features file not found at path: {txt_path}. Please check the file path.")
            st.stop()
        except Exception as e:
            st.error(f"An error occurred while loading the TXT file: {e}")
            st.stop()
