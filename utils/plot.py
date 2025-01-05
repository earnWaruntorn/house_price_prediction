import streamlit as st
import pandas as pd
import plotly.graph_objects as go

class PlotData:

    def __init__(self, dataset, categorical_mappings):
        self.dataset = dataset
        self.categorical_mappings = categorical_mappings

    def visualize_feature_vs_prediction(self, feature, prediction, user_input):

        if feature not in self.dataset.columns:
            st.error(f"The feature '{feature}' does not exist in the dataset.")
            return

        # Determine if the feature is categorical or numerical
        is_categorical = self.dataset[feature].dtype == 'object' or feature in self.categorical_mappings

        # Initialize Plotly figure
        fig = go.Figure()

        if is_categorical:
            # Plot for categorical features: bar plot
            avg_price_per_feature = self.dataset.groupby(feature)["SalePrice"].mean().sort_index()
            fig.add_trace(
                go.Bar(
                    x=avg_price_per_feature.index,
                    y=avg_price_per_feature.values,
                    name="Average Price",
                    marker=dict(color="skyblue"),
                )
            )
            # Highlight the predicted price
            if user_input is not None:
                fig.add_trace(
                    go.Bar(
                        x=[user_input],
                        y=[prediction],
                        name="Predicted Price",
                        marker=dict(color="red"),
                    )
                )
        else:
            # Plot for numerical features: line plot
            avg_price_per_feature = self.dataset.groupby(feature)["SalePrice"].mean().sort_index()
            fig.add_trace(
                go.Scatter(
                    x=avg_price_per_feature.index,
                    y=avg_price_per_feature.values,
                    mode="lines",
                    name="Average Price",
                    line=dict(color="skyblue"),
                )
            )
            # Highlight the predicted price with a horizontal line
            fig.add_trace(
                go.Scatter(
                    x=[user_input],  # Single x value (user input)
                    y=[prediction],  # Single y value (predicted price)
                    mode="markers",  # Use markers to display the point
                    name=f"Predicted Price (${prediction:,.2f})",
                    marker=dict(size=5, color="red", symbol="circle"),  # Custom marker style
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[user_input, user_input],
                    y=[0, prediction],
                    mode="lines",
                    line=dict(dash="dash", color="red"),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=[min(avg_price_per_feature.index), user_input],
                    y=[prediction, prediction],
                    mode="lines",
                    line=dict(dash="dash", color="red"),
                )
            )

        # Update layout
        fig.update_layout(
            title=f"{feature} vs. Average Price",
            xaxis_title=feature,
            yaxis_title="Average Sale Price ($)",
            legend=dict(title="Legend", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white",
        )

        # Show the plot in Streamlit
        st.plotly_chart(fig)