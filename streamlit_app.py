import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout='wide')

st.title('Crop Yield Prediction App')
st.write('Predict crop yield based on environmental factors and crop types.')

# Load the trained model and training columns
@st.cache_resource
def load_data():
    model = joblib.load('best_rf_model_tuned.pkl')
    training_columns = joblib.load('training_columns.pkl')
    feature_importances_df = joblib.load('feature_importances.pkl')
    per_crop_mean_yield = joblib.load('per_crop_mean_yield.pkl')
    original_df = joblib.load('original_df_for_streamlit.pkl') # Load original df
    return model, training_columns, feature_importances_df, per_crop_mean_yield, original_df

model, training_columns, feature_importances_df, per_crop_mean_yield, original_df = load_data()

# Extract unique Area and Item names from training_columns for select boxes
all_areas = sorted([col.replace('Area_', '') for col in training_columns if col.startswith('Area_')])
all_items = sorted([col.replace('Item_', '') for col in training_columns if col.startswith('Item_')])

# Sidebar for user input
st.sidebar.header('Input Features')

# User inputs
selected_area = st.sidebar.selectbox('Area', all_areas)
selected_item = st.sidebar.selectbox('Item', all_items)

year = st.sidebar.slider('Year', min_value=1990, max_value=2013, value=2010) # Adjusted max_value to 2013
rain_fall = st.sidebar.number_input('Average Rainfall (mm/year)', min_value=0.0, max_value=20000.0, value=1500.0)
pesticides = st.sidebar.number_input('Pesticides (tonnes)', min_value=0.0, max_value=100000.0, value=5000.0)
temperature = st.sidebar.number_input('Average Temperature (°C)', min_value=-20.0, max_value=40.0, value=20.0)


if st.sidebar.button('Predict Yield'):
    # Create a DataFrame for the current input, matching the structure of X_train
    input_data = pd.DataFrame(0, index=[0], columns=training_columns)

    # Populate numerical features
    input_data['Year'] = year
    input_data['average_rain_fall_mm_per_year'] = rain_fall
    input_data['pesticides_tonnes'] = pesticides
    input_data['avg_temp'] = temperature

    # Populate one-hot encoded categorical features
    if f'Area_{selected_area}' in input_data.columns:
        input_data[f'Area_{selected_area}'] = 1
    if f'Item_{selected_item}' in input_data.columns:
        input_data[f'Item_{selected_item}'] = 1

    # Ensure the order of columns is the same as during training
    input_data = input_data[training_columns]

    # Make prediction
    prediction = model.predict(input_data)[0]

    st.subheader('Prediction Result')

    # Display prediction prominently
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Predicted Yield", f"{prediction:,.2f} hg/ha")
    with col2:
        # Get historical mean for the selected crop
        historical_yield_for_item = per_crop_mean_yield.get(selected_item, 0) # Use .get with default 0 if item not found

        # --- New: Find and display historical data for comparison ---
        historical_record = original_df[
            (original_df['Area'] == selected_area) &
            (original_df['Item'] == selected_item) &
            (original_df['Year'] == year)
        ]

        plot_labels = ['Predicted', f'Avg Historical ({selected_item})']
        plot_values = [prediction, historical_yield_for_item]

        if not historical_record.empty:
            actual_yield = historical_record['hg/ha_yield'].iloc[0]
            historical_rain = historical_record['average_rain_fall_mm_per_year'].iloc[0]
            historical_pesticides = historical_record['pesticides_tonnes'].iloc[0]
            historical_temp = historical_record['avg_temp'].iloc[0]

            st.markdown(f"**Historical Data for {selected_item} in {selected_area} ({year}):**")
            st.write(f"- **Actual Yield:** {actual_yield:,.2f} hg/ha")
            st.write(f"- **Rainfall:** {historical_rain:,.2f} mm/year")
            st.write(f"- **Pesticides:** {historical_pesticides:,.2f} tonnes")
            st.write(f"- **Temperature:** {historical_temp:,.2f} °C")
            st.markdown("--- ")

            plot_labels.insert(1, 'Actual')
            plot_values.insert(1, actual_yield)

        else:
            st.info(f"No exact historical record found for {selected_item} in {selected_area} for {year}. Displaying predicted vs. average historical.")

        # Contextual plot: Predicted vs. Actual vs. Average Historical Yield
        fig_pred_vs_avg, ax_pred_vs_avg = plt.subplots(figsize=(6, 3))
        bars = ax_pred_vs_avg.bar(plot_labels, plot_values, color=['skyblue', 'lightgreen', 'lightgray'][:len(plot_labels)])
        ax_pred_vs_avg.set_ylabel('Yield (hg/ha)')
        ax_pred_vs_avg.set_title(f'Yield Comparison for {selected_item}')
        for bar in bars:
            yval = bar.get_height()
            ax_pred_vs_avg.text(bar.get_x() + bar.get_width()/2, yval + 500, round(yval, 0), ha='center', va='bottom', fontsize=8)
        st.pyplot(fig_pred_vs_avg)

    st.markdown("--- ")

# General Model Insights (Feature Importance)
st.subheader('Model Insights: Feature Importance')
with st.expander("View Feature Importance", expanded=False):
    fig_feature_imp, ax_feature_imp = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importances_df.head(10), ax=ax_feature_imp)
    ax_feature_imp.set_title("Top 10 Most Important Features")
    ax_feature_imp.set_xlabel("Importance")
    ax_feature_imp.set_ylabel("Feature")
    st.pyplot(fig_feature_imp)
