import streamlit as st
import pandas as pd
# import numpy as np
import joblib

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="wide")

st.markdown("""
    <style>
    .car-card {
        border-radius: 12px;
        color: black;
        padding: 25px;
        background-color: white;
        border: 1px solid #e0e0e0;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    .car-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        border-color: #4A90E2;
    }
    .price-badge {
        background-color: #E3F2FD;
        color: #1976D2;
        padding: 5px 12px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin-top: 10px;
    }
    .hero-section {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 60px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 40px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    try:
        model = joblib.load('car_price_model.joblib')
        metadata = joblib.load('model_metadata.joblib')
        test_data = pd.read_csv('test_data_raw.csv')
        return model, metadata, test_data
    except Exception as e:
        st.error(f"Error loading model or data: {e}. Did you run 'train_and_save.py' first?")
        return None, None, None

model, metadata, test_data = load_assets()

if model and metadata and test_data is not None:
    st.markdown("""
        <div class="hero-section">
            <h1>🚗 AI Car Price Estimator</h1>
            <p style="font-size: 1.2rem; opacity: 0.9;">Trained on Cars24 Data</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.write("### Select a car to see how the the ML model values it")

    st.sidebar.header("Filters")
    make_filter = st.sidebar.multiselect("Select Car Make (Brand)", options=test_data['make'].unique())
    
    filtered_df = test_data
    if make_filter:
        filtered_df = test_data[test_data['make'].isin(make_filter)]

    st.subheader(f"Showing {len(filtered_df)} cars from test data")
    st.write("These cars were never seen by the model")
    
    cols = st.columns(3)
    
    for idx, row in filtered_df.iterrows():
        col_idx = idx % 3
        with cols[col_idx]:
            with st.container():
                st.markdown(f"""
                <div class="car-card">
                    <h3 style="margin-bottom: 5px; color: black;">{row['make']}</h3>
                    <h4 style="color: #666; margin-top: 0;">{row['model']}</h4>
                    <p><b>Age:</b> {int(row['age'])} yrs | <b>Mileage:</b> {row['mileage']} kmpl</p>
                    <div class="price-badge">Actual Market Price: ₹{row['selling_price']:,} Lac</div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"Predict Price", key=f"btn_{idx}"):
                    
                    
                    input_df = pd.DataFrame([row])
                    
                    input_df['make'] = input_df['make'].map(metadata['make_wise_mean']).fillna(metadata['global_mean'])
                    input_df['model'] = input_df['model'].map(metadata['model_wise_mean']).fillna(metadata['global_mean'])
                    
                    input_df = input_df[metadata['all_columns']]
                    scaled_data = metadata['feature_scaler'].transform(input_df)
                    scaled_df = pd.DataFrame(scaled_data, columns=metadata['all_columns'])
                    
                    x_input = scaled_df[metadata['selected_features']]
                    
                    scaled_prediction = model.predict(x_input)[0]
                    
                    min_p = metadata['price_min']
                    max_p = metadata['price_max']
                    actual_prediction = scaled_prediction * (max_p - min_p) + min_p
                    
                    st.success(f"### Predicted Price: ₹{actual_prediction:,.2f}")
                    st.balloons()

# st.info("Note: Please run 'train_and_save.py' to generate the model files.")
