import streamlit as st
import pandas as pd
import numpy as np
import joblib
# import pickle  # Use this if your model was saved with pickle

# Load your model (adjust the path and loading method as needed)
@st.cache_resource
def load_model():
    # For joblib:
    model = joblib.load('random_forest_model.pkl')
    # For pickle:
    # with open('random_forest_model.pkl', 'rb') as f:
    #     model = pickle.load(f)
    return model

# Load any preprocessors if you have them
@st.cache_resource
def load_preprocessor():
    # Example for a scaler
    # scaler = joblib.load('scaler.pkl')
    # return scaler
    return None

# Main app
def main():
    st.title("🍷 Wine Quality Prediction")
    st.write("Enter wine chemical properties to predict if the quality is **Very Good** (≥7) or **Not Good** (<7)")
    
    # Add some info about the model
    st.info("This model predicts wine quality based on 11 chemical properties. Enter the values below to get a prediction.")
    
    # Load model
    model = load_model()
    preprocessor = load_preprocessor()  # Optional
    
    # Wine Quality Features Input
    st.subheader("Wine Chemical Properties")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, max_value=20.0, value=7.4, step=0.1)
        volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, max_value=2.0, value=0.7, step=0.01)
        citric_acid = st.number_input("Citric Acid", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
        residual_sugar = st.number_input("Residual Sugar", min_value=0.0, max_value=50.0, value=1.9, step=0.1)
        chlorides = st.number_input("Chlorides", min_value=0.0, max_value=1.0, value=0.076, step=0.001)
        free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, max_value=100.0, value=11.0, step=1.0)
    
    with col2:
        total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, max_value=300.0, value=34.0, step=1.0)
        density = st.number_input("Density", min_value=0.0, max_value=2.0, value=0.9978, step=0.0001)
        pH = st.number_input("pH", min_value=0.0, max_value=14.0, value=3.51, step=0.01)
        sulphates = st.number_input("Sulphates", min_value=0.0, max_value=2.0, value=0.56, step=0.01)
        alcohol = st.number_input("Alcohol", min_value=0.0, max_value=20.0, value=9.4, step=0.1)
    
    # Create feature array in the correct order
    features = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, 
                         chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, 
                         pH, sulphates, alcohol]])
    
    # Apply preprocessing if you have it
    if preprocessor:
        features = preprocessor.transform(features)
    
    # Make prediction
    if st.button("🔮 Predict Wine Quality", type="primary"):
        try:
            prediction = model.predict(features)
            prediction_proba = model.predict_proba(features)
            
            st.subheader("🎯 Prediction Results")
            
            # Display prediction with styling
            if prediction[0] == 1:
                st.success("🌟 **Very Good Wine** (Quality ≥ 7)")
                quality_text = "Very Good"
            else:
                st.warning("⚠️ **Not Good Wine** (Quality < 7)")
                quality_text = "Not Good"
            
            # Display probabilities with a progress bar
            st.subheader("📊 Confidence Levels")
            prob_not_good = prediction_proba[0][0]
            prob_very_good = prediction_proba[0][1]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Not Good (<7)", f"{prob_not_good:.1%}")
                st.progress(prob_not_good)
            
            with col2:
                st.metric("Very Good (≥7)", f"{prob_very_good:.1%}")
                st.progress(prob_very_good)
                
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.write("Please check that your model file is in the correct location and format.")
    
    # Optional: Display model information
    with st.expander("🔍 Model Information"):
        st.write(f"**Model Type:** {type(model).__name__}")
        if hasattr(model, 'n_estimators'):
            st.write(f"**Number of Trees:** {model.n_estimators}")
        if hasattr(model, 'feature_importances_'):
            st.write("**Feature Importances:**")
            importances = model.feature_importances_
            feature_names = ['Fixed Acidity', 'Volatile Acidity', 'Citric Acid', 'Residual Sugar',
                           'Chlorides', 'Free Sulfur Dioxide', 'Total Sulfur Dioxide', 'Density',
                           'pH', 'Sulphates', 'Alcohol']
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            # Display as a bar chart
            st.bar_chart(importance_df.set_index('Feature'))

if __name__ == "__main__":
    main()