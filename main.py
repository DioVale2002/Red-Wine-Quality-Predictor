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
    try:
        # Try to load a scaler - adjust filename as needed
        scaler = joblib.load('scaler.pkl')
        return scaler
    except FileNotFoundError:
        st.warning("⚠️ No scaler file found. If your model was trained with scaling, please include 'scaler.pkl'")
        return None
    except Exception as e:
        st.error(f"Error loading scaler: {str(e)}")
        return None

# Main app
def main():
    st.title("🍷 Wine Quality Prediction")
    st.write("Enter wine chemical properties to predict if the quality is **Very Good** (≥7) or **Not Good** (<7)")
    
    # Add some info about the model
    st.info("This model predicts wine quality based on 11 chemical properties. Enter the values below to get a prediction.")
    
    # Load model
    try:
        model = load_model()
        preprocessor = load_preprocessor()  # Optional
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()
    
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
    if preprocessor is not None:
        st.info("✅ Applying feature scaling...")
        try:
            features = preprocessor.transform(features)
        except Exception as e:
            st.error(f"Error applying scaler: {str(e)}")
            st.stop()
    else:
        st.info("ℹ️ No preprocessing applied - using raw feature values")
    
    # Make prediction
    if st.button("🔮 Predict Wine Quality", type="primary"):
        try:
            # Make prediction - extract single values from arrays
            prediction = model.predict(features)
            prediction_value = prediction[0]  # Extract single value
            
            # Get prediction probabilities
            if hasattr(model, 'predict_proba'):
                prediction_proba = model.predict_proba(features)
                
                # Handle different return types (numpy array vs list)
                if isinstance(prediction_proba, list):
                    # If it's a list, take the first element
                    proba_array = prediction_proba[0] if len(prediction_proba) > 0 else [0.5, 0.5]
                    prob_not_good = float(proba_array[0])
                    prob_very_good = float(proba_array[1])
                else:
                    # If it's a numpy array
                    prob_not_good = float(prediction_proba[0][0])
                    prob_very_good = float(prediction_proba[0][1])
            else:
                # Fallback if model doesn't support predict_proba
                prob_not_good = 0.5
                prob_very_good = 0.5
                st.warning("⚠️ Model doesn't support probability predictions. Using default values.")
            
            st.subheader("🎯 Prediction Results")
            
            # Display prediction with styling
            if prediction_value == 1:
                st.success("🌟 **Very Good Wine** (Quality ≥ 7)")
                quality_text = "Very Good"
            else:
                st.warning("⚠️ **Not Good Wine** (Quality < 7)")
                quality_text = "Not Good"
            
            # Try to get actual quality score if model supports it
            try:
                if hasattr(model, 'predict_proba') and hasattr(model, 'classes_'):
                    # Estimate quality: if prob >= 0.5, quality likely 7+, else likely < 7
                    estimated_quality = 6.0 + (prob_very_good * 3.0)  # Scale between 6-9
                    st.info(f"📊 **Estimated Quality Score: {estimated_quality:.1f}/10**")
                else:
                    st.info("📊 Quality score estimation not available for this model type")
            except Exception as e:
                st.info(f"📊 Quality score estimation error: {str(e)}")
            
            # Better confidence calculation: distance from decision boundary (0.5)
            try:
                confidence_score = abs(prob_very_good - 0.5) * 2  # Scale to 0-1 range
            except Exception as e:
                st.error(f"Error calculating confidence: {str(e)}")
                confidence_score = 0.5  # Default confidence
            
            # Display confidence score prominently
            st.subheader("🎯 Confidence Score")
            
            # Adjusted confidence thresholds for better user experience
            if confidence_score >= 0.6:  # Was 0.8
                confidence_color = "green"
                confidence_level = "Very High"
                confidence_emoji = "🟢"
            elif confidence_score >= 0.4:  # Was 0.65
                confidence_color = "blue"
                confidence_level = "High"
                confidence_emoji = "🔵"
            elif confidence_score >= 0.2:  # Was 0.55
                confidence_color = "orange"
                confidence_level = "Moderate"
                confidence_emoji = "🟡"
            else:
                confidence_color = "red"
                confidence_level = "Low"
                confidence_emoji = "🔴"
            
            # Display confidence with visual indicators
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown(f"""
                <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: rgba(255,255,255,0.1);">
                    <h2 style="color: {confidence_color};">{confidence_emoji} {confidence_score:.1%}</h2>
                    <p style="font-size: 18px; margin: 0;"><strong>{confidence_level} Confidence</strong></p>
                </div>
                """, unsafe_allow_html=True)
            
            # Display probability details
            st.subheader("📊 Probability Breakdown")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Not Good Wine", f"{prob_not_good:.1%}")
            with col2:
                st.metric("Very Good Wine", f"{prob_very_good:.1%}")
            
            # Confidence interpretation with adjusted thresholds
            st.subheader("💡 Decision Support")
            if confidence_score >= 0.6:
                st.success("**Highly reliable prediction** - You can trust this result with high confidence.")
            elif confidence_score >= 0.4:
                st.info("**Good prediction reliability** - This result is quite trustworthy for decision-making.")
            elif confidence_score >= 0.2:
                st.warning("**Moderate reliability** - Consider additional factors or expert opinion before making critical decisions.")
            else:
                st.error("**Low confidence prediction** - This result is uncertain. Recommend seeking expert evaluation or additional testing.")
            
            # Additional decision-making context
            with st.expander("🤔 How to Interpret This Confidence Score"):
                st.markdown("""
                **Confidence Score Explanation:**
                - **60-100%**: Very High Confidence - The model is very certain about its prediction
                - **40-59%**: High Confidence - The model is quite confident, good for most decisions
                - **20-39%**: Moderate Confidence - The prediction has some uncertainty
                - **Below 20%**: Low Confidence - The model is very uncertain, close to a coin flip
                
                **Confidence Calculation:**
                This confidence score measures how far the prediction is from being uncertain (50/50). 
                A 90% "Very Good" prediction has higher confidence than a 60% "Very Good" prediction.
                
                **For Wine Quality Decisions:**
                - **High confidence "Very Good"**: Excellent choice for special occasions or recommendations
                - **High confidence "Not Good"**: Consider alternative wines or further evaluation
                - **Low confidence results**: May benefit from professional sommelier assessment or additional chemical analysis
                """)
                
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.write("Debug information:")
            st.write(f"Features shape: {features.shape}")
            st.write(f"Model type: {type(model).__name__}")
            # Add more debug info
            try:
                pred_result = model.predict(features)
                st.write(f"Model prediction output type: {type(pred_result)}")
                st.write(f"Prediction result: {pred_result}")
                if hasattr(model, 'predict_proba'):
                    proba_result = model.predict_proba(features)
                    st.write(f"Model probability output type: {type(proba_result)}")
                    st.write(f"Probability result: {proba_result}")
            except Exception as debug_e:
                st.write(f"Debug error: {str(debug_e)}")
    
    # Optional: Display model information
    with st.expander("🔍 Model Information"):
        try:
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
        except Exception as e:
            st.write(f"Error displaying model information: {str(e)}")

if __name__ == "__main__":
    main()
