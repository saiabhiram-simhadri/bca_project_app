import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
import joblib
import sklearn


# Modular Classes
class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        try:
            self.data = pd.read_csv(self.file_path)
            return self.data
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None

class FeatureEngineer:
    def __init__(self, categorical_cols, numerical_cols):
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        self.preprocessor = None

    def create_preprocessor(self):
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ]), self.numerical_cols),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder(handle_unknown='ignore'))
                ]), self.categorical_cols)
            ])
        return self.preprocessor

    def preprocess_data(self, X):
        if self.preprocessor is None:
            self.create_preprocessor()
        return self.preprocessor.fit_transform(X)

    def transform(self, X):
        if self.preprocessor is None:
            raise ValueError("Preprocessor not initialized.")
        return self.preprocessor.transform(X)

class ModelTrainer:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.model_pipeline = None

    def create_pipeline(self, estimators=None):
        if estimators is None:
            xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42, learning_rate=0.1, max_depth=7, n_estimators=200)
            rf_model = RandomForestRegressor(random_state=42)
            lr_model = LinearRegression()
            estimators = [('xgb', xgb_model), ('rf', rf_model), ('lr', lr_model)]
        stacked_model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
        self.model_pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', stacked_model)
        ])
        return self.model_pipeline

    def train(self, X_train, y_train):
        if self.model_pipeline is None:
            self.create_pipeline()
        self.model_pipeline.fit(X_train, y_train)
        return self.model_pipeline

    def save_model(self, file_path):
        if self.model_pipeline is not None:
            joblib.dump(self.model_pipeline, file_path)

class Predictor:
    def __init__(self, model_path=None, model_pipeline=None):
        if model_path:
            try:
                self.model_pipeline = joblib.load(model_path)
            except Exception as e:
                st.error(f"Failed to load model from {model_path}: {e}")
                self.model_pipeline = None
        elif model_pipeline:
            self.model_pipeline = model_pipeline
        else:
            raise ValueError("Either model_path or model_pipeline must be provided.")

    def predict(self, X):
        if self.model_pipeline is None:
            raise ValueError("No valid model loaded.")
        return self.model_pipeline.predict(X)

# Streamlit Application
def main():
    st.set_page_config(page_title="CO2 Emissions Predictor", page_icon="🌍", layout="wide")

    st.markdown("""
        <style>
        .main {background-color: #f0f2f6;}
        .stButton>button {background-color: #4CAF50; color: white; border-radius: 5px;}
        .stTextInput>input, .stNumberInput>input {border-radius: 5px;}
        .stSelectbox {border-radius: 5px;}
        .prediction-box {background-color: #e0f7fa; padding: 20px; border-radius: 10px; text-align: center;}
        </style>
    """, unsafe_allow_html=True)

    st.title("🌍 CO2 Emissions Predictor")
    st.markdown("Predict CO2 emissions for vehicles based on their characteristics using machine learning.")

    # Sidebar for Training
    st.sidebar.header("Model Training")
    train_model = st.sidebar.button("Train Model")
    dataset_path = st.sidebar.text_input("/input/CO2 Emissions_Canada.csv")
    model_path = "optimized_model.pkl"

    # Define dataset parameters
    categorical_cols = ['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type']
    numerical_cols = ['Engine Size(L)', 'Cylinders', 'Fuel Consumption City (L/100 km)', 
                      'Fuel Consumption Hwy (L/100 km)', 'Fuel Consumption Comb (L/100 km)', 
                      'Fuel Consumption Comb (mpg)']

    # Load dataset to populate dropdowns (only if available)
    if 'make_options' not in st.session_state or 'model_options' not in st.session_state:
        loader = DataLoader(dataset_path)
        data = loader.load_data()
        if data is not None:
            st.session_state['make_options'] = sorted(data['Make'].unique().tolist())
            st.session_state['model_options'] = sorted(data['Model'].unique().tolist())
        else:
            # Fallback options if dataset isn't available
            st.session_state['make_options'] = ['Toyota', 'Ford', 'Honda', 'Chevrolet', 'BMW']
            st.session_state['model_options'] = ['Camry', 'F-150', 'Civic', 'Silverado', 'X5']
            st.warning("Dataset not found. Using default options for Make and Model. For accurate options, train the model with the correct dataset.")

    # Train model if requested
    if train_model:
        with st.spinner("Training model... This may take a few minutes."):
            loader = DataLoader(dataset_path)
            data = loader.load_data()
            if data is not None:
                X = data.drop(columns=['CO2 Emissions(g/km)'])
                y = data['CO2 Emissions(g/km)']
                feature_engineer = FeatureEngineer(categorical_cols, numerical_cols)
                preprocessor = feature_engineer.create_preprocessor()
                trainer = ModelTrainer(preprocessor)
                model_pipeline = trainer.train(X, y)
                trainer.save_model(model_path)
                # Update dropdown options after training
                st.session_state['make_options'] = sorted(data['Make'].unique().tolist())
                st.session_state['model_options'] = sorted(data['Model'].unique().tolist())
                st.success(f"Model trained and saved as {model_path} with scikit-learn {sklearn.__version__}")
            else:
                st.error("Failed to load dataset for training.")

    # Main Input Section
    st.header("Enter Vehicle Details")
    st.markdown("Provide the vehicle parameters below to predict CO2 emissions.")

    col1, col2 = st.columns(2)

    defaults = {
        'Make': 'Toyota',
        'Model': 'Camry',
        'Vehicle Class': 'Compact',
        'Engine Size(L)': 2.5,
        'Cylinders': 4.0,
        'Transmission': 'Automatic',
        'Fuel Type': 'Gasoline',
        'Fuel Consumption City (L/100 km)': 8.5,
        'Fuel Consumption Hwy (L/100 km)': 6.0,
        'Fuel Consumption Comb (L/100 km)': 7.3,
        'Fuel Consumption Comb (mpg)': 32.0
    }
    ranges = {
        'Engine Size(L)': (1.0, 8.0),
        'Cylinders': (2.0, 16.0),
        'Fuel Consumption City (L/100 km)': (0.0, 30.0),
        'Fuel Consumption Hwy (L/100 km)': (0.0, 30.0),
        'Fuel Consumption Comb (L/100 km)': (0.0, 30.0),
        'Fuel Consumption Comb (mpg)': (0.0, 60.0)
    }

    with col1:
        make = st.selectbox("Make", 
                            options=st.session_state['make_options'], 
                            index=st.session_state['make_options'].index(defaults['Make']) if defaults['Make'] in st.session_state['make_options'] else 0, 
                            help="Select the vehicle brand")
        model = st.selectbox("Model", 
                             options=st.session_state['model_options'], 
                             index=st.session_state['model_options'].index(defaults['Model']) if defaults['Model'] in st.session_state['model_options'] else 0, 
                             help="Select the vehicle model")
        vehicle_class = st.selectbox("Vehicle Class", 
                                     options=['Compact', 'SUV', 'Pickup', 'Sedan', 'Minivan'], 
                                     index=0, help="Select the vehicle class")
        engine_size = st.number_input("Engine Size (L)", 
                                      min_value=float(ranges['Engine Size(L)'][0]), 
                                      max_value=float(ranges['Engine Size(L)'][1]), 
                                      value=float(defaults['Engine Size(L)']), 
                                      step=0.1)
        cylinders = st.number_input("Cylinders", 
                                    min_value=float(ranges['Cylinders'][0]), 
                                    max_value=float(ranges['Cylinders'][1]), 
                                    value=float(defaults['Cylinders']), 
                                    step=1.0)
        transmission = st.selectbox("Transmission", 
                                    options=['Automatic', 'Manual', 'Semi-automatic'], 
                                    index=0, help="Select the transmission type")

    with col2:
        fuel_type = st.selectbox("Fuel Type", 
                                 options=['Gasoline', 'Diesel', 'Ethanol', 'Hybrid'], 
                                 index=0, help="Select the fuel type")
        fuel_city = st.number_input("Fuel Consumption City (L/100 km)", 
                                    min_value=float(ranges['Fuel Consumption City (L/100 km)'][0]), 
                                    max_value=float(ranges['Fuel Consumption City (L/100 km)'][1]), 
                                    value=float(defaults['Fuel Consumption City (L/100 km)']), 
                                    step=0.1)
        fuel_hwy = st.number_input("Fuel Consumption Hwy (L/100 km)", 
                                   min_value=float(ranges['Fuel Consumption Hwy (L/100 km)'][0]), 
                                   max_value=float(ranges['Fuel Consumption Hwy (L/100 km)'][1]), 
                                   value=float(defaults['Fuel Consumption Hwy (L/100 km)']), 
                                   step=0.1)
        fuel_comb = st.number_input("Fuel Consumption Comb (L/100 km)", 
                                    min_value=float(ranges['Fuel Consumption Comb (L/100 km)'][0]), 
                                    max_value=float(ranges['Fuel Consumption Comb (L/100 km)'][1]), 
                                    value=float(defaults['Fuel Consumption Comb (L/100 km)']), 
                                    step=0.1)
        fuel_comb_mpg = st.number_input("Fuel Consumption Comb (mpg)", 
                                        min_value=float(ranges['Fuel Consumption Comb (mpg)'][0]), 
                                        max_value=float(ranges['Fuel Consumption Comb (mpg)'][1]), 
                                        value=float(defaults['Fuel Consumption Comb (mpg)']), 
                                        step=1.0)

    # Predict button
    if st.button("Predict CO2 Emissions", key="predict_button"):
        input_data = pd.DataFrame({
            'Make': [make],
            'Model': [model],
            'Vehicle Class': [vehicle_class],
            'Engine Size(L)': [engine_size],
            'Cylinders': [cylinders],
            'Transmission': [transmission],
            'Fuel Type': [fuel_type],
            'Fuel Consumption City (L/100 km)': [fuel_city],
            'Fuel Consumption Hwy (L/100 km)': [fuel_hwy],
            'Fuel Consumption Comb (L/100 km)': [fuel_comb],
            'Fuel Consumption Comb (mpg)': [fuel_comb_mpg]
        })

        try:
            predictor = Predictor(model_path=model_path)
            prediction = predictor.predict(input_data)[0]
            st.markdown(f"""
                <div class="prediction-box">
                    <h3>Predicted CO2 Emissions</h3>
                    <p style="font-size: 24px; color: #00796b;">{prediction:.2f} g/km</p>
                </div>
            """, unsafe_allow_html=True)
        except ValueError as e:
            st.error(f"Prediction error: {e}")
        except Exception as e:
            st.error(f"Error during prediction: {e}. Please ensure the model is compatible with scikit-learn {sklearn.__version__}. Try retraining the model using the sidebar.")

    st.markdown("---")
    st.markdown("Developed for BCA Project | Powered by Streamlit & Machine Learning")

if __name__ == "__main__":
    main()