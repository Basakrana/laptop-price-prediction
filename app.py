import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sys

# Page configuration
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide"
)

# Load the model
@st.cache_resource
def load_model():
    try:
        # Try loading with pickle first
        with open('best_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model with pickle: {str(e)}")
        try:
            # Try with joblib as alternative
            import joblib
            model = joblib.load('best_model.pkl')
            return model
        except Exception as e2:
            st.error(f"Error loading model with joblib: {str(e2)}")
            return None

model = load_model()

if model is None:
    st.error("⚠️ Could not load the model. Please check the following:")
    st.markdown("""
    1. Ensure `best_model.pkl` is in the same directory as this script
    2. The model was saved using the same Python/library versions
    3. Try re-saving the model with: `joblib.dump(model, 'best_model.pkl')`
    """)
    st.stop()

# Title and description
st.title("💻 Laptop Price Predictor")
st.markdown("### Predict laptop prices using XGBoost Machine Learning Model")
st.markdown("---")

# Create two columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Hardware Specifications")
    
    graphic_card_type = st.selectbox(
        "Graphics Card Type",
        ["dedicated graphics", "integrated graphics"]
    )
    
    resolution = st.selectbox(
        "Screen Resolution",
        ["1280 x 800", "1366 x 768", "1600 x 900", "1920 x 1080", 
         "1920 x 1200", "1920 x 1280", "2160 x 1440", "2560 x 1440", 
         "2560 x 1600", "2880 x 1620", "3200 x 1800", "3840 x 2160", "other"]

    )
    
    cpu_cores = st.selectbox(
        "CPU Cores",
        ["2", "4", "3", "8", "not applicable"]
    )
    
    ram_size = st.selectbox(
        "RAM Size",
        ["256 mb", "2 gb", "4 gb", "6 gb", "8 gb", "12 gb", "16 gb", 
         "20 gb", "24 gb", "32 gb", "64 gb"]

    )
    
    drive_type = st.selectbox(
        "Drive Type",
        ["ssd", "hdd", "ssd + hdd", "hybrid", "emmc", "other"]
    )
    
    ram_type = st.selectbox(
        "RAM Type",
        ["ddr4", "ddr3", "other"]
    )
    
    cpu_model = st.selectbox(
        "CPU Model",
        [
        "intel celeron m",
        "intel celeron dual-core",
        "intel celeron quad core",
        "intel pentium 4",
        "intel pentium dual-core",
        "intel pentium quad-core",
        "amd e1",
        "amd a4",
        "amd a6",
        "amd a8",
        "amd a10",
        "amd a12",
        "intel core m",
        "intel core i3",
        "intel core i5",
        "intel core i7",
        "other",
        "other CPU"
    ]

    )

with col2:
    st.subheader("Display & Physical")
    
    warranty = st.selectbox(
        "Warranty",
        ["producer warranty", "seller warranty", "no warranty"]
    )
    
    screen_size = st.selectbox(
        "Screen Size",
        ['11.9" and less', '12" - 12.9"', '13" - 13.9"', 
         '14" - 14.9"', '15" - 15.9"', '17" - 17.9"']

    )
    
    operating_system = st.selectbox(
        "Operating System",
        [
        "no system",
        "linux",
        "windows 7 home 64-bit",
        "windows 7 professional 64-bit",
        "windows 8.1 home 32-bit",
        "windows 8.1 home 64-bit",
        "windows 8.1 professional 32-bit",
        "windows 8.1 professional 64-bit",
        "windows 10 home",
        "windows 10 professional",
        "other"
        ]

    )
    
    wifi = st.selectbox(
        "WiFi",
        ["no wifi", "wi-fi", "wi-fi 802.11 b/g/n/ac", "wi-fi 802.11 a/b/g/n/ac",
         "wi-fi 802.11 b/g/n", "wi-fi 802.11 a/b/g/n"]
    )
    
    modem = st.selectbox(
        "Cellular Modem",
        ["None", "3g (wwan)", "3g(wwan) + 4g(lte)", "4g (lte)"]
    )
    
    lan = st.selectbox(
        "LAN Speed",
        ["10/100/1000 mbps", "10/100 mbps", "None"]
    )
    
    keyboard = st.selectbox(
        "Keyboard Type",
        ["keyboard", "illuminated keyboard", "None"]
    )

with col3:
    st.subheader("Performance & Storage")
    
    cpu_speed = st.selectbox(
        "CPU Speed (GHz)",
        ["0-0.5", "0.5-1.0", "1.0-1.5", "1.5-2.0", 
         "2.0-2.5", "2.5-3.0", "3.0-3.5", "3.5-4.0"]

    )
    
    drive_memory_size = st.selectbox(
        "Storage Capacity",
        ["0-256 GB", "256-512 GB", "512-1024 GB", "1024-2048 GB", "2048+ GB"]

    )
    
    st.markdown("### Additional Features")
    
    bluetooth = st.checkbox("Bluetooth", value=True)
    gps = st.checkbox("GPS", value=False)
    widi = st.checkbox("Intel WiDi", value=False)
    nfc = st.checkbox("NFC", value=False)
    numeric_keyboard = st.checkbox("Numeric Keyboard", value=True)
    touchpad = st.checkbox("Touchpad", value=True)
    camera = st.checkbox("Camera", value=True)
    microphone = st.checkbox("Microphone", value=True)
    sd_card_reader = st.checkbox("SD Card Reader", value=True)
    speakers = st.checkbox("Speakers", value=True)

st.markdown("---")

# Predict button
if st.button("🔮 Predict Price", type="primary", use_container_width=True):
    # Prepare input data
    input_data = pd.DataFrame({
        'graphic card type': [graphic_card_type],
        'resolution (px)': [resolution],
        'CPU cores': [cpu_cores],
        'RAM size': [ram_size],
        'drive type': [drive_type],
        'RAM type': [ram_type],
        'CPU model': [cpu_model],
        'warranty': [warranty],
        'screen size': [screen_size],
        'bluetooth': [1 if bluetooth else 0],
        'gps': [1 if gps else 0],
        'intel wireless display (widi)': [1 if widi else 0],
        'nfc (near field communication)': [1 if nfc else 0],
        'numeric keyboard': [1 if numeric_keyboard else 0],
        'touchpad': [1 if touchpad else 0],
        'camera': [1 if camera else 0],
        'microphone': [1 if microphone else 0],
        'sd card reader': [1 if sd_card_reader else 0],
        'speakers': [1 if speakers else 0],
        'wifi': [wifi],
        'operating_system': [operating_system],
        'modem': [modem],
        'lan': [lan],
        'keyboard': [keyboard],
        'CPU_Speed': [cpu_speed],
        'drive_memory_size(GB)': [drive_memory_size]
    })
    
    # Make prediction
    try:
        log_prediction = model.predict(input_data)
        
        # Apply exponential to get actual price (since target was log-transformed)
        predicted_price = np.exp(log_prediction[0])
        
        # Display result
        st.success("### Prediction Complete!")
        st.metric(
            label="Predicted Laptop Price",
            value=f"Rs. {predicted_price:,.2f}",
            delta=None
        )
        
        # Additional info
        st.markdown("#### Model Created by Rana Basak")
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.error("Please ensure the model file 'best_model.pkl' is in the same directory.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Powered by XGBoost Machine Learning Model</p>
        <p><small>Model applies exponential transformation to log-transformed predictions</small></p>
    </div>
    """,
    unsafe_allow_html=True

)




