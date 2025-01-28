import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as plt
import plotly.express as px
import io
import numpy as np
import time
from genai_components import GenAIComponents

# Initialize GenAI components
genai_components = GenAIComponents()

st.set_page_config(
    page_title="ProAct", 
    page_icon="🛠",  # Custom emoji favicon
    layout="wide"
)

st.title("ProAct - AI Predictive Maintenance Dashboard")

st.write("""
This dashboard provides real-time predictive maintenance analysis for industrial equipment.
It uses machine learning to predict potential breakdowns and their causes based on operational parameters.
""")

# Add tabs for single and batch prediction
tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

def make_prediction(input_data, max_retries=3):
    base_url = "http://localhost:8000"
    for attempt in range(max_retries):
        try:
            response = requests.post(f"{base_url}/predict", json=input_data, timeout=10)
            response.raise_for_status()
            return response.json()
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt < max_retries - 1:
                st.warning(f"Connection attempt {attempt + 1} failed. Retrying...")
                time.sleep(2)
            else:
                st.error(f"Failed to connect to prediction server: {str(e)}")
                raise

def make_batch_prediction(file, max_retries=3):
    base_url = "http://localhost:8000"
    for attempt in range(max_retries):
        try:
            # Ensure the file is at the beginning
            file.seek(0)
            
            # Prepare files for upload
            files = {'file': ('data.csv', file, 'text/csv')}
            
            # Make the prediction request
            response = requests.post(f"{base_url}/batch-predict", files=files, timeout=30)
            response.raise_for_status()
            return response.json()
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt < max_retries - 1:
                st.warning(f"Batch prediction attempt {attempt + 1} failed. Retrying...")
                time.sleep(2)
            else:
                st.error(f"Failed to connect to prediction server: {str(e)}")
                raise

API_BASE_URL = "http://localhost:8000"

def generate_feature_importance(results_df):
    """
    Generate feature importance and correlation insights
    """
    # Select numerical features
    features = ['air_temperature', 'process_temperature', 
                'rotational_speed', 'torque', 'tool_wear']
    
    # Correlation with breakdown probability
    correlations = results_df[features + ['breakdown_probability']].corr()['breakdown_probability'][:-1]
    
    # Visualize feature importance
    fig_importance = px.bar(
        x=correlations.index, 
        y=abs(correlations.values),
        title='Feature Importance for Breakdown Probability',
        labels={'x': 'Features', 'y': 'Absolute Correlation Strength'}
    )
    return fig_importance

def generate_breakdown_insights(results_df):
    """
    Generate insights and recommendations based on batch prediction results
    """
    # Breakdown probability distribution
    fig_breakdown_dist = px.line(
        results_df.sort_values('breakdown_probability'), 
        x=results_df.index, 
        y='breakdown_probability', 
        title='Breakdown Probability Distribution',
        labels={'x': 'Sample Index', 'breakdown_probability': 'Breakdown Probability'}
    )
    
    # Add horizontal line for high-risk threshold
    fig_breakdown_dist.add_hline(
        y=0.7, 
        line_dash="dash", 
        line_color="red", 
        annotation_text="High Risk Threshold (0.7)"
    )
    
    return fig_breakdown_dist

def generate_maintenance_recommendations(results_df):
    """
    Generate maintenance recommendations based on prediction results
    """
    # High-risk samples analysis
    high_risk_samples = results_df[results_df['breakdown_probability'] > 0.7]
    
    # Insights generation
    insights = []
    
    # Temperature insights
    temp_diff = results_df['process_temperature'] - results_df['air_temperature']
    if temp_diff.mean() > 10:
        insights.append("🌡️ High temperature differential detected. Consider improving cooling systems.")
    
    # Rotational speed insights
    if results_df['rotational_speed'].mean() > 1800:
        insights.append("⚙️ High average rotational speed. Recommend regular lubrication and bearing checks.")
    
    # Tool wear insights
    if results_df['tool_wear'].mean() > 100:
        insights.append("🛠️ Significant tool wear observed. Consider more frequent tool replacements.")
    
    # Breakdown probability insights
    breakdown_rate = len(high_risk_samples) / len(results_df) * 100
    if breakdown_rate > 20:
        insights.append(f"⚠️ High breakdown risk: {breakdown_rate:.2f}% of machines show elevated failure probability.")
    
    return insights, high_risk_samples

def generate_feature_line_graphs(results_df):
    """
    Generate line graphs for key features and predictions
    """
    # Ensure UDI is available, if not create a sequential index
    if 'UDI' not in results_df.columns:
        results_df['UDI'] = range(len(results_df))
    
    # Prepare line graphs for different features
    feature_graphs = []
    
    # 1. Breakdown and Probability Line Graph
    fig_breakdown = px.line(
        results_df, 
        x='UDI', 
        y=['breakdown_prediction', 'breakdown_probability'],
        title='Breakdown Prediction and Probability by UDI',
        labels={'value': 'Value', 'variable': 'Metric'},
        color_discrete_map={
            'breakdown_prediction': 'red', 
            'breakdown_probability': 'blue'
        }
    )
    feature_graphs.append(fig_breakdown)
    
    # 2. Temperature Line Graphs
    fig_temp = px.line(
        results_df, 
        x='UDI', 
        y=['air_temperature', 'process_temperature'],
        title='Temperature Variations by UDI',
        labels={'value': 'Temperature (K)', 'variable': 'Temperature Type'},
        color_discrete_map={
            'air_temperature': 'green', 
            'process_temperature': 'orange'
        }
    )
    feature_graphs.append(fig_temp)
    
    # 3. Rotational Speed and Torque Line Graph
    fig_speed_torque = px.line(
        results_df, 
        x='UDI', 
        y=['rotational_speed', 'torque'],
        title='Rotational Speed and Torque by UDI',
        labels={'value': 'Value', 'variable': 'Metric'},
        color_discrete_map={
            'rotational_speed': 'purple', 
            'torque': 'brown'
        }
    )
    feature_graphs.append(fig_speed_torque)
    
    # 4. Tool Wear Line Graph
    fig_tool_wear = px.line(
        results_df, 
        x='UDI', 
        y='tool_wear',
        title='Tool Wear by UDI',
        labels={'tool_wear': 'Tool Wear (min)'},
        color_discrete_sequence=['magenta']
    )
    feature_graphs.append(fig_tool_wear)
    
    return feature_graphs

def generate_visualizations(results_df):
    """Generate all visualizations for the batch predictions"""
    
    # Create risk level trend
    time_window = min(20, len(results_df))
    recent_data = results_df.tail(time_window)
    
    fig_trend = px.line(
        recent_data,
        y='breakdown_probability',
        title=f'Risk Level Trend (Last {time_window} Readings)',
        labels={'breakdown_probability': 'Risk Level'}
    )
    fig_trend.add_hline(y=0.7, line_dash="dash", line_color="red", 
                       annotation_text="High Risk Threshold")
    
    # Create feature correlation heatmap
    feature_cols = ['air_temperature', 'process_temperature', 'rotational_speed', 'torque', 'tool_wear']
    correlation_matrix = results_df[feature_cols].corr()
    
    fig_correlations = px.imshow(
        correlation_matrix,
        title='Feature Correlation Heatmap',
        labels=dict(color="Correlation"),
        color_continuous_scale='RdBu'
    )
    
    # Create parameter box plots
    fig_params = px.box(
        results_df,
        y=feature_cols,
        title='Parameter Distributions',
        labels={'variable': 'Parameter', 'value': 'Value'}
    )
    
    return fig_trend, fig_correlations, fig_params

def batch_prediction_page():
    st.header("Batch Prediction")
    
    # Show data format instructions
    with st.expander("📋 Data Format Instructions", expanded=True):
        st.markdown("""
        ### Required CSV Format
        Your CSV file should contain the following columns (case-insensitive):
        
        | Column Name | Alternative Names | Description | Example |
        |------------|-------------------|-------------|---------|
        | air_temperature | Air Temp, AirTemp, Temperature [K] | Air temperature in Kelvin | 298.1 |
        | process_temperature | Process Temp, ProcessTemp, Process Temperature [K] | Process temperature in Kelvin | 308.6 |
        | rotational_speed | RPM, Speed, Rotational Speed [rpm] | Rotational speed in RPM | 1500 |
        | torque | Torque [Nm], TorqueNm | Torque in Newton meters | 40 |
        | tool_wear | Wear, Tool Wear [min], ToolWear | Tool wear in minutes | 0 |
        | type | Machine Type, MachineType | Machine type (integer) | 0 |
        """)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload CSV file with time series data", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            input_df = pd.read_csv(uploaded_file)
            
            # Define column name mappings (case-insensitive)
            column_mappings = {
                'air_temperature': ['air temp', 'airtemp', 'temperature [k]', 'air temperature [k]', 'air_temp'],
                'process_temperature': ['process temp', 'processtemp', 'process temperature [k]', 'process_temp'],
                'rotational_speed': ['rpm', 'speed', 'rotational speed [rpm]', 'rotation_speed'],
                'torque': ['torque [nm]', 'torquenm'],
                'tool_wear': ['wear', 'tool wear [min]', 'toolwear'],
                'type': ['machine type', 'machinetype']
            }
            
            # Convert column names to lowercase for matching
            input_df.columns = [col.lower().strip() for col in input_df.columns]
            
            # Try to map columns to expected names
            column_rename = {}
            for expected_name, alternatives in column_mappings.items():
                # Check if the exact name exists
                if expected_name in input_df.columns:
                    continue
                # Check alternatives
                for alt in alternatives:
                    if alt in input_df.columns:
                        column_rename[alt] = expected_name
                        break
            
            # Rename columns if matches found
            if column_rename:
                input_df = input_df.rename(columns=column_rename)
            
            # Check for missing columns
            required_columns = ['air_temperature', 'process_temperature', 'rotational_speed', 'torque', 'tool_wear', 'type']
            missing_columns = [col for col in required_columns if col not in input_df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                st.markdown("""
                ⚠️ Your CSV file is missing some required columns. Please check:
                1. Column names match the required format (see Data Format Instructions above)
                2. All required columns are present
                3. Column names don't have extra spaces or special characters
                
                Need help? Click 'Data Format Instructions' above for details and examples.
                """)
                
                # Show current columns for debugging
                st.warning("Your CSV file contains these columns:")
                st.write(list(input_df.columns))
                return
            
            with st.spinner("Processing batch predictions..."):
                try:
                    # Ensure numeric data types
                    for col in required_columns:
                        input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
                    
                    # Reset the file pointer
                    uploaded_file.seek(0)
                    
                    # Make batch prediction
                    predictions = make_batch_prediction(uploaded_file)
                    
                    if predictions and 'results' in predictions:
                        # Convert results to DataFrame
                        results_df = pd.DataFrame(predictions['results'])
                        
                        # Display summary statistics
                        st.subheader("Current Status")
                        latest = results_df.iloc[-1]
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Current Risk Level",
                                f"{latest['breakdown_probability']:.1%}",
                                delta="High Risk" if latest['breakdown_probability'] > 0.7 else "Normal"
                            )
                        with col2:
                            st.metric("Predicted Cause", latest['major_cause_name'])
                        with col3:
                            readings = len(results_df)
                            st.metric("Total Readings", readings)
                        
                        try:
                            # Display AI insights and recommendations
                            genai_components.render_batch_insights(results_df)
                        except Exception as insight_error:
                            print(f"Error rendering insights: {str(insight_error)}")
                            st.warning("AI insights temporarily unavailable. Showing data analysis only.")
                        
                        try:
                            # Display risk analysis
                            genai_components.render_risk_analysis(results_df)
                        except Exception as risk_error:
                            print(f"Error rendering risk analysis: {str(risk_error)}")
                            st.warning("Risk analysis temporarily unavailable.")
                        
                        # Display visualizations
                        st.subheader("📊 Data Analysis")
                        try:
                            fig_trend, fig_correlations, fig_params = generate_visualizations(results_df)
                            
                            viz_tab1, viz_tab2, viz_tab3 = st.tabs([
                                "Risk Trend",
                                "Feature Correlations",
                                "Parameter Analysis"
                            ])
                            
                            with viz_tab1:
                                st.plotly_chart(fig_trend, use_container_width=True)
                            with viz_tab2:
                                st.plotly_chart(fig_correlations, use_container_width=True)
                            with viz_tab3:
                                st.plotly_chart(fig_params, use_container_width=True)
                        except Exception as viz_error:
                            print(f"Error generating visualizations: {str(viz_error)}")
                            st.error("Unable to generate visualizations. Please check the data format.")
                    else:
                        st.error("Error processing predictions. Please check the data format and try again.")

                    # Display chatbot after predictions
                    st.markdown("---")
                    genai_components.render_chatbot(results_df)
                    
                    # Display predictions table
                    st.subheader("Detailed Readings")
                    st.dataframe(results_df)
                    
                    # Add download button for results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "Download Predictions CSV",
                        csv,
                        "time_series_predictions.csv",
                        "text/csv",
                        key='download-csv'
                    )

                except Exception as pred_error:
                    print(f"Error in batch prediction: {str(pred_error)}")
                    st.error("Error processing batch prediction. Please ensure your data is in the correct format and try again.")
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            st.error("Error reading the uploaded file. Please ensure it's a valid CSV file with the required columns.")

def single_prediction_page():
    st.header("Single Machine Prediction")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            air_temp = st.number_input("Air Temperature (K)", min_value=290.0, max_value=320.0, value=300.0)
            process_temp = st.number_input("Process Temperature (K)", min_value=305.0, max_value=320.0, value=310.0)
            rot_speed = st.number_input("Rotational Speed (rpm)", min_value=1000, max_value=3000, value=1500)
        
        with col2:
            torque = st.number_input("Torque (Nm)", min_value=3.0, max_value=77.0, value=40.0)
            tool_wear = st.number_input("Tool Wear (min)", min_value=0, max_value=250, value=0)
            machine_type = st.selectbox("Machine Type", options=[0, 1, 2], format_func=lambda x: f"Type {x}")
        
        submitted = st.form_submit_button("Make Prediction")
        
    if submitted:
        input_data = {
            "air_temperature": air_temp,
            "process_temperature": process_temp,
            "rotational_speed": rot_speed,
            "torque": torque,
            "tool_wear": tool_wear,
            "type": machine_type
        }
        
        try:
            prediction = make_prediction(input_data)
            
            if prediction:
                st.subheader("Prediction Results")
                
                # Create three columns for metrics
                col1, col2, col3 = st.columns(3)
                
                # Breakdown Probability with gauge chart
                with col1:
                    fig = plt.Figure(data=[plt.Indicator(
                        mode="gauge+number+delta",
                        value=prediction['breakdown_probability'] * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Breakdown Probability (%)"},
                        delta={'reference': 70, 'increasing': {'color': "red"}},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 30], 'color': "green"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 70
                            }
                        }
                    )])
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.metric(
                        "Risk Level",
                        "High Risk" if prediction['breakdown_probability'] > 0.7 else "Low Risk",
                        delta="⚠️ Attention Needed" if prediction['breakdown_probability'] > 0.7 else "✅ Normal",
                        delta_color="inverse"
                    )
                
                with col3:
                    st.metric(
                        "Predicted Cause",
                        prediction['major_cause_name'],
                        delta=None
                    )
                
                # Parameter Analysis
                st.subheader("Parameter Analysis")
                params_df = pd.DataFrame({
                    'Parameter': ['Air Temperature', 'Process Temperature', 'Rotational Speed', 'Torque', 'Tool Wear'],
                    'Value': [air_temp, process_temp, rot_speed, torque, tool_wear],
                    'Status': ['Normal'] * 5
                })
                
                # Update status based on thresholds
                if process_temp - air_temp > 25:
                    params_df.loc[params_df['Parameter'].isin(['Air Temperature', 'Process Temperature']), 'Status'] = 'Warning'
                if rot_speed > 2000:
                    params_df.loc[params_df['Parameter'] == 'Rotational Speed', 'Status'] = 'Warning'
                if torque > 70:
                    params_df.loc[params_df['Parameter'] == 'Torque', 'Status'] = 'Warning'
                if tool_wear > 200:
                    params_df.loc[params_df['Parameter'] == 'Tool Wear', 'Status'] = 'Warning'
                
                # Create parameter status chart
                fig = px.bar(
                    params_df,
                    x='Parameter',
                    y='Value',
                    color='Status',
                    color_discrete_map={'Normal': 'green', 'Warning': 'orange'},
                    title='Parameter Status Analysis'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display AI insights and chatbot
                genai_components.render_batch_insights(pd.DataFrame([prediction]))
                genai_components.render_chatbot(current_data=pd.DataFrame([prediction]))
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

# Display appropriate page based on selected tab
with tab1:
    single_prediction_page()
    
with tab2:
    batch_prediction_page()
