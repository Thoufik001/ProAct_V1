import streamlit as st
import pandas as pd
from genai_helper import GenAIHelper
import plotly.express as px

class GenAIComponents:
    def __init__(self):
        self.genai_helper = GenAIHelper()
        
    def render_batch_insights(self, predictions_df):
        """Render insights for batch predictions"""
        try:
            with st.expander("🤖 AI Analysis & Insights", expanded=True):
                try:
                    with st.spinner("Generating insights..."):
                        insights = self.genai_helper.analyze_batch_predictions(predictions_df)
                        if insights and isinstance(insights, str):
                            st.markdown(insights)
                        else:
                            # Fallback to basic statistics if AI insights fail
                            high_risk = len(predictions_df[predictions_df['breakdown_probability'] > 0.7])
                            avg_prob = predictions_df['breakdown_probability'].mean()
                            st.markdown(f"""
                            ### Key Statistics
                            - {high_risk} machines have high risk (>70% probability)
                            - Average breakdown probability: {avg_prob:.1%}
                            - Total machines analyzed: {len(predictions_df)}
                            """)
                except Exception as e:
                    print(f"Error generating insights: {str(e)}")
                    st.warning("AI insights temporarily unavailable. Showing basic statistics instead.")
                    # Show basic statistics
                    high_risk = len(predictions_df[predictions_df['breakdown_probability'] > 0.7])
                    avg_prob = predictions_df['breakdown_probability'].mean()
                    st.markdown(f"""
                    ### Key Statistics
                    - {high_risk} machines have high risk (>70% probability)
                    - Average breakdown probability: {avg_prob:.1%}
                    - Total machines analyzed: {len(predictions_df)}
                    """)
                
                # Add maintenance recommendations with error handling
                st.subheader("🔧 Maintenance Recommendations")
                try:
                    with st.spinner("Generating recommendations..."):
                        recommendations = self.genai_helper.get_maintenance_recommendations(predictions_df)
                        if recommendations and isinstance(recommendations, str):
                            st.markdown(recommendations)
                        else:
                            # Fallback recommendations based on data
                            high_risk_machines = predictions_df[predictions_df['breakdown_probability'] > 0.7]
                            st.markdown(f"""
                            ### Priority Actions
                            1. Immediate inspection needed for {len(high_risk_machines)} high-risk machines
                            2. Monitor machines with rising risk trends
                            3. Regular maintenance checks for all equipment
                            """)
                except Exception as e:
                    print(f"Error generating recommendations: {str(e)}")
                    st.warning("AI recommendations temporarily unavailable. Showing basic guidance instead.")
                    # Show basic recommendations
                    high_risk_machines = predictions_df[predictions_df['breakdown_probability'] > 0.7]
                    st.markdown(f"""
                    ### Priority Actions
                    1. Immediate inspection needed for {len(high_risk_machines)} high-risk machines
                    2. Monitor machines with rising risk trends
                    3. Regular maintenance checks for all equipment
                    """)
        except Exception as e:
            print(f"Error in render_batch_insights: {str(e)}")
            st.error("An error occurred while analyzing the batch predictions. Please try again.")
    
    def render_chatbot(self, current_data=None):
        """Render enhanced chatbot interface for user queries"""
        st.subheader("💬 Ask Me Anything")
        
        # Initialize chat history and suggestions
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "suggestions" not in st.session_state:
            st.session_state.suggestions = [
                "What's the current machine status?",
                "Show me high-risk machines",
                "What are the maintenance recommendations?",
                "Analyze recent trends",
                "What are the common issues?"
            ]

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Quick action buttons
        if not st.session_state.messages:
            st.markdown("#### Quick Actions")
            cols = st.columns(3)
            with cols[0]:
                if st.button("📊 Machine Status"):
                    prompt = "What's the current machine status?"
                    self._process_chat(prompt, current_data)
            with cols[1]:
                if st.button("⚠️ Risk Analysis"):
                    prompt = "Show me high-risk machines"
                    self._process_chat(prompt, current_data)
            with cols[2]:
                if st.button("🔧 Maintenance"):
                    prompt = "What are the maintenance recommendations?"
                    self._process_chat(prompt, current_data)

        # Suggested queries
        if st.session_state.suggestions:
            st.markdown("#### Suggested Questions")
            for i in range(0, len(st.session_state.suggestions), 2):
                col1, col2 = st.columns(2)
                with col1:
                    if i < len(st.session_state.suggestions):
                        if st.button(f"🔍 {st.session_state.suggestions[i]}", key=f"sug_{i}"):
                            self._process_chat(st.session_state.suggestions[i], current_data)
                with col2:
                    if i + 1 < len(st.session_state.suggestions):
                        if st.button(f"🔍 {st.session_state.suggestions[i+1]}", key=f"sug_{i+1}"):
                            self._process_chat(st.session_state.suggestions[i+1], current_data)

        # Chat input
        if prompt := st.chat_input("Ask about machine status, maintenance, or predictions..."):
            self._process_chat(prompt, current_data)

    def _process_chat(self, prompt, current_data):
        """Process chat messages and update UI"""
        try:
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Generate and display response
            with st.chat_message("assistant"):
                if current_data is not None and not current_data.empty:
                    with st.spinner("Analyzing..."):
                        response = self.genai_helper.chat_response(current_data, prompt)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        # Update suggestions based on context
                        if 'risk' in prompt.lower():
                            st.session_state.suggestions = [
                                "What's causing the high risks?",
                                "Show me risk trends",
                                "Recommend preventive actions",
                                "Which machines need immediate attention?"
                            ]
                        elif 'maintenance' in prompt.lower():
                            st.session_state.suggestions = [
                                "What are the priority actions?",
                                "Show maintenance schedule",
                                "What issues need attention?",
                                "How can we prevent failures?"
                            ]
                        elif 'trend' in prompt.lower():
                            st.session_state.suggestions = [
                                "Show detailed trend analysis",
                                "What's causing these trends?",
                                "Predict future risks",
                                "How can we improve?"
                            ]
                else:
                    st.warning("No data available for analysis. Please ensure predictions are made first.")
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": "No data available for analysis. Please ensure predictions are made first."
                    })
        except Exception as e:
            print(f"Chat processing error: {e}")
            with st.chat_message("assistant"):
                st.error("I encountered an error. Please try again or rephrase your question.")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "I encountered an error. Please try again or rephrase your question."
                })
    
    def render_risk_analysis(self, predictions_df):
        """Render risk analysis dashboard for time series predictions"""
        with st.expander("🏥 Risk Analysis Dashboard", expanded=True):
            # Get latest prediction
            latest_prediction = predictions_df.iloc[-1]
            
            # Calculate risk metrics based on current reading
            risk_level = "High Risk" if latest_prediction['breakdown_probability'] > 0.7 else \
                        "Medium Risk" if latest_prediction['breakdown_probability'] > 0.3 else \
                        "Low Risk"
            
            # Display current status
            st.subheader("Current Status")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Risk Level", 
                    risk_level,
                    delta=f"{latest_prediction['breakdown_probability']:.1%}"
                )
            with col2:
                st.metric(
                    "Predicted Cause",
                    latest_prediction['major_cause_name']
                )
            
            # Display trend analysis
            st.subheader("Risk Trend Analysis")
            time_window = st.slider("Time Window (last N readings)", 
                                  min_value=5, 
                                  max_value=len(predictions_df), 
                                  value=min(20, len(predictions_df)))
            
            recent_data = predictions_df.tail(time_window)
            fig_trend = px.line(recent_data, 
                              y='breakdown_probability',
                              title='Breakdown Probability Trend',
                              labels={'breakdown_probability': 'Risk Level'})
            fig_trend.add_hline(y=0.7, line_dash="dash", line_color="red", 
                              annotation_text="High Risk Threshold")
            st.plotly_chart(fig_trend, use_container_width=True)
            
            # Display risk assessment
            trend_direction = 'increasing' if recent_data['breakdown_probability'].is_monotonic_increasing else \
                            'decreasing' if recent_data['breakdown_probability'].is_monotonic_decreasing else \
                            'fluctuating'
            
            prompt = f"""
            Analyze the current equipment status:
            - Current Risk Level: {risk_level} ({latest_prediction['breakdown_probability']:.1%})
            - Predicted Cause: {latest_prediction['major_cause_name']}
            - Risk Trend: {trend_direction} over the last {time_window} readings
            
            Provide a brief risk assessment focusing on:
            1. Current status severity
            2. Trend analysis implications
            3. Recommended monitoring/maintenance actions
            
            Keep the analysis to 3-4 sentences.
            """
            analysis = self.genai_helper.model.generate_content(prompt).text
            st.markdown("### Risk Assessment")
            st.markdown(analysis)
