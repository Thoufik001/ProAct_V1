import os
import google.generativeai as genai
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

class GenAIHelper:
    def __init__(self):
        # Configure Gemini API and initialize
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.model = genai.GenerativeModel('gemini-pro')
        self.cache = {}  # Cache for quick responses
        self.context = {
            'last_query': None,
            'machine_stats': None,
            'high_risk_machines': None,
            'common_issues': None,
            'trends': None
        }

    def _update_context(self, data):
        """Update context with latest data for quick responses"""
        try:
            self.context['machine_stats'] = {
                'total': len(data),
                'high_risk': len(data[data['breakdown_probability'] > 0.7]),
                'avg_prob': data['breakdown_probability'].mean(),
                'latest': data.iloc[-1].to_dict()
            }
            
            # Cache common issues
            self.context['common_issues'] = (
                data['major_cause_name'].value_counts()
                .head(3).to_dict()
            )
            
            # Quick trend analysis
            recent = data.tail(10)
            self.context['trends'] = {
                'rising_risk': recent['breakdown_probability'].is_monotonic_increasing,
                'avg_recent_risk': recent['breakdown_probability'].mean()
            }
            
        except Exception as e:
            print(f"Context update error: {e}")

    def _quick_response(self, query):
        """Generate quick responses without API calls"""
        stats = self.context.get('machine_stats', {})
        if not stats:
            return None
            
        query = query.lower()
        
        # Quick response patterns
        patterns = {
            'status': f"""Current Status:
- {stats['high_risk']} machines need attention
- Average risk: {stats.get('avg_prob', 0):.1%}
- Latest reading shows {stats['latest'].get('breakdown_probability', 0):.1%} risk""",

            'risk': f"""Risk Analysis:
- High risk machines: {stats['high_risk']}
- Current trend: {'⚠️ Rising' if self.context['trends']['rising_risk'] else '✅ Stable'}
- Recent average: {self.context['trends']['avg_recent_risk']:.1%}""",

            'cause': f"""Common Issues:
{chr(10).join(f'- {cause}: {count} instances' for cause, count in self.context['common_issues'].items())}""",

            'help': """I can help with:
- Machine status and risks
- Maintenance recommendations
- Trend analysis
- Specific machine details
Just ask naturally!""",

            'hi': "Hello! I'm your AI maintenance assistant. How can I help you analyze the machines today?"
        }

        # Check for pattern matches
        for key, response in patterns.items():
            if key in query:
                return response
                
        return None

    def chat_response(self, current_data, query):
        """Enhanced interactive chat with quick responses"""
        try:
            # Update context if data changed
            if self.context['last_query'] != query:
                self._update_context(current_data)
                self.context['last_query'] = query

            # Try quick response first
            quick_reply = self._quick_response(query)
            if quick_reply:
                return quick_reply

            # Process natural language queries
            query_type = self._classify_query(query)
            
            if query_type == 'machine_specific':
                return self._handle_machine_query(query, current_data)
            elif query_type == 'trend':
                return self._handle_trend_query(query, current_data)
            elif query_type == 'recommendation':
                return self._handle_recommendation_query(query, current_data)
            
            # Fallback to API for complex queries
            prompt = self._create_focused_prompt(query, current_data)
            response = self._generate_response(prompt, timeout=5)  # Reduced timeout
            
            if response:
                return response
            
            return f"I understand you're asking about {query_type}. Could you be more specific about what you'd like to know?"

        except Exception as e:
            print(f"Chat error: {e}")
            return "I'm having trouble understanding. Could you rephrase your question?"

    def _classify_query(self, query):
        """Quickly classify query type for focused responses"""
        query = query.lower()
        
        if any(word in query for word in ['machine', 'unit', 'equipment', 'specific', 'number']):
            return 'machine_specific'
        elif any(word in query for word in ['trend', 'pattern', 'history', 'time']):
            return 'trend'
        elif any(word in query for word in ['recommend', 'suggest', 'should', 'action']):
            return 'recommendation'
        elif any(word in query for word in ['risk', 'probability', 'chance']):
            return 'risk'
        else:
            return 'general'

    def _handle_machine_query(self, query, data):
        """Handle queries about specific machines"""
        try:
            high_risk = data[data['breakdown_probability'] > 0.7]
            return f"""Machine Analysis:
- Total machines: {len(data)}
- High risk machines: {len(high_risk)}
- Highest risk: {data['breakdown_probability'].max():.1%}
- Most common issue: {data['major_cause_name'].mode().iloc[0]}

Need specific details about any machine?"""
        except Exception as e:
            print(f"Machine query error: {e}")
            return "I couldn't analyze the specific machine data. What else would you like to know?"

    def _handle_trend_query(self, query, data):
        """Handle trend analysis queries"""
        try:
            recent = data.tail(10)
            trend = "increasing" if recent['breakdown_probability'].is_monotonic_increasing else "stable"
            return f"""Trend Analysis:
- Recent trend: {trend}
- Average risk: {recent['breakdown_probability'].mean():.1%}
- Risk range: {recent['breakdown_probability'].min():.1%} to {recent['breakdown_probability'].max():.1%}

Would you like to know about any specific patterns?"""
        except Exception as e:
            print(f"Trend analysis error: {e}")
            return "I couldn't analyze the trends. What else would you like to know?"

    def _handle_recommendation_query(self, query, data):
        """Handle recommendation queries"""
        try:
            high_risk = data[data['breakdown_probability'] > 0.7]
            causes = data['major_cause_name'].value_counts()
            
            return f"""Quick Recommendations:
1. Immediate: Check {len(high_risk)} high-risk machines
2. Focus on {causes.index[0]} issues ({causes.iloc[0]} instances)
3. Monitor machines with >50% risk

Need more specific recommendations?"""
        except Exception as e:
            print(f"Recommendation error: {e}")
            return "I couldn't generate recommendations. What specific aspect would you like to know about?"

    def _create_focused_prompt(self, query, data):
        """Create context-aware prompts for complex queries"""
        stats = self.context.get('machine_stats', {})
        return f"""Query: {query}
Context: Analyzing {stats.get('total', 0)} machines, {stats.get('high_risk', 0)} high risk.
Latest risk: {stats.get('latest', {}).get('breakdown_probability', 0):.1%}
Provide a brief, focused response."""

    def _generate_response(self, prompt, timeout=5):
        """Generate API response with shorter timeout"""
        try:
            response = self.model.generate_content(prompt, timeout=timeout)
            return response.text.strip() if response and response.text else None
        except Exception as e:
            print(f"API error: {e}")
            return None

    def analyze_batch_predictions(self, predictions_df):
        """
        Generate actionable insights from batch predictions interactively.
        """
        try:
            high_risk_count = len(predictions_df[predictions_df['breakdown_probability'] > 0.7])
            avg_prob = predictions_df['breakdown_probability'].mean()
            common_causes = predictions_df['major_cause_name'].value_counts().nlargest(3).to_dict()

            self.context["high_risk_count"] = high_risk_count
            self.context["avg_prob"] = avg_prob
            self.context["common_causes"] = common_causes

            # Construct interactive response
            message = (
                f"We analyzed the data:\n\n"
                f"- {high_risk_count} machines are at high risk (>70%).\n"
                f"- Average breakdown probability is {avg_prob:.1%}.\n"
                f"- Top causes include: {', '.join([f'{k}: {v}' for k, v in common_causes.items()])}.\n\n"
                "Would you like recommendations to address these risks?"
            )
            return message
        except Exception as e:
            print(f"Error in analyze_batch_predictions: {e}")
            return "Unable to analyze predictions. Please check the data format."

    def get_maintenance_recommendations(self, predictions_df):
        """
        Generate and provide interactive maintenance recommendations.
        """
        try:
            high_risk = len(predictions_df[predictions_df['breakdown_probability'] > 0.7])
            common_causes = predictions_df['major_cause_name'].value_counts().nlargest(3).to_dict()

            prompt = (
                f"There are {high_risk} machines with >70% breakdown risk.\n"
                f"Frequent causes include: {', '.join([f'{k}: {v}' for k, v in common_causes.items()])}.\n\n"
                "Suggest immediate, short-term, and long-term maintenance actions."
            )

            response = self._generate_response(prompt)
            if response:
                self.context["follow_up"] = (
                    "Would you like to focus on any specific cause or group of machines?"
                )
                return response + "\n\n" + self.context["follow_up"]
            return "Unable to generate recommendations right now."
        except Exception as e:
            print(f"Error in get_maintenance_recommendations: {e}")
            return "Failed to generate recommendations. Please check your data."

    def _generate_dataset_summary(self, dataset):
        """Create and cache a dataset summary."""
        if self.dataset_summary:
            return self.dataset_summary

        self.dataset_summary = {
            "records": len(dataset),
            "features": dataset.columns.tolist(),
            "ranges": {
                col: (dataset[col].min(), dataset[col].max())
                for col in ["Air temperature [K]", "Process temperature [K]",
                            "Rotational speed [rpm]", "Torque [Nm]"]
            },
        }
        return self.dataset_summary
