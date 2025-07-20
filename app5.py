#!/usr/bin/env python3
"""
Streamlit Chat Interface for Enhanced Titanic Analysis System
"""

import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List, Literal
from langchain_openai import AzureChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import numpy as np
import time
from pathlib import Path

# Load environment variables
load_dotenv()

# Configure LangSmith tracing - map LANGSMITH_ variables to LANGCHAIN_ variables
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "Titanic-AI-Analyst")
    print(f"🔍 LangSmith tracing enabled for project: {os.environ['LANGCHAIN_PROJECT']}")
else:
    print("⚠️ LANGSMITH_API_KEY not found - LangSmith tracing disabled")

# Streamlit page config
st.set_page_config(
    page_title="🚢 Titanic AI Chat Analyst",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    query: str
    analysis: str
    response: str
    chart_path: str
    route_decision: str
    chart_code: str

class EnhancedTitanicSystem:
    """Enhanced system with data agent, insights agent, and dynamic chart agent"""
    
    def __init__(self):
        self.llm = None
        self.df = None
        self.app = None
        self._setup_llm()
        if self.llm:
            self.app = self._build()
    
    def _setup_llm(self):
        """Setup Azure OpenAI with validation and LangSmith tracing"""
        required_vars = [
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT", 
            "AZURE_OPENAI_DEPLOYMENT_NAME"
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            st.error(f"❌ Missing environment variables: {', '.join(missing_vars)}")
            st.info("Please set these in your .env file")
            return
        
        # Check LangSmith configuration
        langsmith_status = "✅ Enabled" if os.getenv("LANGSMITH_API_KEY") else "❌ Disabled"
        project_name = os.getenv("LANGSMITH_PROJECT", "Titanic-AI-Analyst")
        
        try:
            self.llm = AzureChatOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                api_version="2024-12-01-preview",
                temperature=0.1
            )
            
            # Test the connection
            test_response = self.llm.invoke("Hello")
            st.success("✅ Azure OpenAI connection successful")
            st.info(f"🔍 LangSmith tracing: {langsmith_status} | Project: {project_name}")
            
        except Exception as e:
            st.error(f"❌ Azure OpenAI setup failed: {str(e)}")
            if "DeploymentNotFound" in str(e):
                st.warning(f"🔍 Check your deployment name: '{os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')}'")
                st.info("Verify it exists in Azure Portal → Azure OpenAI → Model deployments")
            self.llm = None
    
    def _build(self):
        """Build the graph with routing"""
        if not self.llm:
            return None
            
        graph = StateGraph(State)
        
        # Add nodes
        graph.add_node("data_agent", self._analyze)
        graph.add_node("router", self._route_decision)
        graph.add_node("insight_agent", self._insights)
        graph.add_node("chart_code_agent", self._generate_chart_code)
        graph.add_node("chart_agent", self._create_chart)
        
        # Add edges
        graph.add_edge(START, "data_agent")
        graph.add_edge("data_agent", "router")
        
        # Conditional routing
        graph.add_conditional_edges(
            "router",
            self._should_create_chart,
            {
                "chart": "chart_code_agent",
                "insight": "insight_agent"
            }
        )
        
        graph.add_edge("chart_code_agent", "chart_agent")
        graph.add_edge("insight_agent", END)
        graph.add_edge("chart_agent", END)
        
        return graph.compile()
    
    def _analyze(self, state: State) -> State:
        """Data analysis agent with error handling"""
        if self.df is None:
            return {"analysis": "❌ No data loaded. Please upload data first."}
        
        if not self.llm:
            return {"analysis": "❌ Azure OpenAI not configured properly"}
        
        try:
            agent = create_pandas_dataframe_agent(
                self.llm, 
                self.df, 
                verbose=False, 
                allow_dangerous_code=True,
                return_intermediate_steps=True
            )
            result = agent.invoke(state["query"])
            return {"analysis": result["output"]}
            
        except Exception as e:
            error_msg = f"❌ Analysis failed: {str(e)}"
            return {"analysis": error_msg}
    
    def _route_decision(self, state: State) -> State:
        """Determine if question asks for chart or insights"""
        if not self.llm:
            return {"route_decision": "insight"}
        
        try:
            prompt = f"""
            Analyze this question and determine if it's asking for a chart/visualization or insights/analysis:
            
            Question: "{state['query']}"
            
            Chart keywords: chart, plot, graph, visualize, show, display, histogram, scatter, bar chart, line plot, heatmap, boxplot, distribution, correlation
            Insight keywords: analyze, explain, what, why, how many, rate, percentage, insights, summary, calculate, find
            
            Respond with only one word: either "chart" or "insight"
            """
            
            response = self.llm.invoke(prompt).content.strip().lower()
            decision = "chart" if "chart" in response else "insight"
            return {"route_decision": decision}
            
        except Exception as e:
            return {"route_decision": "insight"}
    
    def _should_create_chart(self, state: State) -> Literal["chart", "insight"]:
        """Routing function for conditional edges"""
        return state.get("route_decision", "insight")
    
    def _insights(self, state: State) -> State:
        """Insights agent with error handling"""
        if not self.llm:
            response = "❌ Azure OpenAI not configured properly"
            return {"response": response, "messages": [AIMessage(content=response)]}
        
        if "❌" in state["analysis"]:
            return {"response": state["analysis"], "messages": [AIMessage(content=state["analysis"])]}
        
        try:
            prompt = f"""
            Question: {state['query']}
            Data Analysis Results: {state['analysis']}
            
            Based on this analysis, provide clear, actionable insights about the dataset.
            Be specific and reference the actual numbers from the analysis.
            Structure your response with key findings and implications.
            """
            
            response = self.llm.invoke(prompt).content
            return {"response": response, "messages": [AIMessage(content=response)]}
            
        except Exception as e:
            error_msg = f"❌ Insights generation failed: {str(e)}"
            return {"response": error_msg, "messages": [AIMessage(content=error_msg)]}
    
    def _generate_chart_code(self, state: State) -> State:
        """Generate chart code dynamically based on question and data"""
        if not self.llm:
            return {"chart_code": "# Error: Azure OpenAI not configured"}
        
        try:
            # Get data info
            data_info = self._get_data_info()
            
            prompt = f"""
            Generate Python code to create a chart/visualization for this question using matplotlib and seaborn.
            
            Question: "{state['query']}"
            Data Analysis: {state['analysis']}
            
            Dataset Information:
            {data_info}
            
            Requirements:
            1. Use the dataframe variable 'df' (already loaded)
            2. Create a figure with plt.figure(figsize=(12, 8))
            3. Use appropriate chart type based on the question
            4. Add proper title, labels, and formatting
            5. Use seaborn style: plt.style.use('seaborn-v0_8')
            6. Handle missing values appropriately with .dropna() when needed
            7. Use clear colors and readable fonts
            8. Add plt.tight_layout() at the end
            9. Do NOT include plt.show() or plt.savefig()
            10. **IMPORTANT: Do NOT include any import statements - all modules (plt, sns, pd, np) are already imported and available**
            
            Available chart types and when to use them:
            - Histogram: for distributions of continuous variables
            - Bar chart: for categorical data or counts
            - Scatter plot: for relationships between two continuous variables
            - Box plot: for distributions across categories
            - Heatmap: for correlation matrices
            - Line plot: for trends over time/ordered data
            - Pie chart: for proportions (use sparingly)
            
            Generate ONLY the Python code without any import statements, no explanations:
            """
            
            response = self.llm.invoke(prompt).content
            # Extract code from response (remove markdown if present)
            code = self._extract_code(response)
            
            # Remove any import statements from the generated code
            code_lines = code.split('\n')
            filtered_lines = []
            for line in code_lines:
                stripped_line = line.strip()
                if not (stripped_line.startswith('import ') or stripped_line.startswith('from ') or 
                       'import' in stripped_line and ('matplotlib' in stripped_line or 'seaborn' in stripped_line or 
                       'pandas' in stripped_line or 'numpy' in stripped_line)):
                    filtered_lines.append(line)
            
            code = '\n'.join(filtered_lines)
            return {"chart_code": code}
            
        except Exception as e:
            error_code = f"# Error generating chart code: {str(e)}"
            return {"chart_code": error_code}
    
    def _get_data_info(self) -> str:
        """Get information about the dataset for chart generation"""
        if self.df is None:
            return "No data available"
        
        info = f"""
        Shape: {self.df.shape}
        Columns: {list(self.df.columns)}
        Numeric columns: {list(self.df.select_dtypes(include=[np.number]).columns)}
        Categorical columns: {list(self.df.select_dtypes(include=['object']).columns)}
        Missing values: {dict(self.df.isnull().sum())}
        """
        return info
    
    def _extract_code(self, response: str) -> str:
        """Extract Python code from LLM response"""
        # Remove markdown code blocks if present
        if "```python" in response:
            start = response.find("```python") + 9
            end = response.find("```", start)
            code = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            code = response[start:end].strip()
        else:
            code = response.strip()
        
        return code
    
    def _create_chart(self, state: State) -> State:
        """Execute the generated chart code"""
        if self.df is None:
            response = "❌ No data loaded for chart creation"
            return {"response": response, "messages": [AIMessage(content=response)]}
        
        if "Error" in state["chart_code"]:
            response = f"❌ Chart code generation failed: {state['chart_code']}"
            return {"response": response, "messages": [AIMessage(content=response)]}
        
        try:
            # Set up the plotting environment
            plt.style.use('seaborn-v0_8')
            plt.figure(figsize=(12, 8))
            
            # Create safe execution environment with necessary built-ins for basic operations
            safe_builtins = {
                'len': len,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'list': list,
                'dict': dict,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'max': max,
                'min': min,
                'sum': sum,
                'sorted': sorted,
                'round': round
            }
            
            safe_globals = {
                'df': self.df,
                'plt': plt,
                'sns': sns,
                'pd': pd,
                'np': np,
                '__builtins__': safe_builtins  # Allow basic operations but not imports
            }
            
            # Execute the generated code
            exec(state["chart_code"], safe_globals)
            
            # Apply final formatting
            plt.tight_layout()
            
            # Save chart with timestamp for uniqueness
            timestamp = int(time.time())
            chart_path = f"chart_{timestamp}_{hash(state['query']) % 10000}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            response = f"""
            📊 **Dynamic Chart Generated Successfully!**
            
            **Question:** {state['query']}
            
            The visualization was created dynamically based on your question and the dataset characteristics.
            """
            
            return {
                "response": response,
                "chart_path": chart_path,
                "chart_code": state["chart_code"],
                "messages": [AIMessage(content=response)]
            }
            
        except Exception as e:
            error_msg = f"""
            ❌ **Chart execution failed:** {str(e)}
            
            **Generated Code:**
            ```python
            {state['chart_code']}
            ```
            
            Please check the code for syntax errors or data compatibility issues.
            """
            return {"response": error_msg, "messages": [AIMessage(content=error_msg)]}
    
    def load_data(self, file_path_or_url: str = None, uploaded_file=None):
        """Load data with error handling"""
        try:
            if uploaded_file is not None:
                # Handle uploaded file
                self.df = pd.read_csv(uploaded_file)
                st.success(f"✅ Data loaded successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            else:
                # Use default Titanic URL if no file provided
                url = file_path_or_url or "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
                self.df = pd.read_csv(url)
                st.success(f"✅ Data loaded successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            
            st.info(f"📊 Columns: {', '.join(self.df.columns.tolist())}")
            return True
            
        except Exception as e:
            st.error(f"❌ Failed to load data: {str(e)}")
            self.df = None
            return False
    
    def ask(self, question: str) -> dict:
        """Ask question with validation and tracing"""
        if not self.app:
            return {"error": "❌ System not initialized properly. Check Azure OpenAI configuration."}
        
        if self.df is None:
            return {"error": "❌ No data loaded. Please upload data first."}
        
        try:
            # Add metadata for tracing
            run_metadata = {
                "user_query": question,
                "dataset_shape": str(self.df.shape) if self.df is not None else "No data",
                "dataset_columns": list(self.df.columns) if self.df is not None else [],
                "session_id": st.session_state.get("session_id", "unknown"),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Execute workflow with tracing context
            result = self.app.invoke(
                {
                    "query": question, 
                    "analysis": "", 
                    "response": "", 
                    "messages": [],
                    "chart_path": "",
                    "route_decision": "",
                    "chart_code": ""
                },
                config={
                    "metadata": run_metadata,
                    "tags": ["titanic-analysis", "streamlit-app"],
                    "run_name": f"Query: {question[:50]}..."
                }
            )
            return result
            
        except Exception as e:
            return {"error": f"❌ Query failed: {str(e)}"}
    
    def get_status(self):
        """Get system status including LangSmith tracing"""
        return {
            "llm_ready": self.llm is not None,
            "data_loaded": self.df is not None,
            "graph_ready": self.app is not None,
            "data_shape": self.df.shape if self.df is not None else None,
            "langsmith_enabled": os.getenv("LANGSMITH_API_KEY") is not None,
            "langsmith_project": os.getenv("LANGSMITH_PROJECT", "Titanic-AI-Analyst"),
            "tracing_endpoint": os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
        }

def initialize_session_state():
    """Initialize Streamlit session state"""
    if "system" not in st.session_state:
        st.session_state.system = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "session_id" not in st.session_state:
        # Generate unique session ID for tracing
        st.session_state.session_id = f"session_{int(time.time())}_{hash(time.time()) % 10000}"

def display_sidebar():
    """Display sidebar with system controls"""
    with st.sidebar:
        st.header("🚢 Titanic AI Analyst")
        
        # System status
        if st.session_state.system:
            status = st.session_state.system.get_status()
            st.subheader("System Status")
            st.write(f"🤖 AI Model: {'✅' if status['llm_ready'] else '❌'}")
            st.write(f"📊 Data: {'✅' if status['data_loaded'] else '❌'}")
            st.write(f"🔧 Graph: {'✅' if status['graph_ready'] else '❌'}")
            
            # LangSmith tracing status
            langsmith_enabled = os.getenv("LANGSMITH_API_KEY") is not None
            st.write(f"🔍 LangSmith: {'✅' if langsmith_enabled else '❌'}")
            
            if langsmith_enabled:
                project_name = os.getenv("LANGSMITH_PROJECT", "Titanic-AI-Analyst")
                st.caption(f"Project: {project_name}")
                st.caption(f"Session: {st.session_state.get('session_id', 'unknown')}")
            
            if status['data_shape']:
                st.write(f"📈 Dataset: {status['data_shape'][0]} rows, {status['data_shape'][1]} columns")
        
        st.divider()
        
        # LangSmith Tracing Information
        if os.getenv("LANGSMITH_API_KEY"):
            st.subheader("🔍 LangSmith Tracing")
            project_name = os.getenv("LANGSMITH_PROJECT", "Titanic-AI-Analyst")
            st.success(f"✅ Active - Project: {project_name}")
            st.caption("Monitor your AI agents in real-time")
            
            # LangSmith dashboard link
            if st.button("🌐 Open LangSmith Dashboard", use_container_width=True):
                st.link_button(
                    "Go to LangSmith", 
                    "https://smith.langchain.com",
                    use_container_width=True
                )
        else:
            st.subheader("🔍 LangSmith Tracing")
            st.warning("❌ Not configured")
            with st.expander("📖 Setup Instructions"):
                st.markdown("""
                **To enable LangSmith tracing:**
                
                1. Sign up at [smith.langchain.com](https://smith.langchain.com)
                2. Get your API key from Settings
                3. Add to your .env file:
                   ```
                   LANGSMITH_API_KEY=your_api_key_here
                   LANGSMITH_PROJECT=Titanic-AI-Analyst
                   LANGSMITH_ENDPOINT=https://api.smith.langchain.com
                   LANGSMITH_TRACING=true
                   ```
                4. Restart the application
                
                **Benefits:**
                - 🔍 Trace AI agent decisions
                - 📊 Monitor costs and performance  
                - 🐛 Debug issues easily
                - 📈 Analyze usage patterns
                """)
        
        st.divider()
        
        # Data loading section
        st.subheader("📁 Data Loading")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV file", 
            type=['csv'],
            help="Upload your own CSV dataset for analysis"
        )
        
        # Load default Titanic data
        if st.button("Load Default Titanic Data"):
            if st.session_state.system:
                if st.session_state.system.load_data():
                    st.session_state.data_loaded = True
                    st.rerun()
        
        # Load uploaded file
        if uploaded_file and st.button("Load Uploaded File"):
            if st.session_state.system:
                if st.session_state.system.load_data(uploaded_file=uploaded_file):
                    st.session_state.data_loaded = True
                    st.rerun()
        
        st.divider()
        
        # Sample questions
        st.subheader("💡 Sample Questions")
        sample_questions = [
            "What is the survival rate?",
            "Show survival by passenger class",
            "Create age distribution histogram",
            "Plot fare vs age correlation",
            "Show me a heatmap of correlations",
            "Analyze survival by gender",
            "Create boxplot of fare by class"
        ]
        
        for question in sample_questions:
            if st.button(question, key=f"sample_{question}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": question})
                st.rerun()
        
        # Clear chat
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

def main():
    """Main Streamlit app"""
    initialize_session_state()
    
    # Initialize system if not done
    if st.session_state.system is None:
        with st.spinner("🔧 Initializing AI system..."):
            st.session_state.system = EnhancedTitanicSystem()
    
    # Display sidebar
    display_sidebar()
    
    # Main content area
    st.title("🚢 Titanic AI Chat Analyst")
    st.markdown("Ask questions about the Titanic dataset and get intelligent analysis with dynamic visualizations!")
    
    # Check system status
    if not st.session_state.system or not st.session_state.system.get_status()["llm_ready"]:
        st.error("❌ AI system not properly initialized. Please check your Azure OpenAI configuration.")
        st.info("Make sure you have set the required environment variables in your .env file")
        return
    
    # Display data preview if loaded
    if st.session_state.data_loaded and st.session_state.system.df is not None:
        with st.expander("📊 Data Preview", expanded=False):
            st.dataframe(st.session_state.system.df.head())
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", st.session_state.system.df.shape[0])
            with col2:
                st.metric("Columns", st.session_state.system.df.shape[1])
            with col3:
                st.metric("Missing Values", st.session_state.system.df.isnull().sum().sum())
    
    # Chat interface
    st.subheader("💬 Chat with your data")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display chart if present
            if message["role"] == "assistant" and "chart_path" in message:
                if message["chart_path"] and Path(message["chart_path"]).exists():
                    st.image(message["chart_path"], caption="Generated Visualization")
                    
                    # Show generated code if available
                    if "chart_code" in message and message["chart_code"]:
                        with st.expander("🔍 View Generated Code"):
                            st.code(message["chart_code"], language="python")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your data..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing your question..."):
                # Show tracing info if enabled
                if os.getenv("LANGSMITH_API_KEY"):
                    with st.expander("🔍 LangSmith Tracing Active", expanded=False):
                        st.info(f"**Project**: {os.getenv('LANGSMITH_PROJECT', 'Titanic-AI-Analyst')}")
                        st.info(f"**Session**: {st.session_state.get('session_id', 'unknown')}")
                        st.caption("This query will be traced in LangSmith for debugging and monitoring.")
                
                result = st.session_state.system.ask(prompt)
                
                if "error" in result:
                    st.error(result["error"])
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": result["error"]
                    })
                else:
                    # Display response
                    st.markdown(result["response"])
                    
                    # Display chart if generated
                    chart_displayed = False
                    if result.get("chart_path") and Path(result["chart_path"]).exists():
                        st.image(result["chart_path"], caption="Generated Visualization")
                        chart_displayed = True
                        
                        # Show generated code
                        if result.get("chart_code"):
                            with st.expander("🔍 View Generated Code"):
                                st.code(result["chart_code"], language="python")
                    
                    # Add assistant message to history
                    assistant_message = {
                        "role": "assistant", 
                        "content": result["response"]
                    }
                    
                    if chart_displayed:
                        assistant_message["chart_path"] = result["chart_path"]
                        assistant_message["chart_code"] = result.get("chart_code", "")
                    
                    st.session_state.messages.append(assistant_message)

if __name__ == "__main__":
    main()