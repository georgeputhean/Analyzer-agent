# 🚢 Titanic AI Chat Analyst

An intelligent data analysis chatbot built with Streamlit, LangChain, and Azure OpenAI that provides conversational insights and dynamic visualizations for the Titanic dataset (or any CSV data).

## ✨ Features

### 🤖 **Intelligent Multi-Agent System**
- **Data Analysis Agent**: Performs pandas-based data analysis
- **Router Agent**: Intelligently routes between chart generation and insights
- **Insights Agent**: Converts raw analysis into human-readable insights
- **Chart Code Generator**: Dynamically creates Python visualization code
- **Chart Execution Agent**: Safely executes generated code to create visualizations

### 📊 **Dynamic Visualizations**
- **Smart Chart Selection**: Automatically chooses appropriate chart types based on your question
- **Custom Code Generation**: Creates matplotlib/seaborn code tailored to your specific query
- **Multiple Chart Types**: Histograms, scatter plots, bar charts, heatmaps, box plots, and more
- **Interactive Code Viewing**: See the generated Python code behind each visualization

### 🔍 **Advanced Monitoring**
- **LangSmith Integration**: Full tracing and monitoring of AI agent decisions
- **Session Tracking**: Conversation grouping and user journey analysis
- **Performance Metrics**: Token usage, latency, and cost monitoring
- **Error Tracking**: Comprehensive debugging capabilities

### 💬 **User-Friendly Interface**
- **Chat-Based Interaction**: Natural language queries about your data
- **Real-Time Responses**: Instant analysis and visualization generation
- **Sample Questions**: Pre-built queries to get you started quickly
- **Data Upload**: Support for custom CSV files or default Titanic dataset

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Azure OpenAI account with GPT-4 deployment
- (Optional) LangSmith account for tracing

### 1. Clone the Repository
```bash
git clone <repository-url>
cd titanic-ai-analyst
```

### 2. Install Dependencies
```bash
pip install streamlit pandas matplotlib seaborn python-dotenv
pip install langchain-openai langchain-experimental langgraph
pip install numpy pathlib
```

### 3. Set Up Environment Variables

Create a `.env` file in the project root:

```env
# Azure OpenAI Configuration (Required)
AZURE_OPENAI_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_DEPLOYMENT_NAME=your_gpt4_deployment_name

# LangSmith Tracing Configuration (Optional)
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=your_project_name
```

### 4. Run the Application
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 🚀 Usage

### Getting Started

1. **Launch the Application**: Run `streamlit run app.py`
2. **Load Data**: 
   - Click "Load Default Titanic Data" for the classic dataset
   - Or upload your own CSV file using the file uploader
3. **Start Asking Questions**: Use natural language to explore your data

### Example Questions

#### 📈 **For Insights & Analysis**
- "What is the survival rate?"
- "How many passengers were in each class?"
- "What's the average age of survivors vs non-survivors?"
- "Analyze survival patterns by gender and class"
- "Calculate the fare statistics by passenger class"

#### 📊 **For Visualizations**
- "Show me a histogram of passenger ages"
- "Create a bar chart of survival by class"
- "Plot the correlation heatmap"
- "Show fare distribution by class as a boxplot"
- "Visualize survival rate by gender"

### Advanced Features

#### **Custom Data Upload**
1. Click the file uploader in the sidebar
2. Select your CSV file
3. Click "Load Uploaded File"
4. Start analyzing your custom dataset

#### **LangSmith Monitoring**
- View real-time traces of AI agent decisions
- Monitor performance and costs
- Debug issues with detailed logs
- Click "🌐 Open LangSmith Dashboard" to access the monitoring interface

## 🏗️ Technical Architecture

### Multi-Agent Workflow
```
User Question → Data Agent → Router → [Chart Path OR Insight Path] → Response
                                   ↘                    ↗
                                    Chart Code Agent → Chart Execution Agent
                                   ↗                    ↘
                                  Insight Agent ────────→ Final Response
```

### Key Components

- **StateGraph (LangGraph)**: Orchestrates the multi-agent workflow
- **Azure OpenAI**: Powers the intelligent agents with GPT-4
- **Pandas DataFrame Agent**: Enables natural language data analysis
- **Dynamic Code Generation**: Creates custom visualization code
- **Safe Code Execution**: Sandboxed environment for running generated code
- **Streamlit Interface**: Provides the chat-based user experience

### Security Features

- **Restricted Execution Environment**: Limited built-in functions for safety
- **Input Validation**: Comprehensive error handling and validation
- **Environment Variable Security**: API keys stored securely in `.env`
- **Code Filtering**: Removes potentially dangerous import statements

## 🔍 LangSmith Setup (Optional)

LangSmith provides powerful monitoring and debugging capabilities for your AI agents.

### 1. Sign Up
Visit [smith.langchain.com](https://smith.langchain.com) and create an account

### 2. Get API Key
- Go to Settings → API Keys
- Create a new API key
- Copy the key (starts with `lsv2_pt_...`)

### 3. Configure Environment
Add to your `.env` file:
```env
LANGSMITH_API_KEY=lsv2_pt_your_api_key_here
LANGSMITH_PROJECT=titanic-ai-analyst
```

### 4. Monitor Your Application
- Restart the application
- All AI agent interactions will be traced
- View traces in the LangSmith dashboard
- Analyze performance, costs, and user patterns

## 🛠️ Customization

### Adding New Chart Types
Modify the `_generate_chart_code` method to include new visualization types:

```python
# Add to the chart types section in the prompt
- Custom chart type: for specific use cases
```

### Extending Agent Capabilities
Add new agents to the workflow:

```python
# In the _build method
graph.add_node("new_agent", self._new_agent_method)
graph.add_edge("existing_agent", "new_agent")
```

### Custom Data Sources
Extend the `load_data` method to support:
- Database connections
- API integrations
- Multiple file formats (Excel, JSON, etc.)

## 🐛 Troubleshooting

### Common Issues

#### "❌ Azure OpenAI setup failed"
- **Check credentials**: Verify your Azure OpenAI endpoint, API key, and deployment name
- **Verify deployment**: Ensure your GPT-4 model is deployed and accessible
- **Test connection**: Use Azure Portal to test your OpenAI resource

#### "❌ Chart execution failed"
- **Data compatibility**: Ensure your dataset has the required columns
- **Missing values**: The system handles NaN values, but extreme cases may fail
- **Code generation**: Check the generated code in the expandable section

#### "❌ No data loaded"
- **File format**: Ensure your CSV file is properly formatted
- **File size**: Very large files may cause memory issues
- **Encoding**: Try UTF-8 encoding for special characters

#### LangSmith Tracing Not Working
- **API Key**: Verify your LangSmith API key is correct
- **Project Name**: Ensure the project exists in your LangSmith account
- **Network**: Check firewall settings for `api.smith.langchain.com`

### Debug Mode
Enable verbose logging by modifying the pandas agent:
```python
agent = create_pandas_dataframe_agent(
    self.llm, 
    self.df, 
    verbose=True,  # Enable detailed logging
    allow_dangerous_code=True
)
```

## 📊 Performance Optimization

### Large Datasets
- **Sampling**: For datasets >100k rows, consider sampling for faster analysis
- **Chunking**: Process large files in chunks
- **Memory Management**: Monitor RAM usage with large datasets

### Response Time
- **Caching**: Implement caching for repeated queries
- **Async Processing**: Use async for non-blocking operations
- **Model Selection**: Consider using smaller models for simple queries

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Submit a pull request with a clear description

### Code Standards
- Follow PEP 8 Python style guidelines
- Add docstrings to all functions
- Include error handling for new features
- Test with both Titanic and custom datasets

### Areas for Contribution
- **New visualization types**: 3D plots, interactive charts
- **Data source integrations**: Databases, APIs, cloud storage
- **Enhanced AI agents**: Statistical testing, prediction models
- **UI improvements**: Better error messages, user onboarding
- **Performance optimizations**: Caching, async processing

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Streamlit**: For the amazing web app framework
- **LangChain**: For the powerful AI agent capabilities
- **Azure OpenAI**: For providing the GPT-4 model
- **LangSmith**: For excellent tracing and monitoring
- **Titanic Dataset**: Classic machine learning dataset from Kaggle

## 📞 Support

For questions, issues, or contributions:

1. **GitHub Issues**: Open an issue for bugs or feature requests
2. **Documentation**: Check this README and inline code comments
3. **LangSmith Support**: Visit [smith.langchain.com](https://smith.langchain.com) for tracing issues
4. **Azure Support**: Check Azure documentation for OpenAI service issues

## 🔮 Future Roadmap

- [ ] **Multi-dataset Support**: Compare and analyze multiple datasets simultaneously
- [ ] **Advanced Statistics**: Integration with scipy for statistical tests
- [ ] **Machine Learning**: Built-in ML model training and evaluation
- [ ] **Real-time Data**: Support for streaming data sources
- [ ] **Collaborative Features**: Share analyses and insights with teams
- [ ] **Export Capabilities**: PDF reports, presentation slides
- [ ] **Voice Interface**: Speech-to-text for voice queries
- [ ] **Mobile Optimization**: Responsive design for mobile devices

---

*Built with ❤️ using Streamlit, LangChain, and Azure OpenAI*