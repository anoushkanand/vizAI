import streamlit as st
import json
import logging
import sqlite3
import os
import pandas as pd
import re
import plotly.express as px
from langchain.agents import create_sql_agent, AgentType
from lida.utils import clean_code_snippet
from lida import Manager, TextGenerationConfig, llm
from lida.datamodel import Goal, Persona
import plotly.figure_factory as ff
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv 

load_dotenv()

os.makedirs("data-db", exist_ok=True)

# System Instructions for LLM
SYSTEM_INSTRUCTIONS = """
You are an experienced data analyst who can generate a number of insightful GOALS based about data, on a database schema or file data,
and a specified persona. The VISUALIZATIONS YOU RECOMMEND MUST FOLLOW VISUALIZATION BEST PRACTICES (e.g., must use bar charts instead 
of pie charts for comparing quantities) AND BE MEANINGFUL. They must also be relevant to the specified persona. 
Each goal must include a question, a visualization (THE VISUALIZATION MUST REFERENCE THE EXACT COLUMN FIELDS FROM THE SCHEMA), 
and a rationale (JUSTIFICATION FOR WHICH database FIELDS ARE USED and what we will learn from the visualization). Each goal MUST 
mention the exact fields from the database schema above.
"""

# Format Instructions
FORMAT_INSTRUCTIONS = """
THE OUTPUT MUST BE A CODE SNIPPET OF A VALID LIST OF JSON OBJECTS USING THIS FORMAT:
[
    { "index": 0,  "question": "What is the distribution of X?", "visualization": "histogram of X", "rationale": "This shows..." },
    { "index": 1,  "question": "...", "visualization": "...", "rationale": "..." }
]
THE OUTPUT SHOULD ONLY USE THE JSON FORMAT ABOVE.
"""

# Logger setup
logger = logging.getLogger("lida")
logging.basicConfig(level=logging.INFO)

openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    st.error("OPENAI_API_KEY not found. Please check your .env file.")
    st.stop()
st.sidebar.write("## Setup")
selected_model = "gpt-3.5-turbo"
temperature = 0.4

# Streamlit UI
st.title("VizAI: AI-Powered Data Insights 📊")

st.sidebar.write("### Choose a database")
uploaded_file = st.sidebar.file_uploader("Upload your database here:", type=["db"])
db_path = None
if uploaded_file is not None:
    db_path = os.path.join("data-db", uploaded_file.name)
    with open(db_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"Database uploaded successfully: {uploaded_file.name}")
    st.session_state["db_path"] = db_path
else:
    st.info("Please upload a database file to proceed.")
    
class GoalExplorer:
    """Generate goals given a summary of data or a database schema."""

    def generate(self, db_path: str = None, textgen_config: TextGenerationConfig = None, text_gen=None, n=5, persona: Persona = None) -> list[Goal]:
        """Generate goals based on the schema of all tables in the database."""
        if not db_path:
            raise ValueError("A database path must be provided.")

        summary = self.extract_db_schema(db_path)

        if not persona:
            persona = Persona(persona="A highly skilled data analyst who generates insightful goals about data.", rationale="")

        user_prompt = f"The number of GOALS to generate is {n}. The goals should be based on the database schema:\n{json.dumps(summary)}\n\n"
        user_prompt += f"The generated goals SHOULD BE FOCUSED ON THE INTERESTS AND PERSPECTIVE of a '{persona.persona} persona, who is insterested in complex, insightful goals about the data. \n"

        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": f"{user_prompt}\n\n {FORMAT_INSTRUCTIONS} \n\nThe {n} goals are:\n"}
        ]

        result = text_gen.generate(messages=messages, config=textgen_config)
        try:
            json_string = clean_code_snippet(result.text[0]["content"])
            parsed_result = json.loads(json_string)

            if isinstance(parsed_result, dict):
                parsed_result = [parsed_result]

            return [Goal(**x) for x in parsed_result]

        except json.decoder.JSONDecodeError:
            logger.error(f"Error decoding JSON: {result.text[0]['content']}")
            raise ValueError("Model did not return valid JSON.")

    def extract_db_schema(self, db_path: str) -> dict:
        """Extract schema details from all tables in the database."""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]

        schema = {}
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [{"name": col[1], "type": col[2]} for col in cursor.fetchall()]
            schema[table] = columns

        conn.close()
        return schema
    
# Initialize LIDA Manager
lida = Manager(text_gen=llm("openai", api_key=openai_key))
textgen_config = TextGenerationConfig(max_tokens=1000, n=5, temperature=temperature, model=selected_model)

# Callback to capture SQL query
class SQLHandler(BaseCallbackHandler):
    def __init__(self):
        self.sql_result = None

    def on_agent_action(self, action, **kwargs):
        """Captures the final SQL query used by the SQL agent."""
        if action.tool == "sql_db_query":
            self.sql_result = action.tool_input

class SQLAgent:
    def __init__(self, db_path: str, openai_key: str):
        """Initialize the SQL agent with database connection and LLM API."""
        self.db = SQLDatabase.from_uri(f"sqlite:///{db_path}") 
        self.llm = ChatOpenAI(openai_api_key=openai_key, model="gpt-3.5-turbo")

        # Initialize callback handler
        self.handler = SQLHandler()

        # Create LangChain SQL Agent
        self.agent_executor = create_sql_agent(
            llm=self.llm,
            db=self.db,
            agent_type=AgentType.OPENAI_FUNCTIONS,  # Use OpenAI Functions for better query handling
            verbose=True,
            agent_executor_kwargs={"handle_parsing_errors": True}
        )

    def generate_query(self, goal: str) -> str:
        """Generate SQL query using LangChain's agent and extract query from callback."""
        try:
            response = self.agent_executor.invoke({"input": goal}, {"callbacks": [self.handler]})
            
            # Debugging: Print response to understand structure
            print("Agent Response:", response)

            # Extract SQL query from response
            if isinstance(self.handler.sql_result, dict):
                query = self.handler.sql_result.get("query")  # Adjust key if needed
            else:
                query = self.handler.sql_result

            if query and isinstance(query, str):
                return query.strip()
            else:
                st.error("Could not extract a valid SQL query.")
                return None
        except Exception as e:
            st.error(f"Error generating query: {e}")
            return None

# Function to Run SQL Query
def run_query(db_path: str, query: str):
    """Execute the SQL query and return a Pandas DataFrame."""
    if not query:
        st.error("No valid SQL query to execute.")
        return None
    
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        st.error(f"Error executing query: {e}")
        return None
    finally:
        conn.close()

# Function to Generate Natural Language Insights
def generate_insights(df: pd.DataFrame, goal: str):
    """Generate natural language insights based on the SQL query result."""

    if df is None or df.empty:
        return "No data available to analyze."

    # Convert a sample of the dataset to JSON for better readability
    sample_data = df.head(5).to_dict(orient="records")

    # Formulate the LLM prompt
    insight_prompt = f"""
    You are a data analyst. Provide insights based on the given dataset. 
    The insights should be in **5 bullet points**, each around **50 words**. 
    Highlight key numbers and findings using bold text. 
    Ensure the insights are meaningful and align with the goal.

    **Goal:** {goal}
    
    **Query Result (Sample Data):**
    {json.dumps(sample_data, indent=2)}

    Focus on key trends, anomalies, and actionable insights.
    """

    messages = [
        {"role": "system", "content": "You are an expert data analyst providing insights based on a dataset."},
        {"role": "user", "content": insight_prompt}
    ]

    # Generate response from LLM
    result = lida.text_gen.generate(messages=messages, config=textgen_config)

    return result.text[0]["content"] if result else "No insights available."

# Function to Generate Visualizations
def generate_visualization(df: pd.DataFrame, visualization_type: str):
    """Generate visualizations based on the recommended type."""
    if df is None or df.empty:
        st.error("No data available to visualize.")
        return

    try:
        columns = df.columns[:2]  # Default to first two columns
        x_col, y_col = columns[0], columns[1] if len(columns) > 1 else columns[0]

        if "bar chart" in visualization_type.lower():
            fig = px.bar(df, x=x_col, y=y_col, title=f"Bar Chart of {x_col} vs {y_col}")
        elif "line chart" in visualization_type.lower():
            fig = px.line(df, x=x_col, y=y_col, title=f"Line Chart of {x_col} vs {y_col}")
        elif "scatter plot" in visualization_type.lower():
            fig = px.scatter(df, x=x_col, y=y_col, title=f"Scatter Plot of {x_col} vs {y_col}")
        elif "pie chart" in visualization_type.lower():
            fig = px.pie(df, names=x_col, values=y_col, title=f"Pie Chart of {x_col} vs {y_col}")
        elif "histogram" in visualization_type.lower():
            fig = px.histogram(df, x=x_col, title=f"Histogram of {x_col}", nbins=20)
        elif "box plot" in visualization_type.lower():
            fig = px.box(df, x=x_col, title=f"Box Plot of {x_col}")
        elif "heatmap" in visualization_type.lower():
            corr_matrix = df.corr()
            fig = ff.create_annotated_heatmap(
                z=corr_matrix.values,
                x=list(corr_matrix.columns),
                y=list(corr_matrix.index),
                colorscale="Viridis",
                showscale=True
            )
        else:
            st.error("Visualization type not recognized.")
            return

        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Error generating visualization: {e}")


# Load Goals
goal_explorer = GoalExplorer()
num_goals = st.sidebar.slider("Number of goals to generate", min_value=1, max_value=50, value=6)

goals = goal_explorer.generate(db_path=db_path, textgen_config=textgen_config, text_gen=lida.text_gen, n=num_goals)

if "goals" not in st.session_state:
    st.session_state.goals = goals

own_goal = st.sidebar.checkbox("Add Your Own Goal")

if own_goal:
    user_goal = st.sidebar.text_input("Describe your goal")
    if user_goal:
        # Ask LLM to determine the best visualization type
        vis_prompt = f"""
        You are a data visualization expert. Based on the following goal, choose the best visualization type 
        from this list: ["bar chart", "line chart", "scatter plot", "pie chart", "histogram", "box plot", "heatmap"].
        
        **User Goal:** {user_goal}
        
        ONLY RETURN THE VISUALIZATION TYPE AS A SINGLE STRING.
        """

        vis_messages = [
            {"role": "system", "content": "You are an expert in data visualization and statistics."},
            {"role": "user", "content": vis_prompt}
        ]

        vis_result = lida.text_gen.generate(messages=vis_messages, config=textgen_config)
        chosen_visualization = vis_result.text[0]["content"].strip().lower()

        st.write(f"**Selected Visualization Type:** {chosen_visualization}")

        # Add the new goal to the existing list
        new_goal = Goal(question=user_goal, visualization=chosen_visualization, rationale="User-defined goal")
        goals.append(new_goal)

goal_questions = [goal.question for goal in goals]

selected_goal_text = st.selectbox("Select a Goal:", goal_questions)
selected_goal = next(goal for goal in goals if goal.question == selected_goal_text)
    
st.write(f"**Visualization Type:** {selected_goal.visualization}")
st.write(f"**Rationale:** {selected_goal.rationale}")

sql_agent = SQLAgent(db_path=db_path, openai_key=openai_key)
db_schema = goal_explorer.extract_db_schema(db_path)
generated_query = sql_agent.generate_query(selected_goal.question)

if generated_query:
    #st.code(generated_query, language="sql")
    df = run_query(db_path, generated_query)

    insights = generate_insights(df, selected_goal.visualization)
    st.markdown("### Insights 📊")
    st.write(insights)

    if "show_viz" not in st.session_state:
        st.session_state.show_viz = False
    if st.button("Show Visualization"):
        st.session_state.show_viz = not st.session_state.show_viz
    if st.session_state.show_viz:
        generate_visualization(df, selected_goal.visualization)