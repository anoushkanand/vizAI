# 📊 VizAI: AI-Powered Data Insights

VizAI is an interactive Streamlit app that lets users upload a SQLite database and automatically generates analytical **goals**, **SQL queries**, **visualizations**, and **natural language insights** — all powered by **OpenAI (via LangChain)** and **LIDA**.

> 💡 Designed for data analysts, PMs, and non-technical users to gain fast, meaningful insights from structured datasets.

---

## 🔧 Features

- 🧠 **Goal Generation**: Automatically generate analytical goals based on your database schema using LLMs.
- 📄 **SQL Query Generation**: Generate accurate SQL queries via LangChain SQL Agent (OpenAI Functions).
- 📈 **Visualization Recommendation & Rendering**: Auto-select the best visualization type and render it using Plotly.
- 🗣 **Natural Language Insights**: Summarize query results into easy-to-understand bullet points.
- 🧍 **Persona-based Goal Customization**: Adapt goals based on target user persona (e.g. Data Analyst).
- 🧩 **Add Your Own Goal**: Manually define goals and get automated support from LLMs.

---

## 🚀 Demo

[![VizAI Demo](https://img.youtube.com/vi/YOUR_VIDEO_ID/0.jpg)](https://drive.google.com/file/d/1Y7V_nPeFDzS46UIMDQtqu8yKwMUWnJep/view?usp=sharing)

---

## 🧰 Tech Stack

| Layer       | Technology                          |
|------------|--------------------------------------|
| Frontend   | [Streamlit](https://streamlit.io/)   |
| Backend    | Python, [LangChain](https://www.langchain.com/), [LIDA](https://microsoft.github.io/lida/) |
| LLM        | OpenAI (`gpt-3.5-turbo`)             |
| Database   | SQLite (.db upload)                  |
| Visualization | Plotly, Plotly Express           |
| Secrets    | `.env` for API keys                  |



