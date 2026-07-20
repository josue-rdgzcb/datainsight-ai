# DataInsight AI 🧠🤖

**DataInsight AI** is an AI-powered assistant that combines automated analysis with interactive visualization. Built on top of a **Streamlit** interface, it is designed to generate an automatic **Exploratory Data Analysis (EDA)** from any CSV file and deliver **immediate AI-assisted insights**. 

The platform processes dataset variables, generates an automated EDA, creates instant interactive charts, and uses an AI module (OpenAI) to analyze data structure and write:

- **Key insights** regarding dataset behavior   

- **Quality alerts** and consistency issues  

- **Relevant correlations** between variables  

- **Feature engineering** recommendations  

- **Model suggestions based on the learning approach:** supervised (regression or classification) when a target variable is selected, and unsupervised (clustering) when no target is defined.

This way, **DataInsight AI** blends immediate visualization with automated interpretation, offering an agile, assisted workflow for analysts, data scientists, and engineers looking to **accelerate initial data exploration**.


---

## 🚀 Live Demo

**Try the application here:**

👉 https://datainsight-ai-assistant.streamlit.app/

<p align="center">
    <img src="assets/demo.gif">
</p>

---

## 🚀 Key Features

- **Automated EDA:** Instant generation of an Exploratory Data Analysis from any CSV file.

- **Interactive Visualization:** Dynamic and customizable charts using Streamlit + Plotly, ready to explore relationships between variables.

- **Automated Narrative:** Fluid natural language summaries that convert statistics into easily understandable insights.

- **Data Quality Alerts:** Identification of missing values, duplicates, and consistency issues.

- **AI-Generated Insights:** The OpenAI module writes key findings, quality problems, and relevant correlations.

- **Feature Engineering Recommendations**: 
    - Automated and specific strategies to transform numerical variables (log, scaling, polynomial).
    - Strategies to encode high-cardinality categorical variables (target/frequency encoding).
    - Proposals for new features with potential explanatory power, including external integrations based on the dataset domain.

- **Modeling Suggestions:** Recommendations aligned with supervised learning (regression or classification when a target is defined) and unsupervised learning (clustering when no target is available).

- **HTML Report Export:** Generation of interactive `.html` documents, self-contained and usable offline, including metrics, visualizations, and AI-generated analysis.


---

## 🛠️ Prerequisites

To run this application locally or deploy it on a private server, you need:
- **Python 3.10** or higher installed on your system.
- A valid **OpenAI API Key** (Make sure your OpenAI developer account has usage balance/credits available).


---

## 📦 Installation

1. **Clone the repository** or navigate to the project folder in your terminal:
   ```bash
   git clone https://github.com/josue-rdgzcb/datainsight-ai.git
   cd datainsight-ai
   ```

2. **Create a virtual environment** to isolate system dependencies safely:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # venv\Scripts\activate   # On Windows
   ```

3. **Install the required packages** using the structured dependency file:
   ```bash
   pip install -r requirements.txt
   ```
4. **(Optional)** Verify dependency installation and versions with the included script:
   ```bash
   python check_requirements.py
   ```



---

## ▶️ Usage

1. Start the platform by running the main orchestrator entry point:
   ```bash
   streamlit run app.py
   ```
   *Note: If you are running Streamlit for the first time, you can press **Enter** to skip the optional email registration step.* 
<br> 

2. The application will automatically open in your default web browser (usually at `http://localhost:8501`).
<br>

3. **In the interface:**

   1. Import or drag and drop any structured dataset in CSV format.
   2. Select your target variable (or 'None' for a general unsupervised EDA).
   3. After selecting your target variable, an automated EDA and visualizations will be generated on the main page.
   4. To use the AI analysis features, enter your OpenAI API Key in the settings panel located in the left sidebar.
   5. Click on **"✨ Analyze with AI"** to activate the Artificial Intelligence analysis integration (OpenAI).
   6. (Optional) Download the generated reports from the Report Export Center in `.html` format.


---

## 📚 Technologies and Architecture

- **Framework and Web Interface:** Streamlit (v1.35.0+)

- **Data Manipulation and Analysis:** Pandas, NumPy, SciPy

- **Interactive Visualization:** Plotly

- **AI Engine:** OpenAI Python SDK + dotenv

- **Report Generation:** HTML export with Markdown2 + Plotly (offline interactive reports)

---

## 💻 Future Improvements

- **Interactive Data Chat (AI Agent):** Upgrading the UI to include a chat interface where users can ask custom questions about the dataset, using an AI Agent to dynamically query the data.
- **One-Click Jupyter Notebook Export:** Allowing users to download the complete analysis, including the source code for the Plotly charts and Pandas transformations, as a `.ipynb` file to continue workflows locally.
- **Executive PDF Report Generation:** Expanding the export center to generate highly polished, print-ready PDF reports combining AI summaries and static chart snapshots.
- **Local LLM Ecosystem (Ollama):** Adding support for local open-source models (like Llama 3) to allow enterprise users to analyze sensitive data completely offline.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

