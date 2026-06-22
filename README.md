# DataInsight AI 🧠🤖

**DataInsight AI** es un asistente impulsado por **Inteligencia Artificial** que combina análisis automático y visualización interactiva. Construido sobre una interfaz de **Streamlit**, está diseñado para generar un **Exploratory Data Analysis (EDA)** automático a partir de cualquier archivo CSV y ofrecer **insights inmediatos asistidos por IA**.

La plataforma procesa las variables del dataset, genera un EDA automático, crea gráficos interactivos instantáneos y utiliza un módulo de inteligencia artificial (OpenAI) para analizar la estructura de los datos y redactar:

- **Key insights** sobre el comportamiento del dataset

- **Alertas de calidad** y problemas de consistencia

- **Correlaciones** relevantes entre variables

- **Recomendaciones de feature engineering**

- **Sugerencias de modelos según el enfoque de aprendizaje:** supervisado (regresión o clasificación) cuando se selecciona un target, y no supervisado (clustering) cuando no se define un target.

De esta forma, **DataInsight AI** combina visualización inmediata con interpretación automática, ofreciendo un flujo de trabajo ágil y asistido para analistas, científicos de datos e ingenieros que buscan **acelerar la exploración inicial** de sus datasets.


---

## 🚀 Funcionalidades Clave

- **EDA automático:** Generación instantánea de un Exploratory Data Analysis a partir de cualquier archivo CSV.

- **Visualización interactiva:** Gráficos dinámicos y personalizables con Streamlit + Plotly, listos para explorar relaciones entre variables.

- **Narrativa automática:** Resúmenes fluidos en lenguaje natural que convierten estadísticas en insights comprensibles.

- **Alertas de calidad de datos:** Identificación de valores faltantes, duplicados y problemas de consistencia.

- **Insights generados por IA:**  El módulo de OpenAI redacta hallazgos clave, problemas de calidad y correlaciones relevantes.

- **Recomendaciones de feature engineering**: 
    - Estrategias automáticas y específicas para transformar variables numéricas (log, escalado, polinomiales).
    - Estrategias para codificar categóricas de alta cardinalidad (target/frequency encoding).
    - Propuestas de nuevas features con potencial poder explicativo, incluyendo integraciones externas según el dominio del dataset.

- **Sugerencia de modelado:** Recomendaciones alineadas con el aprendizaje supervisado (regresión o clasificación cuando se define un target) y con el aprendizaje no supervisado (clustering cuando no se dispone de un target).

- **Exportación de reportes HTML:** Generación de documentos interactivos en formato `.html`, autocontenidos y utilizables sin conexión, que incluyen métricas, visualizaciones y análisis generado por IA.

---

## 🛠️ Requisitos Previos

Para ejecutar esta aplicación localmente o desplegarla en un servidor privado, necesitas:
- **Python 3.10** o superior instalado en tu sistema.
- Una **OpenAI API Key** válida (Asegúrate de que tu cuenta de desarrollador de OpenAI cuente con saldo/créditos de uso disponibles).

---

## 📦 Instalación

1. **Clona el repositorio** o navega hasta la carpeta del proyecto en tu terminal:
   ```bash
   git clone https://github.com/josue-rdgzcb/datainsight-ai.git
   cd datainsight-ai
   ```

2. **Crea un entorno virtual** para aislar las dependencias del sistema de forma segura:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En macOS/Linux
   # venv\Scripts\activate   # En Windows
   ```

3. **Instala los paquetes requeridos** utilizando el archivo de dependencias estructurado:
   ```bash
   pip install -r requirements.txt
   ```
4. **(Opcional)** Verifica la instalación de dependencias y versiones con el script incluido:
      ```bash
   python check_requirements.py
   ```


---

## ▶️ Uso

1. Inicia la plataforma ejecutando el punto de entrada principal del orquestador:
   ```bash
   streamlit run app.py
   ```
   *Nota: Si ejecutas Streamlit por primera vez, puedes presionar **Enter** para omitir el paso opcional de registro de correo electrónico.* 
<br> 

2. La aplicación se abrirá automáticamente en tu navegador web predeterminado (usualmente en `http://localhost:8501`).
<br>
3. **En la interfaz:**

   1. Importa o arrastra y suelta cualquier dataset estructurado en formato CSV.
   2. Selecciona tu variable objetivo / target (o 'None' para un EDA no supervisado general).
   3. Después de seleccionar tu variable objetivo se generará un EDA automático y visualizaciones en la página principal.
   4. Para utilizar la funcionalidad de análisis con IA, ingresa tu OpenAI API Key en el panel de configuración de la barra lateral izquierda.
   5. Haz clic en **"✨ Analyze with AI"** para activar la integración de análisis con Inteligencia Artificial (OpenAI).
   6. (Opcional) Descarga los reportes generados desde el Report Export Center en formato `.html`

---

## 📚 Tecnologías y Arquitectura

- **Framework e Interfaz Web:** Streamlit (v1.35.0+)

- **Manipulación y análisis de datos:** Pandas, NumPy, SciPy

- **Visualización interactiva:** Plotly

- **Motor de IA:** OpenAI Python SDK + dotenv

- **Generación de reportes:** Exportación en HTML con Markdown2 + Plotly (reportes interactivos sin conexión)
