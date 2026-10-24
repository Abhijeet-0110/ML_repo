# 🌱 Will My Plant Survive Me?

> **A playful, botanical machine learning app that predicts whether your houseplants will Thrive, Survive, or meet their Doom based on your actual care habits.**

---

## 📖 Overview & How It Works

Have you ever wondered if that stylish fiddle-leaf fig or cute succulent stands a fighting chance on your windowsill? **"Will My Plant Survive Me?"** turns your honest care habits into a botanical fate prediction with probability scores and personalized care advice.

### 🧪 Synthetic Data Generation
Because no public tabular dataset exists mapping everyday habits to houseplant outcomes, this project generates a realistic synthetic dataset (`~2,200` samples) founded on:
1. **Biological Care Rules**: Each plant type (`succulent`, `fern`, `cactus`, `tropical`, `flowering`) adheres to grounded botanical tolerances across watering frequency, sunlight exposure, ambient humidity, temperature, pot drainage, and owner forgetfulness.
2. **Calibrated Gaussian Noise**: Real life is messy—drafts, microclimates, and irregular pest visits happen. We inject calibrated Gaussian noise (`scale = 5.0`) into the health scoring function. This creates realistic class boundary overlap and prevents an artificial 100% test accuracy.
3. **Three-Tier Labeling**:
   - 🌿 **Thrives**: Optimal light, watering, and environmental conditions.
   - 🍂 **Survives**: Suboptimal care; the plant hangs on but is stressed.
   - 💀 **Doomed**: Fatal mistakes (e.g. overwatering cacti, no drainage, scorching delicate ferns).

### 🤖 Machine Learning Pipeline
- **Algorithm**: `RandomForestClassifier` (200 estimators, stratified 80/20 train/test split).
- **Target Accuracy**: **~81–85%** (intentionally not 100%, reflecting realistic overlap).
- **Inference Robustness**: Saves both the trained classifier and exact one-hot encoded training feature column ordering to prevent inference column-mismatch bugs.

---

## 🛠️ Tech Stack

- **Python 3.10+** (Tested and compatible across 3.10, 3.11, 3.12, 3.14)
- **Streamlit**: Interactive web interface with live probability meters and instant feedback
- **scikit-learn**: Random Forest modeling, stratification, and evaluation metrics
- **pandas & numpy**: Vectorized data synthesis, preprocessing, and encoding
- **joblib**: Model artifact serialization and schema persistence

---

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/will-my-plant-survive-me.git
cd will-my-plant-survive-me
```

### 2. Set Up a Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Note: If you are running Python 3.14+, simply run `pip install pandas numpy scikit-learn streamlit joblib`.)*

### 4. Generate Synthetic Data
```bash
python src/generate_data.py
```
*Output: Generates `data/plants.csv` with 2,200 rows and prints class distributions.*

### 5. Train the Model
```bash
python src/train.py
```
*Output: Evaluates accuracy (~81%), confusion matrix, and feature importances, then saves `plant_model.joblib` and `model_columns.joblib` into `models/`.*

### 6. Launch the Streamlit App
```bash
streamlit run src/app.py
```
Open your browser to `http://localhost:8501` to test your care routine!

---

## ☁️ Deployment to Streamlit Community Cloud

Deploy your working app for free in less than 3 minutes:

1. **Push to GitHub**:
   Make sure your repo has the following structure and commit the generated model artifacts:
   ```bash
   git add .
   git commit -m "Initial commit for Will My Plant Survive Me"
   git branch -M main
   git remote add origin https://github.com/<your-username>/will-my-plant-survive-me.git
   git push -u origin main
   ```

2. **Connect to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
   - Click **"New app"**.
   - Select your repository (`will-my-plant-survive-me`), branch (`main`), and set **Main file path** to:
     ```text
     src/app.py
     ```

3. **Deploy**:
   - Click **"Deploy!"**.
   - Streamlit Cloud will install `requirements.txt` and launch your live web app with a shareable URL!

---

## ⚠️ Disclaimer

This is a **fun toy machine learning model** built for educational and demonstration purposes. It does not replace real horticultural expertise or diagnosing diseased foliage. Please do not blame this app if your fern decides to give up on life anyway!
