
# 🏆 ML_Code_Craft – CodeCraft ML Competition 2025 (1st Place 🥇)

Welcome to the official repository of **Team T10** – Winners of the **CodeCraft ML Competition 2025** organized by *Bajaj Institute of Technology, Wardha*. This project showcases our end-to-end approach to solving a real-world machine learning challenge using the dataset provided by the organizers.

## 📌 Problem Statement

Given a dataset with multiple business-related attributes, the goal was to develop a predictive ML model that could extract meaningful insights and deliver accurate predictions. The entire solution was developed **without external data** and evaluated on criteria including accuracy, interpretability, creativity, and robustness.

## 🧠 Approach

Our pipeline consisted of:
- **Data Cleaning & Preprocessing**: Handling null values, encoding categorical variables, scaling, etc.
- **Feature Engineering**: Selecting and transforming the most impactful features.
- **Model Selection**: Multiple models were tested; the final selection was based on performance and interpretability.
- **Hyperparameter Tuning**: Applied GridSearchCV & RandomizedSearchCV for optimization.
- **Evaluation Metrics**: R² score, MAE, RMSE, and visualization-based interpretability.
- **Deployment (Optional)**: Simple Flask app to demonstrate real-time predictions.

## 📂 Repository Structure

```
ML_Code_Craft/
├── data/                     # Raw and processed datasets
│   └── CodeCraft_Dataset.xlsx
├── notebooks/                # Jupyter Notebooks with exploration & modeling
│   └── analysis_modeling.ipynb
├── app/                      # Flask App (Optional Demo)
│   ├── templates/
│   │   └── index.html
│   ├── static/
│   │   └── style.css
│   └── app.py
├── models/                   # Saved model files (.pkl)
│   └── final_model.pkl
├── README.md                 # You're here!
└── requirements.txt          # Python package requirements
```

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/AyushNigam005/ML_Code_Craft.git
   cd ML_Code_Craft
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask app**
   ```bash
   cd app
   python app.py
   ```

## 📈 Results

- Achieved a high model accuracy on test data.
- Visualized important features for interpretability.
- Delivered a user-friendly interface for deployment.

## 🧑‍💻 Team Members – Team T10

- Komal Punwatkar
- Ayush Nigam  
- Shruti Malode  
- Omkar Chauhan  
- Asmi Bhandekar
- Shrawan chambhare

## 🏅 Achievement

🥇 **1st Position** – CodeCraft ML Competition 2025  
Organized by: *Department of Computer Engineering, BIT Wardha*  
Date: 03/06/2025

