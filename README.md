# Diabetes Prediction Web Application
This project is a **Machine Learning and Web Application** designed to predict whether a person has diabetes based on health metrics. It includes:
- A model training script (`train_models.py`) that trains and compares multiple ML models.
- A Flask-based web application (`app.py`) for user input and predictions.
- A dataset (`diabetes_prediction_dataset.csv`) used for training.
- Deployment steps via **Anaconda Prompt**.

## Dataset
- **File:** `diabetes_prediction_dataset.csv`
- **Columns:**
  - Gender
  - Age
  - Hypertension
  - Heart Disease
  - Smoking History
  - BMI
  - HbA1c Level
  - Blood Glucose Level
  - Diabetes (Target variable)

### Preprocessing
1. Removed duplicate records and entries with "Other" gender.
2. One-hot encoding applied to categorical variables (`gender`, `smoking_history`).
3. Feature scaling applied to `age`, `bmi`, `HbA1c_level`, and `blood_glucose_level`.

## Machine Learning Models
The `train_models.py` script trains and evaluates:
- Logistic Regression
- Random Forest Classifier
- K-Nearest Neighbors (KNN)
- Gradient Boosting Classifier

### Model Performance
- Metrics used: **Accuracy** and **AUC**.
- **Best Model:** Gradient Boosting (based on AUC score).
- Saved models:
  ```
  logistic_regression_model.joblib
  random_forest_model.joblib
  k_nearest_neighbors_model.joblib
  gradient_boosting_model.joblib
  ```

## Web Application
- **Framework:** Flask
- **Frontend:** `index.html` and `result.html`
- **Functionality:**
  1. Input patient data (Name, ID, Gender, Age, BMI, HbA1c, Blood Glucose, Hypertension, Heart Disease, Smoking History).
  2. Output prediction (Diabetic / Not Diabetic) with **confidence percentage**.

## Usage and Applications
- **Healthcare Screening:** Helps healthcare professionals quickly screen patients for diabetes risk using common health metrics.
- **Telemedicine:** Can be integrated into online platforms for remote patient monitoring.
- **Education:** Demonstrates how machine learning can be applied to real-world health datasets.
- **Research:** A baseline project for experimenting with ML models on healthcare prediction tasks.

## How to Run
1. **Clone the Repository**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```
2. **Set Up the Environment**
   ```bash
   conda create -n diabetes-app python=3.8
   conda activate diabetes-app
   pip install -r requirements.txt
   ```
   If `requirements.txt` is missing:
   ```bash
   pip install flask numpy pandas scikit-learn joblib
   ```
3. **Train the Models**
   ```bash
   python train_models.py
   ```
   This generates `.joblib` files for each model.
4. **Run the Web App**
   ```bash
   python app.py
   ```
5. **Open the App**
   ```
   http://localhost:5000
   ```

## Screenshots
### Home Page
![Input Form](d1388d92-0afb-4da7-b6c6-88e0715f0f9f.png)

## Project Structure
```plaintext
.
├── app.py
├── train_models.py
├── diabetes_prediction_dataset.csv
├── templates/
│   ├── index.html
│   └── result.html
├── *.joblib
└── README.md
```

## Future Improvements
- Add database storage for predictions.
- Improve UI/UX design.
- Implement SHAP-based model explainability.

## License
This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for more details.
