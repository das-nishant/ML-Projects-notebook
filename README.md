# Machine Learning Projects Showcase

Welcome to my repository of machine learning projects! This collection showcases a variety of models I've built to solve different prediction and classification problems. Each project is contained within its own Jupyter Notebook and includes data preprocessing, model training, and evaluation steps.

## Projects Overview

Here is a summary of the projects included in this repository:

| # | Project Name | Description | Model Used | Key Libraries |
|---|--------------|-------------|------------|---------------|
| 1 | **[Rock vs. Mine Prediction](#1-rock-vs-mine-prediction)** | A binary classification model to distinguish between sonar signals bounced off rocks and mines. | Logistic Regression | Scikit-learn, Pandas, Numpy |
| 2 | **[Car Price Prediction](#2-car-price-prediction)** | A regression model to predict the selling price of used cars based on their features. | Linear Regression | Scikit-learn, Pandas, Seaborn |
| 3 | **[Fake News Prediction](#3-fake-news-prediction)** | A text classification model to identify whether a news article is real or fake based on its title and URL. | Logistic Regression, TfidfVectorizer | Scikit-learn, Pandas, NLTK |
| 4 | **[Gold Price Prediction](#4-gold-price-prediction)** | A regression model to predict the price of gold using other financial market indicators. | Random Forest Regressor | Scikit-learn, Pandas, Seaborn |
| 5 | **[Loan Status Prediction](#5-loan-status-prediction)** | A binary classification model to predict whether a loan application will be approved. | Support Vector Machine (SVM) | Scikit-learn, Pandas, Seaborn |
| 6 | **[House Price Prediction](#6-house-price-prediction)** | A regression project to predict median house values in California districts using various features. | Linear Regression, Decision Tree | Scikit-learn, Pandas, Seaborn |
| 7 | **[Wine Quality Prediction](#7-wine-quality-prediction)** | A classification model that predicts whether a red wine is of "good quality" based on its chemical properties. | Random Forest Classifier | Scikit-learn, Pandas, Seaborn |
| 8 | **[Diabetes Prediction](#8-diabetes-prediction)** | A binary classification model to predict the onset of diabetes based on diagnostic measures. | Support Vector Machine (SVM) | Scikit-learn, Pandas, Numpy |

---

## Detailed Project Descriptions

### 1. Rock vs. Mine Prediction

*   **Objective:** To build a model that can accurately classify sonar returns as either a Rock (R) or a Mine (M). This is a classic binary classification problem.
*   **Dataset:** The "Sonar, Mines vs. Rocks" dataset, which contains 208 instances. Each instance is a set of 60 sonar readings at different angles, with a corresponding label (R or M).
*   **Methodology:**
    1.  **Data Processing:** The dataset was loaded into a Pandas DataFrame. The features (60 sonar readings) and the target labels were separated.
    2.  **Train-Test Split:** The data was split into a training set (90%) and a testing set (10%) to evaluate the model's performance on unseen data.
    3.  **Model Training:** A **Logistic Regression** model was chosen for this classification task and trained on the training data.
    4.  **Evaluation:** The model's accuracy was measured on both the training and test data. It achieved approximately **83.4% accuracy on the training data** and **76.2% on the test data**.
    5.  **Predictive System:** A simple function was created to take a new set of 60 sonar readings, reshape the data, and predict whether the object is a rock or a mine.
*   **Libraries Used:** `numpy`, `pandas`, `sklearn.model_selection`, `sklearn.linear_model`, `sklearn.metrics`.

### 2. Car Price Prediction

*   **Objective:** To predict the selling price of used cars based on various attributes like manufacturing year, kilometers driven, fuel type, and transmission.
*   **Dataset:** A dataset of used car sales containing features such as `name`, `year`, `selling_price`, `km_driven`, `fuel`, `seller_type`, `transmission`, and `owner`.
*   **Methodology:**
    1.  **Data Processing:** The dataset was loaded and inspected for missing values.
    2.  **Feature Encoding:** Categorical features like `fuel`, `seller_type`, and `transmission` were converted into numerical values using label encoding to make them suitable for the regression model.
    3.  **Train-Test Split:** The dataset was divided into features (X) and the target variable (`selling_price`). It was then split into a 90% training set and a 10% testing set.
    4.  **Model Training:** A **Linear Regression** model was trained to learn the relationship between the car's features and its selling price.
*   **Libraries Used:** `pandas`, `matplotlib`, `seaborn`, `sklearn.model_selection`, `sklearn.linear_model`.

### 3. Fake News Prediction

*   **Objective:** To develop a system that can distinguish between "Real News" and "Fake News" using Natural Language Processing (NLP) techniques.
*   **Dataset:** The FakeNewsNet dataset, which includes information like the news title and URL. The label is binary: 1 for Fake News and 0 for Real News.
*   **Methodology:**
    1.  **Data Preprocessing:** The `news_url` and `title` columns were combined to form the primary text content.
    2.  **Text Processing (Stemming):** The text data was processed using Porter Stemming to reduce words to their root form (e.g., "running" becomes "run"). Stop words (common words like "a", "the", "is") were also removed.
    3.  **Vectorization:** The processed text was converted into numerical feature vectors using **TF-IDF (Term Frequency-Inverse Document Frequency)**, which reflects the importance of a word in the text.
    4.  **Model Training:** A **Logistic Regression** model was trained on the vectorized text data.
    5.  **Evaluation:** The model showed high accuracy, achieving **94.8% on the training data** and **92.7% on the test data**.
*   **Libraries Used:** `numpy`, `pandas`, `re`, `nltk`, `sklearn.feature_extraction.text`, `sklearn.linear_model`, `sklearn.metrics`.

### 4. Gold Price Prediction

*   **Objective:** To predict the price of gold (represented by the GLD ETF) based on other market variables.
*   **Dataset:** A time-series dataset containing daily prices for GLD, SPX (S&P 500), USO (United States Oil Fund), SLV (Silver ETF), and the EUR/USD exchange rate.
*   **Methodology:**
    1.  **Data Analysis:** Correlation between the features was analyzed using a heatmap to understand their relationships. A strong positive correlation was observed between the prices of Gold (GLD) and Silver (SLV).
    2.  **Feature Selection:** All columns except for 'Date' and the target 'GLD' were used as input features.
    3.  **Model Training:** A **Random Forest Regressor** was trained on an 80/20 train-test split of the data. This model is well-suited for capturing non-linear relationships.
    4.  **Evaluation:** The model's performance was evaluated using the R-squared error metric, achieving an impressive score of **98.87%** on the test data, indicating a very accurate fit.
*   **Libraries Used:** `numpy`, `pandas`, `matplotlib`, `seaborn`, `sklearn.ensemble`, `sklearn.metrics`.

### 5. Loan Status Prediction

*   **Objective:** To build a classification model that predicts whether a loan will be approved ('Y') or rejected ('N') based on applicant details.
*   **Dataset:** A loan prediction dataset with applicant information such as `Gender`, `Married`, `Dependents`, `Education`, `ApplicantIncome`, `Credit_History`, etc.
*   **Methodology:**
    1.  **Data Preprocessing:** Missing data points were handled by dropping the respective rows. Categorical data (e.g., 'Married', 'Gender', 'Education') was converted into numerical format (0s and 1s). The '3+' value in the 'Dependents' column was mapped to the number 4.
    2.  **Train-Test Split:** The data was split into a 90% training set and a 10% testing set.
    3.  **Model Training:** A **Support Vector Machine (SVM)** with a linear kernel was trained to classify the loan applications.
    4.  **Evaluation:** The model achieved **79.9% accuracy on the training data** and **83.3% on the test data**.
*   **Libraries Used:** `numpy`, `pandas`, `seaborn`, `sklearn.model_selection`, `sklearn.svm`, `sklearn.metrics`.

### 6. House Price Prediction

*   **Objective:** To predict the median value of homes in California districts using data from the 1990 census.
*   **Dataset:** The California Housing dataset, which includes features like median income, housing median age, total rooms, and location (latitude/longitude) for each district.
*   **Methodology:**
    1.  **Data Preprocessing:** Missing values in the `total_bedrooms` feature were imputed using the median value. The categorical `ocean_proximity` feature was converted to numerical data using one-hot encoding.
    2.  **Feature Scaling:** All numerical features were standardized using `StandardScaler` to ensure they have a similar scale, which is important for many ML algorithms.
    3.  **Model Training:** Two different regression models were trained and compared:
        *   **Linear Regression**
        *   **Decision Tree Regressor**
    4.  **Evaluation:** Both models were evaluated using Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R2) Score. The **Decision Tree Regressor performed slightly better with an R2 score of 63.7%** compared to the Linear Regression's 62.5%.
*   **Libraries Used:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn.linear_model`, `sklearn.tree`, `sklearn.preprocessing`.

### 7. Wine Quality Prediction

*   **Objective:** To classify red wine into two categories: "good quality" (quality score of 7 or higher) or "bad quality" (quality score below 7).
*   **Dataset:** The Red Wine Quality dataset from the UCI Machine Learning Repository. It contains 11 physicochemical input variables and one output variable, 'quality' (score from 0 to 10).
*   **Methodology:**
    1.  **Data Analysis:** Exploratory data analysis was performed to understand the distribution of wine quality and the correlation between different features.
    2.  **Label Binarization:** The multiclass 'quality' target variable was converted into a binary classification problem. Wines with a quality score of 7 or above were labeled as '1' (good), and the rest were labeled as '0' (bad).
    3.  **Model Training:** A **Random Forest Classifier**, an ensemble model known for its high accuracy, was trained on the data.
    4.  **Evaluation:** The model's performance was evaluated on the test set, achieving a high **accuracy of 92.2%**.
*   **Libraries Used:** `numpy`, `pandas`, `matplotlib`, `seaborn`, `sklearn.ensemble`, `sklearn.metrics`.

### 8. Diabetes Prediction

*   **Objective:** To create a model that can predict whether a person has diabetes based on certain diagnostic medical measurements.
*   **Dataset:** The PIMA Indians Diabetes Dataset, which includes 8 diagnostic features like `Glucose`, `BloodPressure`, `BMI`, and `Age` for female patients. The outcome is binary: 1 for diabetic, 0 for non-diabetic.
*   **Methodology:**
    1.  **Data Standardization:** The feature data was standardized using `StandardScaler`. This is crucial because features are on different scales (e.g., Age vs. Glucose), and standardization ensures that each feature contributes equally to the model's decision.
    2.  **Train-Test Split:** The dataset was split into training (80%) and testing (20%) sets.
    3.  **Model Training:** A **Support Vector Machine (SVM)** with a linear kernel was trained to find the optimal hyperplane that separates diabetic and non-diabetic patients.
    4.  **Evaluation:** The model was evaluated for accuracy, achieving **78.7% on the training data** and **77.3% on the test data**.
*   **Libraries Used:** `numpy`, `pandas`, `sklearn.preprocessing`, `sklearn.model_selection`, `sklearn.svm`, `sklearn.metrics`.