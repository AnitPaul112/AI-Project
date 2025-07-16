# AI-Project
# 🍄 Mushroom Classification Using Machine Learning

This project analyzes a mushroom dataset to classify mushrooms as **edible** or **poisonous** based on their physical characteristics using various machine learning models.

## 📌 Objective

To build and compare machine learning models that accurately identify poisonous mushrooms using real-world data, enabling safer foraging and consumption.

## 📊 Dataset

- **Source:** [`mushroom_dataset.csv`](https://drive.google.com/file/d/19i3IoW38KBgMF96Rcnpu2ZUoO_UBwcdX/view)
- **Initial Entries:** 61,069 → **After Cleaning:** 60,923
- **Features:** Originally 21 → Used 15 after preprocessing
- **Target Variable:** `class` (`e` = edible, `p` = poisonous)
- **Feature Types:**
  - Quantitative: `cap-diameter`, `stem-height`, `stem-width`
  - Categorical (label-encoded): shape, color, bruises, gill features, habitat, season, etc.

## 🧹 Data Preprocessing

- **Missing Values:**
  - Dropped features with >40% nulls
  - Imputed remaining with **most frequent** strategy
- **Duplicate Removal:** Removed using `df.drop_duplicates()`
- **Encoding:** Label Encoding applied to categorical features
- **Feature Scaling:** Standardized numerical features using `StandardScaler`
- **Train-Test Split:** 70/30 with stratification on `class`

## 📈 Exploratory Data Analysis (EDA)

- **Correlation Heatmap**: Shows feature relationships
- **Distributions:**
  - Histograms
  - Boxplots
  - Density plots
- **Class Balance:** Slightly imbalanced (33.7k poisonous vs. 27.1k edible)

## 🧠 Models Trained & Evaluated

| Model               | Accuracy |
|--------------------|----------|
| Logistic Regression| 58%      |
| Random Forest      | 95%      |
| K-Nearest Neighbors| 95%      |
| Neural Network     | 92%      |

### 🔬 Neural Network Architecture

- 7 Dense layers + 1 Output layer
- Activation: ReLU (hidden), Sigmoid (output)
- Framework: Keras + TensorFlow backend
- Trained for 25 epochs (batch size: 64, val split: 0.2)

## 📊 Evaluation Metrics

- Accuracy, Precision, Recall, F1-Score
- Confusion Matrices (visualized)
- ROC Curve & AUC scores (plotted for all models)

## 🏁 Conclusion

- **Top Performers:** Random Forest & KNN (95%)
- **Neural Network:** Performed well (92%)
- **Logistic Regression:** Underperformed (58%) due to linearity
- **Key Factor:** Proper preprocessing + model selection led to high accuracy

## 📁 Project Structure

├── data/
│ └── mushroom_dataset.csv
├── notebooks/
│ └── mushroom_classification.ipynb
├── plots/
│ └── (EDA, heatmap, ROC, CM plots)
├── README.md
└── requirements.txt


