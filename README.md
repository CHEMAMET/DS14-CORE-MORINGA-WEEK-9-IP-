# Spam Email Classification using Naive Bayes

A machine learning project that implements Gaussian Naive Bayes algorithm to classify emails as spam or non-spam with high accuracy.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Model Performance](#model-performance)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Project Overview

This project addresses the problem of spam email detection using machine learning techniques. Spam emails can clutter inboxes with irrelevant marketing material, fake news, and unsavory content. This classifier helps identify spam characteristics and flag unwanted emails while maintaining high accuracy to avoid misclassifying legitimate emails.

### Research Question
Use the Naive Bayes algorithm to predict the probability of an email being spam or non-spam.

### Success Metric
The model is considered successful if it achieves an accuracy score of at least 80%.

## 📊 Dataset

- **Source**: UCI Machine Learning Repository
- **Creators**: Mark Hopkins, Erik Reeber, George Forman, Jaap Suermondt (Hewlett-Packard Labs)
- **Generated**: June-July 1999
- **Size**: 4,601 emails with 58 features
- **Features**: 
  - 54 continuous features (0-100): percentage of words matching specific terms
  - 3 run-length attributes (55-57): length of consecutive capital letters
  - 1 target variable: spam classification (0=not spam, 1=spam)

### Dataset Characteristics
- **No missing values**: Complete dataset requiring no imputation
- **Class distribution**: 2,531 non-spam emails, 1,679 spam emails
- **Data cleaning**: 391 duplicate records removed

## 🛠 Installation

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

### Required Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
```

## 🚀 Usage

1. **Load the dataset**:
```python
df = pd.read_csv("spambase.data", names=spam_columns)
```

2. **Run the preprocessing**:
```python
# Remove duplicates
df.drop_duplicates(keep='first', inplace=True)

# Convert target to categorical
data = df.astype({'spam': 'category'})
```

3. **Train the model**:
```python
# Split features and target
X = data.drop(columns=['spam'])
y = data['spam']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=12)

# Initialize and train Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Make predictions
y_pred = gnb.predict(X_test)
```

## 🔬 Methodology

### 1. Data Preparation
- **Duplicate removal**: 391 duplicate records identified and removed
- **Missing value check**: No missing values found
- **Data type conversion**: Target variable converted to categorical

### 2. Exploratory Data Analysis
- **Distribution analysis**: All features showed positive skewness
- **Class imbalance**: 60.1% non-spam, 39.9% spam emails
- **Feature correlation**: Identified multicollinear variables

### 3. Algorithm Selection
- **Gaussian Naive Bayes**: Chosen due to continuous feature values
- **Baseline comparison**: Logistic Regression used as baseline (92.5% accuracy)

### 4. Model Optimization
- **Multicollinearity handling**: Removed highly correlated features (`word_freq_857`, `word_freq_415`)
- **Train-test split optimization**: Tested various split ratios (10%, 20%, 30%, 40%, 50%)
- **Normalization testing**: Found that normalization decreased performance

## 📈 Results

### Model Performance Summary

| Model Configuration | Accuracy Score |
|-------------------|---------------|
| Baseline (Logistic Regression) | 92.5% |
| Gaussian NB (no split) | 82.8% |
| Gaussian NB (80-20 split) | 84.2% |
| Gaussian NB (optimized, 80-20 split) | **84.6%** |

### Key Findings

1. **Success Criteria Met**: Final model achieved 84.6% accuracy, exceeding the 80% threshold
2. **Optimal Split**: 80-20 train-test split provided the best performance
3. **Feature Selection Impact**: Removing multicollinear variables improved accuracy by 0.4%
4. **Normalization Effect**: Normalization reduced model performance significantly

### Detailed Performance by Test Size

| Test Size | Accuracy (Original) | Accuracy (Optimized) |
|-----------|-------------------|---------------------|
| 10% | 83.3% | 84.3% |
| 20% | 84.2% | **84.6%** |
| 30% | 83.3% | 83.8% |
| 40% | 83.5% | 84.3% |
| 50% | 83.3% | 84.1% |

## 🏆 Model Performance

### Strengths
- ✅ Meets success criteria (>80% accuracy)
- ✅ Good balance between precision and recall
- ✅ Computationally efficient
- ✅ Handles continuous features well

### Limitations
- ⚠️ Lower accuracy compared to baseline logistic regression
- ⚠️ Assumes feature independence (Naive Bayes assumption)
- ⚠️ Dataset from 1999 may not reflect modern spam patterns

### Recommendations for Production
1. Consider ensemble methods for improved accuracy
2. Implement cross-validation for robust performance estimation
3. Regular model retraining with current spam patterns
4. Monitor for concept drift in spam characteristics

## 🔧 Technical Details

### Feature Engineering
- **VIF Analysis**: Identified multicollinear features using Variance Inflation Factor
- **Correlation Matrix**: Visualized feature relationships using heatmaps
- **Feature Removal**: Eliminated `word_freq_857` and `word_freq_415` due to high correlation

### Algorithm Justification
- **Multinomial NB**: Rejected due to non-discrete features
- **Bernoulli NB**: Rejected due to non-binary features
- **Gaussian NB**: Selected for continuous feature handling

## 📝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- UCI Machine Learning Repository for providing the dataset
- Hewlett-Packard Labs for the original research and data collection
- Mark Hopkins, Erik Reeber, George Forman, and Jaap Suermondt for dataset creation

---

**Note**: This model was trained on data from 1999. For production use, consider training on more recent spam data to account for evolving spam patterns and techniques.
