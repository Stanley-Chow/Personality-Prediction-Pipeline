# Behavioral Personality Analytics

This data science project explores the fascinating intersection between day-to-day human behavioral data and underlying personality traits, focusing on predicting introversion and extraversion based on social patterns.

---

## Project Overview

Our personalities heavily influence how we interact with the world—from the size of our friend circles to our social media habits. Using the **Extrovert vs. Introvert Personality Traits Dataset**, this project uncovers the structural links between offline social habits and online behavior. We structured our investigation around four core analytical problems:

1. **Predicting Stage Fright** using underlying personality indicators.


2. **Mapping Offline Social Exposure to Online Activity** via social media posting frequency regression.


3. **Optimizing Feature Selection** to identify the most parsimonious subset of behaviors for personality classification.


4. **Full-Feature Personality Classification** to evaluate the maximum predictive upper bound of our models.



---

## Dataset & Preprocessing

The dataset originates from the Kaggle course competition *Extrovert vs Introvert Behavior Data*, containing a training set of **18,524 rows** and a test set of **6,175 rows**. The observational unit represents an individual respondent.

### Features Handled:

* `Time_spent_Alone`: Average hours spent alone per day (Numerical).


* `Stage_fear`: Fear of public speaking/performing (Categorical).


* `Social_event_attendance`: Weekly social event count (Scaled 0-10).


* `Going_outside`: Weekly frequency of going outside (Scaled 0-7).


* `Drained_after_socializing`: Post-socializing exhaustion (Categorical).


* `Friends_circle_size`: Count of close friends (Scaled 0-15).


* `Post_frequency`: Weekly social media post count (Scaled 0-10).


* `Personality`: Target label (**1 = Extrovert**, **0 = Introvert**).



### Data Preparation Pipeline:

* **Imputation**: Categorical missing entries were filled using mode imputation (e.g., `Stage_fear`, `Drained_after_socializing`). Numerical missing features utilized median imputation to guarantee robustness against outliers.


* **Encoding & Scaling**: Binary categorical features were mapped to $1/0$. Continuous variables were normalized using `StandardScaler`.


* **Leakage Prevention**: All transformations, scaling, and hyperparameter tuning were securely wrapped inside `scikit-learn` Pipelines to guarantee zero data leakage during cross-validation.



---

## Methodology & Results

### Problem 1: Prediction of Stage Fright (Classification)

* **Goal**: Find the top 3 independent variables to predict public speaking anxiety.


* **Predictors**: `Personality`, `Drained_after_socializing`, `Time_spent_Alone`.


* **Top Model**: **K-Nearest Neighbors (KNN)** via `GridSearchCV`.



| Model | Accuracy | Precision | Recall | F1-Score |
| --- | --- | --- | --- | --- |
| **KNN Classifier** | **0.9933** | **0.9842** | **0.9883** | **0.9862** |
| Logistic Regression | 0.9929 | 0.9832 | 0.9876 | 0.9854 |
| Random Forest Classifier | 0.9930 | 0.9827 | 0.9883 | 0.9855 |

(Note: Data sourced from Table 1 of the report )

### Problem 2: Offline Social Activity vs. Post Frequency (Regression)

* **Goal**: Predict weekly social media posting frequency using only two offline actions (`Social_event_attendance` and `Going_outside`).


* **Top Model**: **Feedforward Neural Network (MLPRegressor)**.



| Model | Cross-Validated MSE | RMSE |
| --- | --- | --- |
| Linear Regression | 5.0059 | 2.2374 |
| Random Forest Regressor | 3.9872 | 1.9968 |
| **Neural Network (MLP)** | **3.9826** | **1.9956** |

(Note: Data sourced from Table 2 of the report )

> 
> **Insight**: Flexible, non-linear algorithms significantly cut down Mean Squared Error (MSE), signaling complex interactions between offline exposure and online posting.
> 
> 

### Problem 3: Feature Selection for Personality Prediction

* **Goal**: Minimize data-collection cost by finding the single most optimal, parsimonious feature subset using exhaustive search across $2^7 - 1 = 127$ combinations.


* **Top Configuration**: **Gradient Boosting Classifier** leveraging exactly **5 core features**.


* **Optimal Feature Set**: `Time_spent_Alone`, `Stage_fear`, `Social_event_attendance`, `Going_outside`, `Drained_after_socializing`.


* **Result**: Achieved an outstanding **ROC-AUC of 0.9650** on the hold-out test set.



### Problem 4: Full-Feature Personality Prediction

* **Goal**: Maximize classification accuracy utilizing all behavioral dimensions available in the dataset.


* **Top Model**: **Random Forest / Neural Network** (Minimal performance differences observed across models).



| Model | Accuracy | Precision | Recall | F1-Score |
| --- | --- | --- | --- | --- |
| KNN Classifier | 0.9689 | 0.9773 | 0.9807 | 0.9790 |
| Logistic Regression | 0.9685 | 0.9770 | 0.9805 | 0.9788 |
| **Random Forest Classifier** | **0.9691** | **0.9777** | **0.9806** | **0.9792** |

(Note: Data sourced from Table 4 of the report )

---

## Key Takeaways & Discussion

1. **High Signal-to-Noise Ratio**: Simple, well-chosen behavioral data carries an incredibly robust signal for identifying latent traits like introversion/extraversion.


2. **Parsimony Wins**: Gathering exhaustive data isn't necessary. Dropping from the full dataset to a compact 5-feature battery of questions yielded nearly identical predictive capacities (AUC 0.965). This is highly practical for fast screening applications.


3. **Interpretability vs. Performance**: While non-linear architectures like Gradient Boosting and Multilayer Perceptrons excelled at capturing subtle behavior interactions, simple models like Logistic Regression and KNN competed aggressively. This presents a useful deployment tradeoff depending on whether transparency or absolute predictive power is the goal.
