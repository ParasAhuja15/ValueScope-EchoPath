# deep_learning_marketing_attribution
This is the project that developed for marketing attribution. It is an interpretable deep learning model for marketing problem.
# Advanced Customer Analytics: CLV Prediction and Multi-Touch Attribution

This project provides a comprehensive suite of tools for advanced customer analytics, designed to offer deep insights into customer behavior and marketing effectiveness. It combines methodologies for predicting Customer Lifetime Value (CLV) and for understanding the impact of marketing channels through Multi-Touch Attribution (MTA). These tools empower businesses to make data-driven decisions, optimize marketing spend, and enhance customer relationships.

### Features
*   **Customer Lifetime Value (CLV) Modeling**: Predicts the future value of customers based on their transaction history.
*   **Customer Segmentation**: Uses the RFM (Recency, Frequency, Monetary) model to classify customers into meaningful groups like "important value customers" and "lost customers".
*   **Marketing Response Prediction**: Employs various machine learning algorithms to predict which customers are most likely to respond to marketing campaigns.
*   **Multi-Touch Attribution (MTA)**: Implements the novel `DeepMTA` model to fairly distribute credit for conversions across all marketing touchpoints in a customer's journey.
*   **Interpretable Deep Learning**: Combines the predictive power of deep learning with cooperative game theory (Shapley values) to provide understandable explanations for attribution results.

## Component 1: Customer Lifetime Value (CLV) Analysis

#### Overview
This component focuses on estimating the total value a customer will bring to a business over their entire relationship. By analyzing past purchasing behavior, the model can identify the most profitable customers, allowing for targeted marketing efforts and improved service. The analysis is tailored for non-contractual business settings, such as online retail, where customers can make purchases at any time.

#### Methodology
*   **RFM Analysis**: The foundation of the customer segmentation is the RFM model, which scores customers based on three key dimensions:
    *   **Recency (R)**: How recently a customer made a purchase.
    *   **Frequency (F)**: How often they make purchases.
    *   **Monetary (M)**: How much money they spend.
*   **Probabilistic Models**: To predict future behavior, the project leverages two sophisticated models:
    *   **BG/NBD (Beta Geometric/Negative Binomial Distribution)**: This model predicts the number of future transactions a customer is expected to make, considering that customers can become inactive at any time.
    *   **Gamma-Gamma Model**: Used in conjunction with the BG/NBD model, it estimates the average monetary value of a customer's transactions.
*   **Machine Learning for Customer Segmentation**: After initial segmentation with RFM, machine learning algorithms like Logistic Regression, Decision Trees, SVM, and Gradient Boosting are used to refine customer groups and predict their response to marketing initiatives.

#### Data Processing
The project includes a robust data mining pipeline for preparing transactional data for analysis. This involves:
1.  **Data Selection**: Identifying relevant attributes from the source dataset.
2.  **Data Preprocessing**: Handling missing values, removing duplicate entries, and correcting outliers (e.g., canceled orders).
3.  **Data Transformation**: Calculating R, F, and M values for each customer from raw transaction logs.

## Component 2: Multi-Touch Attribution (MTA) with DeepMTA

#### Overview
In a typical customer journey, users interact with multiple marketing channels (e.g., paid search, social media, affiliates) before making a purchase. The MTA component aims to solve the critical problem of assigning proper credit to each of these touchpoints. Unlike traditional models like last-click, the `DeepMTA` model provides a data-driven, interpretable solution.

#### Methodology: The DeepMTA Model
`DeepMTA` is a novel, two-stage model that combines deep learning for prediction with a framework for interpretation:

1.  **Conversion Prediction Model**:
    *   A **Phased Long Short-Term Memory (Phased-LSTM)** network is used to model the customer journey. This type of RNN is specifically chosen to handle the varying time intervals between touchpoints and to capture the sequence, frequency, and time-decay effects of marketing interactions.
    *   The model is trained to predict the likelihood of conversion at each step of the journey, achieving 91% Area Under the Curve (AUC) on a real-world dataset.

2.  **Interpretation Model**:
    *   To explain the predictions of the "black box" deep learning model, an **additive feature attribution model** based on **Shapley values** is used.
    *   This method, borrowed from cooperative game theory, calculates the marginal contribution of each marketing touchpoint to the conversion outcome. It effectively assigns an "importance" weight to each channel in the journey.
    *   This makes `DeepMTA` the first model to combine deep learning with game theory for interpretable multi-touch attribution.

### Getting Started

To use this project, you would typically follow these steps:

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/advanced-customer-analytics.git
    ```
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Prepare your data**:
    *   For CLV analysis, provide a transactional dataset with customer ID, order date, and sales amount.
    *   For MTA, provide a customer journey dataset with user ID, touchpoint channel, and timestamps.
4.  **Run the analysis**:
    *   Execute the Jupyter notebooks or Python scripts for either the CLV or MTA analysis.

### Key Contributions
*   **Holistic Customer View**: Provides a dual perspective on customer value, looking at both future potential (CLV) and the marketing drivers behind it (MTA).
*   **Practical and Accurate CLV**: Implements a robust workflow for CLV calculation that addresses common theoretical and data-related challenges, improving the accuracy of predictions.
*   **Interpretable AI for Marketing**: Introduces `DeepMTA`, a pioneering deep learning model for attribution that is not only highly accurate but also fully interpretable, explaining *why* it makes certain predictions[2].
