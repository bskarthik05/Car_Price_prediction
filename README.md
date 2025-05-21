# Car Price Prediction using Linear and Lasso Regression

## Project Overview

This project aims to accurately predict car prices based on various features using machine learning techniques. We leverage both **Linear Regression** and **Lasso Regression** to build robust predictive models, explore their performance, and understand the factors influencing car pricing. The goal is to provide insights into the automotive market and offer a tool for potential buyers or sellers to estimate car values.

## Table of Contents

  - [Project Overview](https://www.google.com/search?q=%23project-overview)
  - [Table of Contents](https://www.google.com/search?q=%23table-of-contents)
  - [Problem Statement](https://www.google.com/search?q=%23problem-statement)
  - [Dataset](https://www.google.com/search?q=%23dataset)
  - [Features](https://www.google.com/search?q=%23features)
  - [Methodology](https://www.google.com/search?q=%23methodology)
      - [1. Data Loading and Initial Exploration](https://www.google.com/search?q=%231-data-loading-and-initial-exploration)
      - [2. Exploratory Data Analysis (EDA)](https://www.google.com/search?q=%232-exploratory-data-analysis-eda)
      - [3. Data Preprocessing](https://www.google.com/search?q=%233-data-preprocessing)
      - [4. Model Training](https://www.google.com/search?q=%234-model-training)
      - [5. Model Evaluation](https://www.google.com/search?q=%235-model-evaluation)
  - [Key Findings and Insights](https://www.google.com/search?q=%23key-findings-and-insights)
  - [Technologies Used](https://www.google.com/search?q=%23technologies-used)
  - [How to Run](https://www.google.com/search?q=%23how-to-run)
  - [Future Enhancements](https://www.google.com/search?q=%23future-enhancements)
  - [Contributing](https://www.google.com/search?q=%23contributing)
  - [License](https://www.google.com/search?q=%23license)
  - [Contact](https://www.google.com/search?q=%23contact)

## Problem Statement

Predicting car prices is a complex task influenced by a multitude of factors such as brand, model, year, mileage, engine specifications, and more. This project addresses the challenge of building a reliable model that can accurately estimate car prices, which can be beneficial for:

  * **Individuals:** To get an unbiased estimate before buying or selling a car.
  * **Car Dealerships:** To price their inventory competitively.
  * **Insurance Companies:** For accurate vehicle valuation.
  * **Financial Institutions:** For loan approvals based on vehicle collateral.

## Dataset

The dataset used for this project contains various attributes related to cars and their corresponding prices.
**Source:** Kaggle
**File Name:** `CAR DETAILS FROM CAR DEKHO.csv`

The dataset includes both numerical and categorical features that capture different aspects of a car.

## Features

The dataset comprises the following key features:

  * `car_name`: The name of the car model.
  * `year`: The year of manufacture of the car.
  * `selling_price`: The selling price of the car (this is our target variable).
  * `km_driven`: The total kilometers driven by the car.
  * `fuel`: The type of fuel the car uses (Diesel, Petrol, CNG, LPG, Electric).
  * `seller_type`: The type of seller (Individual, Dealer, Trustmark Dealer).
  * `transmission`: The type of transmission (Manual, Automatic).
  * `owner`: The number of previous owners of the car (First Owner, Second Owner, Third Owner, Fourth & Above Owner, Test Drive Car).

**Note:** The 'Test Drive Car' owner type might represent cars primarily used for test drives and could have unique pricing characteristics. This will be considered during analysis.

## Methodology

The project follows a standard machine learning pipeline:

### 1\. Data Loading and Initial Exploration

  * Loading the dataset using Pandas.
  * Initial inspection of data types, missing values, and basic statistics (`.info()`, `.describe()`, `.isnull().sum()`).

### 2\. Exploratory Data Analysis (EDA)

  * **Univariate Analysis:**
      * Distribution of numerical features (`year`, `km_driven`, `selling_price`) using histograms and box plots.
      * Frequency counts of categorical features (`fuel`, `seller_type`, `transmission`, `owner`) using bar plots.
  * **Bivariate Analysis:**
      * Relationship between `km_driven`, `year` and `selling_price` (scatter plots, correlation matrix/heatmap).
      * Relationship between categorical features (`fuel`, `seller_type`, `transmission`, `owner`) and `selling_price` (box plots, violin plots to compare price distributions across categories).
  * **Feature Engineering:**
      * Created a "Car\_Age" feature from the "year" column by subtracting the car's manufacturing year from the current year (or a chosen reference year, e.g., 2024 or 2025). This directly represents the car's age, which is often a strong predictor of price.
      * Considered extracting brand from `car_name` for further analysis if deemed relevant.

### 3\. Data Preprocessing

  * **Handling Missing Values:** 
  * **Categorical Feature Encoding:**
      * Applied **One-Hot Encoding** to nominal categorical features such as `fuel`, `seller_type`, and `transmission`. This converts categorical variables into a format that can be provided to ML algorithms to improve prediction accuracy.
      * Applied **Ordinal Encoding** (or a similar approach if numerical mapping was more suitable) to the `owner` feature, as it has an inherent order (First Owner \< Second Owner, etc.). 'Test Drive Car' might be treated as a separate category or mapped carefully.
  * **Feature Scaling:**
      * Applied **Standard Scaling** to numerical features (`Car_Age`, `km_driven`) to normalize their range and ensure no single feature dominates the model training due to its scale.
  * **Train-Test Split:** Divided the dataset into training and testing sets (typically 80% training, 20% testing) to evaluate model performance on unseen data.

### 4\. Model Training

  * **Linear Regression:**
      * Trained a standard **Linear Regression** model. This serves as a baseline model, assuming a linear relationship between features and the target variable.
  * **Lasso Regression:**
      * Trained a **Lasso Regression** (Least Absolute Shrinkage and Selection Operator) model. Lasso is a regularization technique that adds a penalty equivalent to the absolute value of the magnitude of coefficients. This helps in:
          * **Feature Selection:** Shrinks the coefficients of less important features to zero, effectively performing feature selection.
          * **Preventing Overfitting:** Reduces model complexity by penalizing large coefficients.
      * Used **Grid Search Cross-Validation** to find the optimal `alpha` parameter for Lasso, which controls the strength of the regularization.

### 5\. Model Evaluation

The trained models were evaluated using the following metrics:

  * **Mean Absolute Error (MAE):** Measures the average absolute difference between predicted and actual values. It's robust to outliers.
  * **Mean Squared Error (MSE):** Measures the average of the squares of the errors. Penalizes larger errors more.
  * **Root Mean Squared Error (RMSE):** The square root of MSE, providing the error in the same units as the target variable.
  * **R-squared ($R^2$):** Represents the proportion of the variance in the dependent variable that is predictable from the independent variables. A higher $R^2$ indicates a better fit.

`[Mention any other evaluation techniques you specifically used, e.g., 'Residual plots were inspected to ensure assumptions of linearity and homoscedasticity were met.']`

## Key Findings and Insights

  * `[Summarize most important findings]`
      * "Car age and `km_driven` show a strong negative correlation with `selling_price`; older cars with more mileage tend to be cheaper."
      * "Certain `fuel` types, like Diesel, often command higher `selling_price` than Petrol in this dataset, indicating market preference or higher initial cost."
      * "`seller_type` significantly impacts the price, with 'Dealer' and 'Trustmark Dealer' potentially listing cars at higher prices than 'Individual' sellers due to added services or guarantees."
      * "The `owner` variable clearly shows a decreasing `selling_price` as the number of owners increases, with 'First Owner' cars having the highest value."
      * "'Test Drive Car' as an owner type might require special handling, as its price distribution could be unique compared to regularly owned vehicles."
      * "Lasso Regression, with its regularization, performed slightly better than standard Linear Regression in terms of `[mention specific metric, e.g., 'RMSE']`, indicating its effectiveness in handling multicollinearity and selecting relevant features."
      * "The optimal `alpha` for Lasso identified during hyperparameter tuning was `[your alpha value]`, leading to `[X number]` features being selected by shrinking the coefficients of less impactful features to zero."
      * "Linear Regression achieved an $R^2$ of `[X]`, MAE of `[Y]`, and RMSE of `[Z]`. Lasso Regression, on the other hand, resulted in an $R^2$ of `[A]`, MAE of `[B]`, and RMSE of `[C]`, demonstrating a `[slight/significant]` improvement."
      * "Lasso Regression assigned higher coefficients to `Car_Age`, `km_driven`, and specific `fuel` and `transmission` types (e.g., 'Diesel', 'Automatic'), confirming their strong influence on car `selling_price`."

## Technologies Used

  * **Python** `3.9`
  * **Jupyter Notebook** 
  * **Pandas**  - For data manipulation and analysis.
  * **NumPy** - For numerical operations.
  * **Scikit-learn**  - For machine learning models (Linear Regression, Lasso, preprocessing, model selection).
  * **Matplotlib** - For data visualization.
  * **Seaborn**  - For enhanced data visualization.

## How to Run

1.  **Open the Jupyter Notebook :**
    ```bash
    jupyter notebook
    ```
    Then navigate to `[car_price_predict.ipynb]` and run all cells.

The notebook will:

  * Load the `CAR DETAILS FROM CAR DEKHO.csv` dataset.
  * Perform data preprocessing, including feature engineering for 'Car\_Age' and encoding of categorical variables.
  * Train Linear and Lasso Regression models.
  * Evaluate models and print performance metrics.

## Future Enhancements

  * **Explore other regression models:** Tree-based models (Random Forest, Gradient Boosting, XGBoost) could potentially capture non-linear relationships better.
  * **Deep Learning Models:** Investigate neural networks for potential improvements, especially with more complex feature interactions.
  * **More Advanced Feature Engineering:** Extract more detailed information from `car_name` (e.g., brand, sub-model) to create new features.
  * **Hyperparameter Tuning:** More extensive hyperparameter tuning using advanced techniques (e.g., Bayesian Optimization) for Lasso and other models.
  * **Deployment:** Deploy the trained model as a web application (e.g., using Flask or FastAPI) for interactive predictions.
  * **Data Acquisition:** Implement web scraping to gather more up-to-date car data, as car prices fluctuate.
  * **Interpretability:** Use techniques like SHAP or LIME to explain individual model predictions and feature contributions.

## Contributing

Contributions are welcome\! If you have suggestions for improvements, new features, or bug fixes, please open an issue or submit a pull request.

1.  Fork the repository.
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

## License

This project is licensed under the `MIT License` - see the `LICENSE` file for details.

## Contact

`Sathya Karthik Byrisetty` - `sathyakarthik1354@gmail.com` - `[LinkedIn (https://www.linkedin.com/in/sathyakarthik05/)]`
Project Link: `https://github.com/bskarthik05/Car_Price_prediction`

-----
