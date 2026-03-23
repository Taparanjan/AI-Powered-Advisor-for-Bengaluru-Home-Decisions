# AI-Powered Advisor for Bengaluru Home Decisions

This project is a machine learning web application that estimates house prices in Bengaluru and gives a short property advisory summary in simple language. It combines a trained Linear Regression model, a Flask web app, and a lightweight chat-style assistant for follow-up guidance.

The goal is simple: enter a location, total square feet, number of bedrooms, and number of bathrooms, and the app predicts the expected house price in lakhs.

## Project Overview

The application is built for Bengaluru house price prediction using a cleaned real-estate dataset. It focuses on a practical user flow:

- predict the price of a property
- explain the result in plain English
- suggest nearby alternative locations with similar market patterns
- support follow-up questions based on the latest prediction

This makes the project useful both as a machine learning demonstration and as a beginner-friendly end-to-end deployment project.

## Features

- House price prediction using a trained `scikit-learn` Linear Regression model
- Clean web interface built with Flask
- Location-aware prediction using one-hot encoding
- Support for square-feet ranges like `1200-1400`
- Market-based alternative location suggestions
- Advisory-style summary generated from the prediction context
- Render deployment support through `render.yaml`

## Dataset

The dataset used in this project is:

- `dataset/Bengaluru_House_Data.csv`

Important dataset details:

- Total rows: `13,320`
- Main columns include: `location`, `size`, `total_sqft`, `bath`, and `price`

The dataset contains missing values, mixed text formats, and non-uniform square-feet entries, so preprocessing is an important part of the pipeline.

## Machine Learning Approach

The trained model is a `LinearRegression` model from `scikit-learn`.

### Input Features

The prediction is based on:

- `total_sqft`
- `bath`
- `bhk`
- encoded location columns

The location is converted into many binary columns using one-hot encoding. For example, if the chosen location is `Whitefield`, then:

- `Whitefield = 1`
- all other location columns = `0`

### Mathematical Model

Linear Regression assumes that price can be written as a weighted sum of the input features.

The prediction formula is:

```text
y_hat = beta_0 + beta_1 x_1 + beta_2 x_2 + beta_3 x_3 + ... + beta_n x_n
```

Where:

- `y_hat` is the predicted house price
- `beta_0` is the intercept
- `beta_1 ... beta_n` are learned coefficients
- `x_1 ... x_n` are the input features

For this project, the idea becomes:

```text
Predicted Price =
intercept
+ (coefficient for total_sqft x total_sqft)
+ (coefficient for bath x bath)
+ (coefficient for bhk x bhk)
+ (coefficient for selected location x 1)
```

This model learns how strongly each feature affects price from historical data.

### Why Linear Regression?

Linear Regression is a good choice here because:

- it is easy to understand
- it is fast to train and predict
- it works well as a baseline for tabular housing data
- the coefficients make the model easier to explain

This project uses the model in a practical way, where interpretability matters as much as prediction.

## Data Preprocessing Logic

Before prediction, the project handles the data carefully:

- square-feet values are cleaned
- ranges like `2100-2850` are converted to their average
- `bhk` is extracted from the `size` column
- invalid or missing rows are removed
- location names are trimmed and normalized
- market statistics are grouped by location for recommendation logic

This preprocessing makes the model input more consistent and helps reduce noisy predictions.

## How the App Works

The main flow of the application is:

1. The user selects a location and enters property details.
2. The app converts the inputs into the exact feature format expected by the trained model.
3. The Linear Regression model predicts the price in lakhs.
4. The app stores the latest prediction in session memory.
5. A property summary is generated for the user.
6. Similar budget-friendly nearby locations are suggested from the dataset.
7. The user can ask follow-up questions in the advisor section.

## Recommendation Logic

The project also computes nearby alternative locations using grouped market data.

For each location, it calculates values such as:

- median price
- median square feet
- median BHK
- median bathrooms
- median price per square foot
- listing count

Then it compares the current property against other locations using a scoring formula based on:

- price difference
- BHK difference
- size difference
- price-per-square-foot difference

The locations with the smallest score are shown as alternatives.

This is not a separate ML model. It is a market similarity rule built on top of the dataset.

## Project Structure

```text
AIML Powered House Price Prediction/
|-- app.py
|-- linear_regression_model.pkl
|-- requirements.txt
|-- render.yaml
|-- LICENSE
|-- README.md
|-- bangalore_house_prices_prediction.ipynb
|-- dataset/
|   `-- Bengaluru_House_Data.csv
`-- templates/
    `-- index.html
```

## Tech Stack

- Python
- Flask
- Pandas
- Scikit-learn
- Joblib / Pickle
- HTML and CSS
- Gunicorn
- Render

## Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd "AIML Powered House Price Prediction"
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

On Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root and define:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_API_URL=https://api.groq.ai/v1/requests
FLASK_SECRET_KEY=your_secret_key
```

Notes:

- `GROQ_API_KEY` is used for the advisory response layer
- if the API is unavailable, the app still returns a local fallback summary
- use a strong `FLASK_SECRET_KEY` in production

## Run the Project Locally

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Deployment

This project includes a `render.yaml` file for easy deployment on Render.

Current deployment setup:

- environment: Python
- build command: `pip install -r requirements.txt`
- start command: `gunicorn app:app`

## Model Notes

The loaded model exposes `feature_names_in_`, which is important because the app builds the prediction input in the exact same column order used during training.

This reduces feature mismatch problems and makes prediction safer.

## Limitations

- The model is only as good as the training data.
- Real market prices change over time.
- Factors like furnishing, amenities, building age, legal status, floor number, and exact locality quality are not fully modeled here.
- The prediction should be treated as an estimate, not a final valuation.

## Future Improvements

- add model evaluation metrics like RMSE, MAE, and R-squared
- compare Linear Regression with Ridge, Lasso, Random Forest, and XGBoost
- add charts and location analytics
- support loan EMI estimation
- improve recommendation scoring with geospatial distance
- retrain the model with newer market data

## Learning Value of This Project

This project is a strong example of a complete machine learning workflow:

- data cleaning
- feature engineering
- model training
- model serialization
- web app integration
- deployment

It is especially useful for students and beginners who want to understand how a mathematical ML model becomes a real user-facing product.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
