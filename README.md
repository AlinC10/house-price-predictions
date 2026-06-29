# House Price Prediction

![Jupyter Notebook](https://img.shields.io/badge/Jupyter-F37626.svg?logo=Jupyter&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.14-blue.svg?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-green.svg?logo=pandas&logoColor=white)

This project was part of the [Intro to Machine Learning Course](https://www.kaggle.com/learn/intro-to-machine-learning)
from [kaggle.com](https://www.kaggle.com/learn).

## Description

Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or
the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price
negotiations than the number of bedrooms or a white-picket fence.

With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition
challenges you to predict the final price of each home.

## How it works

This project is built as an automated script that cleans the data, finds the most useful information,
and trains a prediction model.

The code is broken down into 4 main steps:

1. **Translating Text to Numbers:** Machine learning models cannot read text. I used Pandas (`pd.get_dummies()`) to
   automatically scan columns containing text (like Neighborhoods or House Styles) and convert them into Yes/No (1 or 0)
   numerical columns.
2. **Finding the "Sweet Spot" for Features:** Throwing too much data at a model can confuse it. I wrote a loop that uses
   Scikit-Learn's `SelectKBest` to test the model using just 1 feature, then 2 features, all the way up to the maximum.
   By
   checking the error rate of every single loop, the script automatically finds the exact number of features that gives
   the most accurate predictions.
3. **Handling Missing Test Data:** In the real world, test data is often missing information. Before making final
   predictions, the script finds any blank spots (`NaN`) in the test file and fills them with the average value of that
   column so the model doesn't crash.
4. **Training the Final Model:** Once the optimal number of features is found, the script trains a
   `RandomForestRegressor` on those specific columns and generates the final price predictions.

## Results

By implementing automated feature selection and categorical encoding, the model's accuracy improved significantly
compared to a raw numerical baseline:
* **First model error:** $19.029
* **Optimized model Error:** $16.821

## How to run

Follow the next step for running the code:

1. Install the necessary libraries from the requirements.txt: `pip install -r requirements.txt`
2. Run **main.py** using the Run button or by using the terminal: `python main.py`. The code can also be run from the
   Jupyter Notebook file **notebook.ipynb**. Also, the notebook contains the first version of the Prediction Model used
   and its improvement.
3. Check **submission.csv** for the results.
