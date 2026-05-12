# Video Game Sales Analysis

## Overview

This project analyzes historical video game sales and review data to identify industry trends, regional preferences, platform performance, and publisher success.

Using Python and Tableau, the analysis explores how factors such as genre, platform, critic ratings, and geographic region influenced global video game sales.

The goal of the project is to generate business insights that could support data-driven decision-making for game publishers and developers.

---

## Business Problem
Video game publishers and developers need to understand:
- Which genres perform best globally
- Regional gaming preferences
- Top-selling platforms
- Publisher market performance
- Sales trends over time

This analysis aims to uncover patterns in video game sales that can support marketing, development, and publishing strategies.

---

## Dataset
Source: [Video Game Sales with Ratings Dataset (Kaggle)](https://www.kaggle.com/datasets/rush4ratio/video-game-sales-with-ratings/data)

The dataset includes:
- Game Title
- Platform
- Release Year
- Genre
- Publisher
- Regional Sales (NA, EU, JP, Other, Global)
- Critic Score
- User Score
- Developer
- ESRB Rating

---

## Tools Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Tableau

---

## Data Cleaning & Preparation
The following preprocessing steps were completed:

- Handled missing values using median imputation
- Converted `User_Score` to numeric format
- Applied one-hot encoding to categorical variables
- Grouped low-frequency publishers into an `"Other"` category (Publishers outside of the top 10 are set to 'Other')
- Removed redundant and non-predictive features
- Standardized numerical variables using `StandardScaler`
- Selected significant predictors using feature selection techniques

---

## Machine Learning

A stacking ensemble regression model was developed to predict global video game sales.

### Models Used

- ElasticNet
- Support Vector Regression (SVR)
- Decision Tree Regressor
- AdaBoost Regressor
- Random Forest Regressor
- Extra Trees Regressor

### Evaluation Metrics

- RMSE (Root Mean Squared Error)
- R² Score
- K-Fold Cross Validation

---

## Exploratory Data Analysis

The analysis focused on:

- Global sales trends
- Top performing video game genres
- Platform popularity
- Correlation between global sales and different data points

---

## Key Insights

- The publisher Nintendo has the largest positive impact on global sales
- Higher critic and user review scores were generally associated with stronger global sales performance
- Classic Nintendo consoles contain a small amount of games within the dataset but carry a large amount of sales
- Platform games demonstrated strong sales performance compared to several other genres

---

## Visualizations

---

## Tableau Dashboard

---

## Project Structure

```text
VideoGameSales/
│
├── data/
│   ├── processed/             
│      └── VideoGameSales_Processed_Full.csv    # Processed dataset
│
│   └── raw/                   
│      └── VideoGameSales.csv                   # Original raw dataset
│
├── scripts/
│   ├── VideoGameSales.py                       # Main Python script
│   ├── VideoGameSalesFF.py                     # Forward Feature script
│   ├── VideoGameSalesFI.py                     # Feature Importance script
│   └── VideoGameSalesRFE.py                    # Recursive Feature Elimination script
│
├── visualizations/
│   ├── dashboard/
│     ├── Critic_User.png                       # Critic and user metrics
│     ├── PlatformGenre.png                     # Platforming genre performance
│     ├── PlatformNegative.png                  # Negative correlating platforms
│     ├── PlatformPositive.png                  # Positive correlating platforms
│     └── Publishers.png                        # Publisher metrics
│
│   └── worksheet/
│     ├── CriticCount.png                       # Critic review count
│     ├── CriticScore.png                       # Critic review scores
│     ├── GenrePlatform.png                     # Platforming genre sales
│     ├── Heatmap.png                           # Heatmap for each metric against global sales
│     ├── PlatformGB.png                        # Nintendo Gameboy sales
│     ├── PlatformNES.png                       # Nintendo Entertainment System sales
│     ├── PlatformPC.png                        # Personal Computer sales
│     ├── PlatformWii.png                       # Nintendo Wii sales
│     ├── Platformx360.png                      # Microsoft Xbox 360 sales
│     ├── PublisherNintendo.png                 # Nintendo Publishing sales
│     ├── PublisherOther.png                    # Other Publishing Sales (Outside of the top 10)
│     ├── UserCount.png                         # User review count
│     └── UserScore.png                         # User review scores
│
├── README.md                                   # Project Documentation
│
└── requirements.txt                            # Project dependencies
```

---

## Future Improvements

- Build a more advanced Tableau dashboard

---

## Author

Kevin Pimentel

Junior Data Analyst  
Graduate of the BCIT ADAC Program

[LinkedIn](https://linkedin.com/in/kevin-pimentel-679085405)  
[GitHub](https://github.com/KPimentel777)
