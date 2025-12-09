# Assignment 4: Bakeoff! XGBoost vs Neural Networks Across City Markets

## Project Overview
This project compares XGBoost and Neural Networks for predicting Airbnb listing prices across 12 cities stratified into three tiers: Big, Medium, and Small cities. The analysis evaluates model performance within and across market segments to determine which approach generalizes better under different market conditions.

## Learning Objectives
- Implement neural networks using Keras across multiple market contexts
- Conduct stratified model comparison across city sizes and market types
- Test model robustness and generalization capabilities
- Perform advanced evaluation comparing performance within and across market segments
- Handle real-world data science complexity of multi-market model validation
- Analyze feature importance in different market contexts

## Dataset
Data sourced from Inside Airbnb summary CSV files for the following cities:

| City          | Tier    | Data Month | Listings Count |
|---------------|---------|------------|----------------|
| New York City | Big     | Sept 2024  | 36,111         |
| Los Angeles   | Big     | Sept 2024  | 45,886         |
| San Francisco | Big     | Sept 2024  | 7,780          |
| Chicago       | Big     | Sept 2024  | 8,604          |
| Austin        | Medium  | Oct 2024   | 15,187         |
| Seattle       | Medium  | Oct 2024   | 6,996          |
| Denver        | Medium  | Oct 2024   | 4,910          |
| Portland      | Medium  | Oct 2024   | 4,425          |
| Asheville     | Small   | Oct 2024   | 2,876          |
| Santa Cruz    | Small   | Oct 2024   | 1,739          |
| Salem         | Small   | Oct 2024   | 351            |
| Columbus      | Small   | Oct 2024   | 2,877          |

## Features Used
**Base Features:**
- accommodates, bedrooms, beds, bathrooms_text
- review_scores_rating, review_scores_accuracy, review_scores_cleanliness, review_scores_checkin, review_scores_communication, review_scores_location, review_scores_value
- number_of_reviews, availability_365, minimum_nights, maximum_nights

**Additional Engineered Features (minimum 4):**
1. price_per_bedroom = price / bedrooms (where bedrooms > 0)
2. avg_review_score = average of all review scores
3. is_entire_home = 1 if room_type == 'Entire home/apt', else 0
4. amenities_count = number of amenities listed
5. property_type_encoded = label encoded property_type
6. neighbourhood_encoded = label encoded neighbourhood_cleansed

## Models Compared
1. **XGBoost**: Gradient boosting with hyperparameter tuning
2. **Neural Networks**: Two architectures experimented with:
   - Architecture 1: 128-64-32-1 layers
   - Architecture 2: 256-128-64-32-1 layers with dropout

## Major Findings

### Individual City Performance
Neural Networks (Architecture 1: 128-64-32-1) generally outperform XGBoost in larger markets, while XGBoost performs better in smaller markets. For example:
- **Big Cities**: NN1 achieves lower RMSE in NYC (325.15 vs 519.48), LA (266.45 vs 569.14), SF (184.04 vs 199.38), Chicago (402.88 vs 888.24)
- **Small Cities**: XGBoost dominates in Asheville (23.81 vs 33.49), Salem (10.98 vs 37.45), Columbus (36.43 vs 659.33)

### Tier-Level Analysis
- **Big Tier**: NN RMSE=176.67, MAE=100.08, R²=1.00 vs XGBoost RMSE=423.75, MAE=30.65, R²=0.98
- **Medium Tier**: XGBoost RMSE=293.39, MAE=21.12, R²=0.99 vs NN RMSE=252.43, MAE=99.12, R²=0.99  
- **Small Tier**: XGBoost RMSE=275.52, MAE=12.88, R²=0.99 vs NN RMSE=779.40, MAE=124.54, R²=0.89

### Cross-Tier Generalization
Models trained on larger markets generalize better to smaller markets:
- Train Big → Test Small: NN RMSE=161.46, MAE=117.98, R²=1.00 vs XGBoost RMSE=236.71, MAE=15.37, R²=0.99
- Train Medium → Test Small: NN RMSE=241.01, MAE=107.14, R²=0.99 vs XGBoost RMSE=521.57, MAE=26.42, R²=0.96
Models trained on smaller markets perform poorly on larger markets (e.g., Small → Big: NN R²=0.13, XGBoost R²=0.77), indicating market-specific patterns.

## Results Summary

### Individual City RMSE Scores
| City | XGBoost | NN1 (128-64-32-1) | NN2 (256-128-64-32-1) |
|------|---------|-------------------|-----------------------|
| New York City | 519.48 | 325.15 | 541.51 |
| Los Angeles | 569.14 | 266.45 | 364.68 |
| San Francisco | 199.38 | 184.04 | 248.81 |
| Chicago | 888.24 | 402.88 | 426.34 |
| Austin | 1327.03 | 284.12 | 389.94 |
| Seattle | 560.51 | 1105.27 | 955.66 |
| Denver | 139.53 | 74.58 | 81.78 |
| Portland | 177.43 | 567.30 | 524.14 |
| Asheville | 23.81 | 33.49 | 50.59 |
| Santa Cruz | 88.49 | 1605.83 | 869.73 |
| Salem | 10.98 | 37.45 | 34.84 |
| Columbus | 36.43 | 659.33 | 719.92 |

### Tier-Level RMSE Scores
| Tier | XGBoost | Neural Network |
|------|---------|----------------|
| Big | 423.75 | 126.04 |
| Medium | 293.39 | 193.57 |
| Small | 275.52 | 720.98 |

### Cross-Tier Generalization RMSE
| Train → Test | XGBoost | Neural Network |
|---------------|---------|----------------|
| Big → Medium | 338.64 | 153.94 |
| Big → Small | 236.71 | 119.06 |
| Medium → Big | 723.09 | 344.61 |
| Medium → Small | 521.57 | 193.61 |
| Small → Big | 1497.27 | 2909.65 |
| Small → Medium | 1430.12 | 1724.50 |

## Instructions to Run the Code

### Prerequisites
- Python 3.8+
- Required libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, tensorflow, shap

### Setup
1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Jupyter notebook: `jupyter notebook notebooks/airbnb_price_prediction.ipynb`

### Data Download
To reproduce this analysis with your own data:

1. Visit http://insideairbnb.com/get-the-data.html
2. Download the summary `listings.csv` files for each city using the months specified in the dataset table above.
3. Rename the downloaded files to `{City_Name}_listings.csv` format (e.g., `New_York_City_listings.csv`, replacing spaces with underscores).
4. Place all CSV files in the `data/` directory.
5. Run the notebook, which will load the data locally from the `data/` folder.

## Repository Structure
```
├── README.md
├── requirements.txt
├── notebooks/
│   └── airbnb_price_prediction.ipynb
├── data/
│   └── (CSV files as described above)
└── src/
    └── (helper functions if any)
```