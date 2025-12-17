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
Neural Networks generally outperform XGBoost in larger markets, while XGBoost performs better in smaller markets. Key findings:
- **Big Cities**: NN1 shows better performance in most cases (e.g., LA: NN1 RMSE=266.45 vs XGB 569.14)
- **Small Cities**: XGBoost dominates (e.g., Salem: XGB RMSE=10.98 vs NN1 37.45)
- **NN Architecture 1** (128-64-32-1) typically performs best among NN variants

### Tier-Level Analysis
- **Big Tier**: XGBoost RMSE=2046.74, MAE=411.06, R²=0.54 vs NN1 RMSE=2764.71, MAE=557.31, R²=0.16
- **Medium Tier**: XGBoost RMSE=2124.02, MAE=345.06, R²=0.56 vs NN1 RMSE=2729.00, MAE=411.45, R²=0.27
- **Small Tier**: XGBoost RMSE=1647.03, MAE=262.55, R²=0.49 vs NN1 RMSE=1990.53, MAE=290.78, R²=0.25

### Cross-Tier Generalization
Models show poor generalization across different market tiers, with negative R² values indicating worse performance than a simple mean predictor:
- Train Big → Test Medium: XGBoost R²=0.09, NN1 R²=0.09
- Train Big → Test Small: XGBoost R²=0.02, NN1 R²=0.04
- Train Medium → Test Big: XGBoost R²=0.05, NN1 R²=0.02
- Train Medium → Test Small: XGBoost R²=-0.48, NN1 R²=-0.09
- Train Small → Test Big: XGBoost R²=-0.07, NN1 R²=-0.02
- Train Small → Test Medium: XGBoost R²=-0.09, NN1 R²=-0.04

## Results Summary

### Individual City Performance Metrics
| City | XGBoost RMSE | XGBoost MAE | XGBoost R² | NN1 RMSE | NN1 MAE | NN1 R² | NN2 RMSE | NN2 MAE | NN2 R² |
|------|--------------|-------------|------------|----------|---------|--------|----------|---------|--------|
| New York City | 2542.53 | 422.43 | 0.68 | 3051.99 | 545.17 | 0.53 | 2918.26 | 518.28 | 0.57 |
| Los Angeles | 1403.37 | 268.41 | 0.45 | 1719.18 | 330.73 | 0.18 | 1739.44 | 299.34 | 0.16 |
| San Francisco | 1418.66 | 255.91 | 0.10 | 1511.73 | 330.24 | -0.02 | 853.16 | 188.03 | 0.67 |
| Chicago | 1982.21 | 391.16 | 0.73 | 2418.39 | 533.31 | 0.60 | 2124.67 | 327.76 | 0.69 |
| Austin | 937.19 | 252.61 | 0.84 | 2191.32 | 428.81 | 0.12 | 2015.49 | 358.09 | 0.26 |
| Seattle | 2313.90 | 318.20 | 0.74 | 3600.92 | 776.54 | 0.37 | 3278.46 | 446.33 | 0.48 |
| Denver | 243.45 | 87.80 | 0.04 | 244.18 | 88.37 | 0.03 | 244.07 | 88.48 | 0.03 |
| Portland | 2310.96 | 307.37 | 0.63 | 3413.36 | 748.47 | 0.20 | 2765.73 | 481.50 | 0.47 |
| Asheville | 169.82 | 92.76 | -0.07 | 155.38 | 89.83 | 0.10 | 155.10 | 91.15 | 0.11 |
| Santa Cruz | 3943.93 | 656.44 | 0.02 | 3626.92 | 781.20 | 0.17 | 3736.18 | 731.74 | 0.12 |
| Salem | 56.80 | 35.33 | -0.27 | 48.39 | 37.10 | 0.08 | 46.45 | 35.83 | 0.15 |
| Columbus | 2986.89 | 285.47 | -0.18 | 2504.39 | 359.68 | 0.17 | 2909.02 | 333.45 | -0.12 |

### Tier-Level Performance Metrics
| Tier | XGBoost RMSE | XGBoost MAE | XGBoost R² | NN1 RMSE | NN1 MAE | NN1 R² | NN2 RMSE | NN2 MAE | NN2 R² |
|------|--------------|-------------|------------|----------|---------|--------|----------|---------|--------|
| Big | 2046.74 | 411.06 | 0.54 | 2764.71 | 557.31 | 0.16 | 2649.48 | 433.16 | 0.23 |
| Medium | 2124.02 | 345.06 | 0.56 | 2729.00 | 411.45 | 0.27 | 2784.74 | 375.99 | 0.24 |
| Small | 1647.03 | 262.55 | 0.49 | 1990.53 | 290.78 | 0.25 | 1946.89 | 269.59 | 0.29 |

### Cross-Tier Generalization Metrics
| Train → Test | XGBoost RMSE | XGBoost MAE | XGBoost R² | NN1 RMSE | NN1 MAE | NN1 R² | NN2 RMSE | NN2 MAE | NN2 R² |
|---------------|--------------|-------------|------------|----------|---------|--------|----------|---------|--------|
| Big → Medium | 2967.84 | 452.15 | 0.09 | 2968.32 | 473.10 | 0.09 | 3030.52 | 391.43 | 0.05 |
| Big → Small | 2540.81 | 421.78 | 0.02 | 2515.55 | 391.31 | 0.04 | 2535.26 | 316.18 | 0.02 |
| Medium → Big | 3041.67 | 529.63 | 0.05 | 3097.65 | 569.96 | 0.02 | 3183.49 | 457.07 | -0.04 |
| Medium → Small | 3113.44 | 524.85 | -0.48 | 2677.39 | 397.03 | -0.09 | 2656.56 | 337.18 | -0.08 |
| Small → Big | 3223.41 | 485.21 | -0.07 | 3155.75 | 392.52 | -0.02 | 3167.42 | 399.52 | -0.03 |
| Small → Medium | 3253.63 | 424.25 | -0.09 | 3169.28 | 375.50 | -0.04 | 3239.49 | 379.38 | -0.08 |

## Major Findings

### Model Performance Comparison
- **XGBoost generally outperforms Neural Networks** across all evaluation levels, with lower RMSE and higher R² scores in most cases
- **Tier-level analysis** shows XGBoost achieving RMSE around 2000 and R² ~0.5-0.6, while Neural Networks score RMSE ~2700 and R² ~0.2-0.3
- **Individual city performance** varies significantly, with some cities showing strong performance (e.g., Austin XGBoost R²=0.84, Denver XGBoost R²=0.04) while others show poor fits (e.g., Columbus XGBoost R²=-0.18)

### Cross-Tier Generalization Challenges
- **Poor generalization across tiers** is evident, with most cross-tier R² scores being low or negative (< 0.1)
- **Small → Big tier transfers** perform worst, suggesting models trained on smaller markets struggle with larger, more complex markets
- **Big → Small transfers** show slightly better performance, indicating some transferability from complex to simpler markets

### Architecture Insights
- **NN1 (128-64-32-1)** and **NN2 (256-128-64-32-1)** show similar performance, with NN2 slightly outperforming NN1 in some cases
- **Dropout regularization** in both architectures helps prevent overfitting but doesn't significantly improve generalization across tiers
- **XGBoost's tree-based approach** appears more robust for this tabular dataset compared to neural networks

### Market Size Effects
- **Small tier cities** show more variable performance, with some excellent results (Salem XGBoost RMSE=56.80) and others poor (Santa Cruz XGBoost RMSE=3943.93)
- **Big tier cities** demonstrate challenging prediction environments, with high RMSE across all models
- **Medium tier** shows more consistent performance between the extremes

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