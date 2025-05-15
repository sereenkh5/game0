import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import logging
from typing import Tuple, Dict, List, Optional, Union
import joblib
import os

# -*- coding: utf-8 -*-
# استيراد المكتبات الأساسية
import cv2  # لمعالجة الصور
import pandas as pd  # للتعامل مع البيانات
import numpy as np  # للحسابات الرياضية
import logging  # لتسجيل الأخطاء والتحذيرات


# تعريف المتغيرات الأساسية
all_features = {}  # سيخزن جميع ميزات السيارات
car_images = {
    "1001": ["images/car1001_1.jpg", "images/car1001_2.jpg"],
    "1002": ["images/car1002_1.jpg"]
}


# تهيئة نظام التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CarFeaturesLogger")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('ImprovedCarPricePredictor')

# Suppress warnings
warnings.filterwarnings('ignore')

# Try to import ML libraries with graceful fallbacks
try:
    from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV, RandomizedSearchCV
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, PowerTransformer, QuantileTransformer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer, KNNImputer
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor, StackingRegressor
    from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
    from sklearn.feature_selection import SelectFromModel, RFE, RFECV
    from sklearn.base import BaseEstimator, TransformerMixin
    ML_LIBRARIES = {'sklearn': True}
except ImportError:
    logger.error("scikit-learn not available! Core functionality will be limited")
    ML_LIBRARIES = {'sklearn': False}

try:
    from xgboost import XGBRegressor
    ML_LIBRARIES['xgboost'] = True
except ImportError:
    logger.warning("XGBoost not available")
    ML_LIBRARIES['xgboost'] = False
    
try:
    import lightgbm as lgb
    from lightgbm import LGBMRegressor
    ML_LIBRARIES['lightgbm'] = True
except ImportError:
    logger.warning("LightGBM not available")
    ML_LIBRARIES['lightgbm'] = False
    
try:
    from catboost import CatBoostRegressor, Pool
    ML_LIBRARIES['catboost'] = True
except ImportError:
    logger.warning("CatBoost not available")
    ML_LIBRARIES['catboost'] = False
    
try:
    import shap
    ML_LIBRARIES['shap'] = True
except ImportError:
    logger.warning("SHAP not available")
    ML_LIBRARIES['shap'] = False
    
try:
    import optuna
    from optuna.integration.sklearn import OptunaSearchCV
    ML_LIBRARIES['optuna'] = True
except ImportError:
    logger.warning("Optuna not available")
    ML_LIBRARIES['optuna'] = False


# Custom transformers
class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target encoder for categorical variables.
    Replaces categorical values with the average target value for each category.
    """
    def __init__(self, cols=None, smoothing=10):
        self.cols = cols
        self.smoothing = smoothing
        self.encodings = {}
        self.global_mean = None
        
    def fit(self, X, y):
        if self.cols is None:
            self.cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            
        self.global_mean = np.mean(y)
        
        for col in self.cols:
            if col in X.columns:
                # Calculate target mean for each category
                group_means = X.groupby(col)[y.name].agg(['mean', 'count'])
                
                # Apply smoothing
                smoothed = (group_means['count'] * group_means['mean'] + self.smoothing * self.global_mean) / \
                           (group_means['count'] + self.smoothing)
                           
                self.encodings[col] = smoothed
        
        return self
        
    def transform(self, X):
        X_new = X.copy()
        
        for col in self.cols:
            if col in X.columns:
                # Replace categories with their encoding, or global mean for unseen categories
                X_new[col + '_encoded'] = X[col].map(self.encodings[col]).fillna(self.global_mean)
                
        return X_new


class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Handle outliers in numeric columns using IQR or percentile-based capping.
    """
    def __init__(self, cols=None, method='iqr', lower_quantile=0.01, upper_quantile=0.99):
        self.cols = cols
        self.method = method
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.boundaries = {}
        
    def fit(self, X, y=None):
        if self.cols is None:
            self.cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
        for col in self.cols:
            if col in X.columns:
                if self.method == 'iqr':
                    # IQR-based boundaries
                    Q1 = X[col].quantile(0.25)
                    Q3 = X[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                else:
                    # Percentile-based boundaries
                    lower_bound = X[col].quantile(self.lower_quantile)
                    upper_bound = X[col].quantile(self.upper_quantile)
                    
                self.boundaries[col] = (lower_bound, upper_bound)
        
        return self
        
    def transform(self, X):
        X_new = X.copy()
        
        for col in self.cols:
            if col in X.columns and col in self.boundaries:
                lower_bound, upper_bound = self.boundaries[col]
                X_new[col] = X_new[col].clip(lower_bound, upper_bound)
                
        return X_new


class ImprovedCarPricePredictor:
    """
    An improved car price predictor that handles data loading, preprocessing, 
    model training, and evaluation for used car price prediction.
    """
    
    def __init__(
        self, 
        data_dir: str = '.', 
        log_transform: bool = True,
        current_year: int = 2025,
        random_state: int = 42,
        output_dir: str = './output',
        use_gpu: bool = False,
        validation_size: float = 0.2,
        use_feature_selection: bool = True,
        feature_selection_method: str = 'rfe',
        image_data_dir: str = None  # New parameter for image data directory
    ):
        """
        Initialize the car price predictor.
        
        Args:
            data_dir: Directory containing the data files
            log_transform: Whether to apply log transformation to the target variable
            current_year: Current year for age calculation
            random_state: Random seed for reproducibility
            output_dir: Directory for saving outputs
            use_gpu: Whether to use GPU for model training if available
            validation_size: Size of the validation set for internal evaluation
            use_feature_selection: Whether to use feature selection
            feature_selection_method: Method for feature selection ('rfe', 'model', 'shap')
            image_data_dir: Directory containing car images (if available)
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.log_transform = log_transform
        self.current_year = current_year
        self.random_state = random_state
        self.use_gpu = use_gpu
        self.validation_size = validation_size
        self.use_feature_selection = use_feature_selection
        self.feature_selection_method = feature_selection_method
        self.image_data_dir = image_data_dir
        
        # Initialize data containers
        self.train_df = None
        self.test_df = None
        self.sample_submission = None
        self.X_train = None
        self.X_test = None
        self.X_val = None  # Validation set
        self.y_train = None
        self.y_val = None  # Validation set targets
        self.y_train_original = None  # For inverse transform
        self.train_id = None
        self.test_id = None
        self.preprocessor = None
        self.model = None
        self.model_name = None
        self.feature_names = None
        self.feature_importances = None
        self.models = {}
        self.ensemble_weights = None
        self.categorical_features = None
        self.numeric_features = None
        self.target_transformer = None
        self.cross_val_scores = None
        
        logger.info(f"ImprovedCarPricePredictor initialized with data directory: {data_dir}")
        
    def load_data(self) -> None:
        """Load the data files from the data directory."""
        try:
            self.train_df = pd.read_csv(self.data_dir / 'train.csv')
            self.test_df = pd.read_csv(self.data_dir / 'test.csv')
            
            try:
                self.sample_submission = pd.read_csv(self.data_dir / 'sample_submission.csv')
                logger.info(f"Sample submission loaded with {len(self.sample_submission)} rows")
            except FileNotFoundError:
                logger.warning("Sample submission file not found, will create one during prediction")
            
            logger.info(f"Data loaded successfully: {len(self.train_df)} training samples, "
                        f"{len(self.test_df)} test samples")
            
            # Store IDs if present
            id_column = None
            for col in ['id', 'ID', 'Id', 'car_id', 'car_ID', 'car_Id']:
                if col in self.train_df.columns:
                    id_column = col
                    break
            
            if id_column:
                self.train_id = self.train_df[id_column]
                self.test_id = self.test_df[id_column]
                logger.info(f"ID column found: {id_column}")
            
            # Print data columns
            logger.info(f"Train columns: {self.train_df.columns.tolist()}")
            logger.info(f"Test columns: {self.test_df.columns.tolist()}")
            
            # Print sample data
            logger.info(f"Sample train data:\n{self.train_df.head(2)}")
            
            # Check for data types and convert if needed
            self._convert_data_types()
            
        except FileNotFoundError as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _convert_data_types(self):
        """Convert columns to appropriate data types."""
        # Find potential numeric columns stored as strings
        for df in [self.train_df, self.test_df]:
            for col in df.columns:
                # Skip obvious non-numeric columns
                if col in ['Car Make', 'Model', 'Trim', 'Body Type', 'Fuel', 'Transmission', 'Color']:
                    continue
                
                # Try to convert to numeric
                if df[col].dtype == 'object':
                    try:
                        # Try to convert removing common symbols
                        cleaned_series = df[col].str.replace(',', '').str.replace('$', '')
                        numeric_series = pd.to_numeric(cleaned_series, errors='coerce')
                        
                        # If conversion was mostly successful, apply it
                        if numeric_series.notna().sum() > 0.5 * len(numeric_series):
                            df[col] = numeric_series
                            logger.info(f"Converted column {col} to numeric")
                    except:
                        pass
    
    def explore_data(self, save_plots: bool = True) -> Dict:
        """
        Explore and understand the data, generate visualizations.
        
        Args:
            save_plots: Whether to save the plots to the output directory
            
        Returns:
            Dict: Summary statistics and insights
        """
        if self.train_df is None:
            logger.error("Data not loaded. Call load_data() first.")
            return {}
            
        # Basic information
        train_shape = self.train_df.shape
        dtypes = self.train_df.dtypes
        missing_values = self.train_df.isnull().sum()
        missing_percent = (missing_values / len(self.train_df)) * 100
        summary_stats = self.train_df.describe(include='all')
        
        logger.info(f"DataFrame Shape: {train_shape}")
        logger.info(f"Missing values: {missing_values[missing_values > 0]}")
        logger.info(f"Missing percentages: {missing_percent[missing_percent > 0]}")
        
        # Get data types
        numeric_features = self.train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = self.train_df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Store feature types
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        
        logger.info(f"Numeric features: {numeric_features}")
        logger.info(f"Categorical features: {categorical_features}")
        
        # Check cardinality of categorical variables
        cat_cardinality = {col: self.train_df[col].nunique() for col in categorical_features}
        logger.info(f"Categorical cardinality: {cat_cardinality}")
        
        # Identify outliers in numeric columns
        outliers_info = {}
        for col in numeric_features:
            if col == 'Price' or col.startswith('id') or col == 'ID':  # Skip target and ID
                continue
                
            Q1 = self.train_df[col].quantile(0.25)
            Q3 = self.train_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            n_outliers = ((self.train_df[col] < lower_bound) | (self.train_df[col] > upper_bound)).sum()
            pct_outliers = n_outliers / len(self.train_df) * 100
            
            if pct_outliers > 1:  # Only report if more than 1% are outliers
                outliers_info[col] = {
                    'n_outliers': n_outliers,
                    'pct_outliers': pct_outliers,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
        
        if outliers_info:
            logger.info("Outliers detected:")
            for col, info in outliers_info.items():
                logger.info(f"  {col}: {info['n_outliers']} outliers ({info['pct_outliers']:.2f}%)")
            
        # Visualizations if Price column exists
        if 'Price' in self.train_df.columns and save_plots:
            fig = plt.figure(figsize=(16, 12))
            
            # Price distribution
            plt.subplot(2, 2, 1)
            sns.histplot(self.train_df['Price'], kde=True, bins=50)
            plt.title('Distribution of Car Prices')
            plt.xlabel('Price (JOD)')
            plt.ylabel('Frequency')
            
            # Log-transformed price distribution
            plt.subplot(2, 2, 2)
            sns.histplot(np.log1p(self.train_df['Price']), kde=True, bins=50)
            plt.title('Log-transformed Price Distribution')
            plt.xlabel('Log(Price + 1)')
            plt.ylabel('Frequency')
            
            # Price boxplot
            plt.subplot(2, 2, 3)
            sns.boxplot(y=self.train_df['Price'])
            plt.title('Price Boxplot')
            
            # Price vs Year (if Year exists)
            if 'Year' in self.train_df.columns:
                plt.subplot(2, 2, 4)
                sns.scatterplot(x='Year', y='Price', data=self.train_df.sample(min(1000, len(self.train_df))), alpha=0.6)
                plt.title('Price vs Year')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'price_analysis.png')
            plt.close()
            
            # Top Car Makes by Average Price
            if 'Car Make' in self.train_df.columns:
                plt.figure(figsize=(14, 8))
                top_makes = self.train_df.groupby('Car Make')['Price'].mean().sort_values(ascending=False).head(15)
                sns.barplot(x=top_makes.index, y=top_makes.values)
                plt.title('Top 15 Car Makes by Average Price')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(self.output_dir / 'car_makes_by_price.png')
                plt.close()
            
            # Correlation matrix for numeric columns
            plt.figure(figsize=(14, 10))
            numeric_df = self.train_df.select_dtypes(include=['int64', 'float64'])
            numeric_df = numeric_df.apply(pd.to_numeric, errors='coerce')
            
            # Drop columns with all NaN values after conversion
            numeric_df = numeric_df.dropna(axis=1, how='all')
            
            # Calculate correlation matrix
            try:
                corr_matrix = numeric_df.corr(method='pearson')
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm',
                          linewidths=0.5, cbar_kws={"shrink": .8})
                plt.title('Correlation Matrix')
                plt.tight_layout()
                plt.savefig(self.output_dir / 'correlation_matrix.png')
                plt.close()
            except Exception as e:
                logger.warning(f"Error creating correlation matrix: {e}")
                
            # Price by Body Type
            if 'Body Type' in self.train_df.columns:
                plt.figure(figsize=(14, 8))
                body_type_prices = self.train_df.groupby('Body Type')['Price'].median().sort_values(ascending=False)
                sns.barplot(x=body_type_prices.index, y=body_type_prices.values)
                plt.title('Median Price by Body Type')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(self.output_dir / 'price_by_body_type.png')
                plt.close()
                
            # Price by Fuel Type
            if 'Fuel' in self.train_df.columns:
                plt.figure(figsize=(12, 8))
                fuel_type_prices = self.train_df.groupby('Fuel')['Price'].median().sort_values(ascending=False)
                sns.barplot(x=fuel_type_prices.index, y=fuel_type_prices.values)
                plt.title('Median Price by Fuel Type')
                plt.tight_layout()
                plt.savefig(self.output_dir / 'price_by_fuel_type.png')
                plt.close()
                
            # Price by Car Age
            if 'Year' in self.train_df.columns:
                self.train_df['Car_Age'] = self.current_year - pd.to_numeric(self.train_df['Year'], errors='coerce')
                plt.figure(figsize=(14, 8))
                sns.regplot(x='Car_Age', y='Price', data=self.train_df.sample(min(1000, len(self.train_df))), 
                           scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
                plt.title('Price vs Car Age')
                plt.tight_layout()
                plt.savefig(self.output_dir / 'price_vs_age.png')
                plt.close()
                
            # Price by Mileage (if exists)
            if 'Kilometers' in self.train_df.columns or 'Mileage' in self.train_df.columns:
                mileage_col = 'Kilometers' if 'Kilometers' in self.train_df.columns else 'Mileage'
                plt.figure(figsize=(14, 8))
                sns.regplot(x=mileage_col, y='Price', data=self.train_df.sample(min(1000, len(self.train_df))),
                           scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
                plt.title('Price vs Mileage')
                plt.tight_layout()
                plt.savefig(self.output_dir / 'price_vs_mileage.png')
                plt.close()
            
        # Return insights
        insights = {
            "shape": train_shape,
            "dtypes": dtypes,
            "missing_values": missing_values,
            "missing_percent": missing_percent,
            "summary_stats": summary_stats,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "categorical_cardinality": cat_cardinality,
            "outliers": outliers_info
        }
        
        return insights
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with missing values handled
        """
        result_df = df.copy()
        missing_stats = result_df.isnull().sum()
        
        # Features with missing values
        features_with_missing = missing_stats[missing_stats > 0].index.tolist()
        
        for feature in features_with_missing:
            missing_percent = missing_stats[feature] / len(result_df) * 100
            
            # Skip features with too many missing values
            if missing_percent > 80:
                logger.info(f"Dropping column {feature} - {missing_percent:.2f}% missing values")
                result_df = result_df.drop(feature, axis=1)
                continue
                
            # For numeric features
            if result_df[feature].dtype in ['int64', 'float64']:
                # For mileage, use median by car make and model if possible
                if feature in ['Kilometers', 'Mileage', 'Engine Size (cc)', 'Engine_Size']:
                    if 'Car Make' in result_df.columns and 'Model' in result_df.columns:
                        # Group by make and model, then fill missing values with group median
                        result_df[feature] = result_df.groupby(['Car Make', 'Model'])[feature].transform(
                            lambda x: x.fillna(x.median() if not pd.isna(x.median()) else result_df[feature].median())
                        )
                    else:
                        # Use median
                        result_df[feature] = result_df[feature].fillna(result_df[feature].median())
                else:
                    # Use KNN imputation for other numeric features if less than 30% missing
                    if missing_percent < 30 and ML_LIBRARIES['sklearn']:
                        # Get numeric columns for KNN imputation
                        numeric_cols = result_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                        
                        # Only use columns with no missing values as features
                        complete_numeric_cols = [col for col in numeric_cols 
                                             if col != feature and result_df[col].isnull().sum() == 0]
                        
                        if len(complete_numeric_cols) >= 2:  # Need at least 2 features for KNN
                            # Create and fit the imputer
                            imputer = KNNImputer(n_neighbors=5)
                            
                            # Impute feature using complete numeric columns
                            result_df[feature] = imputer.fit_transform(
                                result_df[complete_numeric_cols + [feature]]
                            )[:, -1]  # Take last column (the imputed feature)
                            logger.info(f"Used KNN imputation for {feature}")
                        else:
                            # Fallback to median imputation
                            result_df[feature] = result_df[feature].fillna(result_df[feature].median())
                    else:
                        # Use median
                        result_df[feature] = result_df[feature].fillna(result_df[feature].median())
            
            # For categorical features
            else:
                if feature in ['Car Make', 'Model', 'Body Type', 'Transmission', 'Fuel']:
                    # For important categorical features, use more sophisticated imputation
                    if missing_percent < 10:
                        # Use most frequent value
                        result_df[feature] = result_df[feature].fillna(result_df[feature].mode()[0])
                    else:
                        # Try to impute based on correlations with other features
                        # For example, impute Model based on Car Make if possible
                        if feature == 'Model' and 'Car Make' in result_df.columns:
                            # Group by Car Make and find most common model for each make
                            most_common_model = result_df.groupby('Car Make')['Model'].agg(
                                lambda x: x.mode()[0] if not x.mode().empty else None
                            )
                            
                            # Apply the imputation
                            for make in most_common_model.index:
                                mask = (result_df['Car Make'] == make) & (result_df['Model'].isnull())
                                result_df.loc[mask, 'Model'] = most_common_model[make]
                        
                        # Fill remaining missing values with a new category
                        result_df[feature] = result_df[feature].fillna(f"Unknown_{feature}")
                else:
                    # Use mode (most frequent value) for other categorical features
                    result_df[feature] = result_df[feature].fillna(result_df[feature].mode()[0])
                
        return result_df

    def preprocess_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess text features to extract additional information.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with processed text features
        """
        result_df = df.copy()
        
        # Process car descriptions if available
        if 'Description' in result_df.columns:
            # Extract key information from descriptions
            result_df['Has_Description'] = (~result_df['Description'].isna()).astype(int)
            
            # Length of description (might correlate with car quality/price)
            result_df.loc[result_df['Description'].notna(), 'Description_Length'] = \
                result_df.loc[result_df['Description'].notna(), 'Description'].str.len()
            result_df['Description_Length'] = result_df['Description_Length'].fillna(0)
            
            # Keywords indicating luxury/premium features
            luxury_keywords = ['leather', 'premium', 'luxury', 'navigation', 'sunroof', 'moonroof', 
                             'heated seats', 'cooled seats', 'bluetooth', 'alloy', 'camera', 'sensors']
            
            # Check for luxury keywords
            result_df['Luxury_Features_Count'] = 0
            for keyword in luxury_keywords:
                mask = result_df['Description'].str.contains(keyword, case=False, na=False)
                result_df.loc[mask, 'Luxury_Features_Count'] += 1
                
            # Maintenance keywords
            maintenance_keywords = ['service', 'maintenance', 'records', 'warranty', 'inspection', 'certified']
            result_df['Maintenance_Keywords_Count'] = 0
            for keyword in maintenance_keywords:
                mask = result_df['Description'].str.contains(keyword, case=False, na=False)
                result_df.loc[mask, 'Maintenance_Keywords_Count'] += 1
                
            # Condition keywords
            good_condition = ['excellent', 'perfect', 'great', 'clean', 'like new']
            result_df['Good_Condition_Keywords'] = 0
            for keyword in good_condition:
                mask = result_df['Description'].str.contains(keyword, case=False, na=False)
                result_df.loc[mask, 'Good_Condition_Keywords'] += 1
        
        # Standardize car make and model names
        if 'Car Make' in result_df.columns:
            # Convert to title case and strip spaces
            result_df['Car Make'] = result_df['Car Make'].str.strip().str.title()
            
            # Fix common variations
            make_mapping = {
                'Mercedes': 'Mercedes-Benz',
                'Mercedes Benz': 'Mercedes-Benz',
                'Bmw': 'BMW',
                'Vw': 'Volkswagen',
                'Chevy': 'Chevrolet',
                'Kia Motors': 'Kia'
            }
            
            for old_name, new_name in make_mapping.items():
                result_df.loc[result_df['Car Make'] == old_name, 'Car Make'] = new_name
                
        # Process model names if available
        if 'Model' in result_df.columns:
            result_df['Model'] = result_df['Model'].str.strip().str.title()
            
            # Extract model series/generation if in format "Model Series" (e.g., "Civic Si", "3 Series")
            result_df['Model_Base'] = result_df['Model'].str.split().str[0]
        
        return result_df
            
    def engineer_features(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Engineer features from the train and test datasets.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Transformed train and test data
        """
        if self.train_df is None or self.test_df is None:
            logger.error("Data not loaded. Call load_data() first.")
            return None, None
            
        # Create copies to avoid modifying original data
        train_processed = self.train_df.copy()
        test_processed = self.test_df.copy()
        
        # Handle missing values first
        logger.info("Handling missing values...")
        train_processed = self.handle_missing_values(train_processed)
        test_processed = self.handle_missing_values(test_processed)
        
        # Process text features
        logger.info("Processing text features...")
        train_processed = self.preprocess_text_features(train_processed)
        test_processed = self.preprocess_text_features(test_processed)
        
        # Add car age feature if Year exists
        if 'Year' in train_processed.columns:
            logger.info(f"Creating car age feature based on current year {self.current_year}")
            train_processed['Car_Age'] = self.current_year - pd.to_numeric(train_processed['Year'], errors='coerce')
            test_processed['Car_Age'] = self.current_year - pd.to_numeric(test_processed['Year'], errors='coerce')
            
            # Fill NA values in car age with median
            train_processed['Car_Age'] = train_processed['Car_Age'].fillna(train_processed['Car_Age'].median())
            test_processed['Car_Age'] = test_processed['Car_Age'].fillna(train_processed['Car_Age'].median())
            
            # Car age squared (for non-linear relationship with price)
            train_processed['Car_Age_Squared'] = train_processed['Car_Age'] ** 2
            test_processed['Car_Age_Squared'] = test_processed['Car_Age'] ** 2
        
        # Calculate price per kilometer ratio for train set if both price and kilometers exist
        if 'Price' in train_processed.columns and 'Kilometers' in train_processed.columns:
            # Avoid division by zero
            train_processed['Price_per_KM'] = train_processed['Price'] / (train_processed['Kilometers'] + 1)
            
            # Log the distribution of this feature
            logger.info(f"Price_per_KM statistics: {train_processed['Price_per_KM'].describe()}")
        
        # Handle mileage-related features (Kilometers or Mileage)
        mileage_col = next((col for col in ['Kilometers', 'Mileage'] if col in train_processed.columns), None)
        if mileage_col:
            # Log transform to handle skewness
            train_processed[f'{mileage_col}_Log'] = np.log1p(train_processed[mileage_col])
            test_processed[f'{mileage_col}_Log'] = np.log1p(test_processed[mileage_col])
            
            # Bin mileage into categories
            mileage_bins = [0, 10000, 50000, 100000, 150000, 200000, float('inf')]
            train_processed[f'{mileage_col}_Bin'] = pd.cut(train_processed[mileage_col], bins=mileage_bins, labels=False)
            test_processed[f'{mileage_col}_Bin'] = pd.cut(test_processed[mileage_col], bins=mileage_bins, labels=False)
            
            # Fill NA values
            train_processed[f'{mileage_col}_Bin'] = train_processed[f'{mileage_col}_Bin'].fillna(-1).astype(int)
            test_processed[f'{mileage_col}_Bin'] = test_processed[f'{mileage_col}_Bin'].fillna(-1).astype(int)
            
        # Handle engine size features
        engine_col = next((col for col in ['Engine Size (cc)', 'Engine_Size', 'CC'] if col in train_processed.columns), None)
        if engine_col:
            # Clean engine size (remove non-numeric characters)
            if train_processed[engine_col].dtype == 'object':
                train_processed[engine_col] = train_processed[engine_col].str.extract(r'(\d+)').astype(float)
                test_processed[engine_col] = test_processed[engine_col].str.extract(r'(\d+)').astype(float)
            
            # Fill missing values
            train_processed[engine_col] = train_processed[engine_col].fillna(train_processed[engine_col].median())
            test_processed[engine_col] = test_processed[engine_col].fillna(train_processed[engine_col].median())
            
            # Create engine size categories
            engine_bins = [0, 1000, 1600, 2000, 2500, 3000, 4000, float('inf')]
            train_processed['Engine_Size_Bin'] = pd.cut(train_processed[engine_col], bins=engine_bins, labels=False)
            test_processed['Engine_Size_Bin'] = pd.cut(test_processed[engine_col], bins=engine_bins, labels=False)
            
            # Fill NA values
            train_processed['Engine_Size_Bin'] = train_processed['Engine_Size_Bin'].fillna(-1).astype(int)
            test_processed['Engine_Size_Bin'] = test_processed['Engine_Size_Bin'].fillna(-1).astype(int)
        
        # Process 'Car Make' and 'Model' features
        if 'Car Make' in train_processed.columns:
            # Group rare makes into 'Other'
            make_counts = train_processed['Car Make'].value_counts()
            rare_makes = make_counts[make_counts < 10].index
            train_processed['Car_Make_Grouped'] = train_processed['Car Make'].copy()
            test_processed['Car_Make_Grouped'] = test_processed['Car Make'].copy()
            
            # Replace rare makes with 'Other'
            train_processed.loc[train_processed['Car_Make_Grouped'].isin(rare_makes), 'Car_Make_Grouped'] = 'Other'
            test_processed.loc[test_processed['Car_Make_Grouped'].isin(rare_makes), 'Car_Make_Grouped'] = 'Other'
            
            # Create brand origin feature
            origin_mapping = {
                'Toyota': 'Japan', 'Honda': 'Japan', 'Nissan': 'Japan', 'Mazda': 'Japan', 'Mitsubishi': 'Japan',
                'Suzuki': 'Japan', 'Subaru': 'Japan', 'Lexus': 'Japan', 'Infiniti': 'Japan', 
                'BMW': 'Germany', 'Mercedes-Benz': 'Germany', 'Audi': 'Germany', 'Volkswagen': 'Germany',
                'Porsche': 'Germany', 'Opel': 'Germany',
                'Hyundai': 'Korea', 'Kia': 'Korea', 'Daewoo': 'Korea', 'SsangYong': 'Korea',
                'Ford': 'USA', 'Chevrolet': 'USA', 'Jeep': 'USA', 'Dodge': 'USA', 'Chrysler': 'USA',
                'GMC': 'USA', 'Cadillac': 'USA', 'Buick': 'USA',
                'Fiat': 'Italy', 'Alfa Romeo': 'Italy', 'Ferrari': 'Italy', 'Lamborghini': 'Italy',
                'Maserati': 'Italy',
                'Peugeot': 'France', 'Renault': 'France', 'Citroen': 'France', 'Bugatti': 'France',
                'Land Rover': 'UK', 'Jaguar': 'UK', 'Bentley': 'UK', 'Aston Martin': 'UK',
                'Rolls-Royce': 'UK', 'MINI': 'UK',
                'Volvo': 'Sweden', 'Saab': 'Sweden',
                'Seat': 'Spain', 'Skoda': 'Czech Republic',
                'BYD': 'China', 'Geely': 'China', 'Chery': 'China', 'Great Wall': 'China'
            }
            
            train_processed['Brand_Origin'] = train_processed['Car Make'].map(origin_mapping).fillna('Other')
            test_processed['Brand_Origin'] = test_processed['Car Make'].map(origin_mapping).fillna('Other')
                
        # Process 'Model' feature if exists
        if 'Model' in train_processed.columns:
            # Group rare models into 'Other'
            model_counts = train_processed['Model'].value_counts()
            rare_models = model_counts[model_counts < 5].index
            train_processed['Model_Grouped'] = train_processed['Model'].copy()
            test_processed['Model_Grouped'] = test_processed['Model'].copy()
            
            # Replace rare models with 'Other'
            train_processed.loc[train_processed['Model_Grouped'].isin(rare_models), 'Model_Grouped'] = 'Other'
            test_processed.loc[test_processed['Model_Grouped'].isin(rare_models), 'Model_Grouped'] = 'Other'
            
            # Create Make_Model combined feature
            if 'Car Make' in train_processed.columns:
                train_processed['Make_Model'] = train_processed['Car Make'] + '_' + train_processed['Model_Grouped']
                test_processed['Make_Model'] = test_processed['Car Make'] + '_' + test_processed['Model_Grouped']
                
                # Group rare Make_Model combinations
                make_model_counts = train_processed['Make_Model'].value_counts()
                rare_make_models = make_model_counts[make_model_counts < 3].index
                
                train_processed.loc[train_processed['Make_Model'].isin(rare_make_models), 'Make_Model'] = 'Other'
                test_processed.loc[test_processed['Make_Model'].isin(rare_make_models), 'Make_Model'] = 'Other'
        
        # Process body type
        if 'Body Type' in train_processed.columns:
            # Group similar body types
            body_type_mapping = {
                'Sedan': 'Sedan',
                'Saloon': 'Sedan',
                '4 Door Sedan': 'Sedan',
                'Hatchback': 'Hatchback',
                '5 Door Hatchback': 'Hatchback',
                '3 Door Hatchback': 'Hatchback',
                'SUV': 'SUV',
                'Crossover': 'SUV',
                'CUV': 'SUV',
                'Sports Utility Vehicle': 'SUV',
                'Truck': 'Truck',
                'Pickup': 'Truck',
                'Pick-up': 'Truck',
                'Pick Up': 'Truck',
                'Coupe': 'Coupe',
                '2 Door Coupe': 'Coupe',
                'Sports Coupe': 'Coupe',
                'Convertible': 'Convertible',
                'Cabriolet': 'Convertible',
                'Roadster': 'Convertible',
                'Van': 'Van',
                'Minivan': 'Van',
                'MPV': 'Van',
                'Wagon': 'Wagon',
                'Estate': 'Wagon',
                'Station Wagon': 'Wagon'
            }
            
            # Apply mapping
            train_processed['Body_Type_Grouped'] = train_processed['Body Type'].str.strip().str.title()
            test_processed['Body_Type_Grouped'] = test_processed['Body Type'].str.strip().str.title()
            
            # Apply mapping
            train_processed['Body_Type_Grouped'] = train_processed['Body_Type_Grouped'].map(
                lambda x: next((v for k, v in body_type_mapping.items() if x and k.lower() in x.lower()), 'Other')
            )
            test_processed['Body_Type_Grouped'] = test_processed['Body_Type_Grouped'].map(
                lambda x: next((v for k, v in body_type_mapping.items() if x and k.lower() in x.lower()), 'Other')
            )
        
        # Process transmission
        if 'Transmission' in train_processed.columns:
            # Simplify transmission types
            transmission_mapping = {
                'Automatic': 'Automatic',
                'Auto': 'Automatic',
                'A/T': 'Automatic',
                'Automated': 'Automatic',
                'CVT': 'Automatic',
                'Steptronic': 'Automatic',
                'DSG': 'Automatic',
                'Manual': 'Manual',
                'M/T': 'Manual',
                'Standard': 'Manual',
                'Semi-Automatic': 'Semi-Automatic',
                'Semi Auto': 'Semi-Automatic',
                'Tiptronic': 'Semi-Automatic',
                'Both': 'Semi-Automatic'
            }
            
            # Apply mapping
            train_processed['Transmission_Grouped'] = train_processed['Transmission'].str.strip().str.title()
            test_processed['Transmission_Grouped'] = test_processed['Transmission'].str.strip().str.title()
            
            train_processed['Transmission_Grouped'] = train_processed['Transmission_Grouped'].map(
                lambda x: next((v for k, v in transmission_mapping.items() if x and k.lower() in x.lower()), 'Other')
            )
            test_processed['Transmission_Grouped'] = test_processed['Transmission_Grouped'].map(
                lambda x: next((v for k, v in transmission_mapping.items() if x and k.lower() in x.lower()), 'Other')
            )
            
            # Create binary features
            train_processed['Is_Automatic'] = (train_processed['Transmission_Grouped'] == 'Automatic').astype(int)
            test_processed['Is_Automatic'] = (test_processed['Transmission_Grouped'] == 'Automatic').astype(int)
        
        # Process fuel type
        if 'Fuel' in train_processed.columns:
            # Simplify fuel types
            fuel_mapping = {
                'Petrol': 'Petrol',
                'Gasoline': 'Petrol',
                'Gas': 'Petrol',
                'Diesel': 'Diesel',
                'Hybrid': 'Hybrid',
                'Electric': 'Electric',
                'Plugin Hybrid': 'Hybrid',
                'LPG': 'Alternative',
                'CNG': 'Alternative',
                'Ethanol': 'Alternative',
                'Biodiesel': 'Alternative'
            }
            
            # Apply mapping
            train_processed['Fuel_Grouped'] = train_processed['Fuel'].str.strip().str.title()
            test_processed['Fuel_Grouped'] = test_processed['Fuel'].str.strip().str.title()
            
            train_processed['Fuel_Grouped'] = train_processed['Fuel_Grouped'].map(
                lambda x: next((v for k, v in fuel_mapping.items() if x and k.lower() in x.lower()), 'Other')
            )
            test_processed['Fuel_Grouped'] = test_processed['Fuel_Grouped'].map(
                lambda x: next((v for k, v in fuel_mapping.items() if x and k.lower() in x.lower()), 'Other')
            )
            
            # Create binary features
            train_processed['Is_Diesel'] = (train_processed['Fuel_Grouped'] == 'Diesel').astype(int)
            test_processed['Is_Diesel'] = (test_processed['Fuel_Grouped'] == 'Diesel').astype(int)
            
            train_processed['Is_Hybrid_Electric'] = (
                (train_processed['Fuel_Grouped'] == 'Hybrid') | 
                (train_processed['Fuel_Grouped'] == 'Electric')
            ).astype(int)
            test_processed['Is_Hybrid_Electric'] = (
                (test_processed['Fuel_Grouped'] == 'Hybrid') | 
                (test_processed['Fuel_Grouped'] == 'Electric')
            ).astype(int)
        
        # Process color
        if 'Color' in train_processed.columns:
            # Group colors into categories
            color_groups = {
                'White': ['white', 'pearl', 'cream', 'ivory'],
                'Black': ['black', 'jet black'],
                'Silver': ['silver', 'gray', 'grey'],
                'Red': ['red', 'burgundy', 'maroon', 'wine', 'cherry'],
                'Blue': ['blue', 'navy', 'ocean', 'sky blue'],
                'Green': ['green', 'olive', 'emerald', 'lime'],
                'Brown': ['brown', 'beige', 'tan', 'champagne', 'gold'],
                'Yellow': ['yellow', 'amber'],
                'Orange': ['orange', 'copper'],
                'Purple': ['purple', 'violet', 'lavender']
            }
            
            # Function to map color to group
            def map_color_to_group(color):
                if pd.isna(color):
                    return 'Unknown'
                color = color.lower()
                for group, color_list in color_groups.items():
                    if any(c in color for c in color_list):
                        return group
                return 'Other'
            
            # Apply color mapping
            train_processed['Color_Group'] = train_processed['Color'].apply(map_color_to_group)
            test_processed['Color_Group'] = test_processed['Color'].apply(map_color_to_group)
            
            # Create dark/light color feature (darker colors often fetch higher prices)
            dark_colors = ['Black', 'Blue', 'Brown', 'Purple']
            train_processed['Is_Dark_Color'] = (train_processed['Color_Group'].isin(dark_colors)).astype(int)
            test_processed['Is_Dark_Color'] = (test_processed['Color_Group'].isin(dark_colors)).astype(int)
        
        # Create feature for luxury brands
        luxury_brands = ['Mercedes-Benz', 'BMW', 'Audi', 'Lexus', 'Porsche', 'Jaguar', 'Land Rover', 
                        'Cadillac', 'Maserati', 'Bentley', 'Ferrari', 'Lamborghini', 'Rolls-Royce',
                        'Aston Martin', 'Tesla', 'Infiniti', 'Lincoln', 'Volvo', 'Acura', 'Genesis']
        
        if 'Car Make' in train_processed.columns:
            train_processed['Is_Luxury_Brand'] = (train_processed['Car Make'].isin(luxury_brands)).astype(int)
            test_processed['Is_Luxury_Brand'] = (test_processed['Car Make'].isin(luxury_brands)).astype(int)
        
        # Create feature for popular models (high demand)
        if 'Model' in train_processed.columns and 'Car Make' in train_processed.columns:
            popular_models = [
                ('Toyota', 'Camry'), ('Toyota', 'Corolla'), ('Toyota', 'Land Cruiser'), ('Toyota', 'RAV4'),
                ('Honda', 'Civic'), ('Honda', 'Accord'), ('Honda', 'CR-V'),
                ('Nissan', 'Altima'), ('Nissan', 'Sentra'), ('Nissan', 'Patrol'),
                ('Hyundai', 'Elantra'), ('Hyundai', 'Sonata'), ('Hyundai', 'Tucson'),
                ('Kia', 'Optima'), ('Kia', 'Rio'), ('Kia', 'Sportage'),
                ('Ford', 'F-150'), ('Ford', 'Focus'), ('Ford', 'Escape'),
                ('Chevrolet', 'Malibu'), ('Chevrolet', 'Cruze'), ('Chevrolet', 'Tahoe')
            ]
            
            train_processed['Is_Popular_Model'] = 0
            test_processed['Is_Popular_Model'] = 0
            
            for make, model in popular_models:
                train_mask = (train_processed['Car Make'] == make) & (train_processed['Model'].str.contains(model, case=False, na=False))
                test_mask = (test_processed['Car Make'] == make) & (test_processed['Model'].str.contains(model, case=False, na=False))
                
                train_processed.loc[train_mask, 'Is_Popular_Model'] = 1
                test_processed.loc[test_mask, 'Is_Popular_Model'] = 1
        
        # Process image features if available
        if self.image_data_dir is not None:
            logger.info("Processing image features...")
            image_features = self._extract_enhanced_image_features()
            
            if image_features is not None:
                # Merge image features with our datasets
                train_processed = train_processed.merge(image_features, how='left', left_on='id', right_index=True)
                test_processed = test_processed.merge(image_features, how='left', left_on='id', right_index=True)
                
                # Fill missing image features with zeros
                image_cols = [col for col in image_features.columns if col.startswith('img_')]
                train_processed[image_cols] = train_processed[image_cols].fillna(0)
                test_processed[image_cols] = test_processed[image_cols].fillna(0)
        
        # Drop columns that are no longer needed
        columns_to_drop = []
        for df in [train_processed, test_processed]:
            for col in df.columns:
                # Drop high cardinality categorical columns that have been processed
                if col in ['Model'] and 'Model_Grouped' in df.columns:
                    columns_to_drop.append(col)
                # Drop redundant columns
                if col == 'Description' and 'Description_Length' in df.columns:
                    columns_to_drop.append(col)
        
        # Remove duplicates from columns_to_drop
        columns_to_drop = list(set(columns_to_drop))

        # تعريف البيانات أولاً
car_images = {
    "1001": ["images/car1001_1.jpg", "images/car1001_2.jpg"],
    "1002": ["images/car1002_1.jpg"]
}
        
        # Second pass: process images for each car
for car_id, img_paths in car_images.items():
    car_features = {}
    multiple_images = len(img_paths) > 1

    # Lists to store features from multiple images
    all_image_features = []

    for img_idx, img_path in enumerate(img_paths):
        try:
            # Read and preprocess image
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"Could not read image: {img_path}")
                continue

            img_resized = cv2.resize(img, (224, 224))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_features = {}

            # Color features
            avg_color = np.mean(img_rgb, axis=(0, 1))
            img_features['img_avg_r'] = avg_color[0]
            img_features['img_avg_g'] = avg_color[1]
            img_features['img_avg_b'] = avg_color[2]

            img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
            avg_hsv = np.mean(img_hsv, axis=(0, 1))
            img_features['img_avg_h'] = avg_hsv[0]
            img_features['img_avg_s'] = avg_hsv[1]
            img_features['img_avg_v'] = avg_hsv[2]

            std_color = np.std(img_rgb, axis=(0, 1))
            img_features['img_std_r'] = std_color[0]
            img_features['img_std_g'] = std_color[1]
            img_features['img_std_b'] = std_color[2]

            hist_r = cv2.calcHist([img_rgb], [0], None, [10], [0, 256])
            hist_g = cv2.calcHist([img_rgb], [1], None, [10], [0, 256])
            hist_b = cv2.calcHist([img_rgb], [2], None, [10], [0, 256])
            for i in range(10):
                img_features[f'img_hist_r_{i}'] = hist_r[i][0]
                img_features[f'img_hist_g_{i}'] = hist_g[i][0]
                img_features[f'img_hist_b_{i}'] = hist_b[i][0]

            gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            hog_features = feature.hog(
                gray,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm='L2-Hys',
                feature_vector=True
            )
            hog_blocks = np.array_split(hog_features, 12)
            for i, block in enumerate(hog_blocks):
                img_features[f'img_hog_mean_{i}'] = np.mean(block)
                img_features[f'img_hog_std_{i}'] = np.std(block)

            from skimage.feature import local_binary_pattern
            lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(60), range=(0, 59))
            lbp_hist = lbp_hist.astype(float) / sum(lbp_hist)
            for i in range(min(10, len(lbp_hist))):
                img_features[f'img_lbp_{i}'] = lbp_hist[i]

            brightness = np.mean(gray)
            contrast = np.std(gray)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = np.var(laplacian)
            median_filtered = cv2.medianBlur(gray, 3)
            noise_level = np.mean(np.abs(gray.astype(np.float32) - median_filtered.astype(np.float32)))

            img_features['img_brightness'] = brightness
            img_features['img_contrast'] = contrast
            img_features['img_sharpness'] = sharpness
            img_features['img_noise'] = noise_level

            edges_strong = cv2.Canny(gray, 100, 200)
            edges_weak = cv2.Canny(gray, 50, 150)
            img_features['img_edge_density_strong'] = np.sum(edges_strong) / (gray.shape[0] * gray.shape[1])
            img_features['img_edge_density_weak'] = np.sum(edges_weak) / (gray.shape[0] * gray.shape[1])

            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
            orientation = np.arctan2(sobely, sobelx) * 180 / np.pi
            orientation_bins = np.linspace(-180, 180, 9)
            orientation_hist, _ = np.histogram(orientation.ravel(), bins=orientation_bins)
            orientation_hist = orientation_hist.astype(float) / sum(orientation_hist)
            for i in range(len(orientation_hist)):
                img_features[f'img_edge_orient_{i}'] = orientation_hist[i]

            if use_deep_features:
                img_tf = tf.keras.applications.mobilenet_v2.preprocess_input(
                    np.expand_dims(img_rgb.astype(np.float32), axis=0)
                )
                mobilenet_features = mobilenet_extractor.predict(img_tf, verbose=0)[0]
                resnet_features = resnet_extractor.predict(img_tf, verbose=0)[0]
                efficientnet_features = efficientnet_extractor.predict(img_tf, verbose=0)[0]

                combined_dl_features = np.concatenate([
                    mobilenet_features, resnet_features, efficientnet_features
                ])

                img_features['img_dl_mean'] = np.mean(combined_dl_features)
                img_features['img_dl_std'] = np.std(combined_dl_features)
                img_features['img_dl_max'] = np.max(combined_dl_features)
                img_features['img_dl_min'] = np.min(combined_dl_features)

                for name, features in [
                    ('mobilenet', mobilenet_features),
                    ('resnet', resnet_features),
                    ('efficientnet', efficientnet_features)
                ]:
                    img_features[f'img_{name}_mean'] = np.mean(features)
                    img_features[f'img_{name}_std'] = np.std(features)
                    img_features[f'img_{name}_skew'] = stats.skew(features)
                    img_features[f'img_{name}_kurtosis'] = stats.kurtosis(features)

                pca = PCA(n_components=5)
                try:
                    pca_result = pca.fit_transform(combined_dl_features.reshape(1, -1))[0]
                    for i in range(5):
                        img_features[f'img_pca_{i}'] = pca_result[i]
                except Exception as e:
                    logger.warning(f"PCA failed: {e}")
                    for i in range(5):
                        img_features[f'img_pca_{i}'] = 0.0

            all_image_features.append(img_features)

        except Exception as e:
            logger.warning(f"Error processing image {img_path}: {e}")
            continue

    # يجب أن يكون في بداية البرنامج (قبل الحلقة)
all_features = {}

# داخل الحلقة:
if all_image_features:
    df_imgs = pd.DataFrame(all_image_features)
    car_features = df_imgs.mean().to_dict()
    # تعريف المتغيرات الأساسية في البداية
all_features = {}
image_features_df = None

def process_image_features(features_dict):
    """
    دالة لمعالجة ميزات الصور وتحويلها لـ DataFrame
    """
    try:
        df = pd.DataFrame.from_dict(features_dict, orient='index')
        
        # تعديل أسماء الأعمدة
        df.columns = ['img_' + col if not col.startswith('img_') else col 
                     for col in df.columns]
        
        # تعويض القيم الناقصة
        df = df.fillna(0)
        
        logger.info(f"Extracted {len(df.columns)} features for {len(df)} cars")
        return df
        
    except Exception as e:
        logger.error(f"Failed to process image features: {e}")
        return None

# الكود الرئيسي
try:
    # ... (الكود الذي يملئ all_features)
    
    # معالجة النتائج
    if all_features:
        image_features_df = process_image_features(all_features)
        
except Exception as e:
    logger.error(f"Error in main processing: {e}")
finally:
    # أي عمليات تنظيف إن وجدت
    pass

   # Call the function if needed
try:
    result_df = process_image_features()
except Exception as e:
    logger.error(f"Error in DataFrame conversion: {e}")
    result_df = None

if condition:  # السطر 1210
    for car_id, img_paths in car_images.items():  # السطر 1212 (مع مسافة بادئة)
        all_image_features = []
        
        for img_path in img_paths:
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    features = extract_image_features(img)
                    all_image_features.append(features)
                else:
                    logger.warning(f"Failed to load image: {img_path}")
            except Exception as e:
                logger.error(f"Error processing {img_path}: {str(e)}")
        
        # باقي الكود هنا مع المسافات البادئة الصحيحة
        # تجميع الميزات لكل سيارة
        if all_image_features:
            image_features_df = pd.DataFrame(all_image_features)
            car_features = image_features_df.mean().to_dict()
            all_features[car_id] = car_features
            logger.info(f"Processed car {car_id} with {len(car_features)} features")
    
    # النتيجة النهائية
    if all_features:
        final_df = pd.DataFrame.from_dict(all_features, orient='index')
        logger.info(f"Successfully processed {len(final_df)} cars")
        print(final_df.head())
    else:
        logger.error("No features were extracted")