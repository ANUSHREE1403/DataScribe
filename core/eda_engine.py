import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for web
import matplotlib.pyplot as plt

# Try to import seaborn, but make it optional
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("Warning: seaborn not available, using matplotlib defaults")

# Try to import plotly, but make it optional
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    # Plotly optional; analysis uses matplotlib. No need to warn.

from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
if SEABORN_AVAILABLE:
    try:
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    except:
        plt.style.use('default')
else:
    plt.style.use('default')

class DataScribeEDA:
    """
    Core EDA Engine for DataScribe MVP
    Handles data analysis, cleaning, and visualization generation
    """
    
    def __init__(self, df: pd.DataFrame, target_col: Optional[str] = None):
        self.df = df.copy()
        self.original_df = df.copy()
        self.target_col = target_col
        self.numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        self.analysis_results = {}
        
    def analyze_dataset(self) -> Dict:
        """Main analysis function that runs all EDA components"""
        try:
            # Basic dataset overview
            self.analysis_results['overview'] = self._get_dataset_overview()
            
            # Data quality analysis
            self.analysis_results['data_quality'] = self._analyze_data_quality()
            
            # Statistical summaries
            self.analysis_results['statistics'] = self._get_statistical_summary()
            
            # Univariate analysis
            self.analysis_results['univariate'] = self._univariate_analysis()
            
            # Bivariate analysis
            self.analysis_results['bivariate'] = self._bivariate_analysis()
            
            # Multivariate analysis
            self.analysis_results['multivariate'] = self._multivariate_analysis()
            
            # Target analysis (if target column specified)
            if self.target_col:
                self.analysis_results['target_analysis'] = self._analyze_target()
            
            # Insights and recommendations
            self.analysis_results['insights'] = self._generate_insights()
            
            return self.analysis_results
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _get_dataset_overview(self) -> Dict:
        """Get basic dataset information"""
        return {
            "shape": self.df.shape,
            "memory_usage": self.df.memory_usage(deep=True).sum() / 1024**2,  # MB
            "data_types": self.df.dtypes.value_counts().to_dict(),
            "columns": {
                "total": len(self.df.columns),
                "numerical": len(self.numerical_cols),
                "categorical": len(self.categorical_cols),
                "datetime": len(self.datetime_cols)
            }
        }
    
    def _analyze_data_quality(self) -> Dict:
        """Analyze data quality issues"""
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        
        # Detect potential duplicates
        duplicates = self.df.duplicated().sum()
        
        # Check for constant columns
        constant_cols = [col for col in self.df.columns if self.df[col].nunique() <= 1]
        
        return {
            "missing_values": {
                "count": missing_data.to_dict(),
                "percentage": missing_percent.to_dict(),
                "total_missing": missing_data.sum(),
                "columns_with_missing": missing_data[missing_data > 0].index.tolist()
            },
            "duplicates": {
                "count": duplicates,
                "percentage": (duplicates / len(self.df)) * 100
            },
            "constant_columns": constant_cols,
            "data_quality_score": self._calculate_quality_score(missing_percent, duplicates)
        }
    
    def _calculate_quality_score(self, missing_percent: pd.Series, duplicates: int) -> float:
        """Calculate overall data quality score (0-100)"""
        missing_penalty = missing_percent.mean() * 2  # Missing data penalty
        duplicate_penalty = (duplicates / len(self.df)) * 100 * 5  # Duplicate penalty
        
        score = 100 - missing_penalty - duplicate_penalty
        return max(0, min(100, score))
    
    def _get_statistical_summary(self) -> Dict:
        """Get comprehensive statistical summary"""
        stats = {}
        
        # Numerical columns statistics
        if self.numerical_cols:
            stats['numerical'] = self.df[self.numerical_cols].describe().to_dict()
            
            # Additional statistics
            for col in self.numerical_cols:
                stats['numerical'][f'{col}_skewness'] = self.df[col].skew()
                stats['numerical'][f'{col}_kurtosis'] = self.df[col].kurtosis()
        
        # Categorical columns statistics
        if self.categorical_cols:
            stats['categorical'] = {}
            for col in self.categorical_cols:
                value_counts = self.df[col].value_counts()
                stats['categorical'][col] = {
                    'unique_count': self.df[col].nunique(),
                    'top_values': value_counts.head(10).to_dict(),
                    'missing_count': self.df[col].isnull().sum()
                }
        
        return stats
    
    def _univariate_analysis(self) -> Dict:
        """Perform univariate analysis on all columns"""
        analysis = {}
        
        # Numerical columns
        for col in self.numerical_cols:
            analysis[col] = {
                'type': 'numerical',
                'distribution': self._analyze_numerical_distribution(col),
                'outliers': self._detect_outliers(col)
            }
        
        # Categorical columns
        for col in self.categorical_cols:
            analysis[col] = {
                'type': 'categorical',
                'distribution': self._analyze_categorical_distribution(col)
            }
        
        return analysis
    
    def _analyze_numerical_distribution(self, col: str) -> Dict:
        """Analyze distribution of numerical column"""
        data = self.df[col].dropna()
        
        return {
            'mean': data.mean(),
            'median': data.median(),
            'std': data.std(),
            'min': data.min(),
            'max': data.max(),
            'q25': data.quantile(0.25),
            'q75': data.quantile(0.75),
            'iqr': data.quantile(0.75) - data.quantile(0.25),
            'skewness': data.skew(),
            'kurtosis': data.kurtosis()
        }
    
    def _detect_outliers(self, col: str, method: str = 'iqr') -> Dict:
        """Detect outliers using specified method"""
        data = self.df[col].dropna()
        
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            
            return {
                'method': 'iqr',
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outlier_count': len(outliers),
                'outlier_percentage': (len(outliers) / len(data)) * 100,
                'outlier_values': outliers.tolist()[:10]  # First 10 outliers
            }
        
        return {"method": method, "error": "Method not implemented"}
    
    def _analyze_categorical_distribution(self, col: str) -> Dict:
        """Analyze distribution of categorical column"""
        value_counts = self.df[col].value_counts()
        
        return {
            'unique_count': self.df[col].nunique(),
            'top_categories': value_counts.head(10).to_dict(),
            'missing_count': self.df[col].isnull().sum(),
            'most_common': value_counts.index[0] if len(value_counts) > 0 else None,
            'least_common': value_counts.index[-1] if len(value_counts) > 0 else None
        }
    
    def _bivariate_analysis(self) -> Dict:
        """Perform bivariate analysis"""
        analysis = {}
        
        if self.target_col and self.target_col in self.df.columns:
            # Target vs features analysis
            for col in self.numerical_cols:
                if col != self.target_col:
                    analysis[f'{col}_vs_target'] = self._analyze_feature_target_relationship(col)
        
        # Correlation analysis for numerical columns
        if len(self.numerical_cols) > 1:
            analysis['correlations'] = self._analyze_correlations()
        
        return analysis
    
    def _analyze_feature_target_relationship(self, feature_col: str) -> Dict:
        """Analyze relationship between feature and target"""
        if self.df[self.target_col].dtype in ['object', 'category']:
            # Classification task
            return self._analyze_classification_relationship(feature_col)
        else:
            # Regression task
            return self._analyze_regression_relationship(feature_col)
    
    def _analyze_classification_relationship(self, feature_col: str) -> Dict:
        """Analyze feature-target relationship for classification"""
        if feature_col in self.numerical_cols:
            # Numerical feature vs categorical target
            groups = self.df.groupby(self.target_col)[feature_col]
            return {
                'type': 'numerical_vs_categorical',
                'group_means': groups.mean().to_dict(),
                'group_stds': groups.std().to_dict(),
                'anova_f_stat': self._calculate_anova_f_stat(feature_col)
            }
        else:
            # Categorical feature vs categorical target
            contingency_table = pd.crosstab(self.df[feature_col], self.df[self.target_col])
            return {
                'type': 'categorical_vs_categorical',
                'contingency_table': contingency_table.to_dict(),
                'chi_square_stat': self._calculate_chi_square(contingency_table)
            }
    
    def _analyze_regression_relationship(self, feature_col: str) -> Dict:
        """Analyze feature-target relationship for regression"""
        correlation = self.df[feature_col].corr(self.df[self.target_col])
        
        return {
            'type': 'numerical_vs_numerical',
            'correlation': correlation,
            'correlation_strength': self._interpret_correlation(correlation)
        }
    
    def _analyze_correlations(self) -> Dict:
        """Analyze correlations between numerical features"""
        corr_matrix = self.df[self.numerical_cols].corr()
        
        # Find high correlations
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:  # High correlation threshold
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'high_correlations': sorted(high_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True),
            'avg_correlation': corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
        }
    
    def _multivariate_analysis(self) -> Dict:
        """Perform multivariate analysis"""
        analysis = {}
        
        # Principal Component Analysis for numerical columns
        if len(self.numerical_cols) > 2:
            analysis['pca'] = self._perform_pca()
        
        # Feature importance (if target specified)
        if self.target_col and self.target_col in self.df.columns:
            analysis['feature_importance'] = self._calculate_feature_importance()
        
        return analysis
    
    def _perform_pca(self) -> Dict:
        """Perform basic PCA analysis"""
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            # Prepare data
            numerical_data = self.df[self.numerical_cols].dropna()
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numerical_data)
            
            # Perform PCA
            pca = PCA()
            pca_result = pca.fit_transform(scaled_data)
            
            # Calculate explained variance
            explained_variance_ratio = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance_ratio)
            
            # Find number of components for 95% variance
            n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
            
            return {
                'n_components': len(self.numerical_cols),
                'explained_variance_ratio': explained_variance_ratio.tolist(),
                'cumulative_variance': cumulative_variance.tolist(),
                'n_components_95_variance': n_components_95,
                'total_explained_variance': cumulative_variance[-1]
            }
        except ImportError:
            return {"error": "scikit-learn not available for PCA"}
    
    def _calculate_feature_importance(self) -> Dict:
        """Calculate basic feature importance"""
        if self.target_col not in self.numerical_cols:
            return {"error": "Target column must be numerical for feature importance"}
        
        importance = {}
        for col in self.numerical_cols:
            if col != self.target_col:
                correlation = abs(self.df[col].corr(self.df[self.target_col]))
                importance[col] = correlation
        
        # Sort by importance
        sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        return {
            'feature_importance': sorted_importance,
            'top_features': list(sorted_importance.keys())[:5]
        }
    
    def _analyze_target(self) -> Dict:
        """Analyze target variable specifically"""
        if self.target_col not in self.df.columns:
            return {"error": "Target column not found"}
        
        target_data = self.df[self.target_col].dropna()
        
        if target_data.dtype in ['object', 'category']:
            # Classification target
            return {
                'type': 'classification',
                'class_distribution': target_data.value_counts().to_dict(),
                'class_balance': self._assess_class_balance(target_data),
                'missing_count': self.df[self.target_col].isnull().sum()
            }
        else:
            # Regression target
            return {
                'type': 'regression',
                'distribution': self._analyze_numerical_distribution(self.target_col),
                'outliers': self._detect_outliers(self.target_col),
                'missing_count': self.df[self.target_col].isnull().sum()
            }
    
    def _assess_class_balance(self, target_data: pd.Series) -> str:
        """Assess if classification target is balanced"""
        value_counts = target_data.value_counts()
        min_count = value_counts.min()
        max_count = value_counts.max()
        
        ratio = min_count / max_count
        
        if ratio > 0.8:
            return "balanced"
        elif ratio > 0.5:
            return "moderately_imbalanced"
        else:
            return "highly_imbalanced"
    
    def _generate_insights(self) -> Dict:
        """Generate actionable insights and recommendations"""
        insights = {
            "data_quality_insights": [],
            "feature_insights": [],
            "recommendations": []
        }
        
        # Data quality insights
        quality = self.analysis_results.get('data_quality', {})
        if quality.get('missing_values', {}).get('total_missing', 0) > 0:
            insights["data_quality_insights"].append(
                f"Dataset has {quality['missing_values']['total_missing']} missing values that need attention"
            )
        
        if quality.get('duplicates', {}).get('count', 0) > 0:
            insights["data_quality_insights"].append(
                f"Found {quality['duplicates']['count']} duplicate rows that should be removed"
            )
        
        # Feature insights
        if self.numerical_cols:
            insights["feature_insights"].append(
                f"Dataset has {len(self.numerical_cols)} numerical features for analysis"
            )
        
        if self.categorical_cols:
            insights["feature_insights"].append(
                f"Dataset has {len(self.categorical_cols)} categorical features for encoding"
            )
        
        # Recommendations
        if quality.get('data_quality_score', 100) < 80:
            insights["recommendations"].append("Consider data cleaning before analysis")
        
        if len(self.numerical_cols) > 10:
            insights["recommendations"].append("Consider dimensionality reduction for high-dimensional data")
        
        if self.target_col:
            target_analysis = self.analysis_results.get('target_analysis', {})
            if target_analysis.get('type') == 'classification':
                balance = target_analysis.get('class_balance', 'unknown')
                if balance == 'highly_imbalanced':
                    insights["recommendations"].append("Target is highly imbalanced - consider resampling techniques")
        
        return insights
    
    # Helper methods for statistical calculations
    def _calculate_anova_f_stat(self, feature_col: str) -> float:
        """Calculate ANOVA F-statistic for feature vs target"""
        try:
            from scipy import stats
            groups = [group.values for name, group in self.df.groupby(self.target_col)[feature_col]]
            groups = [g for g in groups if len(g) > 0]  # Remove empty groups
            if len(groups) > 1:
                f_stat, _ = stats.f_oneway(*groups)
                return f_stat
            return 0
        except ImportError:
            return 0
    
    def _calculate_chi_square(self, contingency_table: pd.DataFrame) -> float:
        """Calculate chi-square statistic for categorical variables"""
        try:
            from scipy import stats
            chi2, _, _, _ = stats.chi2_contingency(contingency_table)
            return chi2
        except ImportError:
            return 0
    
    def _interpret_correlation(self, corr: float) -> str:
        """Interpret correlation strength"""
        abs_corr = abs(corr)
        if abs_corr >= 0.8:
            return "very_strong"
        elif abs_corr >= 0.6:
            return "strong"
        elif abs_corr >= 0.4:
            return "moderate"
        elif abs_corr >= 0.2:
            return "weak"
        else:
            return "very_weak"

def convert_numpy_types(obj):
    """Convert NumPy types to Python native types for JSON serialization"""
    import numpy as np
    import pandas as pd
    
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def run_eda(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict:
    """
    Main function to run EDA analysis
    """
    eda_engine = DataScribeEDA(df, target_col)
    results = eda_engine.analyze_dataset()
    return convert_numpy_types(results)
