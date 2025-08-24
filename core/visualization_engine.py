import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for web
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

class DataScribeVisualizer:
    """
    Visualization Engine for DataScribe MVP
    Generates comprehensive, publication-ready visualizations
    """
    
    def __init__(self, df: pd.DataFrame, target_col: Optional[str] = None):
        self.df = df.copy()
        self.target_col = target_col
        self.numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Color schemes
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'accent': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd'
        }
        
    def generate_all_visualizations(self, analysis_results: Dict) -> Dict[str, str]:
        """
        Generate all visualizations for the dataset
        Returns: Dict with plot file paths
        """
        plots = {}
        
        try:
            # Overview plots
            plots['overview'] = self._create_overview_plots()
            
            # Data quality plots
            plots['data_quality'] = self._create_data_quality_plots(analysis_results.get('data_quality', {}))
            
            # Univariate plots
            plots['univariate'] = self._create_univariate_plots(analysis_results.get('univariate', {}))
            
            # Bivariate plots
            plots['bivariate'] = self._create_bivariate_plots(analysis_results.get('bivariate', {}))
            
            # Multivariate plots
            plots['multivariate'] = self._create_multivariate_plots(analysis_results.get('multivariate', {}))
            
            # Target analysis plots
            if self.target_col:
                plots['target_analysis'] = self._create_target_analysis_plots(analysis_results.get('target_analysis', {}))
            
            return plots
            
        except Exception as e:
            return {"error": f"Visualization generation failed: {str(e)}"}
    
    def _create_overview_plots(self) -> str:
        """Create dataset overview visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Dataset Overview', fontsize=16, fontweight='bold')
        
        # 1. Data types distribution
        dtype_counts = self.df.dtypes.value_counts()
        axes[0, 0].pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Data Types Distribution')
        
        # 2. Missing values heatmap
        missing_data = self.df.isnull().sum()
        if missing_data.sum() > 0:
            missing_pct = (missing_data / len(self.df)) * 100
            axes[0, 1].bar(range(len(missing_pct)), missing_pct.values, color=self.colors['warning'])
            axes[0, 1].set_title('Missing Values by Column (%)')
            axes[0, 1].set_xlabel('Columns')
            axes[0, 1].set_ylabel('Missing %')
            axes[0, 1].tick_params(axis='x', rotation=45)
        else:
            axes[0, 1].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Missing Values by Column (%)')
        
        # 3. Numerical vs Categorical columns
        col_types = ['Numerical', 'Categorical', 'Datetime']
        col_counts = [len(self.numerical_cols), len(self.categorical_cols), len(self.datetime_cols)]
        axes[1, 0].bar(col_types, col_counts, color=[self.colors['primary'], self.colors['secondary'], self.colors['accent']])
        axes[1, 0].set_title('Column Types Distribution')
        axes[1, 0].set_ylabel('Count')
        
        # 4. Dataset shape info
        axes[1, 1].text(0.5, 0.7, f'Rows: {self.df.shape[0]:,}', ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=14)
        axes[1, 1].text(0.5, 0.5, f'Columns: {self.df.shape[1]:,}', ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=14)
        axes[1, 1].text(0.5, 0.3, f'Memory: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB', ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('Dataset Information')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save plot
        filename = f"overview_plots.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def _create_data_quality_plots(self, quality_data: Dict) -> str:
        """Create data quality visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Data Quality Analysis', fontsize=16, fontweight='bold')
        
        # 1. Missing values heatmap
        if quality_data.get('missing_values', {}).get('total_missing', 0) > 0:
            missing_matrix = self.df.isnull()
            sns.heatmap(missing_matrix, cbar=True, ax=axes[0, 0], cmap='viridis')
            axes[0, 0].set_title('Missing Values Heatmap')
        else:
            axes[0, 0].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Missing Values Heatmap')
        
        # 2. Missing values by column
        missing_counts = quality_data.get('missing_values', {}).get('count', {})
        if missing_counts:
            missing_df = pd.DataFrame(list(missing_counts.items()), columns=['Column', 'Missing_Count'])
            missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=True)
            
            if len(missing_df) > 0:
                axes[0, 1].barh(range(len(missing_df)), missing_df['Missing_Count'], color=self.colors['warning'])
                axes[0, 1].set_yticks(range(len(missing_df)))
                axes[0, 1].set_yticklabels(missing_df['Column'])
                axes[0, 1].set_title('Missing Values by Column')
                axes[0, 1].set_xlabel('Missing Count')
            else:
                axes[0, 1].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('Missing Values by Column')
        else:
            axes[0, 1].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Missing Values by Column')
        
        # 3. Data quality score
        quality_score = quality_data.get('data_quality_score', 100)
        axes[1, 0].pie([quality_score, 100-quality_score], labels=['Quality Score', 'Issues'], 
                       colors=[self.colors['accent'], self.colors['warning']], autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title(f'Data Quality Score: {quality_score:.1f}/100')
        
        # 4. Duplicates and constant columns
        duplicate_count = quality_data.get('duplicates', {}).get('count', 0)
        constant_cols = quality_data.get('constant_columns', [])
        
        issues = ['Duplicates', 'Constant Columns']
        issue_counts = [duplicate_count, len(constant_cols)]
        colors = [self.colors['warning'] if duplicate_count > 0 else self.colors['accent'], 
                 self.colors['warning'] if constant_cols else self.colors['accent']]
        
        axes[1, 1].bar(issues, issue_counts, color=colors)
        axes[1, 1].set_title('Data Issues Summary')
        axes[1, 1].set_ylabel('Count')
        
        plt.tight_layout()
        
        # Save plot
        filename = f"data_quality_plots.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def _create_univariate_plots(self, univariate_data: Dict) -> str:
        """Create univariate analysis visualizations"""
        if not univariate_data:
            return "no_univariate_data.png"
        
        # Calculate number of subplots needed
        total_cols = len(univariate_data)
        cols_per_row = 3
        rows = (total_cols + cols_per_row - 1) // cols_per_row
        
        fig, axes = plt.subplots(rows, cols_per_row, figsize=(15, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Univariate Analysis', fontsize=16, fontweight='bold')
        
        plot_idx = 0
        for col_name, col_data in univariate_data.items():
            row = plot_idx // cols_per_row
            col = plot_idx % cols_per_row
            
            if col_data['type'] == 'numerical':
                self._plot_numerical_distribution(axes[row, col], col_name, col_data)
            else:
                self._plot_categorical_distribution(axes[row, col], col_name, col_data)
            
            plot_idx += 1
        
        # Hide empty subplots
        for i in range(plot_idx, rows * cols_per_row):
            row = i // cols_per_row
            col = i % cols_per_row
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        # Save plot
        filename = f"univariate_plots.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def _plot_numerical_distribution(self, ax, col_name: str, col_data: Dict):
        """Plot distribution for numerical column"""
        data = self.df[col_name].dropna()
        
        # Histogram with KDE
        ax.hist(data, bins=30, alpha=0.7, color=self.colors['primary'], density=True)
        
        # Add KDE
        try:
            from scipy import stats
            kde_x = np.linspace(data.min(), data.max(), 100)
            kde = stats.gaussian_kde(data)
            ax.plot(kde_x, kde(kde_x), color=self.colors['secondary'], linewidth=2)
        except ImportError:
            pass
        
        # Add statistics
        mean_val = col_data['distribution']['mean']
        median_val = col_data['distribution']['median']
        
        ax.axvline(mean_val, color=self.colors['warning'], linestyle='--', label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color=self.colors['accent'], linestyle='--', label=f'Median: {median_val:.2f}')
        
        ax.set_title(f'{col_name} Distribution')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_categorical_distribution(self, ax, col_name: str, col_data: Dict):
        """Plot distribution for categorical column"""
        value_counts = self.df[col_name].value_counts().head(10)  # Top 10 categories
        
        if len(value_counts) > 0:
            bars = ax.bar(range(len(value_counts)), value_counts.values, color=self.colors['primary'])
            ax.set_xticks(range(len(value_counts)))
            ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
            ax.set_title(f'{col_name} Distribution (Top 10)')
            ax.set_ylabel('Count')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{col_name} Distribution')
    
    def _create_bivariate_plots(self, bivariate_data: Dict) -> str:
        """Create bivariate analysis visualizations"""
        if not bivariate_data:
            return "no_bivariate_data.png"
        
        # Focus on correlations if available
        if 'correlations' in bivariate_data:
            return self._create_correlation_plots(bivariate_data['correlations'])
        
        return "no_correlation_data.png"
    
    def _create_correlation_plots(self, correlation_data: Dict) -> str:
        """Create correlation analysis visualizations"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Correlation Analysis', fontsize=16, fontweight='bold')
        
        # 1. Correlation heatmap
        corr_matrix = pd.DataFrame(correlation_data['correlation_matrix'])
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0, 
                   square=True, ax=axes[0], cbar_kws={"shrink": .8})
        axes[0].set_title('Correlation Matrix')
        
        # 2. High correlations bar plot
        high_corrs = correlation_data.get('high_correlations', [])
        if high_corrs:
            # Take top 10 high correlations
            top_corrs = high_corrs[:10]
            feature_pairs = [f"{pair['feature1']}\n{pair['feature2']}" for pair in top_corrs]
            corr_values = [pair['correlation'] for pair in top_corrs]
            
            colors = [self.colors['warning'] if abs(val) > 0.8 else self.colors['primary'] for val in corr_values]
            
            bars = axes[1].barh(range(len(feature_pairs)), corr_values, color=colors)
            axes[1].set_yticks(range(len(feature_pairs)))
            axes[1].set_yticklabels(feature_pairs)
            axes[1].set_title('Top High Correlations')
            axes[1].set_xlabel('Correlation Coefficient')
            axes[1].axvline(x=0, color='black', linestyle='-', alpha=0.3)
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'No High Correlations\n(threshold: 0.7)', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Top High Correlations')
        
        plt.tight_layout()
        
        # Save plot
        filename = f"correlation_plots.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def _create_multivariate_plots(self, multivariate_data: Dict) -> str:
        """Create multivariate analysis visualizations"""
        if not multivariate_data:
            return "no_multivariate_data.png"
        
        plots_created = []
        
        # PCA plots
        if 'pca' in multivariate_data and 'error' not in multivariate_data['pca']:
            pca_plot = self._create_pca_plots(multivariate_data['pca'])
            plots_created.append(pca_plot)
        
        # Feature importance plots
        if 'feature_importance' in multivariate_data and 'error' not in multivariate_data['feature_importance']:
            importance_plot = self._create_feature_importance_plots(multivariate_data['feature_importance'])
            plots_created.append(importance_plot)
        
        if not plots_created:
            return "no_multivariate_plots.png"
        
        return plots_created[0]  # Return first plot for now
    
    def _create_pca_plots(self, pca_data: Dict) -> str:
        """Create PCA analysis visualizations"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Principal Component Analysis', fontsize=16, fontweight='bold')
        
        # 1. Explained variance ratio
        explained_variance = pca_data['explained_variance_ratio']
        cumulative_variance = pca_data['cumulative_variance']
        components = range(1, len(explained_variance) + 1)
        
        axes[0].bar(components, explained_variance, alpha=0.7, color=self.colors['primary'])
        axes[0].set_xlabel('Principal Component')
        axes[0].set_ylabel('Explained Variance Ratio')
        axes[0].set_title('Explained Variance by Component')
        axes[0].grid(True, alpha=0.3)
        
        # 2. Cumulative explained variance
        axes[1].plot(components, cumulative_variance, 'o-', color=self.colors['secondary'], linewidth=2, markersize=6)
        axes[1].axhline(y=0.95, color=self.colors['warning'], linestyle='--', label='95% Variance Threshold')
        axes[1].axhline(y=0.80, color=self.colors['accent'], linestyle='--', label='80% Variance Threshold')
        
        # Mark the 95% variance point
        n_components_95 = pca_data.get('n_components_95_variance', 0)
        if n_components_95 > 0:
            axes[1].axvline(x=n_components_95, color=self.colors['warning'], linestyle=':', alpha=0.7)
            axes[1].text(n_components_95, 0.95, f'{n_components_95} components\nfor 95% variance', 
                        ha='center', va='bottom', fontsize=10)
        
        axes[1].set_xlabel('Principal Component')
        axes[1].set_ylabel('Cumulative Explained Variance')
        axes[1].set_title('Cumulative Explained Variance')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        filename = f"pca_plots.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def _create_feature_importance_plots(self, importance_data: Dict) -> str:
        """Create feature importance visualizations"""
        feature_importance = importance_data['feature_importance']
        top_features = importance_data['top_features']
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Feature importance bar plot
        features = list(feature_importance.keys())
        importance_values = list(feature_importance.values())
        
        # Sort by importance
        sorted_indices = np.argsort(importance_values)[::-1]
        features = [features[i] for i in sorted_indices]
        importance_values = [importance_values[i] for i in sorted_indices]
        
        bars = axes[0].bar(range(len(features)), importance_values, color=self.colors['primary'])
        axes[0].set_xticks(range(len(features)))
        axes[0].set_xticklabels(features, rotation=45, ha='right')
        axes[0].set_title('Feature Importance (Correlation with Target)')
        axes[0].set_ylabel('Absolute Correlation')
        axes[0].grid(True, alpha=0.3)
        
        # 2. Top features pie chart
        if top_features:
            top_importance = {k: feature_importance[k] for k in top_features if k in feature_importance}
            if top_importance:
                axes[1].pie(top_importance.values(), labels=top_importance.keys(), autopct='%1.1f%%', 
                           startangle=90, colors=plt.cm.Set3(np.linspace(0, 1, len(top_importance))))
                axes[1].set_title('Top 5 Most Important Features')
            else:
                axes[1].text(0.5, 0.5, 'No Feature Importance\nData Available', 
                            ha='center', va='center', transform=axes[1].transAxes)
                axes[1].set_title('Top Features Distribution')
        else:
            axes[1].text(0.5, 0.5, 'No Top Features\nIdentified', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Top Features Distribution')
        
        plt.tight_layout()
        
        # Save plot
        filename = f"feature_importance_plots.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def _create_target_analysis_plots(self, target_data: Dict) -> str:
        """Create target variable analysis visualizations"""
        if not target_data or 'error' in target_data:
            return "no_target_data.png"
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Target Variable Analysis: {self.target_col}', fontsize=16, fontweight='bold')
        
        if target_data['type'] == 'classification':
            # Classification target
            class_dist = target_data['class_distribution']
            class_balance = target_data['class_balance']
            
            # 1. Class distribution
            classes = list(class_dist.keys())
            counts = list(class_dist.values())
            
            bars = axes[0].bar(classes, counts, color=self.colors['primary'])
            axes[0].set_title('Class Distribution')
            axes[0].set_ylabel('Count')
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].grid(True, alpha=0.3)
            
            # 2. Class balance assessment
            balance_colors = {
                'balanced': self.colors['accent'],
                'moderately_imbalanced': self.colors['warning'],
                'highly_imbalanced': self.colors['warning']
            }
            
            axes[1].pie([1], labels=[f'Class Balance:\n{class_balance.replace("_", " ").title()}'], 
                       colors=[balance_colors.get(class_balance, self.colors['primary'])], 
                       autopct='', startangle=90)
            axes[1].set_title('Class Balance Assessment')
            
        else:
            # Regression target
            target_col = self.target_col
            target_data_clean = self.df[target_col].dropna()
            
            # 1. Target distribution
            axes[0].hist(target_data_clean, bins=30, alpha=0.7, color=self.colors['primary'], density=True)
            axes[0].set_title(f'{target_col} Distribution')
            axes[0].set_xlabel('Value')
            axes[0].set_ylabel('Density')
            axes[0].grid(True, alpha=0.3)
            
            # 2. Target statistics
            mean_val = target_data['distribution']['mean']
            median_val = target_data['distribution']['median']
            
            axes[1].text(0.5, 0.7, f'Mean: {mean_val:.2f}', ha='center', va='center', transform=axes[1].transAxes, fontsize=14)
            axes[1].text(0.5, 0.5, f'Median: {median_val:.2f}', ha='center', va='center', transform=axes[1].transAxes, fontsize=14)
            axes[1].text(0.5, 0.3, f'Std: {target_data["distribution"]["std"]:.2f}', ha='center', va='center', transform=axes[1].transAxes, fontsize=14)
            axes[1].text(0.5, 0.1, f'Range: {target_data["distribution"]["min"]:.2f} - {target_data["distribution"]["max"]:.2f}', ha='center', va='center', transform=axes[1].transAxes, fontsize=12)
            axes[1].set_title('Target Statistics')
            axes[1].axis('off')
        
        plt.tight_layout()
        
        # Save plot
        filename = f"target_analysis_plots.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename

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

def generate_visualizations(df: pd.DataFrame, analysis_results: Dict, target_col: Optional[str] = None) -> Dict[str, str]:
    """
    Main function to generate all visualizations
    """
    visualizer = DataScribeVisualizer(df, target_col)
    results = visualizer.generate_all_visualizations(analysis_results)
    return convert_numpy_types(results)
