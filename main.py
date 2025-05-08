import pandas as pd
import numpy as np
import logging
from scipy.stats import entropy, shapiro, anderson, normaltest


class DatasetAnalyzer:
    def __init__(self, data: pd.DataFrame, log_level=logging.INFO):
        self.data = data.copy()
        self.datetime_cols = self._detect_datetime_columns()
        self.analysis = {}

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=log_level)

    def _detect_datetime_columns(self):
        detected = []
        for col in self.data.columns:
            if self.data[col].dtype == 'object':

                try:
                    self.data[col] = pd.to_datetime(
                        self.data[col], errors='coerce')

                    if self.data[col].isnull().sum() < len(self.data[col]):
                        detected.append(col)
                except Exception as e:
                    self.logger.warning(
                        f"Column '{col}' could not be converted to datetime. Error: {str(e)}")
                    continue
        return detected

    def analyze(self, target_col=None):
        self.analysis = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.astype(str).to_dict(),
            'memory_usage_MB': round(self.data.memory_usage(deep=True).sum() / (1024 ** 2), 2),
            'missing_values': self.data.isnull().sum().to_dict(),
            'missing_percentage': (self.data.isnull().mean() * 100).round(2).to_dict(),
            'duplicates': int(self.data.duplicated().sum()),
            'duplicate_percentage': round(self.data.duplicated().mean() * 100, 2),
            'high_missing_columns': [col for col, pct in (self.data.isnull().mean() * 100).items() if pct > 50],
            'numeric_stats': self._analyze_numeric(),
            'categorical_stats': self._analyze_categorical(),
            'datetime_stats': self._analyze_datetime(),
            'correlation_matrix': self._compute_correlation(),
            'target_imbalance': self._check_imbalance(target_col) if target_col else None
        }
        return self.analysis

    def _analyze_numeric(self):
        numeric_data = self.data.select_dtypes(include='number')
        if numeric_data.empty:
            return {}

        stats = {
            'describe': numeric_data.describe().to_dict(),
            'skewness': numeric_data.skew().to_dict(),
            'kurtosis': numeric_data.kurtosis().to_dict(),
            'modes': numeric_data.mode().iloc[0].to_dict() if not numeric_data.mode().empty else {},
            'outliers': {},
            'zero_counts': (numeric_data == 0).sum().to_dict(),
            'negative_counts': (numeric_data < 0).sum().to_dict(),
            'variance': numeric_data.var().to_dict(),
            'coefficient_of_variation': (numeric_data.std() / numeric_data.mean()).fillna(0).to_dict(),
            'distributions': self._analyze_distributions(numeric_data),
            'normality_tests': self._test_normality(numeric_data)
        }

        for col in numeric_data.columns:
            Q1 = numeric_data[col].quantile(0.25)
            Q3 = numeric_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outlier_mask = (numeric_data[col] < lower) | (
                numeric_data[col] > upper)

            stats['outliers'][col] = {
                'count': int(outlier_mask.sum()),
                'lower_bound': round(lower, 2),
                'upper_bound': round(upper, 2)
            }

        return stats

    def _analyze_distributions(self, numeric_data):
        distributions = {}
        for col in numeric_data.columns:
            data = numeric_data[col].dropna()
            if len(data) == 0:
                continue
            skew = data.skew()
            kurt = data.kurtosis()
            if abs(skew) < 0.5 and abs(kurt) < 0.5:
                dist_type = "Normal"
            elif skew > 1:
                dist_type = "Right-skewed"
            elif skew < -1:
                dist_type = "Left-skewed"
            else:
                dist_type = "Moderate"
            distributions[col] = {
                'distribution_type': dist_type,
                'skewness': round(skew, 3),
                'kurtosis': round(kurt, 3)
            }
        return distributions

    def _test_normality(self, numeric_data):
        results = {}
        for col in numeric_data.columns:
            data = numeric_data[col].dropna()
            if len(data) >= 3:
                if len(data) > 5000:
                    data = data.sample(5000, random_state=1)
                try:
                    stat, p = shapiro(data)
                    results[col] = {
                        'shapiro_stat': round(stat, 4),
                        'p_value': round(p, 4),
                        'is_normal': p > 0.05
                    }
                except:
                    try:
                        result = anderson(data)
                        results[col] = {
                            'anderson_stat': round(result.statistic, 4),
                            'anderson_critical_values': result.critical_values.round(4),
                            # 5% significance
                            'is_normal': result.statistic < result.critical_values[2]
                        }
                    except:
                        try:
                            stat, p = normaltest(data)
                            results[col] = {
                                'normaltest_stat': round(stat, 4),
                                'normaltest_p_value': round(p, 4),
                                'is_normal': p > 0.05
                            }
                        except:
                            self.logger.warning(
                                f"Failed normality tests for {col}")
        return results

    def _analyze_categorical(self):
        cat_data = self.data.select_dtypes(include=['object', 'category'])
        if cat_data.empty:
            return {}

        stats = {
            'unique_counts': cat_data.nunique().to_dict(),
            'top_values': {col: cat_data[col].value_counts().head(5).to_dict() for col in cat_data.columns},
            'top_percentages': {col: (cat_data[col].value_counts(normalize=True) * 100).head(5).round(2).to_dict() for col in cat_data.columns},
            'entropy': {},
            'cardinality': {},
            'most_frequent_percentage': {},
            'least_frequent_percentage': {}
        }

        for col in cat_data.columns:
            value_counts = cat_data[col].value_counts(normalize=True)
            if len(value_counts) > 1:
                stats['entropy'][col] = round(entropy(value_counts), 3)
            else:
                stats['entropy'][col] = 0.0

            unique_count = cat_data[col].nunique()
            total_count = len(cat_data[col])

            if unique_count == 1:
                cardinality = "Constant"
            elif unique_count == 2:
                cardinality = "Binary"
            elif unique_count <= 10:
                cardinality = "Low"
            elif unique_count <= 100:
                cardinality = "Medium"
            else:
                cardinality = "High"

            stats['cardinality'][col] = cardinality

            stats['most_frequent_percentage'][col] = round(
                value_counts.iloc[0] * 100, 2)
            stats['least_frequent_percentage'][col] = round(
                value_counts.iloc[-1] * 100, 2)

        return stats

    def _analyze_datetime(self):
        summary = {}
        for col in self.datetime_cols:
            s = self.data[col].dropna()
            if s.empty:
                continue
            summary[col] = {
                'min': str(s.min()),
                'max': str(s.max()),
                'range_days': (s.max() - s.min()).days,
                'missing': int(self.data[col].isnull().sum()),
                'weekend_percentage': round((s.dt.dayofweek >= 5).mean() * 100, 2),
                'has_time_component': any(s.dt.hour != 0)
            }
        return summary

    def _compute_correlation(self):
        numeric = self.data.select_dtypes(include='number')
        if numeric.shape[1] < 2:
            return None
        return numeric.corr().round(3).to_dict()

    def _check_imbalance(self, target_col):
        if target_col not in self.data.columns:
            return None
        target = self.data[target_col]
        if target.dtype not in ['object', 'category'] or target.nunique() > 10:
            return None
        counts = target.value_counts(normalize=True).round(3)
        return {
            'distribution': counts.to_dict(),
            'majority_class': counts.index[0],
            'majority_percentage': round(counts.iloc[0] * 100, 2),
            'minority_class': counts.index[-1] if len(counts) > 1 else None,
            'minority_percentage': round(counts.iloc[-1] * 100, 2) if len(counts) > 1 else None,
            'imbalance_ratio': round(counts.iloc[0] / counts.iloc[-1], 2) if len(counts) > 1 else None
        }
