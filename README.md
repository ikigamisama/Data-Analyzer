# CleanDatasetAnalyzer

`CleanDatasetAnalyzer` is a Python class designed to analyze datasets and provide valuable insights before performing any data preprocessing. It offers a comprehensive set of features to explore numeric, categorical, and datetime data, check for missing values, duplicates, and calculate various statistics. It can also assess the distribution and normality of numerical columns, and help detect imbalance in the target column.

## Features

- **Data Overview**: Shape, columns, data types, memory usage, and missing values.
- **Missing Values**: Identifies columns with a high percentage of missing values (greater than 50%).
- **Duplicates**: Detects duplicate rows and calculates their percentage.
- **Numeric Stats**: Calculates statistics such as mean, median, variance, skewness, kurtosis, and more for numeric columns.
- **Categorical Stats**: Calculates unique counts, top values, entropy, cardinality, and frequencies of categorical columns.
- **Datetime Stats**: Provides summary information for datetime columns, including min/max, range, missing values, and weekend percentage.
- **Correlation Matrix**: Computes the correlation matrix for numeric columns.
- **Target Imbalance**: Checks for class imbalance in the target column, if specified.

## Installation

You can install the required dependencies using `pip`:

```bash
pip install pandas numpy scipy
```

## Usage

### 1. Import the class and create an instance:

```python
from clean_dataset_analyzer import CleanDatasetAnalyzer
import pandas as pd

# Sample dataset
data = pd.DataFrame({
    'age': [22, 25, 27, 30, 22, 23, 23, 28],
    'gender': ['male', 'female', 'female', 'male', 'male', 'female', 'female', 'male'],
    'date_of_birth': ['1998-01-01', '1997-05-12', '1996-11-23', '1995-03-18', '1998-05-14', '1997-07-21', '1996-09-30', '1997-10-15']
})

# Instantiate the analyzer
analyzer = CleanDatasetAnalyzer(data)
```

### 2. Analyze the dataset:

```python
# Perform analysis
analysis_results = analyzer.analyze()

# Print the results
print(analysis_results)
```

### 3. Specific analysis of target imbalance:

```python
# Analyze the target column for class imbalance (for example: 'gender')
target_analysis = analyzer.analyze(target_col='gender')
print(target_analysis)
```

## Functions

- `__init__(self, data: pd.DataFrame)` - Initializes the analyzer with a pandas DataFrame.
- `analyze(self, target_col=None)` - Analyzes the dataset and returns a dictionary of analysis results.
- `_analyze_numeric(self)` - Analyzes numeric columns, computing descriptive statistics and more.
- `_analyze_categorical(self)` - Analyzes categorical columns and computes top values, entropy, and more.
- `_analyze_datetime(self)` - Analyzes datetime columns and returns a summary of the data.
- `_compute_correlation(self)` - Computes the correlation matrix for numeric columns.
- `_check_imbalance(self, target_col)` - Checks for class imbalance in the target column.

## License

MIT License
