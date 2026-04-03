"""
DS Mentor Pro — Knowledge Base Generator (pure stdlib).
Produces 210+ QA pairs across 7 pipeline stages with enriched fields:
  query, stage, answer, code, why_explanation, when_to_use,
  common_pitfall, related_questions, difficulty
"""
import csv
import json
import os

FIELDS = [
    "query", "stage", "answer", "code",
    "why_explanation", "when_to_use", "common_pitfall",
    "related_questions", "difficulty",
]

# ---------------------------------------------------------------------------
# HAND-CRAFTED SEED DATASET  (6 per stage = 42 entries)
# ---------------------------------------------------------------------------

SEED = [
    # ── Stage 1: Problem Understanding ──
    {
        "query": "What is the goal of a Titanic survival prediction task?",
        "stage": 1, "difficulty": "beginner",
        "answer": "The goal is to predict whether a passenger survived the Titanic disaster based on features like Age, Sex, Pclass, and Fare. It is a binary classification problem.",
        "code": "target = 'Survived'\nprint(f'Target column: {target}')\nprint(f'Classes: 0 = died, 1 = survived')",
        "why_explanation": "Framing the problem clearly determines every downstream choice: what metric to optimise, what data to collect, and what model family fits. Without a clear objective you risk solving the wrong problem.",
        "when_to_use": "Always start here. Use binary classification framing when the target has two outcomes. Use regression framing when the target is continuous. Use multi-class when there are 3+ categories.",
        "common_pitfall": "Jumping straight to modelling without defining a success metric leads to models that optimise the wrong thing — e.g. accuracy on an imbalanced dataset.",
        "related_questions": ["How do I choose the right evaluation metric?", "What is the difference between classification and regression?", "How do I establish a baseline model?"],
    },
    {
        "query": "How do I define a baseline model?",
        "stage": 1, "difficulty": "beginner",
        "answer": "A baseline model is the simplest reasonable predictor. For classification use the majority-class predictor; for regression use the mean of the target.",
        "code": "from sklearn.dummy import DummyClassifier\nbaseline = DummyClassifier(strategy='most_frequent')\nbaseline.fit(X_train, y_train)\nprint('Baseline accuracy:', baseline.score(X_test, y_test))",
        "why_explanation": "A baseline gives you a lower bound. Any useful model must beat it. If your fancy model barely outperforms predicting the majority class, the added complexity is not justified.",
        "when_to_use": "Compute a baseline before trying any real model. Use majority-class for classification, mean/median for regression. For time series, use naive persistence (predict last value).",
        "common_pitfall": "Skipping the baseline makes it impossible to know if your model is genuinely learning patterns or just memorising noise.",
        "related_questions": ["What evaluation metric should I use for imbalanced data?", "How do I load a CSV file with pandas?", "What is overfitting and how do I detect it?"],
    },
    {
        "query": "How do I choose the right evaluation metric?",
        "stage": 1, "difficulty": "intermediate",
        "answer": "Choose the metric that aligns with the business cost. Use accuracy for balanced classes, F1 for imbalanced data, AUC-ROC when ranking matters, and RMSE/MAE for regression.",
        "code": "# Example: classification on imbalanced data\nfrom sklearn.metrics import f1_score, roc_auc_score\nprint('F1:', f1_score(y_test, preds))\nprint('AUC:', roc_auc_score(y_test, preds_proba))",
        "why_explanation": "Different metrics penalise different types of errors. Accuracy treats all mistakes equally, while F1 balances precision and recall. AUC measures ranking quality regardless of threshold.",
        "when_to_use": "Use accuracy only when classes are balanced. Use F1 or precision/recall when false positives and false negatives have different costs. Use AUC-ROC when you need threshold-independent evaluation.",
        "common_pitfall": "Using accuracy on a dataset where 95% of samples belong to one class gives a misleading 95% score even if the model never predicts the minority class.",
        "related_questions": ["What is the difference between precision and recall?", "How do I handle class imbalance?", "How do I calculate AUC-ROC score?"],
    },
    {
        "query": "What is the difference between classification and regression?",
        "stage": 1, "difficulty": "beginner",
        "answer": "Classification predicts discrete categories (spam/not-spam). Regression predicts continuous values (house price). The choice depends on whether your target variable is categorical or numeric.",
        "code": "# Classification target\nprint(df['Survived'].value_counts())\n# Regression target\nprint(df['SalePrice'].describe())",
        "why_explanation": "The distinction determines the loss function, evaluation metric, and model family. Cross-entropy loss suits classification; MSE suits regression. Using the wrong type produces meaningless predictions.",
        "when_to_use": "Use classification when the target is categorical (yes/no, A/B/C). Use regression when the target is a real number (price, temperature). Use ordinal regression for ordered categories (low/medium/high).",
        "common_pitfall": "Treating an ordinal variable (ratings 1-5) as unordered classification throws away the ordering information. Consider ordinal regression or treating it as regression.",
        "related_questions": ["What is the goal of a Titanic survival prediction task?", "How do I define a baseline model?", "How do I load a CSV file with pandas?"],
    },
    {
        "query": "How do I scope a data science project?",
        "stage": 1, "difficulty": "intermediate",
        "answer": "Define the business question, identify the target variable, decide on success criteria, estimate data availability, and set a timeline. Use the CRISP-DM framework as a guide.",
        "code": "project = {\n    'objective': 'Predict customer churn',\n    'target': 'churn (binary)',\n    'metric': 'F1-score > 0.75',\n    'data': 'CRM export, 50k rows',\n    'deadline': '4 weeks'\n}\nfor k, v in project.items():\n    print(f'{k}: {v}')",
        "why_explanation": "Scoping prevents scope creep and ensures stakeholders agree on what success looks like before work begins. It also surfaces data gaps early, saving wasted modelling effort.",
        "when_to_use": "Scope every project before writing code. Use CRISP-DM for general DS projects. Use design docs for ML systems going to production. Keep it lightweight for exploratory analysis.",
        "common_pitfall": "Starting to code without a clear target variable or success metric leads to aimless exploration and models that nobody can evaluate.",
        "related_questions": ["How do I define a baseline model?", "How do I choose the right evaluation metric?", "How do I load a CSV file with pandas?"],
    },
    {
        "query": "How do I establish a baseline for regression?",
        "stage": 1, "difficulty": "beginner",
        "answer": "For regression, the simplest baseline predicts the mean (or median) of the target for every sample. Compare your model's RMSE/MAE against this baseline.",
        "code": "import numpy as np\nbaseline_pred = np.full(len(y_test), y_train.mean())\nmae = np.mean(np.abs(y_test - baseline_pred))\nprint('Baseline MAE:', round(mae, 2))",
        "why_explanation": "The mean predictor minimises MSE by definition. If your model cannot beat the mean, it has learned nothing useful. The baseline quantifies the 'value added' by the model.",
        "when_to_use": "Use mean baseline when target distribution is symmetric. Use median baseline when the target is skewed or has outliers. For time series, use the last observed value.",
        "common_pitfall": "Reporting model MAE without comparing to the baseline makes it impossible to judge whether the model is actually useful.",
        "related_questions": ["How do I define a baseline model?", "What is the goal of a Titanic survival prediction task?", "How do I check the distribution of a numeric column?"],
    },

    # ── Stage 2: Data Loading ──
    {
        "query": "How do I load a CSV file with pandas?",
        "stage": 2, "difficulty": "beginner",
        "answer": "Use pandas read_csv(). Pass the file path and optional parameters like encoding, sep, and na_values to handle different CSV dialects.",
        "code": "import pandas as pd\ndf = pd.read_csv('titanic.csv')\nprint(df.shape)\nprint(df.dtypes)",
        "why_explanation": "read_csv parses delimited text into a DataFrame with typed columns. Pandas infers types automatically but may guess wrong for dates or mixed-type columns, so always verify with df.dtypes.",
        "when_to_use": "Use read_csv for comma/tab-delimited files. Use read_excel for .xlsx files. Use read_json for JSON. Use read_sql for database queries. Use read_parquet for large columnar datasets.",
        "common_pitfall": "Not specifying encoding (e.g. encoding='latin-1') causes UnicodeDecodeError on files with special characters. Always check encoding if you get garbled text.",
        "related_questions": ["How do I check the shape and dtypes of a DataFrame?", "How do I handle encoding errors when loading data?", "How do I create a correlation heatmap?"],
    },
    {
        "query": "How do I check the shape and dtypes of a DataFrame?",
        "stage": 2, "difficulty": "beginner",
        "answer": "Use df.shape for dimensions, df.dtypes for column types, and df.info() for a combined summary including non-null counts and memory usage.",
        "code": "print('Shape:', df.shape)\nprint('\\nColumn types:')\nprint(df.dtypes)\nprint('\\nInfo:')\ndf.info()",
        "why_explanation": "Knowing the shape tells you how many samples and features you have. dtypes reveal type mismatches (e.g. numeric column stored as object due to stray characters) that must be fixed before analysis.",
        "when_to_use": "Run these checks immediately after loading any dataset. Use df.describe() alongside for summary statistics of numeric columns. Use df.head() to visually inspect the first rows.",
        "common_pitfall": "Ignoring object-type columns that should be numeric leads to silent errors in calculations. A single non-numeric entry can cause pandas to cast the entire column as object.",
        "related_questions": ["How do I load a CSV file with pandas?", "How do I detect missing values in a DataFrame?", "Show the distribution of a numeric column."],
    },
    {
        "query": "How do I handle encoding errors when loading data?",
        "stage": 2, "difficulty": "intermediate",
        "answer": "Specify the correct encoding in read_csv. Common encodings are utf-8, latin-1, and cp1252. Use errors='replace' as a last resort.",
        "code": "import pandas as pd\n# Try utf-8 first, fall back to latin-1\ntry:\n    df = pd.read_csv('data.csv', encoding='utf-8')\nexcept UnicodeDecodeError:\n    df = pd.read_csv('data.csv', encoding='latin-1')\nprint(df.head())",
        "why_explanation": "Text files use different byte-to-character mappings. UTF-8 covers all Unicode but older systems often produce Latin-1 or Windows cp1252 files. Mismatched encoding causes decode errors or garbled text.",
        "when_to_use": "Try utf-8 first (most common). Use latin-1 for European legacy data. Use cp1252 for Windows-exported CSVs. Use chardet library to auto-detect if unsure.",
        "common_pitfall": "Using errors='ignore' silently drops characters, corrupting your data without warning. Prefer errors='replace' so you can spot and fix problems.",
        "related_questions": ["How do I load a CSV file with pandas?", "How do I check the shape and dtypes of a DataFrame?", "How do I detect missing values in a DataFrame?"],
    },
    {
        "query": "How do I load data from a SQL database?",
        "stage": 2, "difficulty": "intermediate",
        "answer": "Use pandas read_sql() with a SQLAlchemy connection string. You can pass a raw SQL query or a table name.",
        "code": "import pandas as pd\nfrom sqlalchemy import create_engine\nengine = create_engine('sqlite:///my_database.db')\ndf = pd.read_sql('SELECT * FROM customers', engine)\nprint(df.head())",
        "why_explanation": "read_sql sends the query to the database engine and converts the result set into a DataFrame. SQLAlchemy provides a unified interface across Postgres, MySQL, SQLite, etc.",
        "when_to_use": "Use read_sql for ad-hoc analysis from production databases. Use read_csv if data is already exported. Use read_parquet for large analytical workloads that need columnar compression.",
        "common_pitfall": "Loading an entire large table into memory with SELECT * can crash your process. Always add WHERE clauses or LIMIT to control the result size.",
        "related_questions": ["How do I load a CSV file with pandas?", "How do I check the shape and dtypes of a DataFrame?", "How do I create a correlation heatmap?"],
    },
    {
        "query": "How do I detect missing values in a DataFrame?",
        "stage": 2, "difficulty": "beginner",
        "answer": "Use df.isnull().sum() to count missing values per column, and df.isnull().mean() to get the fraction missing.",
        "code": "print('Missing counts:')\nprint(df.isnull().sum())\nprint('\\nMissing fractions:')\nprint(df.isnull().mean().round(3))",
        "why_explanation": "NaN values propagate silently through calculations, producing wrong results. Knowing which columns have missing data and how much tells you whether to impute, drop, or investigate further.",
        "when_to_use": "Check for missing values right after loading and again after any merge/join. Use df.isnull().sum() for a quick count. Use missingno library for visual missing-data patterns.",
        "common_pitfall": "Some datasets encode missing values as empty strings, 'N/A', -999, or 'null' rather than NaN. Use na_values parameter in read_csv to catch these.",
        "related_questions": ["How do I fill missing values in a numeric column?", "How do I check the shape and dtypes of a DataFrame?", "Plot the missing pattern for a DataFrame."],
    },
    {
        "query": "How do I load a JSON file into pandas?",
        "stage": 2, "difficulty": "beginner",
        "answer": "Use pd.read_json() for flat JSON or pd.json_normalize() for nested JSON structures.",
        "code": "import pandas as pd\ndf = pd.read_json('data.json')\nprint(df.head())\n# For nested JSON:\n# df = pd.json_normalize(data, record_path='items')",
        "why_explanation": "JSON stores data as nested key-value pairs. read_json handles flat arrays of objects directly. json_normalize flattens nested structures into a tabular DataFrame, which is necessary for analysis.",
        "when_to_use": "Use read_json for simple JSON arrays. Use json_normalize for API responses with nested objects. Use read_csv if the data is already tabular.",
        "common_pitfall": "Deeply nested JSON produces columns with dot-separated names like 'address.city'. You may need to rename or restructure these for downstream use.",
        "related_questions": ["How do I load a CSV file with pandas?", "How do I check the shape and dtypes of a DataFrame?", "How do I handle encoding errors when loading data?"],
    },

    # ── Stage 3: Exploratory Data Analysis ──
    {
        "query": "How do I create a correlation heatmap?",
        "stage": 3, "difficulty": "beginner",
        "answer": "Use seaborn heatmap on the DataFrame correlation matrix. Select only numeric columns first.",
        "code": "import seaborn as sns\nimport matplotlib.pyplot as plt\ncorr = df.select_dtypes('number').corr()\nsns.heatmap(corr, annot=True, cmap='coolwarm', center=0)\nplt.title('Correlation Matrix')\nplt.tight_layout()\nplt.show()",
        "why_explanation": "Pearson correlation measures linear relationships between numeric features. Values near +1 or -1 indicate strong linear dependence. The heatmap makes it easy to spot highly correlated feature pairs that may cause multicollinearity.",
        "when_to_use": "Use Pearson correlation for numeric features with linear relationships. Use Spearman for monotonic non-linear relationships. Use Cramér's V for categorical-categorical associations.",
        "common_pitfall": "Correlation does not imply causation, and Pearson misses non-linear relationships. Two features with zero Pearson correlation can still be strongly dependent (e.g. quadratic relationship).",
        "related_questions": ["How do I check for multicollinearity in features?", "Show the distribution of a numeric column.", "How do I detect outliers using IQR?"],
    },
    {
        "query": "How do I check the distribution of a numeric column?",
        "stage": 3, "difficulty": "beginner",
        "answer": "Use a histogram or KDE plot. seaborn histplot or displot show the shape, skewness, and modality of the distribution.",
        "code": "import seaborn as sns\nimport matplotlib.pyplot as plt\nsns.histplot(df['Age'], kde=True, bins=30)\nplt.title('Age Distribution')\nplt.xlabel('Age')\nplt.show()",
        "why_explanation": "The distribution shape determines which statistical methods and transformations are appropriate. A skewed distribution may need log-transform; a bimodal distribution may indicate two sub-populations.",
        "when_to_use": "Use histplot for a single numeric variable. Use boxplot or violinplot to compare distributions across groups. Use pairplot to visualise all pairwise relationships at once.",
        "common_pitfall": "Using too few bins hides the distribution shape; too many bins create noise. Start with 30 bins and adjust. Also check for spikes at specific values (e.g. 0 or -1 often indicate encoded missing data).",
        "related_questions": ["How do I create a correlation heatmap?", "How do I detect outliers using IQR?", "How do I fill missing values in a numeric column?"],
    },
    {
        "query": "How do I detect outliers using IQR?",
        "stage": 3, "difficulty": "intermediate",
        "answer": "Calculate Q1 and Q3, then IQR = Q3 - Q1. Values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR are outliers.",
        "code": "Q1 = df['Fare'].quantile(0.25)\nQ3 = df['Fare'].quantile(0.75)\nIQR = Q3 - Q1\nlower = Q1 - 1.5 * IQR\nupper = Q3 + 1.5 * IQR\noutliers = df[(df['Fare'] < lower) | (df['Fare'] > upper)]\nprint(f'Outliers: {len(outliers)} rows')",
        "why_explanation": "IQR is robust to extreme values because it uses quartiles, not the mean/std. The 1.5×IQR rule flags roughly 0.7% of normally-distributed data, making it a conservative but reliable detector.",
        "when_to_use": "Use IQR for roughly symmetric distributions. Use Z-score when data is approximately normal. Use Isolation Forest or DBSCAN for high-dimensional outlier detection.",
        "common_pitfall": "Automatically removing all IQR outliers can discard valid extreme values (e.g. high-income earners). Always investigate outliers before removal — they may carry important signal.",
        "related_questions": ["How do I check the distribution of a numeric column?", "How do I cap outliers using winsorisation?", "How do I create a correlation heatmap?"],
    },
    {
        "query": "How do I visualise missing data patterns?",
        "stage": 3, "difficulty": "intermediate",
        "answer": "Use a heatmap of df.isnull() or the missingno library to see which columns have missing values and whether missingness is correlated across columns.",
        "code": "import seaborn as sns\nimport matplotlib.pyplot as plt\nsns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')\nplt.title('Missing Data Pattern')\nplt.tight_layout()\nplt.show()",
        "why_explanation": "Correlated missingness (e.g. if Cabin is missing then Embarked is often missing too) suggests the data is not Missing Completely At Random (MCAR). This affects which imputation strategy is valid.",
        "when_to_use": "Use a heatmap for a quick visual check. Use missingno.matrix for larger datasets. Use Little's MCAR test for a statistical test of missingness mechanism.",
        "common_pitfall": "Assuming data is MCAR when it is actually MAR (Missing At Random) or MNAR (Missing Not At Random) leads to biased imputation and biased models.",
        "related_questions": ["How do I detect missing values in a DataFrame?", "How do I fill missing values in a numeric column?", "What is the difference between MCAR, MAR, and MNAR?"],
    },
    {
        "query": "How do I create a pairplot for feature exploration?",
        "stage": 3, "difficulty": "beginner",
        "answer": "Use seaborn pairplot to visualise pairwise scatterplots and marginal distributions for all numeric columns, coloured by the target variable.",
        "code": "import seaborn as sns\nimport matplotlib.pyplot as plt\nsns.pairplot(df[['Age', 'Fare', 'Pclass', 'Survived']], hue='Survived')\nplt.suptitle('Pairplot', y=1.02)\nplt.show()",
        "why_explanation": "Pairplots reveal clusters, separability, and non-linear relationships that correlation numbers alone cannot capture. Colouring by target shows which features discriminate between classes.",
        "when_to_use": "Use pairplot for datasets with fewer than 10 numeric features (otherwise it gets unwieldy). Use PCA + 2D scatter for high-dimensional data. Use individual boxplots for categorical features.",
        "common_pitfall": "Running pairplot on a dataset with 30+ features produces a huge grid that takes minutes to render and is unreadable. Select a subset of features first.",
        "related_questions": ["How do I create a correlation heatmap?", "How do I check the distribution of a numeric column?", "How do I apply PCA for dimensionality reduction?"],
    },
    {
        "query": "How do I check for multicollinearity in features?",
        "stage": 3, "difficulty": "advanced",
        "answer": "Compute the Variance Inflation Factor (VIF) for each feature. VIF > 5 suggests moderate multicollinearity; VIF > 10 indicates severe multicollinearity.",
        "code": "from sklearn.linear_model import LinearRegression\nimport numpy as np\n\ndef calc_vif(X):\n    vifs = []\n    for i in range(X.shape[1]):\n        others = np.delete(X.values, i, axis=1)\n        r2 = LinearRegression().fit(others, X.iloc[:, i]).score(others, X.iloc[:, i])\n        vifs.append(1 / (1 - r2) if r2 < 1 else float('inf'))\n    return vifs\n\nX_num = df.select_dtypes('number').dropna()\nfor col, vif in zip(X_num.columns, calc_vif(X_num)):\n    print(f'{col}: VIF={vif:.1f}')",
        "why_explanation": "Multicollinearity inflates coefficient variance in linear models, making them unstable and hard to interpret. VIF measures how much a feature's variance is explained by other features.",
        "when_to_use": "Check VIF before fitting linear regression, logistic regression, or any model where you interpret coefficients. Tree-based models handle collinearity naturally so VIF is less critical for them.",
        "common_pitfall": "Dropping one of two correlated features without checking VIF may not fix the problem — there could be multicollinearity among 3+ features. Always recheck VIF after removing a feature.",
        "related_questions": ["How do I create a correlation heatmap?", "How do I apply PCA for dimensionality reduction?", "How do I select the most important features?"],
    },

    # ── Stage 4: Preprocessing ──
    {
        "query": "How do I fill missing values in a numeric column?",
        "stage": 4, "difficulty": "beginner",
        "answer": "Use pandas fillna() with a strategy like mean, median, or a constant value. For more sophisticated imputation, use sklearn SimpleImputer or KNNImputer.",
        "code": "from sklearn.impute import SimpleImputer\nimp = SimpleImputer(strategy='median')\ndf[['Age', 'Fare']] = imp.fit_transform(df[['Age', 'Fare']])",
        "why_explanation": "Mean/median imputation preserves the central tendency but assumes data is MCAR. It reduces variance and can weaken correlations. KNN imputation uses similar rows to preserve local structure but is computationally expensive.",
        "when_to_use": "Use median for skewed numeric data or data with outliers. Use mean for roughly normal distributions. Use KNN imputation when missingness depends on other features (MAR). If >40% is missing, consider dropping the column.",
        "common_pitfall": "Fitting the imputer on the entire dataset before train/test split causes data leakage. Always fit on training data only, then transform both train and test.",
        "related_questions": ["What is the difference between MCAR, MAR, and MNAR?", "How do I detect missing values in a DataFrame?", "How do I scale numeric features?"],
    },
    {
        "query": "How do I scale numeric features?",
        "stage": 4, "difficulty": "beginner",
        "answer": "Use StandardScaler (zero mean, unit variance) or MinMaxScaler (range 0-1). Fit on training data only.",
        "code": "from sklearn.preprocessing import StandardScaler\nscaler = StandardScaler()\nX_train_scaled = scaler.fit_transform(X_train)\nX_test_scaled = scaler.transform(X_test)",
        "why_explanation": "Many algorithms (SVM, KNN, neural nets, PCA) are sensitive to feature scale because they use distance metrics. Without scaling, features with large ranges dominate those with small ranges.",
        "when_to_use": "Use StandardScaler when features are roughly Gaussian. Use MinMaxScaler when you need bounded values (e.g. neural networks). Use RobustScaler when data has many outliers. Tree-based models do not need scaling.",
        "common_pitfall": "Calling fit_transform on the full dataset before splitting leaks test statistics into training. Always fit the scaler on X_train only, then transform both X_train and X_test.",
        "related_questions": ["How do I fill missing values in a numeric column?", "How do I apply one-hot encoding?", "How do I cap outliers using winsorisation?"],
    },
    {
        "query": "How do I cap outliers using winsorisation?",
        "stage": 4, "difficulty": "intermediate",
        "answer": "Clip values at the 1st and 99th percentiles (or 5th/95th). This keeps the data range reasonable without discarding rows.",
        "code": "lower = df['Fare'].quantile(0.01)\nupper = df['Fare'].quantile(0.99)\ndf['Fare'] = df['Fare'].clip(lower, upper)\nprint(df['Fare'].describe())",
        "why_explanation": "Winsorisation reduces the influence of extreme values on the mean and variance without removing observations. Unlike trimming, it replaces outliers rather than dropping them, preserving sample size.",
        "when_to_use": "Use winsorisation when you want to keep all rows but limit extreme influence. Use IQR removal when outliers are clearly erroneous. Use log-transform when the distribution is right-skewed.",
        "common_pitfall": "Capping at fixed quantiles on the full dataset before train/test split is a subtle form of data leakage. Compute percentiles on training data only.",
        "related_questions": ["How do I detect outliers using IQR?", "How do I scale numeric features?", "How do I fill missing values in a numeric column?"],
    },
    {
        "query": "How do I encode categorical variables?",
        "stage": 4, "difficulty": "beginner",
        "answer": "Use pd.get_dummies for one-hot encoding or sklearn LabelEncoder/OrdinalEncoder for ordinal variables.",
        "code": "df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)\nprint(df.head())",
        "why_explanation": "Most ML algorithms require numeric input. One-hot encoding creates binary columns for each category, avoiding false ordinal relationships. Label encoding assigns integers, which tree models handle but linear models may misinterpret.",
        "when_to_use": "Use one-hot encoding for nominal categories with few unique values (<15). Use label/ordinal encoding for tree-based models or ordinal features. Use target encoding for high-cardinality features.",
        "common_pitfall": "One-hot encoding a column with 1000 unique values creates 1000 new columns, causing the curse of dimensionality. Use target encoding or frequency encoding for high cardinality.",
        "related_questions": ["How do I apply one-hot encoding?", "How do I scale numeric features?", "How do I handle high-cardinality categorical features?"],
    },
    {
        "query": "What is the difference between MCAR, MAR, and MNAR?",
        "stage": 4, "difficulty": "advanced",
        "answer": "MCAR: missingness is completely random. MAR: missingness depends on observed data. MNAR: missingness depends on the missing value itself. The mechanism determines which imputation methods are valid.",
        "code": "# Check if missingness correlates with another column (suggests MAR)\nimport pandas as pd\nprint('Age missing rate by Pclass:')\nprint(df.groupby('Pclass')['Age'].apply(lambda x: x.isnull().mean()))",
        "why_explanation": "Under MCAR, any imputation method works. Under MAR, you need methods that condition on observed variables (e.g. KNN, MICE). Under MNAR, no standard imputation is unbiased — you need domain knowledge or sensitivity analysis.",
        "when_to_use": "Test for MCAR first (Little's test). If rejected, assume MAR and use multivariate imputation (MICE/KNN). If you suspect MNAR (e.g. high earners hide income), document the assumption and use sensitivity analysis.",
        "common_pitfall": "Treating MNAR data as MCAR leads to biased estimates. If the reason for missingness is related to the missing value, no imputation can fully fix the bias.",
        "related_questions": ["How do I fill missing values in a numeric column?", "How do I visualise missing data patterns?", "How do I detect missing values in a DataFrame?"],
    },
    {
        "query": "How do I handle high-cardinality categorical features?",
        "stage": 4, "difficulty": "advanced",
        "answer": "Use target encoding, frequency encoding, or hashing. Avoid one-hot encoding which creates thousands of sparse columns.",
        "code": "# Target encoding (mean of target per category)\nmeans = df.groupby('City')['Survived'].mean()\ndf['City_encoded'] = df['City'].map(means)\nprint(df[['City', 'City_encoded']].head())",
        "why_explanation": "Target encoding replaces each category with the mean target value, compressing information into a single numeric column. It captures the category-target relationship directly but risks overfitting on rare categories.",
        "when_to_use": "Use target encoding for 50+ categories. Use frequency encoding when category frequency itself is informative. Use one-hot encoding only for <15 categories. Use hashing for very high cardinality in online settings.",
        "common_pitfall": "Target encoding without regularisation overfits on rare categories. Use smoothing (blend category mean with global mean) or compute it within cross-validation folds to avoid leakage.",
        "related_questions": ["How do I encode categorical variables?", "How do I apply one-hot encoding?", "How do I select the most important features?"],
    },

    # ── Stage 5: Feature Engineering ──
    {
        "query": "How do I apply one-hot encoding?",
        "stage": 5, "difficulty": "beginner",
        "answer": "Use pd.get_dummies to convert categorical columns into binary indicator columns. Use drop_first=True to avoid the dummy variable trap.",
        "code": "df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)\nprint(df.columns.tolist())",
        "why_explanation": "One-hot encoding represents each category as an orthogonal binary vector, preventing the model from assuming an ordinal relationship between categories. drop_first removes one column per feature to avoid perfect multicollinearity in linear models.",
        "when_to_use": "Use one-hot for nominal categories with few unique values. Use get_dummies for pandas workflows. Use sklearn OneHotEncoder for pipeline integration with fit/transform API.",
        "common_pitfall": "Forgetting drop_first=True creates a multicollinearity trap for linear/logistic regression. Tree models handle redundant columns gracefully, but linear models do not.",
        "related_questions": ["How do I encode categorical variables?", "How do I create polynomial features?", "How do I apply PCA for dimensionality reduction?"],
    },
    {
        "query": "How do I create polynomial features?",
        "stage": 5, "difficulty": "intermediate",
        "answer": "Use sklearn PolynomialFeatures to generate interaction terms and polynomial terms up to a given degree.",
        "code": "from sklearn.preprocessing import PolynomialFeatures\npoly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)\nX_poly = poly.fit_transform(X_train[['Age', 'Fare']])\nprint(f'Original features: 2, Polynomial features: {X_poly.shape[1]}')",
        "why_explanation": "Polynomial features let linear models capture non-linear relationships by adding squared terms and interactions. A degree-2 polynomial of [a, b] produces [a, b, a², ab, b²], expanding the model's capacity.",
        "when_to_use": "Use polynomial features when you suspect non-linear effects and use a linear model. Use degree 2 to start — higher degrees cause combinatorial explosion. Tree models capture non-linearity natively and rarely need this.",
        "common_pitfall": "Degree 3+ on 20 features creates thousands of columns, causing severe overfitting and slow training. Always use regularisation (Ridge/Lasso) alongside polynomial expansion.",
        "related_questions": ["How do I apply one-hot encoding?", "How do I apply PCA for dimensionality reduction?", "How do I select the most important features?"],
    },
    {
        "query": "How do I apply PCA for dimensionality reduction?",
        "stage": 5, "difficulty": "intermediate",
        "answer": "Use sklearn PCA after scaling. Choose the number of components that explain 95% of variance, or use a scree plot.",
        "code": "from sklearn.decomposition import PCA\nfrom sklearn.preprocessing import StandardScaler\n\nX_scaled = StandardScaler().fit_transform(X_train)\npca = PCA(n_components=0.95)\nX_pca = pca.fit_transform(X_scaled)\nprint(f'Reduced from {X_scaled.shape[1]} to {X_pca.shape[1]} components')\nprint(f'Explained variance: {pca.explained_variance_ratio_.sum():.2%}')",
        "why_explanation": "PCA finds orthogonal directions of maximum variance. By keeping only the top components, you reduce noise and dimensionality while retaining most of the signal. This speeds up training and can prevent overfitting.",
        "when_to_use": "Use PCA when you have many correlated numeric features. Set n_components=0.95 to retain 95% variance. Don't use PCA on already-sparse one-hot encoded data — use truncated SVD instead.",
        "common_pitfall": "Applying PCA without scaling first means high-variance features dominate the components. Always StandardScaler before PCA.",
        "related_questions": ["How do I create polynomial features?", "How do I scale numeric features?", "How do I check for multicollinearity in features?"],
    },
    {
        "query": "How do I extract features from datetime columns?",
        "stage": 5, "difficulty": "intermediate",
        "answer": "Convert to datetime type, then extract year, month, day of week, hour, and derived features like is_weekend or days_since_event.",
        "code": "df['date'] = pd.to_datetime(df['date'])\ndf['year'] = df['date'].dt.year\ndf['month'] = df['date'].dt.month\ndf['day_of_week'] = df['date'].dt.dayofweek\ndf['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)\nprint(df[['date', 'year', 'month', 'day_of_week', 'is_weekend']].head())",
        "why_explanation": "Raw datetime objects are not numeric and cannot be fed into ML models. Extracting components captures cyclic patterns (seasonality, day-of-week effects) and trends (year) as numeric features the model can learn from.",
        "when_to_use": "Extract datetime features when your data has timestamps and you suspect time-based patterns. For cyclic features (month, hour), consider sine/cosine encoding to preserve continuity between December and January.",
        "common_pitfall": "Treating month as a plain integer implies 12 is closer to 11 than to 1. Use cyclical encoding (sin/cos transform) for features that wrap around.",
        "related_questions": ["How do I apply one-hot encoding?", "How do I create polynomial features?", "How do I select the most important features?"],
    },
    {
        "query": "How do I select the most important features?",
        "stage": 5, "difficulty": "intermediate",
        "answer": "Use model-based feature importance (Random Forest), mutual information, or recursive feature elimination (RFE).",
        "code": "from sklearn.ensemble import RandomForestClassifier\nimport pandas as pd\n\nmodel = RandomForestClassifier(n_estimators=100, random_state=42)\nmodel.fit(X_train, y_train)\nimportances = pd.Series(model.feature_importances_, index=X_train.columns)\nprint(importances.sort_values(ascending=False).head(10))",
        "why_explanation": "Feature importance measures how much each feature contributes to the model's predictions. Tree-based importance counts how often a feature is used for splitting and how much it reduces impurity. Removing low-importance features reduces overfitting and speeds up training.",
        "when_to_use": "Use tree-based importance for a quick ranking. Use mutual information for non-linear relationships without fitting a model. Use RFE when you want to find the optimal feature subset for a specific model.",
        "common_pitfall": "Tree-based importance is biased toward high-cardinality features. Use permutation importance for a more reliable ranking. Also, correlated features split importance between them, underestimating both.",
        "related_questions": ["How do I apply PCA for dimensionality reduction?", "How do I check for multicollinearity in features?", "How do I create polynomial features?"],
    },
    {
        "query": "How do I create interaction features between columns?",
        "stage": 5, "difficulty": "intermediate",
        "answer": "Multiply two features together, ratio them, or use PolynomialFeatures with interaction_only=True to generate all pairwise products.",
        "code": "df['Age_x_Fare'] = df['Age'] * df['Fare']\ndf['Fare_per_person'] = df['Fare'] / (df['SibSp'] + df['Parch'] + 1)\nprint(df[['Age_x_Fare', 'Fare_per_person']].head())",
        "why_explanation": "Interaction features capture relationships that individual features cannot express. For instance, Fare_per_person normalises fare by family size, which better represents individual wealth than raw Fare alone.",
        "when_to_use": "Create interactions when domain knowledge suggests features combine meaningfully. Use PolynomialFeatures(interaction_only=True) for automated exploration. Limit to 2-way interactions to avoid explosion.",
        "common_pitfall": "Blindly creating all pairwise interactions for N features produces N*(N-1)/2 new columns. Most will be noise. Use domain knowledge or feature importance to filter after creation.",
        "related_questions": ["How do I create polynomial features?", "How do I select the most important features?", "How do I apply PCA for dimensionality reduction?"],
    },

    # ── Stage 6: Modeling ──
    {
        "query": "Train a Random Forest classifier",
        "stage": 6, "difficulty": "beginner",
        "answer": "Use scikit-learn RandomForestClassifier. Split data first, then fit on training data and evaluate on test data.",
        "code": "from sklearn.ensemble import RandomForestClassifier\nfrom sklearn.model_selection import train_test_split\n\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\nmodel = RandomForestClassifier(n_estimators=100, random_state=42)\nmodel.fit(X_train, y_train)\nprint('Train accuracy:', model.score(X_train, y_train))\nprint('Test accuracy:', model.score(X_test, y_test))",
        "why_explanation": "Random Forest builds many decision trees on random subsets of data and features, then averages their predictions. This ensemble approach reduces overfitting compared to a single tree and handles non-linear relationships naturally.",
        "when_to_use": "Use Random Forest as a strong baseline for both classification and regression. It handles mixed feature types, missing values (in some implementations), and requires minimal hyperparameter tuning. Use gradient boosting (XGBoost/LightGBM) when you need higher accuracy.",
        "common_pitfall": "Evaluating only on training data gives inflated accuracy. Always use a held-out test set or cross-validation. A big gap between train and test accuracy indicates overfitting.",
        "related_questions": ["How do I tune hyperparameters with GridSearchCV?", "How do I set up cross-validation?", "How do I calculate AUC-ROC score?"],
    },
    {
        "query": "How do I set up cross-validation?",
        "stage": 6, "difficulty": "intermediate",
        "answer": "Use sklearn cross_val_score for quick evaluation. Use StratifiedKFold for classification to preserve class proportions in each fold.",
        "code": "from sklearn.model_selection import cross_val_score, StratifiedKFold\nfrom sklearn.ensemble import RandomForestClassifier\n\nmodel = RandomForestClassifier(n_estimators=100, random_state=42)\ncv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\nscores = cross_val_score(model, X, y, cv=cv, scoring='f1')\nprint(f'CV F1: {scores.mean():.3f} (+/- {scores.std():.3f})')",
        "why_explanation": "Cross-validation trains and evaluates the model on multiple different train/test splits, giving a more reliable performance estimate than a single split. Stratified CV ensures each fold has the same class ratio as the full dataset.",
        "when_to_use": "Use 5-fold CV as default. Use 10-fold for smaller datasets. Use Leave-One-Out for very small datasets (<50 samples). Use TimeSeriesSplit when data has temporal ordering.",
        "common_pitfall": "Performing feature selection or hyperparameter tuning outside the CV loop leaks information. All preprocessing should happen inside each fold using sklearn Pipelines.",
        "related_questions": ["How do I tune hyperparameters with GridSearchCV?", "Train a Random Forest classifier", "How do I calculate AUC-ROC score?"],
    },
    {
        "query": "How do I tune hyperparameters with GridSearchCV?",
        "stage": 6, "difficulty": "intermediate",
        "answer": "Define a parameter grid and use GridSearchCV to exhaustively search combinations. Use RandomizedSearchCV for larger spaces.",
        "code": "from sklearn.model_selection import GridSearchCV\nfrom sklearn.ensemble import RandomForestClassifier\n\nparam_grid = {\n    'n_estimators': [50, 100, 200],\n    'max_depth': [5, 10, None],\n    'min_samples_split': [2, 5]\n}\ngrid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='f1', n_jobs=-1)\ngrid.fit(X_train, y_train)\nprint('Best params:', grid.best_params_)\nprint('Best F1:', round(grid.best_score_, 3))",
        "why_explanation": "Grid search tries every combination in the parameter grid and picks the one with the best CV score. This automates the manual trial-and-error process. The CV inside GridSearchCV prevents overfitting to a single validation set.",
        "when_to_use": "Use GridSearchCV for small parameter spaces (<100 combinations). Use RandomizedSearchCV for large spaces. Use Bayesian optimisation (Optuna) for very expensive models.",
        "common_pitfall": "Tuning on the test set inflates your reported score. Always tune on training data with CV, then evaluate the final model on the held-out test set exactly once.",
        "related_questions": ["How do I set up cross-validation?", "Train a Random Forest classifier", "What is the difference between bagging and boosting?"],
    },
    {
        "query": "How do I handle class imbalance?",
        "stage": 6, "difficulty": "intermediate",
        "answer": "Use class_weight='balanced', SMOTE oversampling, or adjust the decision threshold. Choose based on the degree of imbalance and model type.",
        "code": "from sklearn.ensemble import RandomForestClassifier\n\nmodel = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)\nmodel.fit(X_train, y_train)\nprint('Balanced model trained')\n# Alternative: SMOTE\n# from imblearn.over_sampling import SMOTE\n# X_res, y_res = SMOTE().fit_resample(X_train, y_train)",
        "why_explanation": "Imbalanced classes cause models to favour the majority class. class_weight='balanced' inversely weights classes by frequency, making minority-class errors more costly. SMOTE creates synthetic minority samples, while threshold adjustment shifts the decision boundary.",
        "when_to_use": "Use class_weight='balanced' as a first attempt (simple, no data augmentation). Use SMOTE when you have enough minority samples to interpolate. Adjust threshold when you need precise control over precision-recall tradeoff.",
        "common_pitfall": "Applying SMOTE before train/test split creates test samples that are similar to synthetic training samples, causing over-optimistic evaluation. Always SMOTE inside the cross-validation loop.",
        "related_questions": ["How do I choose the right evaluation metric?", "How do I set up cross-validation?", "How do I calculate AUC-ROC score?"],
    },
    {
        "query": "What is the difference between bagging and boosting?",
        "stage": 6, "difficulty": "advanced",
        "answer": "Bagging trains independent models on random subsets and averages them (reduces variance). Boosting trains sequentially, each model correcting the previous one's errors (reduces bias).",
        "code": "from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier\n\nbag = BaggingClassifier(n_estimators=100, random_state=42)\nbag.fit(X_train, y_train)\nprint('Bagging accuracy:', bag.score(X_test, y_test))\n\nboost = GradientBoostingClassifier(n_estimators=100, random_state=42)\nboost.fit(X_train, y_train)\nprint('Boosting accuracy:', boost.score(X_test, y_test))",
        "why_explanation": "Bagging reduces variance by averaging noisy models (Random Forest is bagging + feature randomisation). Boosting reduces bias by learning from residual errors sequentially. Boosting typically achieves higher accuracy but is more prone to overfitting.",
        "when_to_use": "Use bagging (Random Forest) when you want robustness and low tuning effort. Use boosting (XGBoost, LightGBM) when you need maximum predictive accuracy and are willing to tune carefully.",
        "common_pitfall": "Boosting with too many iterations or too high a learning rate overfits badly. Use early stopping with a validation set to find the optimal number of boosting rounds.",
        "related_questions": ["Train a Random Forest classifier", "How do I tune hyperparameters with GridSearchCV?", "How do I calculate AUC-ROC score?"],
    },
    {
        "query": "How do I train a Logistic Regression model?",
        "stage": 6, "difficulty": "beginner",
        "answer": "Use sklearn LogisticRegression. Scale features first since logistic regression is sensitive to feature scale.",
        "code": "from sklearn.linear_model import LogisticRegression\nfrom sklearn.preprocessing import StandardScaler\n\nscaler = StandardScaler()\nX_train_s = scaler.fit_transform(X_train)\nX_test_s = scaler.transform(X_test)\n\nmodel = LogisticRegression(max_iter=1000, random_state=42)\nmodel.fit(X_train_s, y_train)\nprint('Test accuracy:', model.score(X_test_s, y_test))",
        "why_explanation": "Logistic regression models the log-odds of the positive class as a linear function of features. Despite its simplicity, it is highly interpretable — each coefficient shows the feature's contribution to the prediction.",
        "when_to_use": "Use logistic regression as a simple interpretable baseline for binary classification. It works well when features are roughly linearly separable. For non-linear boundaries, use kernel SVM, tree models, or add polynomial features.",
        "common_pitfall": "Not scaling features causes convergence issues (ConvergenceWarning). Always scale inputs. Also increase max_iter if the solver does not converge at the default 100 iterations.",
        "related_questions": ["Train a Random Forest classifier", "How do I scale numeric features?", "How do I set up cross-validation?"],
    },

    # ── Stage 7: Evaluation ──
    {
        "query": "How do I calculate AUC-ROC score?",
        "stage": 7, "difficulty": "beginner",
        "answer": "Use roc_auc_score from sklearn.metrics. Pass predicted probabilities, not hard class labels.",
        "code": "from sklearn.metrics import roc_auc_score, roc_curve\nimport matplotlib.pyplot as plt\n\npreds = model.predict_proba(X_test)[:, 1]\nauc = roc_auc_score(y_test, preds)\nprint(f'AUC-ROC: {auc:.3f}')\n\nfpr, tpr, _ = roc_curve(y_test, preds)\nplt.plot(fpr, tpr, label=f'AUC={auc:.2f}')\nplt.plot([0,1],[0,1],'--', color='gray')\nplt.xlabel('FPR'); plt.ylabel('TPR')\nplt.legend(); plt.show()",
        "why_explanation": "AUC-ROC measures how well the model ranks positive samples above negative samples across all decision thresholds. An AUC of 0.5 is random guessing; 1.0 is perfect separation. It is threshold-independent, making it ideal for comparing models.",
        "when_to_use": "Use AUC-ROC when ranking quality matters and classes are roughly balanced. Use AUC-PR (precision-recall) when the positive class is rare (<5%). Use F1 when you need a single threshold-specific metric.",
        "common_pitfall": "Passing hard labels (0/1 from model.predict) instead of probabilities to roc_auc_score gives a misleading score. Always use predict_proba for AUC.",
        "related_questions": ["How do I plot a confusion matrix?", "How do I compute precision, recall, and F1 score?", "How do I plot a learning curve?"],
    },
    {
        "query": "How do I plot a confusion matrix?",
        "stage": 7, "difficulty": "beginner",
        "answer": "Use sklearn ConfusionMatrixDisplay or compute the matrix with confusion_matrix and plot with seaborn heatmap.",
        "code": "from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay\nimport matplotlib.pyplot as plt\n\ncm = confusion_matrix(y_test, model.predict(X_test))\ndisp = ConfusionMatrixDisplay(cm, display_labels=['Died', 'Survived'])\ndisp.plot(cmap='Blues')\nplt.title('Confusion Matrix')\nplt.show()",
        "why_explanation": "The confusion matrix shows true positives, false positives, true negatives, and false negatives. It reveals whether the model's errors are skewed toward one class, which aggregate metrics like accuracy can hide.",
        "when_to_use": "Always plot a confusion matrix alongside aggregate metrics. It is especially important for imbalanced datasets where accuracy can be misleading. Use normalised version (normalize='true') to compare across datasets.",
        "common_pitfall": "Reading the matrix backwards — rows are true labels, columns are predictions (in sklearn default). Misinterpreting which axis is which leads to wrong conclusions about error types.",
        "related_questions": ["How do I calculate AUC-ROC score?", "How do I compute precision, recall, and F1 score?", "How do I handle class imbalance?"],
    },
    {
        "query": "How do I compute precision, recall, and F1 score?",
        "stage": 7, "difficulty": "beginner",
        "answer": "Use sklearn classification_report for a full summary, or individual functions precision_score, recall_score, f1_score.",
        "code": "from sklearn.metrics import classification_report\nprint(classification_report(y_test, model.predict(X_test)))",
        "why_explanation": "Precision = what fraction of positive predictions are correct. Recall = what fraction of actual positives are found. F1 = harmonic mean of both. These metrics matter when the cost of false positives differs from false negatives.",
        "when_to_use": "Use precision when false positives are costly (spam filter). Use recall when false negatives are costly (disease screening). Use F1 when you need a single metric that balances both.",
        "common_pitfall": "Using 'micro' averaging on imbalanced data gives the same result as accuracy, hiding poor performance on the minority class. Use 'macro' or 'weighted' averaging to surface per-class issues.",
        "related_questions": ["How do I plot a confusion matrix?", "How do I calculate AUC-ROC score?", "How do I handle class imbalance?"],
    },
    {
        "query": "How do I plot a learning curve?",
        "stage": 7, "difficulty": "intermediate",
        "answer": "Use sklearn learning_curve to plot training and validation scores as a function of training set size. This diagnoses overfitting vs underfitting.",
        "code": "from sklearn.model_selection import learning_curve\nimport matplotlib.pyplot as plt\nimport numpy as np\n\ntrain_sizes, train_scores, val_scores = learning_curve(\n    model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10), scoring='f1')\n\nplt.plot(train_sizes, train_scores.mean(axis=1), label='Train')\nplt.plot(train_sizes, val_scores.mean(axis=1), label='Validation')\nplt.xlabel('Training Set Size')\nplt.ylabel('F1 Score')\nplt.legend(); plt.title('Learning Curve'); plt.show()",
        "why_explanation": "If both curves converge at a low score, the model underfits (needs more capacity). If training score is high but validation is low, the model overfits (needs more data or regularisation). The gap between curves shows generalisation error.",
        "when_to_use": "Plot learning curves after initial modelling to decide next steps. If underfitting, try a more complex model or better features. If overfitting, try regularisation, pruning, or more training data.",
        "common_pitfall": "Interpreting a small gap as good performance when both curves are at 0.55. A small gap means no overfitting, but the model may still be underfitting if absolute performance is low.",
        "related_questions": ["How do I calculate AUC-ROC score?", "How do I tune hyperparameters with GridSearchCV?", "What is overfitting and how do I detect it?"],
    },
    {
        "query": "How do I interpret SHAP values for a model?",
        "stage": 7, "difficulty": "advanced",
        "answer": "SHAP (SHapley Additive exPlanations) assigns each feature a contribution to a specific prediction. Use shap.TreeExplainer for tree models.",
        "code": "import shap\nexplainer = shap.TreeExplainer(model)\nshap_values = explainer.shap_values(X_test)\nshap.summary_plot(shap_values[1], X_test)",
        "why_explanation": "SHAP values are based on cooperative game theory — they fairly distribute the prediction among features. They satisfy consistency (if a feature contributes more, its SHAP value increases) and local accuracy (SHAP values sum to the prediction minus the base value).",
        "when_to_use": "Use SHAP for post-hoc model explanation, especially for black-box models. Use TreeExplainer for tree models (fast), KernelExplainer for any model (slow). Use LIME as a faster but less theoretically grounded alternative.",
        "common_pitfall": "SHAP values explain a specific model's behaviour, not causal relationships in the data. A high SHAP value for a feature does not mean changing that feature will change the outcome.",
        "related_questions": ["How do I plot a learning curve?", "How do I calculate AUC-ROC score?", "How do I select the most important features?"],
    },
    {
        "query": "What is overfitting and how do I detect it?",
        "stage": 7, "difficulty": "beginner",
        "answer": "Overfitting means the model memorises training data instead of learning general patterns. Detect it by comparing train vs test performance — a large gap indicates overfitting.",
        "code": "train_score = model.score(X_train, y_train)\ntest_score = model.score(X_test, y_test)\nprint(f'Train: {train_score:.3f}')\nprint(f'Test:  {test_score:.3f}')\nprint(f'Gap:   {train_score - test_score:.3f}')\nif train_score - test_score > 0.1:\n    print('Warning: possible overfitting')",
        "why_explanation": "An overfit model has learned noise in the training data that does not generalise. It performs well on training data but poorly on unseen data. Regularisation, cross-validation, and more data are the standard remedies.",
        "when_to_use": "Always check train-test gap after fitting. If gap > 10%, try: reducing model complexity, adding regularisation, using dropout (neural nets), or getting more data. If both scores are low, the model underfits.",
        "common_pitfall": "Tuning hyperparameters on the test set until test accuracy is high is itself a form of overfitting to the test set. Use a separate validation set or nested cross-validation.",
        "related_questions": ["How do I plot a learning curve?", "How do I set up cross-validation?", "How do I tune hyperparameters with GridSearchCV?"],
    },
]

# ---------------------------------------------------------------------------
# TEMPLATE BANKS — per-stage templates for generating additional QA pairs
# ---------------------------------------------------------------------------

STAGE_META = {
    1: {
        "name": "Problem Understanding",
        "queries": [
            "What is the objective of {var} prediction?",
            "How do I frame the problem for {var} analysis?",
            "Define the target variable for {var}.",
            "What baseline should I use for {var}?",
        ],
        "answer_t": "For {var}, define the objective clearly, choose a target variable, select an appropriate metric, and establish a simple baseline before building complex models.",
        "code_t": "project = {{'task': '{var}', 'type': 'classification'}}\nprint(project)",
        "why_t": "Defining the problem before modelling ensures you optimise the right objective. A clear problem statement guides feature selection, model choice, and evaluation criteria.",
        "when_t": "Always start with problem framing regardless of the task. Use classification framing for categorical targets and regression for continuous targets.",
        "pitfall_t": "Jumping to modelling without defining what success looks like for {var} wastes effort on poorly targeted solutions.",
        "difficulties": ["beginner", "beginner", "beginner", "intermediate"],
    },
    2: {
        "name": "Data Loading",
        "queries": [
            "Load the {var} dataset into pandas.",
            "How do I read a {var} data file?",
            "Check the shape and dtypes of {var} data.",
            "Inspect the first rows of {var} data.",
        ],
        "answer_t": "Use pandas to load {var} data with the appropriate reader function. Always verify shape, dtypes, and check for obvious data quality issues immediately after loading.",
        "code_t": "import pandas as pd\ndf = pd.read_csv('{var}_data.csv')\nprint(df.shape)\nprint(df.head())",
        "why_t": "Loading data correctly is the foundation — wrong delimiters, encodings, or type inference can silently corrupt your analysis downstream.",
        "when_t": "Use read_csv for CSV/TSV. Use read_excel for spreadsheets. Use read_json for JSON APIs. Always check dtypes after loading to catch misdetected types.",
        "pitfall_t": "Not verifying column types after loading the {var} dataset can lead to silent errors when numeric columns are read as strings.",
        "difficulties": ["beginner", "beginner", "beginner", "beginner"],
    },
    3: {
        "name": "Exploratory Data Analysis",
        "queries": [
            "Show the distribution of {var}.",
            "Check for outliers in {var}.",
            "Plot the missing pattern for {var} data.",
            "Analyse the relationship between {var} and the target.",
        ],
        "answer_t": "Explore {var} by visualising its distribution, checking for outliers and missing values, and measuring its relationship with the target variable.",
        "code_t": "import matplotlib.pyplot as plt\nimport seaborn as sns\nsns.histplot(df['{var}'], kde=True)\nplt.title('{var} Distribution')\nplt.show()",
        "why_t": "EDA reveals the data's structure, anomalies, and relationships before modelling. Skipping it leads to models built on flawed assumptions about the data distribution.",
        "when_t": "Always perform EDA before preprocessing. Use histograms for distributions, boxplots for outlier detection, and heatmaps for correlation analysis.",
        "pitfall_t": "Skipping EDA for {var} means you might miss outliers or distribution issues that invalidate your chosen preprocessing and modelling approach.",
        "difficulties": ["beginner", "intermediate", "intermediate", "intermediate"],
    },
    4: {
        "name": "Preprocessing",
        "queries": [
            "Impute missing values in {var}.",
            "Scale the {var} feature appropriately.",
            "Remove or cap outliers in {var}.",
            "Clean the {var} column for modelling.",
        ],
        "answer_t": "Preprocess {var} by handling missing values, scaling numeric features, and treating outliers. Always fit transformers on training data only.",
        "code_t": "from sklearn.impute import SimpleImputer\nimp = SimpleImputer(strategy='median')\ndf[['{var}']] = imp.fit_transform(df[['{var}']])",
        "why_t": "Preprocessing transforms raw data into a format suitable for ML algorithms. Unscaled or missing data causes convergence issues, biased predictions, or outright errors in many algorithms.",
        "when_t": "Preprocess after EDA and before feature engineering. Use median imputation for skewed data, mean for normal data, and mode for categorical columns.",
        "pitfall_t": "Fitting the preprocessor on the entire dataset including test data before splitting causes data leakage in the {var} column.",
        "difficulties": ["beginner", "beginner", "intermediate", "beginner"],
    },
    5: {
        "name": "Feature Engineering",
        "queries": [
            "Create polynomial features from {var}.",
            "Extract useful features from {var}.",
            "Apply dimensionality reduction to {var} features.",
            "Engineer interaction features with {var}.",
        ],
        "answer_t": "Engineer features from {var} by creating polynomial terms, extracting domain-specific attributes, or reducing dimensionality to capture the most informative signal.",
        "code_t": "df['{var}_squared'] = df['{var}'] ** 2\nprint(df[['{var}', '{var}_squared']].head())",
        "why_t": "Feature engineering amplifies the signal in your data. Good features let even simple models achieve strong performance, while poor features limit even the most complex models.",
        "when_t": "Engineer features after preprocessing and before modelling. Start with domain-knowledge features, then try automated approaches like polynomial expansion or PCA.",
        "pitfall_t": "Creating too many features from {var} without validation can lead to overfitting and slow training. Always evaluate feature importance after engineering.",
        "difficulties": ["intermediate", "intermediate", "advanced", "intermediate"],
    },
    6: {
        "name": "Modeling",
        "queries": [
            "Train a model for {var} prediction.",
            "Tune hyperparameters for {var} model.",
            "Set up cross-validation for {var}.",
            "Compare multiple models for {var}.",
        ],
        "answer_t": "Train a model for {var} using scikit-learn. Start with a simple baseline, then try ensemble methods. Use cross-validation for reliable evaluation and grid search for tuning.",
        "code_t": "from sklearn.ensemble import RandomForestClassifier\nmodel = RandomForestClassifier(n_estimators=100, random_state=42)\nmodel.fit(X_train, y_train)\nprint('Accuracy:', model.score(X_test, y_test))",
        "why_t": "Model selection involves trading off complexity against generalisation. Ensemble methods like Random Forest reduce variance by averaging many weak learners, while boosting reduces bias by learning sequentially.",
        "when_t": "Start with logistic regression or Random Forest as baselines. Try gradient boosting for higher accuracy. Use neural networks only when data is large and features are unstructured (images, text).",
        "pitfall_t": "Evaluating the {var} model only on training data gives an overoptimistic estimate. Always use held-out test data or cross-validation.",
        "difficulties": ["beginner", "intermediate", "intermediate", "advanced"],
    },
    7: {
        "name": "Evaluation",
        "queries": [
            "Evaluate the {var} model performance.",
            "Plot the learning curve for {var}.",
            "Calculate the evaluation metrics for {var}.",
            "Interpret the results of {var} predictions.",
        ],
        "answer_t": "Evaluate the {var} model using appropriate metrics (accuracy, F1, AUC-ROC). Plot learning curves to diagnose overfitting/underfitting. Use confusion matrix for error analysis.",
        "code_t": "from sklearn.metrics import classification_report\ny_pred = model.predict(X_test)\nprint(classification_report(y_test, y_pred))",
        "why_t": "Evaluation reveals whether the model generalises or memorises. Multiple metrics provide different perspectives — accuracy alone can be misleading on imbalanced datasets.",
        "when_t": "Evaluate after training. Use AUC for ranking quality, F1 for balanced precision/recall, and learning curves for diagnosing capacity issues. Report confidence intervals when possible.",
        "pitfall_t": "Using only accuracy to evaluate the {var} model on imbalanced data gives a misleading picture of true performance. Check per-class metrics.",
        "difficulties": ["beginner", "intermediate", "beginner", "advanced"],
    },
}

DOMAIN_VARS = [
    "Titanic", "House Prices", "Sales", "Churn", "Credit Risk",
    "Fraud", "Customer Segmentation", "Recommendation",
    "Text Classification", "Time Series",
    "Medical Diagnosis", "Stock Price", "Energy Consumption",
    "Sentiment", "Anomaly Detection",
]

# ---------------------------------------------------------------------------
# BUILDER
# ---------------------------------------------------------------------------

def _make_related(query, stage, all_queries_by_stage):
    """Build a 3-item related_questions list: lateral, deeper, connected."""
    related = []
    same_stage = [q for q in all_queries_by_stage.get(stage, []) if q != query]
    if same_stage:
        related.append(same_stage[hash(query) % len(same_stage)])
    if len(same_stage) > 1:
        related.append(same_stage[(hash(query) + 7) % len(same_stage)])
    next_stage = stage + 1 if stage < 7 else 1
    next_qs = all_queries_by_stage.get(next_stage, [])
    if next_qs:
        related.append(next_qs[hash(query) % len(next_qs)])
    while len(related) < 3:
        prev_stage = stage - 1 if stage > 1 else 7
        prev_qs = all_queries_by_stage.get(prev_stage, [])
        if prev_qs:
            related.append(prev_qs[len(related) % len(prev_qs)])
        else:
            related.append(query)
    return related[:3]


def build_dataset():
    data = []

    for entry in SEED:
        row = dict(entry)
        row["related_questions"] = json.dumps(entry["related_questions"], ensure_ascii=True)
        row["stage"] = int(row["stage"])
        data.append(row)

    target_total = 210
    idx = 0
    while len(data) < target_total:
        stage = (idx % 7) + 1
        meta = STAGE_META[stage]
        q_idx = idx % len(meta["queries"])
        v_idx = idx % len(DOMAIN_VARS)
        var = DOMAIN_VARS[v_idx]

        query = meta["queries"][q_idx].replace("{var}", var)

        if any(d["query"] == query for d in data):
            idx += 1
            continue

        diff = meta["difficulties"][q_idx % len(meta["difficulties"])]
        data.append({
            "query": query,
            "stage": stage,
            "answer": meta["answer_t"].replace("{var}", var),
            "code": meta["code_t"].replace("{var}", var),
            "why_explanation": meta["why_t"].replace("{var}", var),
            "when_to_use": meta["when_t"].replace("{var}", var),
            "common_pitfall": meta["pitfall_t"].replace("{var}", var),
            "related_questions": "[]",
            "difficulty": diff,
        })
        idx += 1

    all_queries_by_stage = {}
    for d in data:
        all_queries_by_stage.setdefault(int(d["stage"]), []).append(d["query"])

    for d in data:
        if d["related_questions"] == "[]":
            rqs = _make_related(d["query"], int(d["stage"]), all_queries_by_stage)
            d["related_questions"] = json.dumps(rqs, ensure_ascii=True)

    _validate(data)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dataset.csv")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

    print(f"Generated {len(data)} QA pairs at {out_path}")

    from collections import Counter
    stage_counts = Counter(int(d["stage"]) for d in data)
    for s in sorted(stage_counts):
        print(f"  Stage {s}: {stage_counts[s]} entries")


def _validate(data):
    from collections import Counter
    assert len(data) >= 200, f"Only {len(data)} entries (need 200+)"

    stage_counts = Counter(int(d["stage"]) for d in data)
    for s in range(1, 8):
        assert stage_counts.get(s, 0) >= 20, f"Stage {s} has only {stage_counts.get(s, 0)} entries (need 20+)"

    for i, d in enumerate(data):
        for field in FIELDS:
            val = d.get(field, "")
            assert val not in (None, ""), f"Row {i} field '{field}' is empty (query: {d.get('query', '?')[:40]})"

        assert d["difficulty"] in ("beginner", "intermediate", "advanced"), \
            f"Row {i} invalid difficulty: {d['difficulty']}"

        assert 1 <= int(d["stage"]) <= 7, f"Row {i} invalid stage: {d['stage']}"

        rq = json.loads(d["related_questions"])
        assert isinstance(rq, list) and len(rq) == 3, \
            f"Row {i} related_questions must be list of 3, got {rq}"

    print("Validation passed.")


if __name__ == "__main__":
    build_dataset()
