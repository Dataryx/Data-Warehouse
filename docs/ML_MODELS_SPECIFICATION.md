# Machine Learning Models Specification

## AI-Powered Self-Optimizing Data Warehouse
### Detailed ML Models Implementation Guide

---

## 📋 OVERVIEW

This document provides detailed specifications for each machine learning model in the project. Use this as a reference when implementing each model.

---

## MODEL 1: QUERY COST PREDICTOR 🎯

### Purpose
Predict the execution time (cost) of SQL queries before they are executed, enabling proactive optimization decisions.

### Problem Type
**Regression** - Predicting a continuous value (execution time in milliseconds)

### Input Features

#### Query Structural Features
- `num_tables`: Number of tables referenced (FROM + JOINs)
- `num_joins`: Number of JOIN operations
- `num_where_conditions`: Number of conditions in WHERE clause
- `num_and_conditions`: Number of AND operators
- `num_or_conditions`: Number of OR operators
- `has_group_by`: Binary (0/1) - presence of GROUP BY
- `has_order_by`: Binary (0/1) - presence of ORDER BY
- `has_distinct`: Binary (0/1) - presence of DISTINCT
- `has_limit`: Binary (0/1) - presence of LIMIT
- `limit_value`: Value of LIMIT (0 if no LIMIT)
- `query_length`: Character count of query
- `subquery_depth`: Maximum nesting level of subqueries

#### Aggregation Features
- `has_count`: Binary (0/1) - presence of COUNT()
- `has_sum`: Binary (0/1) - presence of SUM()
- `has_avg`: Binary (0/1) - presence of AVG()
- `has_max`: Binary (0/1) - presence of MAX()
- `has_min`: Binary (0/1) - presence of MIN()
- `total_aggregations`: Total number of aggregation functions

#### Join Features
- `has_inner_join`: Binary (0/1)
- `has_left_join`: Binary (0/1)
- `has_right_join`: Binary (0/1)
- `has_outer_join`: Binary (0/1)

#### Data Volume Indicators
- `estimated_scan_rows`: Estimated rows to scan (from table statistics)
- `max_table_size`: Largest table size in query (rows)
- `total_table_rows`: Sum of all table sizes

#### Temporal Features
- `hour_of_day`: Hour when query executed (0-23)
- `day_of_week`: Day of week (0-6)
- `is_business_hours`: Binary (9 AM - 5 PM = 1, else 0)
- `is_weekend`: Binary (0/1)

#### System Load Features
- `concurrent_queries`: Number of queries running simultaneously
- `system_cpu_usage`: CPU usage percentage at query start
- `system_memory_usage`: Memory usage percentage at query start

### Output
- `execution_time_ms`: Actual execution time in milliseconds (target variable)

### Feature Engineering Code Example

```python
import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Where
from sqlparse.tokens import Keyword, DML

def extract_query_features(query_text, system_metrics=None):
    """
    Extract features from SQL query for cost prediction.
    
    Args:
        query_text (str): SQL query string
        system_metrics (dict): Optional system metrics at query time
        
    Returns:
        dict: Feature dictionary
    """
    query_upper = query_text.upper()
    
    features = {
        # Structural features
        'num_tables': query_text.count('FROM') + query_text.count('JOIN'),
        'num_joins': query_upper.count('JOIN'),
        'num_where_conditions': query_text.count('WHERE'),
        'num_and_conditions': query_upper.count(' AND '),
        'num_or_conditions': query_upper.count(' OR '),
        'has_group_by': 1 if 'GROUP BY' in query_upper else 0,
        'has_order_by': 1 if 'ORDER BY' in query_upper else 0,
        'has_distinct': 1 if 'DISTINCT' in query_upper else 0,
        'has_limit': 1 if 'LIMIT' in query_upper else 0,
        'query_length': len(query_text),
        
        # Aggregation features
        'has_count': 1 if 'COUNT(' in query_upper else 0,
        'has_sum': 1 if 'SUM(' in query_upper else 0,
        'has_avg': 1 if 'AVG(' in query_upper else 0,
        'has_max': 1 if 'MAX(' in query_upper else 0,
        'has_min': 1 if 'MIN(' in query_upper else 0,
        
        # Join types
        'has_inner_join': 1 if 'INNER JOIN' in query_upper else 0,
        'has_left_join': 1 if 'LEFT JOIN' in query_upper else 0,
        'has_right_join': 1 if 'RIGHT JOIN' in query_upper else 0,
        'has_outer_join': 1 if 'OUTER JOIN' in query_upper else 0,
    }
    
    # Parse LIMIT value
    if features['has_limit']:
        import re
        limit_match = re.search(r'LIMIT\s+(\d+)', query_upper)
        features['limit_value'] = int(limit_match.group(1)) if limit_match else 100
    else:
        features['limit_value'] = 0
    
    # Calculate subquery depth
    features['subquery_depth'] = query_text.count('(SELECT')
    
    # Total aggregations
    features['total_aggregations'] = sum([
        features['has_count'],
        features['has_sum'],
        features['has_avg'],
        features['has_max'],
        features['has_min']
    ])
    
    # Add system metrics if provided
    if system_metrics:
        features.update({
            'concurrent_queries': system_metrics.get('concurrent_queries', 0),
            'system_cpu_usage': system_metrics.get('cpu_usage', 0),
            'system_memory_usage': system_metrics.get('memory_usage', 0)
        })
    
    return features
```

### Model Training Code Example

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

# Load data
df = pd.read_sql("SELECT * FROM metadata.query_logs WHERE status='success'", conn)

# Extract features
features_list = df['query_text'].apply(extract_query_features).tolist()
X = pd.DataFrame(features_list)
y = df['execution_time_ms']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features (optional but recommended)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Best model
best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# Evaluate
y_pred = best_model.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f} ms")
print(f"MAE: {mae:.2f} ms")
print(f"R² Score: {r2:.4f}")

# Save model and scaler
joblib.dump(best_model, 'query_cost_predictor_v1.pkl')
joblib.dump(scaler, 'query_cost_scaler.pkl')
joblib.dump(X.columns.tolist(), 'query_cost_features.pkl')
```

### Success Criteria
- **R² Score**: > 0.75 (explains 75%+ of variance)
- **MAE**: < 20% of average query time
- **RMSE**: < 30% of average query time

### Usage in System
```python
# Load model
model = joblib.load('query_cost_predictor_v1.pkl')
scaler = joblib.load('query_cost_scaler.pkl')
feature_names = joblib.load('query_cost_features.pkl')

# Predict
query = "SELECT * FROM sales WHERE region='North'"
features = extract_query_features(query)
features_df = pd.DataFrame([features])[feature_names]
features_scaled = scaler.transform(features_df)
predicted_time = model.predict(features_scaled)[0]

print(f"Predicted execution time: {predicted_time:.2f} ms")
```

---

## MODEL 2: WORKLOAD FORECASTER 📈

### Purpose
Predict future query workload (queries per time window) to enable proactive resource allocation.

### Problem Type
**Time Series Forecasting** - Predicting future values based on historical patterns

### Input Data
- Historical query counts aggregated by time window (e.g., 5-minute intervals)
- Time features: hour, day of week, is_holiday, etc.

### Time Series Preparation

```python
import pandas as pd
from datetime import datetime, timedelta

# Load query logs
df = pd.read_sql("""
    SELECT 
        timestamp,
        COUNT(*) as query_count
    FROM metadata.query_logs
    GROUP BY DATE_TRUNC('minute', timestamp)
    ORDER BY timestamp
""", conn)

# Ensure continuous time series
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Resample to 5-minute intervals (fill missing with 0)
df_resampled = df.resample('5T').sum().fillna(0)

# Add time features
df_resampled['hour'] = df_resampled.index.hour
df_resampled['day_of_week'] = df_resampled.index.dayofweek
df_resampled['is_weekend'] = df_resampled['day_of_week'].isin([5, 6]).astype(int)
df_resampled['is_business_hours'] = df_resampled['hour'].between(9, 17).astype(int)
```

### Model Implementation with Prophet

```python
from prophet import Prophet
import pandas as pd

# Prepare data for Prophet (requires 'ds' and 'y' columns)
prophet_df = pd.DataFrame({
    'ds': df_resampled.index,
    'y': df_resampled['query_count']
})

# Create and configure model
model = Prophet(
    changepoint_prior_scale=0.05,  # Flexibility of trend
    seasonality_prior_scale=10,     # Strength of seasonality
    seasonality_mode='multiplicative',
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=False  # Not enough data for yearly
)

# Add custom seasonality (business hours)
model.add_seasonality(
    name='business_hours',
    period=1,  # Daily
    fourier_order=5
)

# Fit model
model.fit(prophet_df)

# Make future predictions (next 6 hours = 72 five-minute intervals)
future = model.make_future_dataframe(periods=72, freq='5T')
forecast = model.predict(future)

# Extract predictions
predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(72)
print(predictions.head())

# Visualize
fig = model.plot(forecast)
fig2 = model.plot_components(forecast)

# Save model
import joblib
joblib.dump(model, 'workload_forecaster_v1.pkl')
```

### Evaluation Metrics

```python
from sklearn.metrics import mean_absolute_percentage_error

# Split into train/test
train_size = int(len(prophet_df) * 0.8)
train_df = prophet_df[:train_size]
test_df = prophet_df[train_size:]

# Train on training data
model = Prophet()
model.fit(train_df)

# Predict on test period
future_test = test_df[['ds']]
forecast_test = model.predict(future_test)

# Calculate MAPE
y_true = test_df['y'].values
y_pred = forecast_test['yhat'].values
mape = mean_absolute_percentage_error(y_true, y_pred) * 100

print(f"MAPE: {mape:.2f}%")
```

### Success Criteria
- **MAPE**: < 15%
- **Peak Hour Detection**: Correctly identify peak hours 80%+ of the time
- **Direction Accuracy**: Correctly predict if load is increasing/decreasing

### Usage in System

```python
# Load model
model = joblib.load('workload_forecaster_v1.pkl')

# Predict next 6 hours
future = model.make_future_dataframe(periods=72, freq='5T')
forecast = model.predict(future)

# Get predictions
next_6_hours = forecast.tail(72)
predicted_load = next_6_hours['yhat'].tolist()

# Determine if peak is coming
current_load = get_current_load()
max_predicted = max(predicted_load)

if max_predicted > current_load * 1.5:
    print("⚠️ High load predicted - prepare resources!")
```

---

## MODEL 3: INDEX RECOMMENDER 🔍

### Purpose
Recommend database indexes based on query patterns to improve performance.

### Problem Type
**Multi-Label Classification / Ranking** - Recommend which columns should be indexed

### Approach: Rule-Based with Scoring (Recommended for Graduate Project)

```python
import pandas as pd
from collections import defaultdict

class IndexRecommender:
    def __init__(self, threshold_score=10):
        self.threshold_score = threshold_score
        self.column_scores = defaultdict(lambda: {
            'where_count': 0,
            'join_count': 0,
            'order_count': 0,
            'group_count': 0,
            'total_queries': 0
        })
    
    def analyze_query_logs(self, conn):
        """Analyze query logs to build recommendation scores"""
        
        # Get all queries
        queries = pd.read_sql("""
            SELECT query_text, execution_time_ms
            FROM metadata.query_logs
            WHERE status = 'success'
        """, conn)
        
        for _, row in queries.iterrows():
            query = row['query_text'].upper()
            
            # Extract columns from WHERE clauses
            where_columns = self._extract_where_columns(query)
            for col in where_columns:
                self.column_scores[col]['where_count'] += 1
                self.column_scores[col]['total_queries'] += 1
            
            # Extract JOIN columns
            join_columns = self._extract_join_columns(query)
            for col in join_columns:
                self.column_scores[col]['join_count'] += 1
                self.column_scores[col]['total_queries'] += 1
            
            # Extract ORDER BY columns
            order_columns = self._extract_order_columns(query)
            for col in order_columns:
                self.column_scores[col]['order_count'] += 1
            
            # Extract GROUP BY columns
            group_columns = self._extract_group_columns(query)
            for col in group_columns:
                self.column_scores[col]['group_count'] += 1
    
    def _extract_where_columns(self, query):
        """Extract column names from WHERE clause"""
        import re
        columns = []
        
        # Find WHERE clause
        where_match = re.search(r'WHERE\s+(.+?)(?:GROUP BY|ORDER BY|LIMIT|$)', query, re.IGNORECASE)
        if where_match:
            where_clause = where_match.group(1)
            # Extract column names (simplified - assumes pattern "column = value")
            col_matches = re.findall(r'(\w+\.\w+|\w+)\s*[=<>]', where_clause)
            columns.extend(col_matches)
        
        return columns
    
    def _extract_join_columns(self, query):
        """Extract columns from JOIN conditions"""
        import re
        columns = []
        
        # Find JOIN ... ON patterns
        join_matches = re.findall(r'ON\s+(\w+\.\w+|\w+)\s*=\s*(\w+\.\w+|\w+)', query, re.IGNORECASE)
        for match in join_matches:
            columns.extend(match)
        
        return columns
    
    def _extract_order_columns(self, query):
        """Extract columns from ORDER BY"""
        import re
        columns = []
        
        order_match = re.search(r'ORDER BY\s+(.+?)(?:LIMIT|$)', query, re.IGNORECASE)
        if order_match:
            order_clause = order_match.group(1)
            col_matches = re.findall(r'(\w+\.\w+|\w+)', order_clause)
            columns.extend(col_matches)
        
        return columns
    
    def _extract_group_columns(self, query):
        """Extract columns from GROUP BY"""
        import re
        columns = []
        
        group_match = re.search(r'GROUP BY\s+(.+?)(?:HAVING|ORDER BY|LIMIT|$)', query, re.IGNORECASE)
        if group_match:
            group_clause = group_match.group(1)
            col_matches = re.findall(r'(\w+\.\w+|\w+)', group_clause)
            columns.extend(col_matches)
        
        return columns
    
    def calculate_scores(self):
        """Calculate recommendation score for each column"""
        recommendations = []
        
        for column, stats in self.column_scores.items():
            # Weighted scoring
            score = (
                stats['where_count'] * 3 +    # WHERE is most important
                stats['join_count'] * 4 +      # JOINs benefit most from indexes
                stats['order_count'] * 2 +     # ORDER BY benefits
                stats['group_count'] * 2       # GROUP BY benefits
            )
            
            if score >= self.threshold_score:
                recommendations.append({
                    'column': column,
                    'score': score,
                    'where_usage': stats['where_count'],
                    'join_usage': stats['join_count'],
                    'order_usage': stats['order_count'],
                    'group_usage': stats['group_count'],
                    'query_frequency': stats['total_queries']
                })
        
        # Sort by score
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return recommendations
    
    def recommend(self, top_k=10):
        """Get top K index recommendations"""
        recommendations = self.calculate_scores()
        return recommendations[:top_k]

# Usage
recommender = IndexRecommender(threshold_score=10)
recommender.analyze_query_logs(conn)
recommendations = recommender.recommend(top_k=10)

for rec in recommendations:
    print(f"Recommend index on: {rec['column']} (Score: {rec['score']})")
    print(f"  Used in WHERE: {rec['where_usage']} times")
    print(f"  Used in JOIN: {rec['join_usage']} times")
    print()
```

### Success Criteria
- **Precision@5**: > 0.70 (70% of top 5 recommendations are beneficial)
- **Performance Improvement**: Average 20%+ speedup for affected queries
- **False Positives**: < 30% (recommendations that don't help)

### Validation Method
1. Apply recommended indexes
2. Re-run affected queries
3. Measure performance improvement
4. Calculate precision based on improvements

---

## MODEL 4: QUERY CLUSTERING 🎨

### Purpose
Group similar queries together for pattern analysis and optimization strategies.

### Problem Type
**Unsupervised Learning - Clustering**

### Implementation

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Extract features for all queries
df = pd.read_sql("SELECT * FROM metadata.query_logs", conn)
features_list = df['query_text'].apply(extract_query_features).tolist()
X = pd.DataFrame(features_list)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal K using elbow method
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    
    from sklearn.metrics import silhouette_score
    score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(score)

# Plot elbow curve
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method')

plt.subplot(1, 2, 2)
plt.plot(K_range, silhouette_scores, 'ro-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis')

plt.tight_layout()
plt.savefig('clustering_analysis.png')

# Choose optimal K (e.g., K=5)
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to dataframe
df['cluster'] = clusters

# Analyze each cluster
for cluster_id in range(optimal_k):
    cluster_queries = df[df['cluster'] == cluster_id]
    
    print(f"\n=== Cluster {cluster_id} ===")
    print(f"Number of queries: {len(cluster_queries)}")
    print(f"Avg execution time: {cluster_queries['execution_time_ms'].mean():.2f} ms")
    print(f"Sample queries:")
    print(cluster_queries['query_text'].head(3).values)
    
    # Analyze cluster characteristics
    cluster_features = X.loc[cluster_queries.index]
    print(f"\nCluster characteristics:")
    print(cluster_features.mean())

# Save model
joblib.dump(kmeans, 'query_clustering_model.pkl')
joblib.dump(scaler, 'query_clustering_scaler.pkl')
```

### Success Criteria
- **Silhouette Score**: > 0.3 (reasonable cluster separation)
- **Interpretability**: Clusters have clear distinguishing characteristics
- **Balance**: No single cluster dominates (>70% of queries)

### Usage in System
- Identify query types (simple reads, complex analytics, writes)
- Apply cluster-specific optimization strategies
- Cache similar queries
- Resource allocation by cluster

---

## MODEL 5: ANOMALY DETECTOR 🚨

### Purpose
Detect unusual query behavior, performance degradation, or system issues.

### Problem Type
**Anomaly Detection - Unsupervised**

### Implementation with Isolation Forest

```python
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load data
df = pd.read_sql("""
    SELECT 
        query_id,
        execution_time_ms,
        rows_returned,
        cpu_usage,
        memory_usage
    FROM metadata.query_logs
    WHERE status = 'success'
""", conn)

# Extract features
features_list = []
for idx, row in df.iterrows():
    query_text = row['query_text']
    query_features = extract_query_features(query_text)
    
    # Combine with execution metrics
    features = {
        **query_features,
        'execution_time_ms': row['execution_time_ms'],
        'rows_returned': row['rows_returned'],
        'cpu_usage': row.get('cpu_usage', 0),
        'memory_usage': row.get('memory_usage', 0)
    }
    features_list.append(features)

X = pd.DataFrame(features_list)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Isolation Forest
iso_forest = IsolationForest(
    contamination=0.05,  # Expect 5% anomalies
    random_state=42,
    n_estimators=100
)

# Fit model
iso_forest.fit(X_scaled)

# Predict anomalies (-1 = anomaly, 1 = normal)
predictions = iso_forest.predict(X_scaled)
anomaly_scores = iso_forest.score_samples(X_scaled)

# Add to dataframe
df['is_anomaly'] = predictions == -1
df['anomaly_score'] = anomaly_scores

# Analyze anomalies
anomalies = df[df['is_anomaly']]
print(f"Detected {len(anomalies)} anomalies ({len(anomalies)/len(df)*100:.2f}%)")

print("\nTop anomalies:")
print(anomalies.nsmallest(10, 'anomaly_score')[['query_id', 'execution_time_ms', 'anomaly_score']])

# Save model
joblib.dump(iso_forest, 'anomaly_detector_model.pkl')
joblib.dump(scaler, 'anomaly_detector_scaler.pkl')
```

### Real-time Anomaly Detection

```python
def detect_anomaly(query_features, execution_metrics):
    """
    Detect if a query execution is anomalous.
    
    Args:
        query_features (dict): Query structural features
        execution_metrics (dict): Execution time, resources, etc.
        
    Returns:
        dict: Anomaly detection results
    """
    # Load model
    model = joblib.load('anomaly_detector_model.pkl')
    scaler = joblib.load('anomaly_detector_scaler.pkl')
    feature_names = joblib.load('anomaly_detector_features.pkl')
    
    # Combine features
    features = {**query_features, **execution_metrics}
    features_df = pd.DataFrame([features])[feature_names]
    
    # Scale
    features_scaled = scaler.transform(features_df)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    anomaly_score = model.score_samples(features_scaled)[0]
    
    is_anomaly = prediction == -1
    
    # Determine severity
    if anomaly_score < -0.5:
        severity = 'high'
    elif anomaly_score < -0.3:
        severity = 'medium'
    else:
        severity = 'low'
    
    return {
        'is_anomaly': is_anomaly,
        'anomaly_score': float(anomaly_score),
        'severity': severity,
        'message': f"Anomaly detected with score {anomaly_score:.3f}" if is_anomaly else "Normal query"
    }
```

### Success Criteria
- **F1 Score**: > 0.70 (on labeled test anomalies)
- **False Positive Rate**: < 10%
- **Detection Speed**: < 100ms per query

### Types of Anomalies to Detect
1. **Performance Anomalies**: Sudden increase in execution time
2. **Volume Anomalies**: Unusual number of rows returned
3. **Resource Anomalies**: Excessive CPU/memory usage
4. **Pattern Anomalies**: Unusual query structure
5. **Temporal Anomalies**: Queries at unusual times

---

## 📊 MODEL COMPARISON & SELECTION

### Model Performance Summary Table

| Model | Algorithm | Training Time | Prediction Time | Accuracy Target | Complexity |
|-------|-----------|---------------|-----------------|-----------------|------------|
| Query Cost Predictor | Random Forest | 2-5 min | <50ms | R² > 0.75 | Medium |
| Workload Forecaster | Prophet | 5-10 min | <100ms | MAPE < 15% | Medium |
| Index Recommender | Rule-based | <1 min | <10ms | Precision > 0.70 | Low |
| Query Clustering | K-Means | 1-3 min | <20ms | Silhouette > 0.3 | Low |
| Anomaly Detector | Isolation Forest | 2-4 min | <50ms | F1 > 0.70 | Low |

---

## 🔄 MODEL RETRAINING STRATEGY

### When to Retrain

1. **Scheduled Retraining**
   - Weekly for Query Cost Predictor
   - Daily for Workload Forecaster
   - Monthly for others

2. **Performance-Based Retraining**
   - When accuracy drops below threshold
   - When prediction errors increase
   - When data distribution changes

3. **Data-Based Retraining**
   - After collecting 10,000+ new queries
   - After significant schema changes
   - After adding new tables

### Retraining Pipeline

```python
def retrain_all_models(conn):
    """
    Retrain all ML models with latest data.
    """
    results = {}
    
    # 1. Query Cost Predictor
    print("Retraining Query Cost Predictor...")
    from training.train_cost_predictor import train_cost_predictor
    metrics = train_cost_predictor(conn)
    results['cost_predictor'] = metrics
    
    # 2. Workload Forecaster
    print("Retraining Workload Forecaster...")
    from training.train_workload_forecaster import train_workload_forecaster
    metrics = train_workload_forecaster(conn)
    results['workload_forecaster'] = metrics
    
    # 3. Anomaly Detector
    print("Retraining Anomaly Detector...")
    from training.train_anomaly_detector import train_anomaly_detector
    metrics = train_anomaly_detector(conn)
    results['anomaly_detector'] = metrics
    
    # 4. Query Clustering
    print("Retraining Query Clustering...")
    from training.train_query_clustering import train_query_clustering
    metrics = train_query_clustering(conn)
    results['query_clustering'] = metrics
    
    # 5. Update Index Recommendations
    print("Updating Index Recommendations...")
    recommender = IndexRecommender()
    recommender.analyze_query_logs(conn)
    recommendations = recommender.recommend()
    results['index_recommender'] = {'recommendations': len(recommendations)}
    
    # Log retraining
    log_retraining(conn, results)
    
    return results

def log_retraining(conn, results):
    """Log retraining results to database"""
    cur = conn.cursor()
    
    for model_name, metrics in results.items():
        cur.execute("""
            INSERT INTO ml.model_registry (model_name, version, performance_metrics, training_date, is_active)
            VALUES (%s, %s, %s, CURRENT_TIMESTAMP, TRUE)
        """, (model_name, 'v_auto', metrics))
    
    conn.commit()
    cur.close()
```

---

## 💡 TIPS FOR SUCCESS

### 1. Start Simple
- Begin with basic features
- Use default hyperparameters
- Get a working model first
- Optimize later

### 2. Feature Engineering is Key
- Spend time on good features
- Domain knowledge matters
- Test features individually
- Remove correlated features

### 3. Validate Properly
- Always use train/test split
- Use cross-validation for small datasets
- Test on realistic scenarios
- Monitor online performance

### 4. Document Everything
- Save training logs
- Document feature definitions
- Record model versions
- Note hyperparameters

### 5. Handle Edge Cases
- Missing data
- Extreme values
- New query types
- System failures

---

## 🎯 GRADUATE-LEVEL EXPECTATIONS

For a strong graduate project, demonstrate:

1. **Understanding of ML Fundamentals**
   - Proper train/test splitting
   - Cross-validation
   - Hyperparameter tuning
   - Model evaluation

2. **Feature Engineering Sophistication**
   - Thoughtful feature selection
   - Domain-specific features
   - Feature interaction consideration

3. **Model Comparison**
   - Try multiple algorithms
   - Compare performance
   - Justify final selection
   - Document trade-offs

4. **Production Considerations**
   - Model versioning
   - Prediction latency
   - Retraining strategy
   - Error handling

5. **Evaluation Rigor**
   - Multiple metrics
   - Statistical significance testing
   - Error analysis
   - Ablation studies (optional but impressive)

---

**This specification provides everything needed to implement the ML models for your project. Follow these guidelines and you'll have a strong, graduate-level machine learning component! 🚀**
