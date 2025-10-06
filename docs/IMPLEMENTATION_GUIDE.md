# Implementation Guide for Self-Optimizing Data Warehouse

## Getting Started

This guide provides step-by-step instructions for implementing your self-optimizing data warehouse project.

## Prerequisites

### Software Requirements
- Python 3.8 or higher
- PostgreSQL 13+ or MySQL 8.0+
- Docker and Docker Compose
- Git
- Node.js 16+ (for frontend components)

### Development Tools
- VS Code or PyCharm
- Postman or similar API testing tool
- Database administration tool (pgAdmin, MySQL Workbench)

## Phase 1: Foundation Setup

### Step 1: Environment Setup

1. **Clone and Initialize Project**
```bash
git clone <your-repo-url>
cd self-optimizing-data-warehouse
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Database Setup**
```bash
# Start PostgreSQL with Docker
docker-compose up -d postgres

# Create database and schemas
psql -h localhost -U postgres -f scripts/database.sql
```

3. **Environment Configuration**
```bash
cp .env.example .env
# Edit .env with your database credentials
```

### Step 2: Basic ETL Pipeline

Create a simple ETL pipeline to populate your data warehouse:

```python
# scripts/etl_pipeline.py
import pandas as pd
import psycopg2
from sqlalchemy import create_engine

def extract_data(source_file):
    """Extract data from CSV files"""
    return pd.read_csv(source_file)

def transform_data(df):
    """Clean and transform data"""
    # Add your transformation logic here
    return df

def load_data(df, table_name, schema='bronze'):
    """Load data into database"""
    engine = create_engine('postgresql://user:password@localhost:5432/DataWarehouse')
    df.to_sql(table_name, engine, schema=schema, if_exists='replace', index=False)

def run_etl():
    """Main ETL process"""
    # Extract
    df = extract_data('datasets/sample_data.csv')
    
    # Transform
    df = transform_data(df)
    
    # Load
    load_data(df, 'sample_table')
    print("ETL process completed successfully")

if __name__ == "__main__":
    run_etl()
```

### Step 3: Monitoring Infrastructure

Set up basic monitoring:

```python
# monitoring/performance_monitor.py
import time
import psycopg2
from datetime import datetime

class PerformanceMonitor:
    def __init__(self, db_connection):
        self.db_connection = db_connection
        
    def log_query_performance(self, query, execution_time, resource_usage):
        """Log query performance metrics"""
        cursor = self.db_connection.cursor()
        cursor.execute("""
            INSERT INTO performance_metrics 
            (query_text, execution_time_ms, cpu_usage, memory_usage, timestamp)
            VALUES (%s, %s, %s, %s, %s)
        """, (query, execution_time, resource_usage['cpu'], 
              resource_usage['memory'], datetime.now()))
        self.db_connection.commit()
        
    def get_performance_summary(self):
        """Get performance summary statistics"""
        cursor = self.db_connection.cursor()
        cursor.execute("""
            SELECT 
                AVG(execution_time_ms) as avg_time,
                MAX(execution_time_ms) as max_time,
                COUNT(*) as total_queries
            FROM performance_metrics
            WHERE timestamp > NOW() - INTERVAL '1 hour'
        """)
        return cursor.fetchone()
```

## Phase 2: Query Analysis Implementation

### Step 1: Query Parser

```python
# analysis/query_parser.py
import re
import hashlib
from sqlparse import parse, format

class QueryParser:
    def __init__(self):
        self.query_patterns = {}
        
    def parse_query(self, sql_query):
        """Parse SQL query and extract features"""
        # Normalize query
        normalized = self.normalize_query(sql_query)
        
        # Extract features
        features = {
            'query_hash': self.get_query_hash(normalized),
            'normalized_query': normalized,
            'tables': self.extract_tables(sql_query),
            'joins': self.count_joins(sql_query),
            'filters': self.count_filters(sql_query),
            'aggregations': self.count_aggregations(sql_query),
            'complexity_score': self.calculate_complexity(sql_query)
        }
        
        return features
    
    def normalize_query(self, sql_query):
        """Normalize SQL query for pattern matching"""
        # Remove comments
        sql_query = re.sub(r'--.*$', '', sql_query, flags=re.MULTILINE)
        
        # Normalize whitespace
        sql_query = re.sub(r'\s+', ' ', sql_query.strip())
        
        # Convert to lowercase
        sql_query = sql_query.lower()
        
        # Replace specific values with placeholders
        sql_query = re.sub(r'\d+', 'N', sql_query)
        sql_query = re.sub(r"'[^']*'", "'STRING'", sql_query)
        
        return sql_query
    
    def get_query_hash(self, normalized_query):
        """Generate hash for query identification"""
        return hashlib.md5(normalized_query.encode()).hexdigest()
    
    def extract_tables(self, sql_query):
        """Extract table names from query"""
        # Simple regex-based extraction
        # In production, use proper SQL parser
        tables = re.findall(r'from\s+(\w+)', sql_query, re.IGNORECASE)
        tables.extend(re.findall(r'join\s+(\w+)', sql_query, re.IGNORECASE))
        return list(set(tables))
    
    def count_joins(self, sql_query):
        """Count number of joins in query"""
        return len(re.findall(r'\bjoin\b', sql_query, re.IGNORECASE))
    
    def count_filters(self, sql_query):
        """Count WHERE conditions"""
        return len(re.findall(r'\bwhere\b', sql_query, re.IGNORECASE))
    
    def count_aggregations(self, sql_query):
        """Count aggregation functions"""
        agg_functions = ['sum', 'count', 'avg', 'min', 'max', 'group_concat']
        count = 0
        for func in agg_functions:
            count += len(re.findall(rf'\b{func}\b', sql_query, re.IGNORECASE))
        return count
    
    def calculate_complexity(self, sql_query):
        """Calculate query complexity score"""
        complexity = 0
        complexity += self.count_joins(sql_query) * 2
        complexity += self.count_filters(sql_query)
        complexity += self.count_aggregations(sql_query) * 1.5
        complexity += len(self.extract_tables(sql_query)) * 0.5
        return complexity
```

### Step 2: Query Clustering

```python
# analysis/query_clustering.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np

class QueryClustering:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.clusterer = None
        self.clusters = {}
        
    def fit_clusters(self, queries, method='kmeans', n_clusters=5):
        """Fit clustering model to queries"""
        # Vectorize queries
        query_vectors = self.vectorizer.fit_transform(queries)
        
        if method == 'kmeans':
            self.clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        elif method == 'dbscan':
            self.clusterer = DBSCAN(eps=0.5, min_samples=2)
        
        cluster_labels = self.clusterer.fit_predict(query_vectors)
        
        # Store cluster assignments
        for i, label in enumerate(cluster_labels):
            if label not in self.clusters:
                self.clusters[label] = []
            self.clusters[label].append(queries[i])
        
        return cluster_labels
    
    def predict_cluster(self, query):
        """Predict cluster for new query"""
        if self.clusterer is None:
            raise ValueError("Model not fitted yet")
        
        query_vector = self.vectorizer.transform([query])
        cluster_label = self.clusterer.predict(query_vector)[0]
        return cluster_label
    
    def get_cluster_centers(self):
        """Get cluster centers for analysis"""
        if hasattr(self.clusterer, 'cluster_centers_'):
            return self.clusterer.cluster_centers_
        return None
    
    def evaluate_clusters(self, query_vectors, cluster_labels):
        """Evaluate clustering quality"""
        if len(set(cluster_labels)) > 1:
            silhouette_avg = silhouette_score(query_vectors, cluster_labels)
            return silhouette_avg
        return 0
```

### Step 3: Performance Prediction

```python
# analysis/performance_predictor.py
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd

class PerformancePredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.feature_columns = []
        
    def prepare_features(self, query_features):
        """Prepare features for model training"""
        features = []
        feature_names = []
        
        # Add query complexity features
        features.append(query_features['joins'])
        feature_names.append('joins')
        
        features.append(query_features['filters'])
        feature_names.append('filters')
        
        features.append(query_features['aggregations'])
        feature_names.append('aggregations')
        
        features.append(query_features['complexity_score'])
        feature_names.append('complexity_score')
        
        # Add table count
        features.append(len(query_features['tables']))
        feature_names.append('table_count')
        
        self.feature_columns = feature_names
        return features
    
    def train(self, query_features_list, execution_times):
        """Train performance prediction model"""
        # Prepare training data
        X = []
        for features in query_features_list:
            X.append(self.prepare_features(features))
        
        X = np.array(X)
        y = np.array(execution_times)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model Performance:")
        print(f"MAE: {mae:.2f}ms")
        print(f"R²: {r2:.2f}")
        
        return mae, r2
    
    def predict(self, query_features):
        """Predict execution time for query"""
        features = self.prepare_features(query_features)
        prediction = self.model.predict([features])[0]
        return max(0, prediction)  # Ensure non-negative prediction
```

## Phase 3: Index Optimization

### Step 1: Index Analyzer

```python
# optimization/index_analyzer.py
import psycopg2
from collections import defaultdict

class IndexAnalyzer:
    def __init__(self, db_connection):
        self.db_connection = db_connection
        
    def analyze_index_usage(self):
        """Analyze current index usage"""
        cursor = self.db_connection.cursor()
        
        # Get index usage statistics
        cursor.execute("""
            SELECT 
                schemaname,
                tablename,
                indexname,
                idx_scan,
                idx_tup_read,
                idx_tup_fetch
            FROM pg_stat_user_indexes
            ORDER BY idx_scan DESC
        """)
        
        index_usage = cursor.fetchall()
        return index_usage
    
    def find_unused_indexes(self, threshold=0):
        """Find indexes that are rarely or never used"""
        index_usage = self.analyze_index_usage()
        unused_indexes = []
        
        for index in index_usage:
            if index[3] <= threshold:  # idx_scan <= threshold
                unused_indexes.append({
                    'schema': index[0],
                    'table': index[1],
                    'index': index[2],
                    'scans': index[3]
                })
        
        return unused_indexes
    
    def analyze_missing_indexes(self):
        """Analyze queries for missing indexes"""
        cursor = self.db_connection.cursor()
        
        # Get slow queries
        cursor.execute("""
            SELECT query, mean_time, calls
            FROM pg_stat_statements
            WHERE mean_time > 1000  -- Queries taking more than 1 second
            ORDER BY mean_time DESC
            LIMIT 100
        """)
        
        slow_queries = cursor.fetchall()
        missing_indexes = []
        
        for query, mean_time, calls in slow_queries:
            # Analyze query for potential missing indexes
            recommendations = self.analyze_query_for_indexes(query)
            if recommendations:
                missing_indexes.extend(recommendations)
        
        return missing_indexes
    
    def analyze_query_for_indexes(self, query):
        """Analyze individual query for index recommendations"""
        recommendations = []
        
        # Simple analysis - look for WHERE clauses
        import re
        where_clauses = re.findall(r'WHERE\s+([^)]+)', query, re.IGNORECASE)
        
        for clause in where_clauses:
            # Extract column names from WHERE clause
            columns = re.findall(r'(\w+)\s*[=<>]', clause)
            if columns:
                recommendations.append({
                    'columns': columns,
                    'reason': 'WHERE clause optimization',
                    'query': query[:100] + '...'
                })
        
        return recommendations
```

### Step 2: Index Recommendation Engine

```python
# optimization/index_recommender.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

class IndexRecommender:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_columns = []
        
    def prepare_training_data(self, query_features, index_benefits):
        """Prepare training data for index recommendation"""
        X = []
        y = []
        
        for features, benefit in zip(query_features, index_benefits):
            # Extract features
            feature_vector = [
                features['joins'],
                features['filters'],
                features['aggregations'],
                features['complexity_score'],
                len(features['tables'])
            ]
            
            X.append(feature_vector)
            y.append(1 if benefit > 0.1 else 0)  # Binary classification
        
        return np.array(X), np.array(y)
    
    def train(self, query_features, index_benefits):
        """Train index recommendation model"""
        X, y = self.prepare_training_data(query_features, index_benefits)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        accuracy = self.model.score(X_test, y_test)
        print(f"Index recommendation accuracy: {accuracy:.2f}")
        
        return accuracy
    
    def recommend_indexes(self, query_features):
        """Recommend indexes for query"""
        feature_vector = [
            query_features['joins'],
            query_features['filters'],
            query_features['aggregations'],
            query_features['complexity_score'],
            len(query_features['tables'])
        ]
        
        probability = self.model.predict_proba([feature_vector])[0][1]
        return probability > 0.5, probability
```

## Phase 4: System Integration

### Step 1: Main Optimization Engine

```python
# core/optimization_engine.py
import asyncio
from datetime import datetime, timedelta
import logging

class OptimizationEngine:
    def __init__(self, db_connection):
        self.db_connection = db_connection
        self.query_parser = QueryParser()
        self.query_clustering = QueryClustering()
        self.performance_predictor = PerformancePredictor()
        self.index_analyzer = IndexAnalyzer(db_connection)
        self.index_recommender = IndexRecommender()
        
        self.logger = logging.getLogger(__name__)
        
    async def start_optimization_loop(self):
        """Start the main optimization loop"""
        self.logger.info("Starting optimization engine...")
        
        while True:
            try:
                await self.run_optimization_cycle()
                await asyncio.sleep(300)  # Run every 5 minutes
            except Exception as e:
                self.logger.error(f"Optimization cycle failed: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    async def run_optimization_cycle(self):
        """Run a single optimization cycle"""
        self.logger.info("Running optimization cycle...")
        
        # 1. Analyze recent queries
        recent_queries = await self.get_recent_queries()
        if not recent_queries:
            return
        
        # 2. Update query patterns
        query_features = []
        for query in recent_queries:
            features = self.query_parser.parse_query(query['query_text'])
            query_features.append(features)
        
        # 3. Update clustering
        queries = [q['query_text'] for q in recent_queries]
        self.query_clustering.fit_clusters(queries)
        
        # 4. Analyze indexes
        unused_indexes = self.index_analyzer.find_unused_indexes()
        missing_indexes = self.index_analyzer.analyze_missing_indexes()
        
        # 5. Make optimization recommendations
        await self.make_optimization_recommendations(
            query_features, unused_indexes, missing_indexes
        )
        
        self.logger.info("Optimization cycle completed")
    
    async def get_recent_queries(self, hours=1):
        """Get recent queries from the database"""
        cursor = self.db_connection.cursor()
        cursor.execute("""
            SELECT query_text, execution_time_ms, timestamp
            FROM performance_metrics
            WHERE timestamp > NOW() - INTERVAL '%s hours'
            ORDER BY timestamp DESC
        """, (hours,))
        
        return cursor.fetchall()
    
    async def make_optimization_recommendations(self, query_features, 
                                              unused_indexes, missing_indexes):
        """Make optimization recommendations"""
        recommendations = []
        
        # Index recommendations
        for features in query_features:
            should_recommend, confidence = self.index_recommender.recommend_indexes(features)
            if should_recommend:
                recommendations.append({
                    'type': 'index',
                    'confidence': confidence,
                    'features': features
                })
        
        # Remove unused indexes
        for index in unused_indexes:
            recommendations.append({
                'type': 'remove_index',
                'index': index,
                'confidence': 0.9
            })
        
        # Apply recommendations
        await self.apply_recommendations(recommendations)
    
    async def apply_recommendations(self, recommendations):
        """Apply optimization recommendations"""
        for rec in recommendations:
            try:
                if rec['type'] == 'index' and rec['confidence'] > 0.7:
                    await self.create_recommended_index(rec)
                elif rec['type'] == 'remove_index' and rec['confidence'] > 0.8:
                    await self.remove_unused_index(rec['index'])
            except Exception as e:
                self.logger.error(f"Failed to apply recommendation: {e}")
    
    async def create_recommended_index(self, recommendation):
        """Create a recommended index"""
        # Implementation for creating indexes
        pass
    
    async def remove_unused_index(self, index):
        """Remove an unused index"""
        # Implementation for removing indexes
        pass
```

### Step 2: API Server

```python
# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Self-Optimizing Data Warehouse API")

class QueryRequest(BaseModel):
    query: str

class OptimizationStatus(BaseModel):
    status: str
    last_optimization: str
    recommendations_count: int

@app.post("/api/queries/analyze")
async def analyze_query(request: QueryRequest):
    """Analyze a query for optimization opportunities"""
    try:
        # Parse query
        features = query_parser.parse_query(request.query)
        
        # Get recommendations
        should_index, confidence = index_recommender.recommend_indexes(features)
        
        return {
            "query_hash": features['query_hash'],
            "complexity_score": features['complexity_score'],
            "index_recommended": should_index,
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/optimization/status")
async def get_optimization_status():
    """Get current optimization status"""
    return {
        "status": "running",
        "last_optimization": "2024-01-01T12:00:00Z",
        "recommendations_count": 5
    }

@app.get("/api/metrics/performance")
async def get_performance_metrics():
    """Get performance metrics"""
    # Implementation to get performance metrics
    return {
        "avg_query_time": 150.5,
        "total_queries": 1000,
        "optimization_improvement": 0.15
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Testing

### Unit Tests

```python
# tests/test_query_parser.py
import unittest
from analysis.query_parser import QueryParser

class TestQueryParser(unittest.TestCase):
    def setUp(self):
        self.parser = QueryParser()
    
    def test_normalize_query(self):
        query = "SELECT * FROM users WHERE id = 123"
        normalized = self.parser.normalize_query(query)
        expected = "select * from users where id = n"
        self.assertEqual(normalized, expected)
    
    def test_extract_tables(self):
        query = "SELECT * FROM users u JOIN orders o ON u.id = o.user_id"
        tables = self.parser.extract_tables(query)
        expected = ['users', 'orders']
        self.assertEqual(set(tables), set(expected))
    
    def test_calculate_complexity(self):
        simple_query = "SELECT * FROM users"
        complex_query = "SELECT COUNT(*) FROM users u JOIN orders o ON u.id = o.user_id WHERE o.status = 'completed' GROUP BY u.id"
        
        simple_complexity = self.parser.calculate_complexity(simple_query)
        complex_complexity = self.parser.calculate_complexity(complex_query)
        
        self.assertGreater(complex_complexity, simple_complexity)

if __name__ == '__main__':
    unittest.main()
```

## Deployment

### Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: DataWarehouse
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  app:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - postgres
    environment:
      DATABASE_URL: postgresql://postgres:password@postgres:5432/DataWarehouse

volumes:
  postgres_data:
```

## Next Steps

1. **Start with Phase 1**: Set up the basic infrastructure
2. **Implement Core Components**: Build the query parser and basic monitoring
3. **Add Machine Learning**: Integrate ML models for optimization
4. **Test and Iterate**: Continuously test and improve the system
5. **Scale Up**: Add more advanced features as you progress

Remember to:
- Keep detailed logs of your progress
- Document your decisions and trade-offs
- Test each component thoroughly
- Maintain clean, readable code
- Follow the roadmap phases systematically