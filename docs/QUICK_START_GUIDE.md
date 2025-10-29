# Quick Start Guide - AI-Powered Self-Optimizing Data Warehouse

## 🚀 Getting Started in 5 Steps

This quick start guide helps you begin your project immediately. Refer to the MASTER_PROJECT_PLAN.md for detailed instructions.

---

## STEP 1: Setup Development Environment (Day 1)

### Install Required Software

```bash
# Python 3.9+
python --version

# Node.js 16+
node --version
npm --version

# PostgreSQL or MySQL
psql --version  # or mysql --version

# Redis
redis-cli --version

# Git
git --version

# Docker (optional but recommended)
docker --version
```

### Create Project Structure

```bash
# Navigate to your workspace
cd /workspace

# Create main directories
mkdir -p backend/api backend/models backend/database backend/services backend/utils backend/config
mkdir -p frontend/src/components frontend/src/pages frontend/src/services
mkdir -p ml_models/training ml_models/trained_models ml_models/datasets ml_models/evaluation
mkdir -p scripts/data_generation scripts/migration
mkdir -p tests docs

# Initialize Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Create requirements.txt
cat > backend/requirements.txt << EOF
fastapi==0.104.1
uvicorn==0.24.0
sqlalchemy==2.0.23
psycopg2-binary==2.9.9  # or pymysql for MySQL
pandas==2.1.3
numpy==1.26.2
scikit-learn==1.3.2
xgboost==2.0.2
redis==5.0.1
python-dotenv==1.0.0
sqlparse==0.4.4
pydantic==2.5.0
websockets==12.0
joblib==1.3.2
matplotlib==3.8.2
seaborn==0.13.0
EOF

# Install Python dependencies
pip install -r backend/requirements.txt

# Initialize frontend (React example)
cd frontend
npx create-react-app . --template typescript
npm install axios recharts @mui/material @emotion/react @emotion/styled socket.io-client

# Return to workspace
cd /workspace
```

---

## STEP 2: Setup Database (Day 1-2)

### Create Database

```sql
-- PostgreSQL example
CREATE DATABASE ai_data_warehouse;

\c ai_data_warehouse;

-- Create schemas
CREATE SCHEMA bronze;
CREATE SCHEMA silver;
CREATE SCHEMA gold;
CREATE SCHEMA metadata;
CREATE SCHEMA ml;
```

### Create Essential Tables

```sql
-- Metadata: Query logs (most important for ML training)
CREATE TABLE metadata.query_logs (
    query_id SERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    execution_time_ms FLOAT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    rows_returned INTEGER,
    tables_accessed TEXT[],
    user_id VARCHAR(50),
    status VARCHAR(20),
    cpu_usage FLOAT,
    memory_usage FLOAT
);

-- Metadata: System metrics
CREATE TABLE metadata.system_metrics (
    metric_id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    cpu_usage FLOAT,
    memory_usage FLOAT,
    active_connections INTEGER,
    queries_per_second FLOAT
);

-- ML: Model registry
CREATE TABLE ml.model_registry (
    model_id SERIAL PRIMARY KEY,
    model_name VARCHAR(100),
    version VARCHAR(20),
    model_path TEXT,
    performance_metrics JSONB,
    training_date TIMESTAMP,
    is_active BOOLEAN DEFAULT FALSE
);

-- Bronze layer: Sample sales data
CREATE TABLE bronze.sales_raw (
    sale_id SERIAL PRIMARY KEY,
    customer_id INTEGER,
    product_id INTEGER,
    sale_date TIMESTAMP,
    amount DECIMAL(10,2),
    quantity INTEGER,
    region VARCHAR(50)
);

-- Bronze layer: Customers
CREATE TABLE bronze.customer_raw (
    customer_id SERIAL PRIMARY KEY,
    customer_name VARCHAR(100),
    email VARCHAR(100),
    age INTEGER,
    location VARCHAR(100),
    segment VARCHAR(50)
);

-- Bronze layer: Products
CREATE TABLE bronze.product_raw (
    product_id SERIAL PRIMARY KEY,
    product_name VARCHAR(100),
    category VARCHAR(50),
    price DECIMAL(10,2),
    supplier VARCHAR(100)
);
```

---

## STEP 3: Generate Sample Data (Day 2-3)

### Create Data Generation Script

Create `scripts/data_generation/generate_sample_data.py`:

```python
import psycopg2
import random
from datetime import datetime, timedelta
from faker import Faker

fake = Faker()

# Database connection
conn = psycopg2.connect(
    host="localhost",
    database="ai_data_warehouse",
    user="your_user",
    password="your_password"
)
cur = conn.cursor()

print("Generating customers...")
# Generate 10,000 customers
for i in range(10000):
    cur.execute("""
        INSERT INTO bronze.customer_raw (customer_name, email, age, location, segment)
        VALUES (%s, %s, %s, %s, %s)
    """, (
        fake.name(),
        fake.email(),
        random.randint(18, 80),
        fake.city(),
        random.choice(['Premium', 'Standard', 'Basic'])
    ))
    
    if i % 1000 == 0:
        print(f"  Generated {i} customers...")
        conn.commit()

conn.commit()
print("Customers generated!")

print("Generating products...")
# Generate 1,000 products
categories = ['Electronics', 'Clothing', 'Food', 'Books', 'Toys', 'Sports']
for i in range(1000):
    cur.execute("""
        INSERT INTO bronze.product_raw (product_name, category, price, supplier)
        VALUES (%s, %s, %s, %s)
    """, (
        fake.word().capitalize() + " " + fake.word().capitalize(),
        random.choice(categories),
        round(random.uniform(10, 1000), 2),
        fake.company()
    ))

conn.commit()
print("Products generated!")

print("Generating sales transactions...")
# Generate 100,000 sales
start_date = datetime.now() - timedelta(days=365)
for i in range(100000):
    sale_date = start_date + timedelta(
        days=random.randint(0, 365),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59)
    )
    
    cur.execute("""
        INSERT INTO bronze.sales_raw (customer_id, product_id, sale_date, amount, quantity, region)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (
        random.randint(1, 10000),
        random.randint(1, 1000),
        sale_date,
        round(random.uniform(10, 5000), 2),
        random.randint(1, 10),
        random.choice(['North', 'South', 'East', 'West', 'Central'])
    ))
    
    if i % 10000 == 0:
        print(f"  Generated {i} sales...")
        conn.commit()

conn.commit()
print("Sales generated!")

cur.close()
conn.close()
print("Data generation complete!")
```

### Install Faker and Run

```bash
pip install faker
python scripts/data_generation/generate_sample_data.py
```

---

## STEP 4: Generate Query Workload (Day 3-4)

### Create Workload Generator

Create `scripts/data_generation/generate_query_workload.py`:

```python
import psycopg2
import random
import time
from datetime import datetime

# Query templates
QUERY_TEMPLATES = [
    # Simple queries
    "SELECT * FROM bronze.sales_raw WHERE region = '{region}' LIMIT 100",
    "SELECT * FROM bronze.customer_raw WHERE age > {age}",
    "SELECT * FROM bronze.product_raw WHERE category = '{category}'",
    
    # Medium complexity
    """SELECT c.customer_name, COUNT(*) as purchase_count 
       FROM bronze.sales_raw s 
       JOIN bronze.customer_raw c ON s.customer_id = c.customer_id 
       WHERE s.sale_date > CURRENT_DATE - INTERVAL '{days} days'
       GROUP BY c.customer_name""",
    
    # Complex queries
    """SELECT p.category, AVG(s.amount) as avg_sale, COUNT(*) as total_sales
       FROM bronze.sales_raw s
       JOIN bronze.product_raw p ON s.product_id = p.product_id
       JOIN bronze.customer_raw c ON s.customer_id = c.customer_id
       WHERE c.segment = '{segment}' AND s.sale_date > CURRENT_DATE - INTERVAL '{days} days'
       GROUP BY p.category
       ORDER BY avg_sale DESC"""
]

conn = psycopg2.connect(
    host="localhost",
    database="ai_data_warehouse",
    user="your_user",
    password="your_password"
)

def execute_and_log_query(query_text):
    cur = conn.cursor()
    start_time = time.time()
    
    try:
        cur.execute(query_text)
        results = cur.fetchall()
        execution_time = (time.time() - start_time) * 1000  # Convert to ms
        rows_returned = len(results)
        status = 'success'
    except Exception as e:
        execution_time = (time.time() - start_time) * 1000
        rows_returned = 0
        status = 'failed'
        print(f"Query failed: {e}")
    
    # Log to metadata
    cur.execute("""
        INSERT INTO metadata.query_logs 
        (query_text, execution_time_ms, rows_returned, status, user_id)
        VALUES (%s, %s, %s, %s, %s)
    """, (query_text, execution_time, rows_returned, status, 'system'))
    
    conn.commit()
    cur.close()
    
    return execution_time

# Generate 10,000 queries
print("Generating query workload...")
for i in range(10000):
    template = random.choice(QUERY_TEMPLATES)
    
    # Fill in template parameters
    query = template.format(
        region=random.choice(['North', 'South', 'East', 'West', 'Central']),
        age=random.randint(20, 70),
        category=random.choice(['Electronics', 'Clothing', 'Food', 'Books', 'Toys', 'Sports']),
        days=random.randint(30, 365),
        segment=random.choice(['Premium', 'Standard', 'Basic'])
    )
    
    execution_time = execute_and_log_query(query)
    
    if i % 100 == 0:
        print(f"Executed {i} queries... Last execution: {execution_time:.2f}ms")
    
    # Simulate realistic timing (don't spam)
    time.sleep(0.1)

conn.close()
print("Query workload generation complete!")
```

### Run Workload Generator

```bash
python scripts/data_generation/generate_query_workload.py
```

**You now have 10,000+ queries logged for ML training!**

---

## STEP 5: Train Your First ML Model (Day 5-7)

### Create Training Script

Create `ml_models/training/train_cost_predictor.py`:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import psycopg2
import sqlparse
import re

# Connect and load data
conn = psycopg2.connect(
    host="localhost",
    database="ai_data_warehouse",
    user="your_user",
    password="your_password"
)

print("Loading query logs...")
df = pd.read_sql("""
    SELECT query_text, execution_time_ms, rows_returned
    FROM metadata.query_logs
    WHERE status = 'success' AND execution_time_ms > 0
""", conn)

print(f"Loaded {len(df)} queries")

# Feature extraction
def extract_query_features(query_text):
    """Extract features from SQL query"""
    query_upper = query_text.upper()
    
    features = {
        'num_tables': query_text.count('FROM') + query_text.count('JOIN'),
        'num_joins': query_upper.count('JOIN'),
        'num_where': query_upper.count('WHERE'),
        'has_group_by': 1 if 'GROUP BY' in query_upper else 0,
        'has_order_by': 1 if 'ORDER BY' in query_upper else 0,
        'has_distinct': 1 if 'DISTINCT' in query_upper else 0,
        'has_aggregation': 1 if any(agg in query_upper for agg in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN']) else 0,
        'query_length': len(query_text),
        'num_conditions': query_text.count('AND') + query_text.count('OR'),
    }
    
    return features

print("Extracting features...")
features_list = df['query_text'].apply(extract_query_features).tolist()
X = pd.DataFrame(features_list)
y = df['execution_time_ms']

print("Features extracted:")
print(X.head())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set: {len(X_train)} queries")
print(f"Test set: {len(X_test)} queries")

# Train model
print("\nTraining Random Forest model...")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate
print("\nEvaluating model...")
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"  RMSE: {rmse:.2f} ms")
print(f"  MAE: {mae:.2f} ms")
print(f"  R² Score: {r2:.4f}")

# Feature importance
print("\nFeature Importance:")
for feature, importance in zip(X.columns, model.feature_importances_):
    print(f"  {feature}: {importance:.4f}")

# Save model
print("\nSaving model...")
joblib.dump(model, 'ml_models/trained_models/query_cost_predictor_v1.pkl')
joblib.dump(X.columns.tolist(), 'ml_models/trained_models/query_cost_predictor_features.pkl')

print("\n✅ Model training complete!")
print(f"Model saved to: ml_models/trained_models/query_cost_predictor_v1.pkl")

# Register model in database
cur = conn.cursor()
cur.execute("""
    INSERT INTO ml.model_registry (model_name, version, model_path, performance_metrics, training_date, is_active)
    VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP, TRUE)
""", (
    'query_cost_predictor',
    'v1.0',
    'ml_models/trained_models/query_cost_predictor_v1.pkl',
    {'rmse': float(rmse), 'mae': float(mae), 'r2': float(r2)}
))
conn.commit()
cur.close()
conn.close()

print("Model registered in database!")
```

### Train the Model

```bash
python ml_models/training/train_cost_predictor.py
```

**Expected output**: R² score > 0.70, indicating your model can predict query costs!

---

## NEXT STEPS

### Week 2-3: Continue with More ML Models
- Train workload forecaster (time series)
- Train index recommender
- Train anomaly detector

### Week 4-6: Build Backend API
- Set up FastAPI
- Create query execution endpoints
- Integrate ML models
- Implement optimization engine

### Week 7-9: Build Frontend Dashboard
- Set up React application
- Create main dashboard
- Add real-time updates
- Build analytics pages

### Week 10-12: Integration & Testing
- Connect all components
- End-to-end testing
- Performance optimization
- Documentation

### Week 13-14: Final Polish & Presentation
- Create demo video
- Prepare presentation
- Practice live demo
- Write final report

---

## 📁 Project File Checklist

After completing Quick Start, you should have:

```
/workspace/
├── venv/                           ✓ Created
├── backend/
│   ├── requirements.txt            ✓ Created
│   └── (other folders ready)       ✓ Created
├── frontend/
│   ├── package.json                ✓ Created (if initialized)
│   └── (React/Vue app)             ✓ Created (if initialized)
├── ml_models/
│   ├── training/
│   │   └── train_cost_predictor.py ✓ Created
│   └── trained_models/
│       └── query_cost_predictor_v1.pkl ✓ Created (after training)
├── scripts/
│   └── data_generation/
│       ├── generate_sample_data.py       ✓ Created
│       └── generate_query_workload.py    ✓ Created
└── docs/
    ├── MASTER_PROJECT_PLAN.md      ✓ Already exists
    └── QUICK_START_GUIDE.md        ✓ This file
```

---

## 💡 Quick Tips

1. **Work Incrementally**: Don't try to build everything at once
2. **Test Often**: Test each component before moving to next
3. **Commit Regularly**: Use git to save progress
4. **Document as You Go**: Write down what works and what doesn't
5. **Ask for Help**: Don't spend more than 2 hours stuck on one issue

---

## 🆘 Common Issues & Quick Fixes

### Issue: Database connection fails
```bash
# Check if PostgreSQL is running
sudo systemctl status postgresql
# Or
ps aux | grep postgres
```

### Issue: Python packages won't install
```bash
# Upgrade pip
pip install --upgrade pip
# Try installing one by one to identify problem package
```

### Issue: Can't import modules
```bash
# Make sure virtual environment is activated
source venv/bin/activate
# Check Python path
which python
```

### Issue: Out of memory during training
```python
# Reduce data size for initial testing
df = df.sample(n=5000)  # Use 5000 instead of all queries
```

---

## 📞 Need More Help?

- **Detailed Instructions**: See MASTER_PROJECT_PLAN.md
- **Stuck on ML**: Review scikit-learn documentation
- **Stuck on API**: Review FastAPI documentation
- **Stuck on Frontend**: Review React/Vue documentation

---

**You're now ready to start building! Follow Phase 1 of the master plan. 🚀**

*Good luck with your project!*
