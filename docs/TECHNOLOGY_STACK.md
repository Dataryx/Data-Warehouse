# Technology Stack Specification

## AI-Powered Self-Optimizing Data Warehouse
### Detailed Technology Recommendations

---

## 📊 DATABASE LAYER

### Primary Database: **PostgreSQL 14+** (Recommended)

**Why PostgreSQL?**
- Excellent query optimization capabilities
- Rich EXPLAIN ANALYZE for learning query plans
- Support for advanced data types (JSONB, Arrays)
- Mature, stable, and well-documented
- Free and open-source
- Built-in full-text search
- Excellent Python support

**Installation**:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install postgresql postgresql-contrib

# macOS
brew install postgresql

# Windows
# Download installer from https://www.postgresql.org/download/windows/
```

**Alternative**: MySQL 8.0+ (also excellent, choose based on familiarity)

### Time-Series Data: **TimescaleDB** (PostgreSQL Extension)

**Why TimescaleDB?**
- Built on PostgreSQL (seamless integration)
- Optimized for time-series metrics
- Automatic partitioning
- Compression for historical data

**Installation**:
```bash
# Add TimescaleDB repository and install
sudo add-apt-repository ppa:timescale/timescaledb-ppa
sudo apt-get update
sudo apt-get install timescaledb-postgresql-14

# Enable extension in your database
CREATE EXTENSION IF NOT EXISTS timescaledb;
```

### Caching: **Redis 7+**

**Why Redis?**
- In-memory storage (extremely fast)
- Perfect for caching ML predictions
- Pub/Sub for real-time updates
- Simple key-value store

**Installation**:
```bash
# Ubuntu/Debian
sudo apt-get install redis-server

# macOS
brew install redis

# Start Redis
redis-server
```

---

## 🐍 BACKEND STACK

### API Framework: **FastAPI 0.104+** (Highly Recommended)

**Why FastAPI?**
- Modern, fast, and high-performance
- Automatic API documentation (Swagger UI)
- Built-in data validation with Pydantic
- Async support for better performance
- Type hints for better code quality
- WebSocket support built-in
- Easy to learn for Python developers

**Installation**:
```bash
pip install fastapi==0.104.1
pip install "uvicorn[standard]==0.24.0"
```

**Basic FastAPI App**:
```python
from fastapi import FastAPI

app = FastAPI(title="AI Data Warehouse API")

@app.get("/")
async def root():
    return {"message": "API is running"}

# Run with: uvicorn main:app --reload
```

**Alternative**: Flask (if you prefer simplicity over features)

### Database ORM: **SQLAlchemy 2.0+**

**Why SQLAlchemy?**
- Industry-standard Python ORM
- Excellent PostgreSQL support
- Powerful query building
- Connection pooling
- Works seamlessly with FastAPI

**Installation**:
```bash
pip install sqlalchemy==2.0.23
pip install psycopg2-binary==2.9.9  # PostgreSQL driver
```

### SQL Parsing: **sqlparse**

**Why sqlparse?**
- Parse and analyze SQL queries
- Format SQL for display
- Extract query components

**Installation**:
```bash
pip install sqlparse==0.4.4
```

---

## 🤖 MACHINE LEARNING STACK

### Core ML Library: **scikit-learn 1.3+**

**Why scikit-learn?**
- Industry standard for classical ML
- Excellent documentation and examples
- All algorithms you need (Random Forest, Isolation Forest)
- Simple API
- Great for graduate-level projects

**Installation**:
```bash
pip install scikit-learn==1.3.2
```

**Key Algorithms for Your Project**:
- `RandomForestRegressor` - Query cost prediction
- `RandomForestClassifier` - Index recommendation
- `IsolationForest` - Anomaly detection
- `KMeans` - Query clustering

### Gradient Boosting: **XGBoost 2.0+**

**Why XGBoost?**
- Often achieves best performance
- Fast training and prediction
- Feature importance built-in
- Handles missing data well

**Installation**:
```bash
pip install xgboost==2.0.2
```

### Time Series: **Prophet** (Facebook)

**Why Prophet?**
- Designed specifically for forecasting
- Handles seasonality automatically
- Robust to missing data
- Easy to use
- Great for business metrics

**Installation**:
```bash
pip install prophet==1.1.5
```

**Alternative**: statsmodels (for ARIMA/SARIMA)

### Data Processing: **Pandas + NumPy**

**Why Pandas + NumPy?**
- Standard for data manipulation in Python
- Essential for feature engineering
- Excellent for working with DataFrames
- Integrates with everything

**Installation**:
```bash
pip install pandas==2.1.3
pip install numpy==1.26.2
```

### Model Persistence: **Joblib**

**Why Joblib?**
- Efficient serialization of large numpy arrays
- Built into scikit-learn
- Faster than pickle for ML models

**Installation**:
```bash
pip install joblib==1.3.2
```

### Visualization: **Matplotlib + Seaborn**

**Why Matplotlib + Seaborn?**
- Create training/evaluation plots
- Feature importance visualization
- Model performance charts
- For your documentation/report

**Installation**:
```bash
pip install matplotlib==3.8.2
pip install seaborn==0.13.0
```

---

## 💻 FRONTEND STACK

### Framework: **React 18+** with **TypeScript** (Recommended)

**Why React?**
- Most popular frontend framework
- Huge ecosystem and community
- Excellent for dashboards
- Component-based architecture
- Great developer tools

**Why TypeScript?**
- Type safety prevents bugs
- Better IDE support
- More professional
- Great for graduate-level projects

**Installation**:
```bash
npx create-react-app frontend --template typescript
cd frontend
```

**Alternative**: Vue.js 3 (also excellent, easier learning curve)

### UI Component Library: **Material-UI (MUI) v5**

**Why MUI?**
- Professional, modern design
- Pre-built components (cards, tables, charts)
- Responsive by default
- Excellent documentation
- Theme customization

**Installation**:
```bash
npm install @mui/material @emotion/react @emotion/styled
npm install @mui/icons-material
```

**Alternative**: Ant Design (more enterprise feel)

### HTTP Client: **Axios**

**Why Axios?**
- Promise-based HTTP client
- Interceptors for request/response
- Error handling
- Works in browser and Node.js

**Installation**:
```bash
npm install axios
```

### Charts: **Recharts**

**Why Recharts?**
- React-specific charting library
- Composable components
- Responsive charts
- Beautiful defaults
- Easy to customize

**Installation**:
```bash
npm install recharts
```

**Alternative**: Chart.js with react-chartjs-2

### Real-time Communication: **socket.io-client**

**Why socket.io?**
- Easy WebSocket implementation
- Automatic reconnection
- Fallback to polling if needed
- Works with FastAPI (with socketio library)

**Installation**:
```bash
npm install socket.io-client
```

### State Management: **Redux Toolkit** (for complex state)

**Why Redux Toolkit?**
- Centralized state management
- Useful for large applications
- Time-travel debugging
- DevTools

**Installation**:
```bash
npm install @reduxjs/toolkit react-redux
```

**Alternative**: React Context API (simpler, built-in)

### Routing: **React Router v6**

**Installation**:
```bash
npm install react-router-dom
```

---

## 🔧 DEVELOPMENT TOOLS

### Version Control: **Git**

**Installation**:
```bash
# Ubuntu/Debian
sudo apt-get install git

# macOS
brew install git

# Configure
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### Python Environment: **venv** (built-in)

**Usage**:
```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Deactivate
deactivate
```

### Code Editor: **VS Code** (Recommended)

**Why VS Code?**
- Free and open-source
- Excellent Python and JavaScript support
- Integrated terminal
- Git integration
- Thousands of extensions

**Recommended Extensions**:
- Python (Microsoft)
- Pylance (Microsoft)
- ES7+ React/Redux/React-Native snippets
- GitLens
- Prettier - Code formatter
- SQLTools
- Docker

### API Testing: **Postman** or **Insomnia**

**Why?**
- Test API endpoints during development
- Save request collections
- Generate code snippets

---

## 📦 CONTAINERIZATION

### Docker & Docker Compose

**Why Docker?**
- Consistent environment across machines
- Easy deployment
- Isolate services
- Professional deployment approach

**Installation**:
```bash
# Ubuntu
sudo apt-get install docker.io docker-compose

# macOS
brew install --cask docker

# Verify
docker --version
docker-compose --version
```

---

## 📊 MONITORING & LOGGING (Optional but Professional)

### Application Monitoring: **Prometheus + Grafana** (Optional)

**Why?**
- Industry-standard monitoring
- Beautiful dashboards
- Real-time metrics
- Professional addition to project

**Note**: For graduate project, custom dashboard is sufficient. This is for extra credit.

### Logging: **Python logging module** (Built-in)

**Usage**:
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("Application started")
```

---

## 🔐 SECURITY & AUTHENTICATION (Optional)

### JWT Authentication: **python-jose** (If adding user auth)

**Installation**:
```bash
pip install python-jose[cryptography]
pip install passlib[bcrypt]
```

**Note**: For graduate project, authentication is optional. Focus on core functionality first.

---

## 📦 COMPLETE REQUIREMENTS FILES

### Backend: `backend/requirements.txt`

```txt
# FastAPI and server
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6

# Database
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
alembic==1.12.1

# Machine Learning
scikit-learn==1.3.2
xgboost==2.0.2
prophet==1.1.5
pandas==2.1.3
numpy==1.26.2
joblib==1.3.2

# Visualization (for model evaluation)
matplotlib==3.8.2
seaborn==0.13.0

# Data processing
sqlparse==0.4.4

# Caching
redis==5.0.1

# Configuration
python-dotenv==1.0.0

# Data validation
pydantic==2.5.0

# WebSocket
websockets==12.0

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.2

# Utilities
python-dateutil==2.8.2

# Data generation (for development)
faker==20.1.0
```

### Frontend: `frontend/package.json` (dependencies section)

```json
{
  "dependencies": {
    "@mui/material": "^5.14.18",
    "@mui/icons-material": "^5.14.18",
    "@emotion/react": "^11.11.1",
    "@emotion/styled": "^11.11.0",
    "axios": "^1.6.2",
    "recharts": "^2.10.3",
    "socket.io-client": "^4.6.0",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.20.1",
    "@reduxjs/toolkit": "^2.0.1",
    "react-redux": "^9.0.4",
    "date-fns": "^2.30.0",
    "react-syntax-highlighter": "^15.5.0"
  }
}
```

---

## 🎯 TECHNOLOGY DECISION MATRIX

| Category | Primary Choice | Alternative | Difficulty | Learning Curve |
|----------|---------------|-------------|------------|----------------|
| Database | PostgreSQL | MySQL | Medium | Moderate |
| Backend | FastAPI | Flask | Medium | Easy |
| ORM | SQLAlchemy | Django ORM | Medium | Moderate |
| ML Library | scikit-learn | TensorFlow | Easy | Easy |
| Frontend | React | Vue.js | Medium | Moderate |
| UI Framework | Material-UI | Ant Design | Easy | Easy |
| Charts | Recharts | Chart.js | Easy | Easy |
| WebSocket | socket.io | Native WS | Easy | Easy |

---

## 💡 RECOMMENDATIONS FOR GRADUATE PROJECT

### Essential (Must Have)
✅ PostgreSQL - Core database
✅ FastAPI - Backend API
✅ SQLAlchemy - Database ORM
✅ scikit-learn - Machine learning
✅ React + TypeScript - Frontend
✅ Material-UI - UI components
✅ Recharts - Visualization
✅ Redis - Caching

### Recommended (Should Have)
🟡 Docker - Containerization
🟡 XGBoost - Better ML performance
🟡 Prophet - Time series forecasting
🟡 socket.io - Real-time updates

### Optional (Nice to Have)
⚪ Prometheus/Grafana - Advanced monitoring
⚪ JWT Auth - User authentication
⚪ TensorFlow - Deep learning (if time permits)

---

## 🚀 GETTING STARTED INSTALLATION SCRIPT

### For Ubuntu/Linux:

```bash
#!/bin/bash

echo "Installing AI Data Warehouse Dependencies..."

# Update system
sudo apt-get update

# Install Python 3.9+
sudo apt-get install -y python3 python3-pip python3-venv

# Install PostgreSQL
sudo apt-get install -y postgresql postgresql-contrib

# Install Redis
sudo apt-get install -y redis-server

# Install Node.js 16+
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install Docker
sudo apt-get install -y docker.io docker-compose

# Install Git
sudo apt-get install -y git

echo "✅ All dependencies installed!"
echo "Next steps:"
echo "1. Create Python virtual environment: python3 -m venv venv"
echo "2. Activate environment: source venv/bin/activate"
echo "3. Install Python packages: pip install -r requirements.txt"
echo "4. Initialize React app: npx create-react-app frontend"
```

### For macOS:

```bash
#!/bin/bash

echo "Installing AI Data Warehouse Dependencies..."

# Install Homebrew if not installed
which brew || /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.11

# Install PostgreSQL
brew install postgresql@14
brew services start postgresql@14

# Install Redis
brew install redis
brew services start redis

# Install Node.js
brew install node@18

# Install Docker
brew install --cask docker

# Install Git
brew install git

echo "✅ All dependencies installed!"
echo "Next steps:"
echo "1. Create Python virtual environment: python3 -m venv venv"
echo "2. Activate environment: source venv/bin/activate"
echo "3. Install Python packages: pip install -r requirements.txt"
echo "4. Initialize React app: npx create-react-app frontend"
```

---

## 📚 LEARNING RESOURCES

### FastAPI
- Official Docs: https://fastapi.tiangolo.com/
- Tutorial: https://fastapi.tiangolo.com/tutorial/

### scikit-learn
- Official Docs: https://scikit-learn.org/
- User Guide: https://scikit-learn.org/stable/user_guide.html

### React
- Official Docs: https://react.dev/
- Tutorial: https://react.dev/learn

### Material-UI
- Official Docs: https://mui.com/
- Components: https://mui.com/material-ui/all-components/

### PostgreSQL
- Official Docs: https://www.postgresql.org/docs/
- Tutorial: https://www.postgresqltutorial.com/

---

**This technology stack is proven, modern, and perfect for a graduate-level final year project. All tools are free, open-source, and have excellent community support.**

**Good luck! 🚀**
