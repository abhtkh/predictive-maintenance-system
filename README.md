# Real-Time Predictive Maintenance System

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95-009688?style=for-the-badge&logo=fastapi)
![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react)
![Docker](https://img.shields.io/badge/Docker-20.10-2496ED?style=for-the-badge&logo=docker)

A full-stack, real-time Predictive Maintenance (PdM) application that simulates industrial machine data, detects anomalies using a PyTorch model, and provides a multi-machine dashboard with a professional, role-based alerting system.

---

## 📸 Screenshot of the Fleet View Dashboard
![Fleet View Screenshot](fleet_view.png) <!-- Replace with an actual screenshot -->

---

## ✨ Features

- **Real-Time Data Pipeline:** Resilient architecture using FastAPI, Redis (Streams & Pub/Sub), and a stateful Python worker.
- **Deep Learning Anomaly Detection:** TorchScript-optimized LSTM Autoencoder detects deviations from normal machine behavior.
- **Multi-Machine Fleet Dashboard:** Modern React UI with live charts and detailed drill-down views.
- **Explainable AI:** "Anomaly Source Contribution" chart highlights the sensor causing the anomaly.
- **Role-Based Alerts:** PostgreSQL-backed role-specific email alerts for sustained anomalies.
- **Secure Backend:** JWT authentication, user management CLI, `.env`-based secure configuration.
- **High-Fidelity Simulator:** Models gradual degradation and acute controllable faults.

---

## 🏗 Architecture

The system is designed as a set of decoupled services communicating via a message broker.

![Architecture Diagram](architecture.png) <!-- Replace with actual diagram -->

---

## 🛠 Technology Stack

- **Backend:** Python, FastAPI, Uvicorn  
- **Frontend:** React, TypeScript, Styled-Components, Recharts, Zustand  
- **Database:** PostgreSQL  
- **Data Pipeline:** Redis  
- **ML Model:** PyTorch (LSTM Autoencoder)  
- **Deployment:** Docker, Docker Compose  

---

## ⚙️ Setup, Running, and Usage (Single Flow)

```bash
# 1. Clone the repository
git clone [Your-Repo-Link-Here]
cd predictive-maintenance-mvp

# 2. Create and activate Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Copy environment variables file and configure
cp .env.example .env
# Edit .env with your database password, JWT secret, SMTP credentials, etc.

# 5. Install frontend dependencies
cd frontend
npm install
cd ..

# 6. Start backend infrastructure (PostgreSQL & Redis)
docker-compose up -d

# 7. Run database migrations (only once)
alembic upgrade head

# 8. Create an admin user
python user_manager.py create \
  --username yourname \
  --email your-email@example.com \
  --role engineer \
  --enable-alerts

# 9. Start the API server (Terminal 1)
uvicorn backend.main:app --reload

# 10. Start the inference worker (Terminal 2)
python backend/worker.py

# 11. Start the frontend (Terminal 3)
cd frontend
npm start

# 12. Start the data simulator (Terminal 4)
python simulator/simulate_realistic.py

# At this point, your dashboard is live at:
# http://localhost:3000

# 13. Log in using the credentials you created earlier

# 14. Inject faults to test the alerting system (optional)
python inject_fault.py \
  --machine-id M1 \
  --fault-type TempRampFault \
  --duration 60
