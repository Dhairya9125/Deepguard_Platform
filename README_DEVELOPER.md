# DeepGuard Platform — Developer Guide

Welcome to the DeepGuard Platform! This guide outlines everything you need to set up the project locally, configure the database, and launch both the Next.js frontend and FastAPI backend.

## Prerequisites

Before getting started, make sure you have the following installed on your machine:
- **Node.js** (v18 or newer)
- **Python** (v3.10 or newer)
- **Git**

## 1. Environment Configuration (`.env`)

The platform relies on a single `.env` file located in the root of the project directory. You **must** create this file before starting the application. 

Create a file named `.env` in the project root (`Deepguard_Platform-main/.env`) and populate it with the following:

```env
# ==========================================
# DATABASE — Neon Serverless PostgreSQL
# ==========================================
# The BACKEND_DATABASE_URL uses `postgresql+asyncpg` for the asynchronous Python backend.
# The DATABASE_URL uses standard `postgresql` format.
BACKEND_DATABASE_URL=postgresql+asyncpg://<DB_USER>:<DB_PASSWORD>@<NEON_HOST>/<DB_NAME>
DATABASE_URL=postgresql://<DB_USER>:<DB_PASSWORD>@<NEON_HOST>/<DB_NAME>

# Required for Neon Database connections
DB_SSL_REQUIRED=True

# ==========================================
# SECURITY — JWT Authentication
# ==========================================
# Generate a secure random string (e.g., using `openssl rand -hex 32`)
SECRET_KEY=98d97be5ee67abf0d01d4a02df2fca35080c326078bb09825b42d54e58b8efbe
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=10080

# ==========================================
# API Configuration
# ==========================================
PROJECT_NAME=TRUX
API_V1_STR=/api/v1

# ==========================================
# MLflow (Model Tracking)
# ==========================================
MLFLOW_TRACKING_URI=file:./mlruns

# ==========================================
# Email Configuration (Nodemailer / SMTP)
# ==========================================
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_SECURE=false
SMTP_USER=your_email@gmail.com
SMTP_PASS=your_app_password

# Application URL (used in email links like Forgot Password)
NEXT_PUBLIC_APP_URL=http://localhost:3000
```

### Understanding the Variables:
* **`BACKEND_DATABASE_URL`**: Used by the FastAPI backend to connect asynchronously using `asyncpg`. 
* **`SECRET_KEY`**: Used to sign and verify JSON Web Tokens (JWT) for user authentication. 
* **`SMTP_*`**: Used to send "Forgot Password" emails. If you are using Gmail, you must generate an **App Password** from your Google Account settings, rather than using your standard password.

---

## 2. Setting Up the Neon Database

The platform uses a serverless PostgreSQL database hosted on [Neon](https://neon.tech/). 

1. **Create an Account**: Go to neon.tech and sign up.
2. **Create a Project**: Create a new project and database.
3. **Get the Connection String**: On your Neon dashboard, you will be provided with a connection string that looks like this:
   `postgresql://neondb_owner:YOUR_PASSWORD@ep-summer-moon-...aws.neon.tech/neondb?sslmode=require`
4. **Update `.env`**: Replace the placeholders in your `.env` file with the exact credentials from your Neon dashboard. **Ensure you keep the `+asyncpg` tag for the `BACKEND_DATABASE_URL` string.**

> **Note:** No manual SQL migrations are required! The FastAPI backend uses SQLModel, and we have configured `apps/api/db/database.py` to automatically detect missing tables and execute `SQLModel.metadata.create_all` during the server startup. 

---

## 3. Installing Dependencies

You need to install dependencies for both the frontend and backend.

### Backend (Python)
Open a terminal in the root directory:
```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install the Python dependencies
pip install -r requirements.txt
```
*(Note: If you plan on running the ML test suites, you may also need to run `pip install insightface onnxruntime imagehash` as they are heavy dependencies used by the detection models).*

### Frontend (Next.js)
Open a new terminal and navigate to the frontend directory:
```bash
cd ADS/frontend
npm install
```

---

## 4. Launching the Platform (`start.bat`)

For a seamless development experience on Windows, we have provided a script named `start.bat` in the root directory. 

### What `start.bat` does:
When you double-click `start.bat` (or run it from the command line), it automatically:
1. Opens a new Command Prompt window, navigates to `ADS/frontend`, and runs `npm run dev` to start the Next.js frontend on **port 3000**.
2. Opens a second Command Prompt window, activates the Python virtual environment (`venv`), and runs `uvicorn apps.api.main:app` to start the FastAPI backend on **port 8001**.

### How to use it:
Simply double click `start.bat` in the Windows File Explorer, or type `start.bat` in your terminal at the root of the project.

Once both servers are running:
- **Frontend Dashboard**: http://localhost:3000
- **Backend API Docs (Swagger UI)**: http://localhost:8001/docs

> **Tip:** If you make changes to the Python code, `uvicorn` will automatically hot-reload the backend server. The same goes for Next.js on the frontend!
