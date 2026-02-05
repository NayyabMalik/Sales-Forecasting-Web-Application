# Sales Forecasting Web Application

This repository contains a Flask-based web application for sales forecasting, allowing users to upload time series data (CSV, Excel, JSON, XML, PDF, or text files), generate statistical visualizations, and predict future sales using ARIMA, Prophet, and LSTM models. The application supports user authentication (registration, login, logout), file management, and interactive forecasting with customizable parameters (e.g., forecast periods, confidence intervals). Visualizations include histograms, box plots, correlation heatmaps, and forecast comparisons, enhancing data analysis and prediction interpretability.



## ✨ Features

- **Secure User Authentication** — Register / Login / Logout with bcrypt hashing + MongoDB
- **Multi-format File Upload** — CSV, Excel (.xls/.xlsx), JSON, XML, PDF, TXT (max 16 MB)
- **Rich Automatic EDA & Visualizations**  
  Histograms, box plots, bar charts (top categories), correlation heatmaps, time series decomposition, distribution plots
- **Multi-model Time Series Forecasting**  
  - ARIMA (statsmodels)  
  - Prophet (with confidence intervals)  
  - Lightweight LSTM (Keras/TensorFlow)  
  - Ensemble average visualization
- **Interactive Forecast Configuration**  
  Choose date column, target value column, forecast horizon (3–24 months), models, confidence level
- **Beautiful Forecast Visualizations**  
  - Historical + forecast comparison  
  - Prophet forecast + uncertainty band  
  - Ensemble overlay  
  - Model error distribution
- **AI-powered Forecast Explainer Chat**  
  GPT-4o-mini (via OpenRouter + LangChain) — explains forecasts, confidence intervals, trends in plain language
- **Observability** — LangSmith tracing for LLM calls (prompts, latency, cost)
- **Smart Session & File Management**  
  - MongoDB-backed sessions (24h lifetime)  
  - Per-user isolated file storage with 24-hour auto-cleanup
- **Download Results** — Original files + generated forecast CSV

## 🛠 Tech Stack

| Layer             | Technologies                                      |
|-------------------|---------------------------------------------------|
| Backend           | Flask, Flask-Session, Flask-CORS                  |
| Database          | MongoDB (users + sessions)                        |
| Authentication    | bcrypt                                            |
| Data Processing   | pandas, numpy, scikit-learn                       |
| Time Series       | statsmodels (ARIMA), prophet, tensorflow/keras (LSTM) |
| Visualization     | matplotlib, seaborn                               |
| LLM Integration   | LangChain + OpenRouter (gpt-4o-mini)              |
| LLM Observability | LangSmith                                         |
| File Handling     | PyPDF2, werkzeug                                  |
| Frontend          | Jinja2 templates, vanilla JS + custom CSS         |


sales-forecasting/
├── app.py                     # Flask application with forecasting and visualization logic
├── requirements.txt           # Python dependencies
├── templates/                # HTML templates for the web interface
│   ├── login.html            # User login/registration page
│   ├── logout.html           # Logout page
│   ├── result.html           # Results display page (for visualizations/forecasts)
│   ├── register.html         # Alias for login.html with registration form
│   ├── file_upload.html      # Main upload interface
│   ├── view_files.html       # File listing and visualization page
│   ├── prediction.html       # Forecasting interface
│   ├── start.html            # Landing page
├── static/                   # Static files for the web interface
│   ├── css/                 # CSS stylesheets
│   │   ├── start.css
│   │   ├── prediction.css
│   │   ├── login.css
│   │   ├── logout.css
│   │   ├── style.css
│   │   ├── result.css
│   │   ├── register.css
│   │   ├── view_files.css
│   │   ├── file_upload.css
│   ├── js/                  # JavaScript files
│   │   ├── file_upload.js
│   │   ├── prediction.js
|   |   ├── view_files.js  
├── uploads/                  # User-uploaded files (temporary, user-specific)
├── static/images/            # Generated visualizations (user-specific)
├── README.md                # This file
├── LICENSE                  # MIT License

## Setup Instructions

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-username/sales-forecasting.git
   cd sales-forecasting
   ```

2. **Install Dependencies**: Install the required Python libraries listed in `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

   Key libraries include: `flask`, `flask-session`, `pymongo`, `bcrypt`, `pandas`, `matplotlib`, `seaborn`, `statsmodels`, `prophet`, `tensorflow`, `numpy`, `scikit-learn`, `PyPDF2`.

3. **Set Up MongoDB**:

   - Install and run MongoDB locally (`mongodb://localhost:27017/`).
   - Ensure the `forecast_app` database and `users` and `sessions` collections are accessible.
   - Alternatively, configure a remote MongoDB instance by updating `app.config['SESSION_MONGODB']` in `app.py`.

4. **Run the Flask Application**:

   ```bash
   python app.py
   ```

   Access the application at `http://localhost:5000` in your browser.

5. **Prepare Data**:

   - Upload CSV or Excel files with at least one date column (e.g., `Date`, or `Year`/`Month`/`Day`) and one numeric column (e.g., `Sales`).
   - Example dataset: [Kaggle Sales Dataset](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting).

## Usage

1. **Access the Application**:

   - Visit `http://localhost:5000` to reach the landing page (`start.html`).
   - Navigate to `/login` to register or log in.

2. **Register/Login**:

   - Register via `/register` (or `/login` with the registration form).
   - Log in with your credentials. Sessions are stored in MongoDB and persist for 24 hours.

3. **Upload Files**:

   - Go to `/upload_page` (`file_upload.html`) to upload CSV, Excel, JSON, XML, PDF, or TXT files.
   - Files are stored in `uploads/<user_id>/` and are automatically deleted after 24 hours.

4. **View Files and Visualizations**:

   - Access `/files` (`view_files.html`) to see uploaded files, their headers, statistical summaries, and visualizations (e.g., histograms, box plots, correlation heatmaps).

5. **Generate Forecasts**:

   - Select a file at `/predict/<filename>` (`prediction.html`).
   - Choose a date column, value column, forecast period (3, 6, 9, or 12 months), models (ARIMA, Prophet, LSTM), and confidence interval (default: 95%).
   - View forecast results, including a table and visualizations (e.g., forecast comparison, Prophet confidence intervals).

6. **Download Results**:

   - Download forecast data as a CSV file via `/download_forecast`.
   - Download original files via `/download/<filename>`.

7. **Clear Files**:

   - Use `/clear` to delete all user-specific files and visualizations.

8. **Logout**:

   - Log out via `/logout`, which clears the session and user files.

## Configuration Parameters (app.py)

- **ALLOWED_EXTENSIONS**: File types allowed for upload (`csv`, `xls`, `xlsx`, `txt`, `json`, `xml`, `pdf`).
- **MAX_FILE_SIZE**: Maximum file size (16MB).
- **SESSION_PERMANENT**: Sessions last 24 hours.
- **Forecast Parameters**:
  - `periods`: Forecast horizon (3, 6, 9, or 12 months).
  - `models`: Forecasting models (`ARIMA`, `Prophet`, `LSTM`).
  - `confidence_interval`: Confidence level for Prophet forecasts (default: 0.95).
- **LSTM Parameters**:
  - `look_back`: 3 time steps for sequence modeling.
  - `epochs`: 20 training epochs.
  - `batch_size`: 1 for training.

## Example

To forecast sales from a CSV file (`sales.csv`) with columns `Date` and `Sales`:

1. Upload `sales.csv` via `/upload_page`.
2. Navigate to `/files` to view the file and its visualizations.
3. Go to `/predict/sales.csv`, select `Date` as the date column, `Sales` as the value column, 6 months as the period, and all models (`ARIMA`, `Prophet`, `LSTM`).
4. View the forecast table and visualizations (e.g., comparison plot).
5. Download the forecast as `forecast.csv` via `/download_forecast`.

## Future Improvements

- **Progress Tracking**: Implement `/progress` (`progress.html`) to show real-time forecasting progress (similar to `progress_data` in `fire_detection.py`).
- **Model Optimization**: Fine-tune LSTM hyperparameters (e.g., `look_back`, layers) for better accuracy.
- **Additional Visualizations**: Add interactive plots using Plotly for the web interface.
- **Error Handling**: Enhance validation for date and numeric columns to prevent NaN issues.
- **Security**: Enable `SESSION_COOKIE_SECURE` and deploy with HTTPS in production.

## Notes

- **MongoDB**: Ensure MongoDB is running and accessible. Update connection settings in `app.py` for remote databases.
- **File Management**: User files and visualizations are stored in `uploads/<user_id>/` and `static/images/<user_id>/`, with automatic cleanup after 24 hours.
- **Permissions**: The code and referenced lab manuals are shared with permission for educational purposes. Contact at nayyabm16@gmail.com for access to original course materials.



