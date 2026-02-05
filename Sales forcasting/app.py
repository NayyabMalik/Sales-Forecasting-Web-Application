import os
import pandas as pd
import json
import xml.etree.ElementTree as ET
import uuid
import logging
import secrets
import shutil
from datetime import datetime, timedelta
import atexit
import hashlib
import io

from flask import (
    Flask, render_template, request, jsonify, redirect,
    url_for, send_from_directory, flash, session, Response, send_file
)
from flask_session import Session
from flask_cors import CORS
from werkzeug.utils import secure_filename

from PyPDF2 import PdfReader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import pymongo
import bcrypt
from functools import wraps

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv(override=True)


app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

# MongoDB client
mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
db = mongo_client["forecast_app"]
users_collection = db["users"]

# Flask-Session with MongoDB
app.config.update(
    SESSION_TYPE                  = "mongodb",
    SESSION_MONGODB               = mongo_client,
    SESSION_MONGODB_DB            = "forecast_app",
    SESSION_MONGODB_COLLECT       = "sessions",
    SESSION_PERMANENT             = True,
    PERMANENT_SESSION_LIFETIME    = timedelta(days=1),
    SESSION_COOKIE_SECURE         = False,           # Important for http://localhost
    SESSION_COOKIE_SAMESITE       = 'Lax',           # Works well locally
    SESSION_COOKIE_HTTPONLY       = True,
    SESSION_COOKIE_NAME           = 'forecast_sid',  # Easier to spot in dev tools
)

Session(app)

# Enable CORS with credentials support
CORS(app, supports_credentials=True)

# Folders
UPLOAD_FOLDER = 'uploads/'
STATIC_FOLDER = 'static/images/'
ALLOWED_EXTENSIONS = {'csv', 'xls', 'xlsx', 'txt', 'json', 'xml', 'pdf'}
MAX_FILE_SIZE = 16 * 1024 * 1024

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO)

# Add this temporary debug code right after load_dotenv()
load_dotenv()

print("=" * 60)
print(f"From .env file: {os.getenv('OPENROUTER_API_KEY', 'NOT FOUND')[:20]}...")
print(f"From system env: {os.environ.get('OPENROUTER_API_KEY', 'NOT FOUND')[:20]}...")
print("=" * 60)


# Memory (per session)
def get_memory():
    if 'chat_memory' not in session:
        session['chat_memory'] = []
    return session['chat_memory']



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_user_upload_folder():
    if 'username' not in session:
        flash('Session expired. Please log in.')
        return redirect(url_for('login'))
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
        logging.info(f"Generated new user_id: {session['user_id']}")
    if 'upload_session_id' not in session:
        session['upload_session_id'] = str(uuid.uuid4())
        logging.info(f"Generated new upload_session_id: {session['upload_session_id']}")
    user_folder = os.path.join(app.config['UPLOAD_FOLDER'], session['user_id'], session['upload_session_id'])
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)
    return user_folder

def get_user_static_folder():
    if 'username' not in session:
        flash('Session expired. Please log in.')
        return redirect(url_for('login'))
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
        logging.info(f"Generated new user_id: {session['user_id']}")
    if 'upload_session_id' not in session:
        session['upload_session_id'] = str(uuid.uuid4())
        logging.info(f"Generated new upload_session_id: {session['upload_session_id']}")
    user_folder = os.path.join(app.config['STATIC_FOLDER'], session['user_id'], session['upload_session_id'])
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)
    return user_folder

def cleanup_user_files():
    now = datetime.now()
    for folder in [UPLOAD_FOLDER, STATIC_FOLDER]:
        for user_folder in os.listdir(folder):
            user_path = os.path.join(folder, user_folder)
            if not os.path.isdir(user_path):
                continue
            for session_folder in os.listdir(user_path):
                session_path = os.path.join(user_path, session_folder)
                try:
                    mod_time = datetime.fromtimestamp(os.path.getmtime(session_path))
                    if now - mod_time > timedelta(hours=24):
                        shutil.rmtree(session_path, ignore_errors=True)
                        logging.info(f"Cleaned up old session folder: {session_path}")
                except:
                    continue

atexit.register(cleanup_user_files)

def get_file_header(filepath, filename):
    ext = filename.rsplit('.', 1)[1].lower()
    header = ""
    try:
        if ext in ['csv', 'xls', 'xlsx']:
            df = pd.read_csv(filepath) if ext == 'csv' else pd.read_excel(filepath)
            header = df.head().to_html(classes='table table-striped', index=False)
        elif ext == 'json':
            with open(filepath, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    header = json.dumps({k: v for k, v in list(data.items())[:5]}, indent=2)
                elif isinstance(data, list):
                    header = json.dumps(data[:5], indent=2)
        elif ext == 'xml':
            tree = ET.parse(filepath)
            root = tree.getroot()
            header = ET.tostring(root, encoding='unicode', method='xml')[:500] + '...'
        elif ext == 'pdf':
            reader = PdfReader(filepath)
            if reader.pages:
                page = reader.pages[0]
                text = page.extract_text()
                header = text[:500] + '...' if text else "No text found in PDF."
        elif ext == 'txt':
            with open(filepath, 'r') as f:
                lines = ''.join([next(f) for _ in range(5)])
                header = f"<pre>{lines}</pre>"
    except Exception as e:
        header = f"Could not read file: {e}"
    return header

def get_file_hash(filepath):
    """Compute SHA-256 hash of a file to check if it has changed."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def generate_statistics_and_visualizations(filepath, filename):
    ext = filename.rsplit('.', 1)[1].lower()
    stats_html = ""
    visualizations = []
    try:
        if ext in ['csv', 'xls', 'xlsx']:
            df = pd.read_csv(filepath) if ext == 'csv' else pd.read_excel(filepath)
            # Downsample large datasets for visualization
            if len(df) > 1000:
                df = df.sample(n=1000, random_state=42)
                logging.info(f"Downsampled {filename} to 1000 rows for visualization")
            desc = df.describe(include='all').to_html(classes='table table-bordered')
            stats_html = desc
            visualizations = create_visualizations(df, filename, filepath)
        elif ext == 'json':
            with open(filepath, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    df = pd.json_normalize(data)
                    if len(df) > 1000:
                        df = df.sample(n=1000, random_state=42)
                        logging.info(f"Downsampled {filename} to 1000 rows for visualization")
                    desc = df.describe(include='all').to_html(classes='table table-bordered')
                    stats_html = desc
                    visualizations = create_visualizations(df, filename, filepath)
                else:
                    stats_html = "JSON data is not in a list format suitable for analysis."
    except Exception as e:
        stats_html = f"Could not generate statistics: {e}"
    return stats_html, visualizations

def create_visualizations(df, filename, filepath):
    visualizations = []
    user_static_folder = get_user_static_folder()
    if isinstance(user_static_folder, Response):
        return visualizations
    
    # Cache visualizations based on file hash
    file_hash = get_file_hash(filepath)
    cache_file = os.path.join(user_static_folder, f"{filename}_{file_hash}_viz_cache.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                visualizations = json.load(f)
                logging.info(f"Loaded cached visualizations for {filename}")
                return visualizations
        except:
            logging.warning(f"Failed to load visualization cache for {filename}")

    try:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()

        # Limit number of columns to process
        numeric_cols = numeric_cols[:5]  # Process up to 5 numeric columns
        categorical_cols = categorical_cols[:3]  # Process up to 3 categorical columns

        # Histogram for numeric columns
        for col in numeric_cols:
            plt.figure(figsize=(8, 5))
            sns.histplot(df[col].dropna(), kde=True, bins=30, color='skyblue')
            plt.title(f'Distribution of {col}')
            img_filename = f"hist_{uuid.uuid4()}.png"
            plt.savefig(os.path.join(user_static_folder, img_filename), bbox_inches='tight')
            plt.close()
            visualizations.append({
                'url': url_for('static', filename=f'images/{session["user_id"]}/{session["upload_session_id"]}/{img_filename}'),
                'title': f'Distribution of {col}',
                'description': f'Histogram showing frequency distribution of {col}'
            })

        # Box plot for numeric columns
        for col in numeric_cols:
            plt.figure(figsize=(8, 5))
            sns.boxplot(x=df[col], color='lightgreen')
            plt.title(f'Box Plot of {col}')
            img_filename = f"box_{uuid.uuid4()}.png"
            plt.savefig(os.path.join(user_static_folder, img_filename), bbox_inches='tight')
            plt.close()
            visualizations.append({
                'url': url_for('static', filename=f'images/{session["user_id"]}/{session["upload_session_id"]}/{img_filename}'),
                'title': f'Spread of {col}',
                'description': f'Box plot showing quartiles and outliers for {col}'
            })

        # Bar plot for categorical columns
        for col in categorical_cols:
            plt.figure(figsize=(10, 6))
            top_categories = df[col].value_counts().nlargest(10)
            sns.barplot(x=top_categories.values, y=top_categories.index, hue=top_categories.index, palette='viridis', legend=False)
            plt.title(f'Top 10 Categories in {col}')
            plt.xlabel('Count')
            plt.ylabel(col)
            img_filename = f"bar_{uuid.uuid4()}.png"
            plt.savefig(os.path.join(user_static_folder, img_filename), bbox_inches='tight')
            plt.close()
            visualizations.append({
                'url': url_for('static', filename=f'images/{session["user_id"]}/{session["upload_session_id"]}/{img_filename}'),
                'title': f'Top {col} Categories',
                'description': f'Most frequent values in {col} column'
            })

        # Correlation heatmap for numeric columns
        if len(numeric_cols) >= 2:
            plt.figure(figsize=(10, 8))
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title('Correlation Between Variables')
            img_filename = f"heatmap_{uuid.uuid4()}.png"
            plt.savefig(os.path.join(user_static_folder, img_filename), bbox_inches='tight')
            plt.close()
            visualizations.append({
                'url': url_for('static', filename=f'images/{session["user_id"]}/{session["upload_session_id"]}/{img_filename}'),
                'title': 'Correlation Heatmap',
                'description': 'Relationships between numeric variables'
            })

        # Time series decomposition if applicable
        if len(datetime_cols) > 0 and len(numeric_cols) > 0:
            time_col = datetime_cols[0]
            num_col = numeric_cols[0]
            try:
                ts = df.set_index(time_col)[num_col].dropna()
                if len(ts) > 30:
                    result = seasonal_decompose(ts, model='additive', period=min(30, len(ts)//2))
                    result.plot()
                    plt.suptitle(f'Time Series Decomposition of {num_col}', y=1.02)
                    img_filename = f"decompose_{uuid.uuid4()}.png"
                    plt.savefig(os.path.join(user_static_folder, img_filename), bbox_inches='tight')
                    plt.close()
                    visualizations.append({
                        'url': url_for('static', filename=f'images/{session["user_id"]}/{session["upload_session_id"]}/{img_filename}'),
                        'title': f'Time Components: {num_col}',
                        'description': 'Trend, seasonal, and residual components'
                    })
            except Exception as e:
                logging.warning(f"Couldn't create decomposition plot: {e}")

        # Cache visualizations
        try:
            with open(cache_file, 'w') as f:
                json.dump(visualizations, f)
                logging.info(f"Cached visualizations for {filename}")
        except Exception as e:
            logging.warning(f"Failed to cache visualizations for {filename}: {e}")

    except Exception as e:
        logging.error(f"Error creating visualizations: {e}")
    
    return visualizations

def train_arima(df, date_col, value_col, periods):
    try:
        series = df.set_index(date_col)[value_col].dropna()
        if len(series) < 2:
            logging.warning("Not enough data for ARIMA (minimum 2 points required)")
            return None, None, None
        model = ARIMA(series, order=(1, 1, 1))
        model_fit = model.fit()
        
        forecast = model_fit.forecast(steps=periods)
        forecast_dates = pd.date_range(start=series.index[-1], periods=periods + 1, freq='ME')[1:]
        
        train_size = int(len(series) * 0.8)
        train, test = series[:train_size], series[train_size:]
        if len(test) > 0:
            model = ARIMA(train, order=(1, 1, 1))
            model_fit = model.fit()
            predictions = model_fit.forecast(steps=len(test))
            errors = test.values - predictions
        else:
            errors = np.array([])
        
        return forecast, forecast_dates, errors
    except Exception as e:
        logging.error(f"ARIMA error: {e}")
        return None, None, None

def train_prophet(df, date_col, value_col, periods, confidence_interval=0.95):
    try:
        prophet_df = df[[date_col, value_col]].rename(columns={date_col: 'ds', value_col: 'y'})
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
        prophet_df = prophet_df.dropna()
        if len(prophet_df) < 2:
            logging.warning("Not enough data for Prophet (minimum 2 points required)")
            return None, None, None, None
        
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True,
                       interval_width=confidence_interval)
        model.fit(prophet_df)
        
        future = model.make_future_dataframe(periods=periods, freq='ME')
        forecast = model.predict(future)
        
        train_size = int(len(prophet_df) * 0.8)
        train, test = prophet_df[:train_size], prophet_df[train_size:]
        if len(test) > 0:
            model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True,
                           interval_width=confidence_interval)
            model.fit(train)
            future_test = model.make_future_dataframe(periods=len(test), freq='ME')
            predictions = model.predict(future_test)
            errors = test['y'].values - predictions['yhat'].tail(len(test)).values
        else:
            errors = np.array([])
        
        return (forecast['yhat'].tail(periods), forecast['ds'].tail(periods),
                forecast[['yhat_lower', 'yhat_upper']].tail(periods), errors)
    except Exception as e:
        logging.error(f"Prophet error: {e}")
        return None, None, None, None

def train_lstm(df, date_col, value_col, periods):
    try:
        series = df.set_index(date_col)[value_col].dropna().values.astype(float)
        if len(series) < 20:
            return None, None, None

        # ── Use most recent data only ────────────────────────────────
        max_train_points = 1200           # adjust 800–2000 depending on patience
        if len(series) > max_train_points:
            series = series[-max_train_points:]

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(series.reshape(-1, 1))

        look_back = 12                    # good for monthly, change to 30 for daily
        X, y = [], []
        for i in range(len(scaled) - look_back):
            X.append(scaled[i:i+look_back])
            y.append(scaled[i+look_back])

        if len(X) < 30:
            return None, None, None

        X, y = np.array(X), np.array(y)

        model = Sequential([
            LSTM(32, activation='tanh', input_shape=(look_back, 1)),  # smaller
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        # Very light training
        model.fit(X, y, epochs=10, batch_size=16, verbose=0)

        # forecasting code remains the same...

    except Exception as e:
        logging.error(f"LSTM failed: {str(e)}")
        return None, None, None
    
def generate_forecast(df, date_col, value_col, periods, models, confidence_interval=0.95):
    """
    Generate time series forecasts and visualizations using selected models.
    Returns a dictionary with forecast table HTML, forecast DataFrame, list of visualizations, and optional error.
    """
    forecasts = {'ARIMA': None, 'Prophet': None, 'LSTM': None}
    errors_dict = {'ARIMA': None, 'Prophet': None, 'LSTM': None}
    prophet_ci = None
    dates = None

    user_static_folder = get_user_static_folder()
    if isinstance(user_static_folder, Response):
        return {
            'forecast_table': None,
            'forecast_data': pd.DataFrame(),
            'visualizations': [],
            'error': 'Session expired. Please log in.'
        }

    visualizations = []

    # ── Historical data ──────────────────────────────────────────────────────
    historical = df.set_index(date_col)[value_col].dropna()
    logging.info(f"Historical data points: {len(historical)}")

    # Minimum reasonable amount of data
    min_required = 10 if 'LSTM' in models else 5
    if len(historical) < min_required:
        return {
            'forecast_table': None,
            'forecast_data': pd.DataFrame(),
            'visualizations': [],
            'error': f"Not enough data points for reliable forecasting (got {len(historical)}, need ≥ {min_required})"
        }

    # ── Train selected models ────────────────────────────────────────────────
    if 'ARIMA' in models:
        arima_fc, arima_dates, arima_err = train_arima(df, date_col, value_col, periods)
        forecasts['ARIMA'] = arima_fc
        errors_dict['ARIMA'] = arima_err
        if arima_dates is not None:
            dates = arima_dates

    if 'Prophet' in models:
        prop_fc, prop_dates, prop_ci, prop_err = train_prophet(
            df, date_col, value_col, periods, confidence_interval
        )
        forecasts['Prophet'] = prop_fc
        errors_dict['Prophet'] = prop_err
        prophet_ci = prop_ci
        if dates is None and prop_dates is not None:
            dates = prop_dates

    if 'LSTM' in models:
        lstm_fc, lstm_dates, lstm_err = train_lstm(df, date_col, value_col, periods)
        forecasts['LSTM'] = lstm_fc
        errors_dict['LSTM'] = lstm_err
        if dates is None and lstm_dates is not None:
            dates = lstm_dates

    # Fallback dates if no model produced valid dates
    if dates is None:
        last_valid_date = historical.index[-1] if len(historical) > 0 else pd.Timestamp.now()
        dates = pd.date_range(start=last_valid_date, periods=periods + 1, freq='ME')[1:]

    # ── Convert all forecasts to NumPy arrays (prevents KeyError & truth value errors) ──
    forecasts_np = {}
    for model_name, fc in forecasts.items():
        if fc is not None:
            if isinstance(fc, pd.Series):
                arr = fc.to_numpy()
            elif isinstance(fc, (list, tuple)):
                arr = np.array(fc)
            else:
                arr = np.asarray(fc).flatten()
            
            # Only keep if length matches expected forecast horizon
            if len(arr) == periods:
                forecasts_np[model_name] = arr
            else:
                logging.warning(f"Length mismatch for {model_name} forecast: got {len(arr)}, expected {periods}")
                forecasts_np[model_name] = None
        else:
            forecasts_np[model_name] = None

    # Prophet confidence intervals (also to NumPy)
    prophet_ci_np = None
    if prophet_ci is not None and len(prophet_ci) > 0:
        prophet_ci_np = {
            'lower': prophet_ci['yhat_lower'].to_numpy(),
            'upper': prophet_ci['yhat_upper'].to_numpy()
        }

    # ── Visualizations ───────────────────────────────────────────────────────
    # Only proceed if we have at least one valid forecast array
    has_valid_forecast = any(
        arr is not None and len(arr) == len(dates)
        for arr in forecasts_np.values()
    )

    if has_valid_forecast:
        # 1. Comparison plot
        plt.figure(figsize=(12, 6))
        plt.plot(historical.index, historical, label='Historical', color='blue', linewidth=2.2)

        for model_name, arr in forecasts_np.items():
            if arr is not None and len(arr) == len(dates):
                plt.plot(dates, arr, label=model_name, linestyle='--', linewidth=1.8)

        plt.title(f'Forecast Comparison – {value_col} (Next {periods} periods)')
        plt.xlabel('Date')
        plt.ylabel(value_col)
        plt.legend()
        plt.grid(True, alpha=0.3)
        img1 = f"comparison_{uuid.uuid4()}.png"
        plt.savefig(os.path.join(user_static_folder, img1), bbox_inches='tight', dpi=120)
        plt.close()

        visualizations.append({
            'url': url_for('static', filename=f'images/{session["user_id"]}/{session["upload_session_id"]}/{img1}'),
            'title': f'Forecast Comparison: {value_col}',
            'description': 'All selected models vs historical data'
        })

        # 2. Prophet with confidence interval (if available)
        if forecasts_np.get('Prophet') is not None and prophet_ci_np is not None:
            plt.figure(figsize=(12, 6))
            plt.plot(historical.index, historical, label='Historical', color='blue', linewidth=2.2)
            plt.plot(dates, forecasts_np['Prophet'], label='Prophet', color='red', linestyle='--', linewidth=1.8)

            plt.fill_between(
                dates,
                prophet_ci_np['lower'],
                prophet_ci_np['upper'],
                color='red', alpha=0.15, label=f'{int(confidence_interval*100)}% CI'
            )

            plt.title(f'Prophet Forecast + Confidence Interval – {value_col}')
            plt.xlabel('Date')
            plt.ylabel(value_col)
            plt.legend()
            plt.grid(True, alpha=0.3)
            img2 = f"prophet_ci_{uuid.uuid4()}.png"
            plt.savefig(os.path.join(user_static_folder, img2), bbox_inches='tight', dpi=120)
            plt.close()

            visualizations.append({
                'url': url_for('static', filename=f'images/{session["user_id"]}/{session["upload_session_id"]}/{img2}'),
                'title': 'Prophet Forecast + Confidence Interval',
                'description': 'Prophet prediction with uncertainty band'
            })

        # 3. Ensemble average overlay
        plt.figure(figsize=(12, 6))
        plt.plot(historical.index, historical, label='Historical', color='blue', linewidth=2.2)

        avg_forecast = []
        for i in range(len(dates)):
            valid_values = [
                forecasts_np[m][i]
                for m in forecasts_np
                if forecasts_np[m] is not None and i < len(forecasts_np[m])
            ]
            mean_value = np.nanmean(valid_values) if valid_values else np.nan
            avg_forecast.append(mean_value)

        plt.plot(dates, avg_forecast, label='Ensemble Average', color='purple',
                 linestyle='--', linewidth=2.5)

        plt.axvspan(historical.index[-1], dates[-1], color='purple', alpha=0.08, label='Forecast Region')

        plt.title(f'Historical + Ensemble Forecast – {value_col}')
        plt.xlabel('Date')
        plt.ylabel(value_col)
        plt.legend()
        plt.grid(True, alpha=0.3)
        img3 = f"ensemble_overlay_{uuid.uuid4()}.png"
        plt.savefig(os.path.join(user_static_folder, img3), bbox_inches='tight', dpi=120)
        plt.close()

        visualizations.append({
            'url': url_for('static', filename=f'images/{session["user_id"]}/{session["upload_session_id"]}/{img3}'),
            'title': 'Ensemble Forecast Overlay',
            'description': 'Historical trend + average of selected models'
        })

        # 4. Error distribution (only if we have errors)
        plt.figure(figsize=(10, 6))
        plotted_any = False
        for model_name, err_data in errors_dict.items():
            if err_data is not None and hasattr(err_data, '__len__') and len(err_data) > 0:
                sns.histplot(err_data, label=model_name, kde=True, stat='density', alpha=0.45)
                plotted_any = True

        if plotted_any:
            plt.title('Model Error Distribution (Validation Set)')
            plt.xlabel('Error')
            plt.ylabel('Density')
            plt.legend()
            img4 = f"error_distribution_{uuid.uuid4()}.png"
            plt.savefig(os.path.join(user_static_folder, img4), bbox_inches='tight', dpi=120)
            plt.close()

            visualizations.append({
                'url': url_for('static', filename=f'images/{session["user_id"]}/{session["upload_session_id"]}/{img4}'),
                'title': 'Model Error Distribution',
                'description': 'Validation errors for each model'
            })
        else:
            plt.close()

    # ────────────────────────────────────────────────
    # Build HTML table
    # ────────────────────────────────────────────────
    table_data = {'Date': [d.strftime('%Y-%m-%d') for d in dates]}

    for model in ['ARIMA', 'Prophet', 'LSTM']:
        if forecasts_np.get(model) is not None:
            values = forecasts_np[model]
            table_data[model] = [
                f"{float(v):,.2f}" if not np.isnan(v) else 'N/A'
                for v in values
            ]
        else:
            table_data[model] = ['N/A'] * len(dates)

    forecast_df = pd.DataFrame(table_data)
    forecast_table = forecast_df.to_html(
        classes='table table-bordered forecast-table',
        index=False,
        escape=False
    )

    return {
        'forecast_table': forecast_table,
        'forecast_data': forecast_df,
        'visualizations': visualizations,
        'error': None
    }


    
@app.route('/')
def start():
    return render_template('start.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'username' in session:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = users_collection.find_one({'username': username})
        if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
            # Create a new session to prevent session fixation
            session.clear()  # Clear any existing session data
            session.permanent = True  # Make session permanent
            session['username'] = username
            session['user_id'] = str(uuid.uuid4())
            logging.info(f"User {username} logged in with user_id: {session['user_id']}")
            flash('Login successful!')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'username' in session:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if users_collection.find_one({'username': username}):
            flash('Username already exists')
        else:
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            users_collection.insert_one({'username': username, 'password': hashed_password})
            flash('Registration successful! Please log in.')
            return redirect(url_for('login'))
    
    return render_template('login.html', register=True)

@app.route('/logout')
def logout():
    if 'username' in session:
        username = session.pop('username', None)
        user_id = session.pop('user_id', None)
        session.pop('upload_session_id', None)
        for folder in [UPLOAD_FOLDER, STATIC_FOLDER]:
            user_folder = os.path.join(folder, user_id)
            if os.path.exists(user_folder):
                shutil.rmtree(user_folder, ignore_errors=True)
        session.clear()  # Clear all session data
        flash('You have been logged out')
    return render_template('logout.html')


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            flash('Please log in to access this page')
            return redirect(url_for('login'))
        # Check if user still exists in database
        user = users_collection.find_one({'username': session['username']})
        if not user:
            session.clear()
            flash('Session invalid. Please log in again.')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/upload_page')
@login_required
def index():
    cleanup_user_files()
    return render_template('file_upload.html')



# Prompt template with memory placeholder (for future chat history)


# Simple chain (no memory yet — can add later)
def get_llm():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENROUTER_API_KEY in .env")

    # Set OPENAI_API_KEY temporarily for LangChain validation
    os.environ["OPENAI_API_KEY"] = api_key

    return ChatOpenAI(
        model="openai/gpt-4o-mini",
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        temperature=0.7,
        max_tokens=800,
        request_timeout=25,
        default_headers={
            "HTTP-Referer": "http://localhost:5000",
            "X-Title": "Sales Forecast App"
        }
    )
# System prompt (shared)
system_prompt = """You are a helpful financial forecasting assistant for a sales prediction app.
You explain time-series forecasts, ARIMA, Prophet, LSTM models, confidence intervals, ensemble averages, etc. in clear, simple, beginner-friendly language.

Current forecast context (use when relevant):
- File: {filename}
- Forecast period: {periods} months
- Models used: {models}
- Target column: {value_col}
- Date column: {date_col}

Always be polite, concise, accurate.
If you don't have enough context or don't know something, say so honestly.
Refer back to previous messages in this conversation when needed."""


@app.route('/chat_explain', methods=['POST'])
@login_required
def chat_explain():
    print("→ chat_explain called | username:", session.get('username'))
    print("→ session contents:", dict(session))

    try:
        data = request.json
        user_message = data.get('message', '').strip()
        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        llm = get_llm()  # must have streaming=True
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful financial forecasting assistant for a sales prediction app.
You explain time-series forecasts, ARIMA, Prophet, LSTM models, confidence intervals, ensemble averages, etc. in clear, simple, beginner-friendly language.

Current forecast context (use when relevant):
- File: {filename}
- Forecast period: {periods} months
- Models used: {models}
- Target column: {value_col}
- Date column: {date_col}

Always be polite, concise, accurate.
If you don't have enough context or don't know something, say so honestly.
Refer back to previous messages in this conversation when needed."""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])

        chain = prompt | llm

        context = {
            "filename": session.get('last_filename', 'unknown'),
            "periods": session.get('last_periods', 'unknown'),
            "models": ", ".join(session.get('last_models', [])) or "none",
            "value_col": session.get('last_value_col', 'unknown'),
            "date_col": session.get('last_date_col', 'unknown'),
        }

        history = get_memory()
        messages = []
        for msg in history:
            if msg['role'] == 'user':
                messages.append(HumanMessage(content=msg['content']))
            else:
                messages.append(AIMessage(content=msg['content']))

        full_input = {
            **context,
            "question": user_message,
            "history": messages
        }

        # ── STREAM instead of invoke ─────────────────────
        print(">>> streaming LLM")
        answer_chunks = []

        for chunk in chain.stream(full_input):
            if chunk.content:
                answer_chunks.append(chunk.content)

        answer = "".join(answer_chunks).strip()
        print(">>> streaming completed")

        # Save history
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": answer})
        session['chat_memory'] = history[-20:]
        session.modified = True

        return jsonify({"response": answer})

    except Exception as e:
        logging.exception("Chat error")
        return jsonify({"error": str(e)}), 500

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    session['upload_session_id'] = str(uuid.uuid4())
    folder = get_user_upload_folder()
    files = request.files.getlist('file')
    count = 0
    for file in files:
        if file and allowed_file(file.filename):
            fn = secure_filename(file.filename)
            file.save(os.path.join(folder, fn))
            count += 1
    flash(f'Uploaded {count} file(s)' if count else 'No valid files uploaded', 'info')
    return redirect(url_for('view_files'))

@app.route('/files')
@login_required
def view_files():
    folder = get_user_upload_folder()
    files_info = []
    for fn in os.listdir(folder):
        path = os.path.join(folder, fn)
        header = get_file_header(path, fn)
        stats, viz = generate_statistics_and_visualizations(path, fn)
        files_info.append({
            'name': fn,
            'header': header,
            'stats': stats,
            'visualizations': viz
        })
    return render_template('view_files.html', files=files_info)

@app.route('/predict/<filename>', methods=['GET', 'POST'])
@login_required
def generate_prediction(filename):
    """
    Handle GET: Show forecast configuration form
    Handle POST: Process form, clean data, run forecast, save results & context for chatbot
    """
    folder = get_user_upload_folder()
    filepath = os.path.join(folder, filename)
    
    if not os.path.exists(filepath):
        flash(f'File not found: {filename}', 'danger')
        return redirect(url_for('view_files'))

    try:
        ext = filename.rsplit('.', 1)[1].lower()
        if ext not in ['csv', 'xls', 'xlsx']:
            flash('Prediction only supports CSV and Excel files', 'warning')
            return redirect(url_for('view_files'))

        # Load data
        df = pd.read_csv(filepath) if ext == 'csv' else pd.read_excel(filepath)
        logging.info(f"Loaded {filename} — shape: {df.shape}, columns: {list(df.columns)}")

        # ── Improved date column detection ───────────────────────────────────
        date_cols = []
        probable_date_col = None

        # 1. Already datetime columns
        dt_cols = df.select_dtypes(include='datetime').columns.tolist()
        if dt_cols:
            probable_date_col = dt_cols[0]
            date_cols.extend(dt_cols)

        # 2. Name-based detection (most reliable)
        date_keywords = ['date', 'sale_date', 'transaction_date', 'order_date', 'time', 'period']
        name_matches = [c for c in df.columns if any(k in c.lower() for k in date_keywords)]
        for col in name_matches:
            if col not in date_cols:
                try:
                    temp = pd.to_datetime(df[col], errors='coerce', dayfirst=False)
                    if temp.notna().mean() >= 0.65:
                        df[col] = temp
                        probable_date_col = col
                        date_cols.append(col)
                        logging.info(f"Auto-converted '{col}' → datetime")
                        break
                except:
                    pass

        value_cols = df.select_dtypes(include=np.number).columns.tolist()

        default_date = probable_date_col or (date_cols[0] if date_cols else None)
        default_value = next(
            (c for c in value_cols if 'sales' in c.lower() or 'amount' in c.lower()),
            value_cols[0] if value_cols else None
        )

        # ── GET: Show form ───────────────────────────────────────────────────
        if request.method == 'GET':
            return render_template('prediction.html',
                                 filename=filename,
                                 date_columns=date_cols,
                                 value_columns=value_cols,
                                 periods_options=[3, 6, 9, 12, 18, 24],
                                 models=['ARIMA', 'Prophet', 'LSTM'],
                                 confidence_interval=95,
                                 date_column=default_date,
                                 value_column=default_value,
                                 periods=3)

        # ── POST: Process forecast request ───────────────────────────────────
        date_col = request.form.get('date_column')
        value_col = request.form.get('value_column')
        periods = request.form.get('periods', type=int, default=3)
        models_list = request.form.getlist('models')
        ci = request.form.get('confidence_interval', type=int, default=95) / 100

        if not all([date_col, value_col, models_list]):
            flash('Please select date column, value column and at least one model', 'warning')
            return render_template('prediction.html',
                                 filename=filename,
                                 date_columns=date_cols,
                                 value_columns=value_cols,
                                 periods_options=[3, 6, 9, 12, 18, 24],
                                 models=['ARIMA', 'Prophet', 'LSTM'],
                                 confidence_interval=95,
                                 date_column=date_col or default_date,
                                 value_column=value_col or default_value,
                                 periods=periods)

        # ── Clean data BEFORE forecasting ─────────────────────────────────────
        df_clean = df[[date_col, value_col]].copy()
        df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors='coerce')
        df_clean = df_clean.dropna(subset=[date_col, value_col])
        df_clean[value_col] = pd.to_numeric(df_clean[value_col], errors='coerce')
        df_clean = df_clean[df_clean[value_col].notna() & np.isfinite(df_clean[value_col])]
        df_clean = df_clean.sort_values(date_col).reset_index(drop=True)

        logging.info(f"Cleaned data shape: {df_clean.shape}")

        if len(df_clean) < 10:
            flash(f'Not enough valid rows after cleaning ({len(df_clean)}). Need at least ~10.', 'danger')
            return render_template('prediction.html', **locals())

        # ── Generate forecast ─────────────────────────────────────────────────
        result = generate_forecast(df_clean, date_col, value_col, periods, models_list, ci)

        if result.get('error'):
            flash(result['error'], 'danger')
        else:
            # Save forecast data for download
            session['forecast_data'] = result['forecast_data'].to_dict(orient='records')

            # Save context for chatbot (very important for memory/context)
            session['last_filename']   = filename
            session['last_periods']    = periods
            session['last_models']     = models_list
            session['last_value_col']  = value_col
            session['last_date_col']   = date_col

            # Optional: Reset chat history when generating a NEW forecast
            # (prevents confusion from old conversation about previous forecast)
            session.pop('chat_memory', None)

        return render_template('prediction.html',
                             filename=filename,
                             date_columns=date_cols,
                             value_columns=value_cols,
                             periods_options=[3, 6, 9, 12, 18, 24],
                             models=['ARIMA', 'Prophet', 'LSTM'],
                             confidence_interval=int(ci * 100),
                             date_column=date_col,
                             value_column=value_col,
                             periods=periods,
                             forecast_table=result.get('forecast_table'),
                             visualizations=result.get('visualizations'),
                             error=result.get('error'))

    except Exception as e:
        logging.exception("Critical error in generate_prediction")
        flash(f"Server error: {str(e)}", 'danger')
        return redirect(url_for('view_files'))
     
@app.route('/download/<filename>')
@login_required
def download_file(filename):
    if 'user_id' not in session or 'upload_session_id' not in session:
        flash('Session expired. Please upload files again.')
        return redirect(url_for('index'))
    
    user_upload_folder = get_user_upload_folder()
    if isinstance(user_upload_folder, Response):
        return user_upload_folder
    return send_from_directory(user_upload_folder, filename, as_attachment=True)

@app.route('/download_forecast')
@login_required
def download_forecast():
    if 'forecast_data' not in session:
        flash('No forecast data available to download')
        return redirect(url_for('view_files'))
    
    forecast_df = pd.DataFrame(session['forecast_data'])
    output = io.StringIO()
    forecast_df.to_csv(output, index=False)
    output.seek(0)
    
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name='forecast.csv'
    )

@app.route('/clear')
@login_required
def clear_files():
    if 'user_id' in session and 'upload_session_id' in session:
        user_id = session['user_id']
        upload_session_id = session['upload_session_id']
        for folder in [UPLOAD_FOLDER, STATIC_FOLDER]:
            session_folder = os.path.join(folder, user_id, upload_session_id)
            if os.path.exists(session_folder):
                shutil.rmtree(session_folder, ignore_errors=True)
        session.pop('upload_session_id', None)
        flash('Your files have been cleared')
    return redirect(url_for('index'))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(threaded=True)
