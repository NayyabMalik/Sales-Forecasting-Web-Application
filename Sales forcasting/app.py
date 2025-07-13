
import os
import pandas as pd
import json
import xml.etree.ElementTree as ET
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, session, Response, send_file
from flask_session import Session
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
import matplotlib.pyplot as plt
import seaborn as sns
import uuid
import logging
import secrets
import shutil
from datetime import datetime, timedelta
import atexit
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import io
import pymongo
import bcrypt
from functools import wraps

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

# Configure session with flask-session
app.config['SESSION_TYPE'] = 'mongodb'
app.config['SESSION_MONGODB'] = pymongo.MongoClient("mongodb://localhost:27017/")
app.config['SESSION_MONGODB_DB'] = 'forecast_app'
app.config['SESSION_MONGODB_COLLECT'] = 'sessions'
app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
Session(app)

# MongoDB setup for user data
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["forecast_app"]
users_collection = db["users"]

# Configure folders
UPLOAD_FOLDER = 'uploads/'
STATIC_FOLDER = 'static/images/'
ALLOWED_EXTENSIONS = {'csv', 'xls', 'xlsx', 'txt', 'json', 'xml', 'pdf'}
MAX_FILE_SIZE = 16 * 1024 * 1024

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create directories
for folder in [UPLOAD_FOLDER, STATIC_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_user_upload_folder():
    if 'username' not in session:
        flash('Session expired. Please log in.')
        return redirect(url_for('login'))
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
        logging.info(f"Generated new user_id: {session['user_id']}")
    user_folder = os.path.join(app.config['UPLOAD_FOLDER'], session['user_id'])
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
    user_folder = os.path.join(app.config['STATIC_FOLDER'], session['user_id'])
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)
    return user_folder

def cleanup_user_files():
    now = datetime.now()
    for folder in [UPLOAD_FOLDER, STATIC_FOLDER]:
        for user_folder in os.listdir(folder):
            folder_path = os.path.join(folder, user_folder)
            try:
                mod_time = datetime.fromtimestamp(os.path.getmtime(folder_path))
                if now - mod_time > timedelta(hours=24):
                    shutil.rmtree(folder_path, ignore_errors=True)
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

def generate_statistics_and_visualizations(filepath, filename):
    ext = filename.rsplit('.', 1)[1].lower()
    stats_html = ""
    visualizations = []
    try:
        if ext in ['csv', 'xls', 'xlsx']:
            df = pd.read_csv(filepath) if ext == 'csv' else pd.read_excel(filepath)
            desc = df.describe(include='all').to_html(classes='table table-bordered')
            stats_html = desc
            visualizations = create_visualizations(df, filename)
        elif ext == 'json':
            with open(filepath, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    df = pd.json_normalize(data)
                    desc = df.describe(include='all').to_html(classes='table table-bordered')
                    stats_html = desc
                    visualizations = create_visualizations(df, filename)
                else:
                    stats_html = "JSON data is not in a list format suitable for analysis."
    except Exception as e:
        stats_html = f"Could not generate statistics: {e}"
    return stats_html, visualizations

def create_visualizations(df, filename):
    visualizations = []
    user_static_folder = get_user_static_folder()
    if isinstance(user_static_folder, Response):
        return visualizations
    
    try:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()

        for col in numeric_cols:
            plt.figure(figsize=(8, 5))
            sns.histplot(df[col].dropna(), kde=True, bins=30, color='skyblue')
            plt.title(f'Distribution of {col}')
            img_filename = f"hist_{uuid.uuid4()}.png"
            plt.savefig(os.path.join(user_static_folder, img_filename), bbox_inches='tight')
            plt.close()
            visualizations.append({
                'url': url_for('static', filename=f'images/{session["user_id"]}/{img_filename}'),
                'title': f'Distribution of {col}',
                'description': f'Histogram showing frequency distribution of {col}'
            })

        for col in numeric_cols:
            plt.figure(figsize=(8, 5))
            sns.boxplot(x=df[col], color='lightgreen')
            plt.title(f'Box Plot of {col}')
            img_filename = f"box_{uuid.uuid4()}.png"
            plt.savefig(os.path.join(user_static_folder, img_filename), bbox_inches='tight')
            plt.close()
            visualizations.append({
                'url': url_for('static', filename=f'images/{session["user_id"]}/{img_filename}'),
                'title': f'Spread of {col}',
                'description': f'Box plot showing quartiles and outliers for {col}'
            })

        for col in categorical_cols:
            plt.figure(figsize=(10, 6))
            top_categories = df[col].value_counts().nlargest(10)
            sns.barplot(x=top_categories.values, y=top_categories.index, palette='viridis')
            plt.title(f'Top 10 Categories in {col}')
            plt.xlabel('Count')
            plt.ylabel(col)
            img_filename = f"bar_{uuid.uuid4()}.png"
            plt.savefig(os.path.join(user_static_folder, img_filename), bbox_inches='tight')
            plt.close()
            visualizations.append({
                'url': url_for('static', filename=f'images/{session["user_id"]}/{img_filename}'),
                'title': f'Top {col} Categories',
                'description': f'Most frequent values in {col} column'
            })

        if len(numeric_cols) >= 2:
            plt.figure(figsize=(10, 8))
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title('Correlation Between Variables')
            img_filename = f"heatmap_{uuid.uuid4()}.png"
            plt.savefig(os.path.join(user_static_folder, img_filename), bbox_inches='tight')
            plt.close()
            visualizations.append({
                'url': url_for('static', filename=f'images/{session["user_id"]}/{img_filename}'),
                'title': 'Correlation Heatmap',
                'description': 'Relationships between numeric variables'
            })

        for col in numeric_cols[:3]:
            plt.figure(figsize=(10, 6))
            sns.violinplot(data=df, y=col, color='lightblue', inner=None)
            sns.swarmplot(data=df, y=col, color='darkred', size=3, alpha=0.5)
            plt.title(f'Detailed Distribution of {col}')
            img_filename = f"violin_swarm_{uuid.uuid4()}.png"
            plt.savefig(os.path.join(user_static_folder, img_filename), bbox_inches='tight')
            plt.close()
            visualizations.append({
                'url': url_for('static', filename=f'images/{session["user_id"]}/{img_filename}'),
                'title': f'Detailed {col} Distribution',
                'description': f'Violin plot with individual data points for {col}'
            })

        if 2 <= len(numeric_cols) <= 5:
            g = sns.pairplot(df[numeric_cols], kind='reg', diag_kind='kde',
                            plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.5}})
            g.fig.suptitle('Pairwise Relationships with Regression', y=1.02)
            img_filename = f"pairplot_{uuid.uuid4()}.png"
            plt.savefig(os.path.join(user_static_folder, img_filename), bbox_inches='tight')
            plt.close()
            visualizations.append({
                'url': url_for('static', filename=f'images/{session["user_id"]}/{img_filename}'),
                'title': 'Variable Relationships',
                'description': 'Scatter plots with regression lines and distributions'
            })

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
                        'url': url_for('static', filename=f'images/{session["user_id"]}/{img_filename}'),
                        'title': f'Time Components: {num_col}',
                        'description': 'Trend, seasonal, and residual components'
                    })
            except Exception as e:
                logging.warning(f"Couldn't create decomposition plot: {e}")

        if len(numeric_cols) >= 3:
            x, y, size = numeric_cols[:3]
            plt.figure(figsize=(10, 8))
            sizes = (df[size] - df[size].min()) / (df[size].max() - df[size].min()) * 1000
            plt.scatter(df[x], df[y], s=sizes, alpha=0.5, c=df[size], cmap='viridis')
            plt.colorbar(label=size)
            plt.xlabel(x)
            plt.ylabel(y)
            plt.title(f'Bubble Chart: {x} vs {y} (Size={size})')
            plt.grid(True, alpha=0.3)
            img_filename = f"bubble_{uuid.uuid4()}.png"
            plt.savefig(os.path.join(user_static_folder, img_filename), bbox_inches='tight')
            plt.close()
            visualizations.append({
                'url': url_for('static', filename=f'images/{session["user_id"]}/{img_filename}'),
                'title': '3D Relationship',
                'description': f'Relationship between {x}, {y} with {size} as bubble size'
            })

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
        forecast_dates = pd.date_range(start=series.index[-1], periods=periods + 1, freq='M')[1:]
        
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
        
        future = model.make_future_dataframe(periods=periods, freq='M')
        forecast = model.predict(future)
        
        train_size = int(len(prophet_df) * 0.8)
        train, test = prophet_df[:train_size], prophet_df[train_size:]
        if len(test) > 0:
            model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True,
                           interval_width=confidence_interval)
            model.fit(train)
            future_test = model.make_future_dataframe(periods=len(test), freq='M')
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
        series = df.set_index(date_col)[value_col].dropna().values
        if len(series) < 4:
            logging.warning("Not enough data for LSTM (minimum 4 points required)")
            return None, None, None
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(series.reshape(-1, 1))
        
        look_back = 3
        X, y = [], []
        for i in range(len(scaled_data) - look_back):
            X.append(scaled_data[i:i + look_back])
            y.append(scaled_data[i + look_back])
        X, y = np.array(X), np.array(y)
        
        if len(X) < 5:
            logging.warning("Not enough training samples for LSTM (minimum 5 required)")
            return None, None, None
        
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(look_back, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        
        model.fit(X, y, epochs=20, batch_size=1, verbose=0)
        
        if len(X_test) > 0:
            predictions = model.predict(X_test, verbose=0)
            errors = scaler.inverse_transform(y_test.reshape(-1, 1)) - scaler.inverse_transform(predictions)
            errors = errors.flatten()
        else:
            errors = np.array([])
        
        last_sequence = scaled_data[-look_back:]
        forecast = []
        for _ in range(periods):
            x_input = last_sequence.reshape((1, look_back, 1))
            yhat = model.predict(x_input, verbose=0)
            forecast.append(yhat[0, 0])
            last_sequence = np.append(last_sequence[1:], yhat)
        
        forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
        
        last_date = df[date_col].iloc[-1]
        if pd.isna(last_date):
            logging.error("Last date is NaN")
            raise ValueError("Last date in date column is NaN")
        if not isinstance(last_date, (pd.Timestamp, datetime)):
            logging.error(f"Last date is not a datetime object: {last_date}")
            raise TypeError(f"Last date must be a datetime object, got {type(last_date)}")
        
        forecast_dates = pd.date_range(start=last_date, periods=periods + 1, freq='M')[1:]
        
        return forecast, forecast_dates, errors
    except Exception as e:
        logging.error(f"LSTM error: {str(e)}")
        return None, None, None

def generate_forecast(df, date_col, value_col, periods, models, confidence_interval=0.95):
    forecasts = {'ARIMA': None, 'Prophet': None, 'LSTM': None}
    errors = {'ARIMA': None, 'Prophet': None, 'LSTM': None}
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
    
    historical = df.set_index(date_col)[value_col].dropna()
    logging.info(f"Historical data length after dropna: {len(historical)}")
    
    min_data_points = 4 if 'LSTM' in models else 2
    
    if len(historical) < min_data_points:
        logging.warning(f"Not enough data points for forecasting (minimum {min_data_points} required)")
        return {
            'forecast_table': None,
            'forecast_data': pd.DataFrame(),
            'visualizations': [],
            'error': f"Not enough data points (minimum {min_data_points} required)"
        }
    
    if 'ARIMA' in models:
        arima_forecast, arima_dates, arima_errors = train_arima(df, date_col, value_col, periods)
        forecasts['ARIMA'] = arima_forecast
        errors['ARIMA'] = arima_errors
        dates = arima_dates
    
    if 'Prophet' in models:
        prophet_forecast, prophet_dates, prophet_ci_data, prophet_errors = train_prophet(df, date_col, value_col, periods, confidence_interval)
        forecasts['Prophet'] = prophet_forecast
        errors['Prophet'] = prophet_errors
        prophet_ci = prophet_ci_data
        if dates is None:
            dates = prophet_dates
    
    if 'LSTM' in models:
        lstm_forecast, lstm_dates, lstm_errors = train_lstm(df, date_col, value_col, periods)
        forecasts['LSTM'] = lstm_forecast
        errors['LSTM'] = lstm_errors
        if dates is None:
            dates = lstm_dates
    
    if dates is None:
        logging.warning("No forecasts generated; all models failed")
        last_date = historical.index[-1] if len(historical) > 0 else pd.to_datetime('2023-01-01')
        dates = pd.date_range(start=last_date, periods=periods + 1, freq='M')[1:]
    
    if any(forecast is not None for forecast in forecasts.values()):
        plt.figure(figsize=(12, 6))
        plt.plot(historical.index, historical, label='Historical', color='blue')
        if forecasts['ARIMA'] is not None:
            plt.plot(dates, forecasts['ARIMA'], label='ARIMA', color='green', linestyle='--')
        if forecasts['Prophet'] is not None:
            plt.plot(dates, forecasts['Prophet'], label='Prophet', color='red', linestyle='--')
        if forecasts['LSTM'] is not None:
            plt.plot(dates, forecasts['LSTM'], label='LSTM', color='yellow', linestyle='--')
        plt.title(f'Sales Forecast Comparison for {value_col} (Next {periods} Months)')
        plt.xlabel('Date')
        plt.ylabel(value_col)
        plt.legend()
        img_filename = f"forecast_comparison_{uuid.uuid4()}.png"
        plt.savefig(os.path.join(user_static_folder, img_filename), bbox_inches='tight')
        plt.close()
        visualizations.append({
            'url': url_for('static', filename=f'images/{session["user_id"]}/{img_filename}'),
            'title': f'Sales Forecast Comparison: {value_col}',
            'description': f'Comparison of forecasts for the next {periods} months'
        })
        
        if 'Prophet' in models and prophet_ci is not None:
            plt.figure(figsize=(12, 6))
            plt.plot(historical.index, historical, label='Historical', color='blue')
            plt.plot(dates, forecasts['Prophet'], label='Prophet Forecast', color='red', linestyle='--')
            plt.fill_between(dates, prophet_ci['yhat_lower'], prophet_ci['yhat_upper'], color='red', alpha=0.2, label='Confidence Interval')
            plt.title(f'Prophet Forecast with Confidence Interval for {value_col}')
            plt.xlabel('Date')
            plt.ylabel(value_col)
            plt.legend()
            img_filename = f"prophet_ci_{uuid.uuid4()}.png"
            plt.savefig(os.path.join(user_static_folder, img_filename), bbox_inches='tight')
            plt.close()
            visualizations.append({
                'url': url_for('static', filename=f'images/{session["user_id"]}/{img_filename}'),
                'title': f'Prophet Forecast with Confidence Interval: {value_col}',
                'description': f'Prophet forecast with {int(confidence_interval*100)}% confidence interval'
            })
        
        plt.figure(figsize=(12, 6))
        plt.plot(historical.index, historical, label='Historical', color='blue')
        forecast_means = []
        for i in range(len(dates)):
            valid_forecasts = [f[i] for f in forecasts.values() if f is not None]
            forecast_means.append(np.mean(valid_forecasts) if valid_forecasts else np.nan)
        forecast_means = np.array(forecast_means)
        plt.plot(dates, forecast_means, label='Average Forecast', color='purple', linestyle='--')
        plt.axvspan(historical.index[-1], dates[-1], color='purple', alpha=0.1, label='Forecast Period')
        plt.title(f'Historical Trend with Forecast Overlay for {value_col}')
        plt.xlabel('Date')
        plt.ylabel(value_col)
        plt.legend()
        img_filename = f"trend_overlay_{uuid.uuid4()}.png"
        plt.savefig(os.path.join(user_static_folder, img_filename), bbox_inches='tight')
        plt.close()
        visualizations.append({
            'url': url_for('static', filename=f'images/{session["user_id"]}/{img_filename}'),
            'title': f'Historical Trend with Forecast Overlay: {value_col}',
            'description': 'Historical data with an average forecast overlay and shaded forecast period'
        })
        
        plt.figure(figsize=(12, 6))
        for model_name, error_data in errors.items():
            if error_data is not None and len(error_data) > 0:
                sns.histplot(error_data, label=model_name, kde=True, stat='density', alpha=0.5)
        plt.title('Model Error Distribution (Validation Set)')
        plt.xlabel('Error')
        plt.ylabel('Density')
        plt.legend()
        img_filename = f"error_distribution_{uuid.uuid4()}.png"
        plt.savefig(os.path.join(user_static_folder, img_filename), bbox_inches='tight')
        plt.close()
        visualizations.append({
            'url': url_for('static', filename=f'images/{session["user_id"]}/{img_filename}'),
            'title': 'Model Error Distribution',
            'description': 'Distribution of errors for each model based on a validation split'
        })
    
    forecast_data = {'Date': [d.strftime('%Y-%m') for d in dates]}
    for model, forecast in forecasts.items():
        forecast_data[model] = [round(f, 2) if f is not None else 'N/A' for f in forecast] if forecast is not None else ['N/A'] * periods
    forecast_df = pd.DataFrame(forecast_data)
    forecast_table = forecast_df.to_html(classes='table table-bordered forecast-table', index=False)
    
    return {
        'forecast_table': forecast_table,
        'forecast_data': forecast_df,
        'visualizations': visualizations,
        'error': None
    }

# Authentication routes
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

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(url_for('index'))

    user_upload_folder = get_user_upload_folder()
    if isinstance(user_upload_folder, Response):
        return user_upload_folder
    files = request.files.getlist('file')
    upload_count = 0

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            if not filename:
                continue
                
            save_path = os.path.join(user_upload_folder, filename)
            try:
                file.save(save_path)
                logging.info(f"Saved file: {save_path}")
                upload_count += 1
            except Exception as e:
                logging.error(f"Error saving {filename}: {e}")
                flash(f'Error uploading {filename}')

    if upload_count > 0:
        flash(f'Successfully uploaded {upload_count} file(s)')
    else:
        flash('No valid files uploaded')
    
    return redirect(url_for('view_files'))

@app.route('/files')
@login_required
def view_files():
    try:
        if 'user_id' not in session:
            session['user_id'] = str(uuid.uuid4())
            logging.info(f"Generated new user_id in view_files: {session['user_id']}")
        
        user_upload_folder = get_user_upload_folder()
        if isinstance(user_upload_folder, Response):
            return user_upload_folder
        files = os.listdir(user_upload_folder) if os.path.exists(user_upload_folder) else []
        logging.info(f"Files in {user_upload_folder}: {files}")
        files_info = []
        
        for file in files:
            filepath = os.path.join(user_upload_folder, file)
            header = get_file_header(filepath, file)
            stats, visualizations = generate_statistics_and_visualizations(filepath, file)
            
            files_info.append({
                'name': file,
                'header': header,
                'stats': stats,
                'visualizations': visualizations if isinstance(visualizations, list) else []
            })
        
        return render_template('view_files.html', files=files_info)
    
    except Exception as e:
        logging.error(f"Error in view_files: {e}")
        flash(f"Error loading files: {e}")
        return redirect(url_for('index'))

@app.route('/download/<filename>')
@login_required
def download_file(filename):
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
        logging.info(f"Generated new user_id in download_file: {session['user_id']}")
    
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
    if 'user_id' in session:
        user_id = session['user_id']
        for folder in [UPLOAD_FOLDER, STATIC_FOLDER]:
            user_folder = os.path.join(folder, user_id)
            if os.path.exists(user_folder):
                shutil.rmtree(user_folder, ignore_errors=True)
        flash('Your files have been cleared')
    return redirect(url_for('index'))

@app.route('/predict/<filename>', methods=['GET', 'POST'])
@login_required
def generate_prediction(filename):
    logging.info(f"Accessing /predict/{filename} for user: {session.get('username')}, user_id: {session.get('user_id')}")
    
    if 'user_id' not in session:
        logging.warning("No user_id in session. Generating new user_id.")
        session['user_id'] = str(uuid.uuid4())
        flash('Session refreshed. Please try again.')
        return redirect(url_for('view_files'))
    
    user_upload_folder = get_user_upload_folder()
    if isinstance(user_upload_folder, Response):
        return user_upload_folder
    
    filepath = os.path.join(user_upload_folder, filename)
    logging.info(f"Checking file at: {filepath}")
    
    if not os.path.exists(filepath):
        logging.warning(f"File not found: {filepath}")
        flash(f'File "{filename}" not found. Please upload it again.')
        return redirect(url_for('view_files'))

    result = {'forecast_table': None, 'forecast_data': pd.DataFrame(), 'visualizations': [], 'error': None}
    date_col = None
    value_col = None
    periods = 3
    models = ['ARIMA']
    confidence_interval = 0.95
    date_cols = []
    value_cols = []

    try:
        ext = filename.rsplit('.', 1)[1].lower()
        logging.info(f"File extension: {ext}")
        if ext not in ['csv', 'xls', 'xlsx']:
            logging.warning(f"Unsupported file extension: {ext}")
            flash('Prediction is only supported for CSV and Excel files.')
            return render_template('prediction.html',
                                 filename=filename,
                                 date_columns=date_cols,
                                 value_columns=value_cols,
                                 periods_options=[3, 6, 9, 12],
                                 models=['ARIMA', 'Prophet', 'LSTM'],
                                 confidence_interval=95,
                                 forecast_table=None,
                                 visualizations=None,
                                 error="Unsupported file format. Please upload a CSV or Excel file.")
        
        logging.info("Loading file into DataFrame")
        try:
            if ext == 'csv':
                df = pd.read_csv(filepath, encoding='utf-8')
            else:
                df = pd.read_excel(filepath)
        except pd.errors.EmptyDataError:
            logging.error("File is empty")
            flash('The uploaded file is empty.')
            return render_template('prediction.html',
                                 filename=filename,
                                 date_columns=date_cols,
                                 value_columns=value_cols,
                                 periods_options=[3, 6, 9, 12],
                                 models=['ARIMA', 'Prophet', 'LSTM'],
                                 confidence_interval=95,
                                 forecast_table=None,
                                 visualizations=None,
                                 error="The file is empty. Please upload a valid file.")
        except UnicodeDecodeError:
            logging.error("Encoding error while reading file")
            flash('Error reading file: Invalid encoding.')
            return render_template('prediction.html',
                                 filename=filename,
                                 date_columns=date_cols,
                                 value_columns=value_cols,
                                 periods_options=[3, 6, 9, 12],
                                 models=['ARIMA', 'Prophet', 'LSTM'],
                                 confidence_interval=95,
                                 forecast_table=None,
                                 visualizations=None,
                                 error="Invalid file encoding. Please ensure the file is UTF-8 encoded.")
        except Exception as e:
            logging.error(f"Error loading file: {e}")
            flash(f"Error loading file: {e}")
            return render_template('prediction.html',
                                 filename=filename,
                                 date_columns=date_cols,
                                 value_columns=value_cols,
                                 periods_options=[3, 6, 9, 12],
                                 models=['ARIMA', 'Prophet', 'LSTM'],
                                 confidence_interval=95,
                                 forecast_table=None,
                                 visualizations=None,
                                 error=f"Error loading file: {str(e)}")
        
        logging.info(f"DataFrame columns: {df.columns.tolist()}")
        logging.info(f"DataFrame dtypes: {df.dtypes}")
        
        year_col = next((col for col in df.columns if col.lower() == 'year'), None)
        month_col = next((col for col in df.columns if col.lower() == 'month'), None)
        day_col = next((col for col in df.columns if col.lower() == 'day'), None)
        if year_col and month_col and day_col:
            logging.info("Found Year, Month, and Day columns; combining into Date column")
            try:
                df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
                df[month_col] = pd.to_numeric(df[month_col], errors='coerce')
                df[day_col] = pd.to_numeric(df[day_col], errors='coerce')
                
                invalid_months = df[month_col].notna() & ((df[month_col] < 1) | (df[month_col] > 12))
                invalid_days = df[day_col].notna() & ((df[day_col] < 1) | (df[day_col] > 31))
                if invalid_months.any():
                    logging.warning(f"Invalid months found in {month_col}")
                    flash(f"Invalid months found in {month_col}. Months must be between 1 and 12.")
                    return render_template('prediction.html',
                                         filename=filename,
                                         date_columns=date_cols,
                                         value_columns=value_cols,
                                         periods_options=[3, 6, 9, 12],
                                         models=['ARIMA', 'Prophet', 'LSTM'],
                                         confidence_interval=95,
                                         forecast_table=None,
                                         visualizations=None,
                                         error=f"Invalid months in {month_col}.")
                if invalid_days.any():
                    logging.warning(f"Invalid days found in {day_col}")
                    flash(f"Invalid days found in {day_col}. Days must be between 1 and 31.")
                    return render_template('prediction.html',
                                         filename=filename,
                                         date_columns=date_cols,
                                         value_columns=value_cols,
                                         periods_options=[3, 6, 9, 12],
                                         models=['ARIMA', 'Prophet', 'LSTM'],
                                         confidence_interval=95,
                                         forecast_table=None,
                                         visualizations=None,
                                         error=f"Invalid days in {day_col}.")
                
                original_len = len(df)
                df = df.dropna(subset=[year_col, month_col, day_col])
                if len(df) == 0:
                    logging.warning("All rows dropped due to NaN values in Year, Month, or Day")
                    flash('No valid data remaining after removing rows with missing Year, Month, or Day.')
                    return render_template('prediction.html',
                                         filename=filename,
                                         date_columns=date_cols,
                                         value_columns=value_cols,
                                         periods_options=[3, 6, 9, 12],
                                         models=['ARIMA', 'Prophet', 'LSTM'],
                                         confidence_interval=95,
                                         forecast_table=None,
                                         visualizations=None,
                                         error="No valid data after processing Year, Month, Day columns.")
                if len(df) < original_len:
                    logging.info(f"Dropped {original_len - len(df)} rows due to NaN values")

                df['Date'] = pd.to_datetime(df[[year_col, month_col, day_col]].rename(columns={year_col: 'year', month_col: 'month', day_col: 'day'}), errors='coerce')
                if df['Date'].isna().all():
                    logging.warning("Failed to create valid Date column")
                    flash('Could not create a valid Date column from Year, Month, and Day.')
                    return render_template('prediction.html',
                                         filename=filename,
                                         date_columns=date_cols,
                                         value_columns=value_cols,
                                         periods_options=[3, 6, 9, 12],
                                         models=['ARIMA', 'Prophet', 'LSTM'],
                                         confidence_interval=95,
                                         forecast_table=None,
                                         visualizations=None,
                                         error="Invalid Year, Month, or Day values.")
                logging.info("Successfully created Date column")
            except Exception as e:
                logging.error(f"Error creating Date column: {e}")
                flash(f"Error creating Date column: {e}")
                return render_template('prediction.html',
                                     filename=filename,
                                     date_columns=date_cols,
                                     value_columns=value_cols,
                                     periods_options=[3, 6, 9, 12],
                                     models=['ARIMA', 'Prophet', 'LSTM'],
                                     confidence_interval=95,
                                     forecast_table=None,
                                     visualizations=None,
                                     error=f"Error processing date columns: {str(e)}")
        
        for col in df.columns:
            if col != 'Date':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        logging.info("Identifying date and value columns")
        date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        
        if not date_cols:
            object_cols = df.select_dtypes(include=['object']).columns.tolist()
            date_formats = [None, '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d', '%m-%d-%Y']
            for col in object_cols:
                if any(keyword in col.lower() for keyword in ['location', 'day of the week', 'store']):
                    continue
                for fmt in date_formats:
                    converted = pd.to_datetime(df[col], errors='coerce', format=fmt, dayfirst=True)
                    if converted.notna().sum() > len(df)/2:
                        date_cols.append(col)
                        df[col] = converted
                        break
        
        if 'Date' in df.columns and df['Date'].dtype in ['datetime64[ns]', 'datetime64']:
            if 'Date' not in date_cols:
                date_cols.append('Date')

        value_cols = df.select_dtypes(include=['number']).columns.tolist()
        logging.info(f"Date columns: {date_cols}")
        logging.info(f"Value columns: {value_cols}")

        if not date_cols or not value_cols:
            logging.warning("No suitable date or numeric columns found")
            flash('No suitable date or numeric columns found for prediction.')
            return render_template('prediction.html',
                                 filename=filename,
                                 date_columns=date_cols,
                                 value_columns=value_cols,
                                 periods_options=[3, 6, 9, 12],
                                 models=['ARIMA', 'Prophet', 'LSTM'],
                                 confidence_interval=95,
                                 forecast_table=None,
                                 visualizations=None,
                                 error="No date or numeric columns found. Please upload a file with appropriate columns.")

        min_data_points = 4 if 'LSTM' in models else 2
        if request.method == 'POST':
            date_col = request.form.get('date_column')
            value_col = request.form.get('value_column')
            periods_str = request.form.get('periods')
            models = request.form.getlist('models')
            confidence_interval_str = request.form.get('confidence_interval')

            logging.info(f"POST request: date_col={date_col}, value_col={value_col}, periods={periods_str}, models={models}, confidence_interval={confidence_interval_str}")

            try:
                periods = int(periods_str)
            except (ValueError, TypeError):
                logging.warning(f"Invalid periods value: {periods_str}")
                flash('Invalid forecast period. Using default (3 months).')
                periods = 3

            try:
                confidence_interval = float(confidence_interval_str) / 100
            except (ValueError, TypeError):
                logging.warning(f"Invalid confidence interval value: {confidence_interval_str}")
                flash('Invalid confidence interval. Using default (95%).')
                confidence_interval = 0.95

            if not date_col or not value_col or periods not in [3, 6, 9, 12] or not models:
                logging.warning("Invalid input parameters in POST request")
                flash('Invalid input parameters. Please select valid columns and models.')
                date_col = date_cols[0] if date_cols else None
                value_col = value_cols[0] if value_cols else None
                periods = 3
                models = ['ARIMA']
            else:
                historical = df.set_index(date_col)[value_col].dropna()
                logging.info(f"Data points available: {len(historical)}")

                if len(historical) < min_data_points:
                    logging.warning(f"Not enough data points: {len(historical)} (minimum {min_data_points} required)")
                    flash(f'Not enough data points for forecasting (minimum {min_data_points} required).')
                    return render_template('prediction.html',
                                         filename=filename,
                                         date_columns=date_cols,
                                         value_columns=value_cols,
                                         periods_options=[3, 6, 9, 12],
                                         models=['ARIMA', 'Prophet', 'LSTM'],
                                         confidence_interval=int(confidence_interval * 100),
                                         forecast_table=None,
                                         visualizations=None,
                                         error=f"Not enough data points (minimum {min_data_points} required).")

                logging.info("Generating forecast")
                try:
                    result = generate_forecast(df, date_col, value_col, periods, models, confidence_interval)
                    if result['error']:
                        flash(result['error'])
                    else:
                        session['forecast_data'] = result['forecast_data'].to_dict()
                except Exception as e:
                    logging.error(f"Error in generate_forecast: {str(e)}")
                    flash(f"Error generating forecast: {str(e)}")
                    return render_template('prediction.html',
                                         filename=filename,
                                         date_columns=date_cols,
                                         value_columns=value_cols,
                                         periods_options=[3, 6, 9, 12],
                                         models=['ARIMA', 'Prophet', 'LSTM'],
                                         confidence_interval=int(confidence_interval * 100),
                                         forecast_table=None,
                                         visualizations=None,
                                         error=f"Forecast error: {str(e)}")
        
        logging.info("Rendering prediction.html")
        return render_template('prediction.html',
                             filename=filename,
                             date_columns=date_cols,
                             value_columns=value_cols,
                             periods_options=[3, 6, 9, 12],
                             models=['ARIMA', 'Prophet', 'LSTM'],
                             confidence_interval=int(confidence_interval * 100),
                             forecast_table=result['forecast_table'],
                             visualizations=result['visualizations'],
                             date_column=date_col,
                             value_column=value_col,
                             periods=periods,
                             error=result['error'])
    
    except Exception as e:
        logging.error(f"Error in generate_prediction: {str(e)}")
        flash(f"Error processing prediction: {str(e)}")
        return render_template('prediction.html',
                             filename=filename,
                             date_columns=date_cols,
                             value_columns=value_cols,
                             periods_options=[3, 6, 9, 12],
                             models=['ARIMA', 'Prophet', 'LSTM'],
                             confidence_interval=95,
                             forecast_table=None,
                             visualizations=None,
                             error=f"Processing error: {str(e)}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run()
