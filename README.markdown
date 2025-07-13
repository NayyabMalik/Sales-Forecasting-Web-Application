# Sales Forecasting Web Application

This repository contains a Flask-based web application for sales forecasting, allowing users to upload time series data (CSV, Excel, JSON, XML, PDF, or text files), generate statistical visualizations, and predict future sales using ARIMA, Prophet, and LSTM models. The application supports user authentication (registration, login, logout), file management, and interactive forecasting with customizable parameters (e.g., forecast periods, confidence intervals). Visualizations include histograms, box plots, correlation heatmaps, and forecast comparisons, enhancing data analysis and prediction interpretability.

This project is part of my deep learning coursework at the National University of Modern Languages, Islamabad, submitted on October 31, 2024, under the supervision of Mam Iqra Nasem. It applies time series forecasting techniques from my deep learning labs, particularly LSTM-based modeling.

## Features

- **User Authentication**: Secure registration and login with MongoDB and bcrypt for password hashing.
- **File Upload**: Supports multiple file formats (CSV, Excel, JSON, XML, PDF, TXT) with a 16MB size limit.
- **Data Visualization**: Generates histograms, box plots, bar charts, correlation heatmaps, violin plots, pair plots, time series decomposition, and bubble charts for uploaded data.
- **Sales Forecasting**: Implements ARIMA, Prophet, and LSTM models for time series prediction with customizable periods (3, 6, 9, 12 months) and confidence intervals.
- **Forecast Visualizations**: Displays forecast comparisons, Prophet confidence intervals, historical trends, and error distributions.
- **Session Management**: Uses MongoDB for session storage with a 24-hour lifetime and automatic cleanup of user files.
- **File Management**: Allows users to view, download, and clear uploaded files, with secure user-specific storage.

## Repository Structure

```
sales-forecasting/
├── app.py                     # Flask application with forecasting and visualization logic
├── requirements.txt           # Python dependencies
├── templates/                # HTML templates for the web interface
│   ├── index.html            # Home page (file upload)
│   ├── login.html            # User login/registration page
│   ├── logout.html           # Logout page
│   ├── progress.html         # Progress tracking page (placeholder, not implemented in app.py)
│   ├── result.html           # Results display page (for visualizations/forecasts)
│   ├── register.html         # Alias for login.html with registration form
│   ├── upload.html           # File upload page (alias for index.html)
│   ├── file_upload.html      # Main upload interface
│   ├── view_files.html       # File listing and visualization page
│   ├── prediction.html       # Forecasting interface
│   ├── start.html            # Landing page
├── static/                   # Static files for the web interface
│   ├── css/                 # CSS stylesheets
│   │   ├── index.css
│   │   ├── login.css
│   │   ├── logout.css
│   │   ├── progress.css
│   │   ├── result.css
│   │   ├── register.css
│   │   ├── upload.css
│   ├── js/                  # JavaScript files
│   │   ├── index.js
│   │   ├── upload.js
├── uploads/                  # User-uploaded files (temporary, user-specific)
├── static/images/            # Generated visualizations (user-specific)
├── README.md                # This file
├── LICENSE                  # MIT License
```

## Related Coursework

This project builds on concepts from my deep learning labs, particularly:

- **Lab 11: RNN_Text** (`lab_manuals/RNN_Text.pdf`): Covers Recurrent Neural Networks (RNNs) for sequence modeling, relevant to LSTM implementation.
- **Lab 12+13: LSTM_TimeSeries** (`lab_manuals/LSTM_TimeSeries.pdf`): Discusses LSTM networks for time series forecasting, used in `app.py` for sales prediction.

See the [deep-learning-labs repository](https://github.com/your-username/deep-learning-labs) for lab manuals and related projects (e.g., `anomaly_detection.ipynb`).

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
- **Permissions**: The code and referenced lab manuals are shared with permission for educational purposes. Contact Mam Iqra Nasem for access to original course materials.
- **File Size**: Use Git LFS for large files (e.g., `git lfs track "*.csv" "*.xlsx" "*.png"`) if uploading to GitHub.

## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.