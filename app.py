import os
import re
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from werkzeug.utils import secure_filename
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import joblib
import datetime
from sklearn.preprocessing import LabelEncoder

# Initialize Flask app
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MODEL_FOLDER"] = "models"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["MODEL_FOLDER"], exist_ok=True)

# ✅ Data Cleaning & Preprocessing Function
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler, LabelEncoder

def clean_and_preprocess_data(df, target_column):
    df = df.copy()

    # ✅ Remove special characters from column names
    df.columns = df.columns.str.replace(r"[^a-zA-Z0-9_]", "_", regex=True)
    df.columns = [col.strip("_") for col in df.columns]  # Remove trailing underscores

    # ✅ Drop duplicates
    df.drop_duplicates(inplace=True)

    # ✅ Fill missing values
    for col in df.columns:
        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "MISSING", inplace=True)
        else:
            df[col].fillna(df[col].median() if not pd.isna(df[col].median()) else 0, inplace=True)

    # ✅ Drop unnecessary columns (like IDs and timestamps)
    drop_cols = [col for col in df.columns if 'id' in col.lower() or 'timestamp' in col.lower()]
    df.drop(columns=drop_cols, errors='ignore', inplace=True)

    # ✅ Split features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # 🚀 **Fix: Label Encode Target Column (y)**
    if y.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y) # Convert categorical labels to numerical values
        y = pd.Series(le.fit_transform(y), name=target_column)  # Keep y as a Series

    # ✅ One-Hot Encode categorical features (excluding target)
    categorical_cols = X.select_dtypes(include=['object']).columns
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # ✅ Standardize numerical features
    scaler = StandardScaler()
    numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
    if not numerical_cols.empty:
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    # ✅ Enhanced column name cleaning for LightGBM compatibility
    X.columns = [col.replace("<", "_lt_").replace(">", "_gt_").replace("=", "_eq_") for col in X.columns]
    X.columns = [col.replace("[", "_").replace("]", "_").replace(":", "_").replace(".", "_") for col in X.columns]
    X.columns = [col.replace(",", "_").replace(";", "_").replace("{", "_").replace("}", "_") for col in X.columns]
    X.columns = [col.replace("(", "_").replace(")", "_").replace("/", "_").replace("\\", "_") for col in X.columns]
    X.columns = [col.replace("\"", "_").replace("'", "_").replace("`", "_") for col in X.columns]
    X.columns = [col.replace("&", "_and_").replace("|", "_or_") for col in X.columns]
    X.columns = [col.replace(" ", "_") for col in X.columns]  # Replace spaces with underscores
    X.columns = [col.replace("-", "_") for col in X.columns]  # Replace hyphens with underscores
    
    # Remove any other non-alphanumeric characters
    X.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col) for col in X.columns]
    
    # Ensure no column name starts with a number (add prefix if needed)
    X.columns = ['x_' + col if col[0].isdigit() else col for col in X.columns]
    
    # Fix duplicate column names if they exist after cleaning
    if len(set(X.columns)) < len(X.columns):
        cols = []
        seen = set()
        for col in X.columns:
            if col in seen:
                i = 1
                new_col = f"{col}_{i}"
                while new_col in seen:
                    i += 1
                    new_col = f"{col}_{i}"
                cols.append(new_col)
                seen.add(new_col)
            else:
                cols.append(col)
                seen.add(col)
        X.columns = cols
        
    return X, y, scaler



    # ✅ Handle class imbalance using SMOTE (only for classification)
    #if len(y.unique()) > 1 and len(y.unique()) < 10:  # Only for classification with reasonable number of classes
    #    try:
    #        smote = SMOTE(random_state=42)
    #        X, y = smote.fit_resample(X, y)
    #    except Exception as e:
    #        print(f"SMOTE error: {e}. Proceeding without SMOTE.")

    #return X, y, scaler

# ✅ Function to Train and Compare Multiple Models with Hyperparameter Optimization
def train_and_compare_models(X, y):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Determine if it's a classification problem
    is_classification = isinstance(y[0], (np.integer, np.bool_)) or len(np.unique(y)) < 10

    # Apply SMOTE for classification tasks with imbalanced data
    if is_classification and len(np.unique(y_train)) > 1 and len(np.unique(y_train)) < 10:
        try:
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
        except Exception as e:
            print(f"SMOTE error: {e}. Proceeding without SMOTE.")

    # Define scoring metric
    scoring = 'f1_weighted' if is_classification else 'neg_mean_squared_error'

    # Model hyperparameter grids
    param_grids = {
        "Random Forest": {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        "XGBoost": {
            'model': XGBClassifier(eval_metric="mlogloss" if is_classification else "rmse", random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        },
        "LightGBM": {
            'model': LGBMClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [3, 5, 7],
                'num_leaves': [31, 50, 70],
                'subsample': [0.8, 1.0]
            }
        },
        "Logistic Regression": {
            'model': LogisticRegression(max_iter=1000, random_state=42),
            'params': {
                'C': [0.01, 0.1, 1.0, 10.0],
                'solver': ['liblinear', 'saga'],
                'penalty': ['l1', 'l2']
            }
        },
        "KNN": {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'p': [1, 2]
            }
        }
    }

    results = {}
    models_info = {}
    best_score = -np.inf
    best_model_name = ""
    best_model = None

    # Train and optimize each model
    for name, model_info in param_grids.items():
        try:
            print(f"\nOptimizing {name}...")

            model = model_info['model']
            param_grid = model_info['params']

            # Step 1: RandomizedSearchCV for broad tuning
            search = RandomizedSearchCV(
                model, param_grid, 
                n_iter=10,  # Increased iterations for better search
                cv=5,
                scoring=scoring,
                n_jobs=-1,
                random_state=42,
                verbose=1
            )
            search.fit(X_train, y_train)

            # Best parameters from RandomizedSearchCV
            best_params = search.best_params_
            best_estimator = search.best_estimator_

            # Step 2: Fine-tuning with GridSearchCV (on best parameters)
            refined_grid = {key: [value] for key, value in best_params.items()}  # Convert to grid format
            grid_search = GridSearchCV(
                best_estimator, refined_grid,
                cv=5,
                scoring=scoring,
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)

            final_best_model = grid_search.best_estimator_

            # Evaluate on test set
            y_pred = final_best_model.predict(X_test)
            if is_classification:
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')

                roc_auc = 0
                if len(np.unique(y)) == 2:
                    if hasattr(final_best_model, "predict_proba"):
                        y_prob = final_best_model.predict_proba(X_test)[:, 1]
                        roc_auc = roc_auc_score(y_test, y_prob)

                model_results = {
                    'accuracy': round(accuracy * 100, 2),
                    'f1_score': round(f1 * 100, 2),
                    'precision': round(precision * 100, 2),
                    'recall': round(recall * 100, 2),
                    'roc_auc': round(roc_auc * 100, 2) if roc_auc > 0 else 'N/A'
                }
                score = f1  # Use F1 score for model comparison
            else:
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                model_results = {
                    'rmse': round(rmse, 4),
                    'mae': round(mae, 4),
                    'r2': round(r2, 4)
                }
                score = -rmse  # Use negative RMSE for comparison

            # Store results
            results[name] = model_results
            models_info[name] = {
                'best_params': best_params,
                'model': final_best_model
            }

            # Check if this is the best model
            if score > best_score:
                best_score = score
                best_model_name = name
                best_model = final_best_model

            print(f"{name} optimized. Best params: {best_params}")
            print(f"Results: {model_results}")

        except Exception as e:
            print(f"Error training {name}: {e}")
            results[name] = {'error': str(e)}

    # Save the best model
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join("models", f"{best_model_name}_{timestamp}.pkl")
    joblib.dump(best_model, model_path)

    return results, best_model_name, model_path, models_info
# ✅ Flask Route for File Upload and Processing
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # File Upload
        file = request.files["file"]
        if file and file.filename.endswith(".csv"):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Read CSV File
            df = pd.read_csv(filepath)

            # Get Target Column from User Input
            target_column = request.form.get("target_column")
            if target_column not in df.columns:
                return jsonify({"error": "Invalid target column"}), 400

            # Get optimization level (fast, balanced, thorough)
            opt_level = request.form.get("optimization_level", "balanced")

            # Clean and Preprocess Data
            X, y, scaler = clean_and_preprocess_data(df, target_column)

            # Train and Compare Models
            results, best_model_name, model_path, models_info = train_and_compare_models(X, y)

            # Save feature names for future inference
            feature_names = X.columns.tolist()
            feature_path = os.path.join(app.config["MODEL_FOLDER"], f"features_{os.path.basename(model_path)}.txt")
            with open(feature_path, 'w') as f:
                f.write('\n'.join(feature_names))

            # Save scaler for future inference
            scaler_path = os.path.join(app.config["MODEL_FOLDER"], f"scaler_{os.path.basename(model_path)}.pkl")
            joblib.dump(scaler, scaler_path)

            # Prepare response
            response = {
                "model_accuracies": results,
                "best_model": best_model_name,
                "best_model_path": model_path,
                "best_model_params": models_info[best_model_name]['best_params'] if best_model_name in models_info else {}
            }

            return jsonify(response)
        
        return jsonify({"error": "Please upload a CSV file"}), 400

    return render_template("index.html")

# ✅ Route for model prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input data
        input_data = request.json.get("data")
        model_path = request.json.get("model_path")
        
        if not input_data or not model_path:
            return jsonify({"error": "Missing input data or model path"}), 400
            
        # Load model
        model = joblib.load(model_path)
        
        # Load feature names
        feature_path = os.path.join(app.config["MODEL_FOLDER"], f"features_{os.path.basename(model_path)}.txt")
        with open(feature_path, 'r') as f:
            feature_names = f.read().splitlines()
        
        # Load scaler
        scaler_path = os.path.join(app.config["MODEL_FOLDER"], f"scaler_{os.path.basename(model_path)}.pkl")
        scaler = joblib.load(scaler_path)
        
        # Prepare input data
        input_df = pd.DataFrame([input_data])
        
        # Apply preprocessing similar to training
        # (would need to match preprocessing steps from training)
        
        # Make prediction
        prediction = model.predict(input_df)
        
        # For classification, get probabilities if available
        probabilities = {}
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_df)[0]
            class_labels = model.classes_
            probabilities = {str(label): float(prob) for label, prob in zip(class_labels, probs)}
        
        return jsonify({
            "prediction": prediction.tolist(),
            "probabilities": probabilities
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
