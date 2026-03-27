import pandas as pd
import joblib
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

def train_and_save_model():
    print("Loading data...")
    df = pd.read_csv('creditcard.csv')
    scaler = RobustScaler()
    df['amount_scaled'] = scaler.fit_transform(df['Amount'].values.reshape(-1,1))
    df['time_scaled'] = scaler.fit_transform(df['Time'].values.reshape(-1,1))
    df.drop(['Time','Amount'], axis=1, inplace=True)
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Applying SMOTE...")
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    print("Training XGBoost...")
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_res, y_res)

    joblib.dump(model, 'fraud_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Model and Scaler saved as .pkl files!")

if __name__ == "__main__":
    train_and_save_model()