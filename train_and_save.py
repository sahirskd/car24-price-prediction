import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


try:
    df = pd.read_csv("cars24-car-price-cleaned-new.csv")
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: 'cars24-car-price-cleaned-new.csv' not found. Please ensure it is in the same directory.")
    exit()



df_train, df_test = train_test_split(df, test_size=0.2, random_state=32)


df_test.head(50).to_csv("test_data_raw.csv", index=False)


model_wise_mean = df_train.groupby('model')['selling_price'].mean()
make_wise_mean = df_train.groupby('make')['selling_price'].mean()
global_mean = df_train['selling_price'].mean()


df_train['make'] = df_train['make'].map(make_wise_mean).fillna(global_mean)
df_train['model'] = df_train['model'].map(model_wise_mean).fillna(global_mean)


scaler = MinMaxScaler()
df_train_scaled = pd.DataFrame(scaler.fit_transform(df_train), columns=df.columns)


price_min = scaler.data_min_[df.columns.get_loc('selling_price')]
price_max = scaler.data_max_[df.columns.get_loc('selling_price')]


y_train = df_train_scaled['selling_price']
x_train = df_train_scaled.drop('selling_price', axis=1)


estimator = LinearRegression()
num_of_features_to_select = x_train.shape[1] // 2
rfe_selector = RFE(estimator=estimator, n_features_to_select=num_of_features_to_select)
rfe_selector.fit(x_train, y_train)
selected_features = x_train.columns[rfe_selector.support_].tolist()

print(f"Selected features: {selected_features}")

x_train_rfe = x_train[selected_features]


BEST_POLY_DEGREE = 3
BEST_ALPHA = 0.01

final_pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('polynomial', PolynomialFeatures(degree=BEST_POLY_DEGREE)),
    ('model', Ridge(alpha=BEST_ALPHA)),
])

final_pipeline.fit(x_train_rfe, y_train)
print("Model trained.")


joblib.dump(final_pipeline, 'car_price_model.joblib')


metadata = {
    'make_wise_mean': make_wise_mean.to_dict(),
    'model_wise_mean': model_wise_mean.to_dict(),
    'global_mean': global_mean,
    'selected_features': selected_features,
    'price_min': price_min,
    'price_max': price_max,
    'all_columns': df.columns.tolist(),
    'feature_scaler': scaler 
}
joblib.dump(metadata, 'model_metadata.joblib')

print("Deployment artifacts saved: 'car_price_model.joblib', 'model_metadata.joblib', 'test_data_raw.csv'")
