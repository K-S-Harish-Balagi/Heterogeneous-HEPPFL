import sys
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def featureEngineering(df):
    print(df)
    df = df.drop('time', axis = 1)
    df['timestamp'] = range(0, df.shape[0])

    # -------- FEATURE SELECTION --------
    drop_cols = ['temp_1', 'temp_2', 'temp_3', 'lc_1', 'lc_2']
    df = df.drop(columns=drop_cols)

    # -------- TIME-DOMAIN FEATURES --------
    def compute_rms(x):
        return np.sqrt(np.mean(x**2))

    window_size = 10  # Adjust based on signal frequency
    for col in ['pleth_1', 'pleth_2', 'pleth_3', 'ecg']:
        df[f'{col}_mean'] = df[col].rolling(window=window_size).mean()
        df[f'{col}_std'] = df[col].rolling(window=window_size).std()
        df[f'{col}_rms'] = df[col].rolling(window=window_size).apply(compute_rms, raw=True)

    df = df.dropna()  # Remove NaN values from rolling calculations

    # -------- FREQUENCY-DOMAIN FEATURES (FFT) --------
    # Compute FFT over sliding windows instead of row-wise
    def compute_fft_windowed(series, window_size=100):
        fft_values = []
        for i in range(0, len(series) - window_size, window_size):
            segment = series[i:i + window_size]
            fft_magnitude = np.abs(np.fft.fft(segment))[:window_size // 2].mean()
            fft_values.append(fft_magnitude)
        
        # Pad with the last computed value to maintain shape
        fft_values.extend([fft_values[-1]] * (len(series) - len(fft_values)))
        return np.array(fft_values)

    for col in ['pleth_1', 'pleth_2', 'pleth_3', 'ecg']:
        df[f'{col}_fft'] = compute_fft_windowed(df[col].values, window_size=100)

    # -------- PEAK-BASED FEATURES --------
    df['RR_interval'] = df['peaks'].diff().fillna(0)

    # -------- DIMENSIONALITY REDUCTION (PCA) --------
    pca = PCA(n_components=10)  # Adjust number of components
    X_pca = pca.fit_transform(df.drop(columns=['ecg']))  # Keep ECG as the target variable
    df_pca = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(10)])
    df_pca['ecg'] = df['ecg'].values  # Add target back
    print(df_pca)
    return df_pca

filename = sys.argv[1]
df = pd.read_csv(f"{filename}.csv")
df_processed = featureEngineering(df)
df_processed.to_csv(f"{filename}_processed.csv", index=False)