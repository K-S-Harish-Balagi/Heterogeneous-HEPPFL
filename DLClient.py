import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

def modelTraining(data, agg_weights=None):

    target_col = ["ecg"]
    # Selecting the feature columns
    features = list(set(list(data.columns)) - set(target_col))

    # Creating a test set
    X = data[features].values
    y = data[target_col].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Applyig Min Max Scaling on the training and test dataset of features
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, kernel_initializer = 'he_normal', input_dim = X_train.shape[1]),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Dense(32, kernel_initializer = 'he_normal'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Dense(16, kernel_initializer = 'he_normal'),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Dense(1, activation = 'linear')
    ])
    # model.summary()

    # Setting the aggregated weights to the model if available
    if not agg_weights is None:
        model.set_weights(agg_weights)

    # Specifying the optimizers and compiling model
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.005, clipnorm = 1.0)
    model.compile(loss = "mean_squared_error" , optimizer = optimizer, metrics = ["mean_squared_error"])

    # Callbacks for training
    early_stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights=True)
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', patience = 3, factor = 0.5, min_lr = 1e-5)

    # Fitting the model
    model.fit(X_train, y_train, epochs = 10, batch_size = 128, validation_data = (X_test, y_test), callbacks=[early_stop, lr_scheduler])

    # Evaluating the error metrics with respect to the predictions on test data
    pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    rscore = r2_score(y_test, pred)
    print("RMSE : ", rmse)
    print("R^2 Score : ", rscore)
    print("Adjusted R^2 Score : ", 1 - (data.shape[0] - 1)/(data.shape[0] - data.shape[1] - 1) * (1 - rscore))

    # Obtaining the optimized weights from each layer of the model
    for i in range(len(model.layers)):
        weights = model.layers[i].get_weights()
        if len(weights) == 0:
            continue
        # print("Optimized Weights in Layer", (i+1), ":", weights)

    # Sending the optimal weights to the client for encryption
    return model.get_weights()