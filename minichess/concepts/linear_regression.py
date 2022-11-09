from sklearn import linear_model
from sklearn.metrics import r2_score, balanced_accuracy_score
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import tensorflow.keras as keras


def perform_regression(points, targets, validation_points, validation_targets, is_binary):
    if is_binary:
        return perform_logistic_regression(points, targets, validation_points, validation_targets)
    else:
        return perform_linear_regression(points, targets, validation_points, validation_targets)


def perform_logistic_regression(points, targets, validation_points, validation_targets):
    inputs = keras.layers.Input((points.shape[1]))
    output = keras.layers.Dense(1, activation="sigmoid", kernel_regularizer=keras.regularizers.L1(l1=0.01))(inputs)

    model = keras.Model(inputs, output)
    model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(learning_rate=0.0001))

    model.fit(points, targets, validation_data=(validation_points, validation_targets), epochs=50)

    train_preds = model.predict(points) > 0.5
    val_preds = model.predict(validation_points) > 0.5
    print(binary_accuracy_metric(targets, train_preds))
    return binary_accuracy_metric(validation_targets, val_preds)


def perform_logistic_regression_stepwise(points, targets, validation_points, validation_targets):
    best = 0
    m = tqdm(range(points.shape[1]), total=points.shape[1])
    for i in m:
        layer_points = points[:, i].reshape((points.shape[0], 8 * 8))
        val_layer_points = validation_points[:, i].reshape((validation_points.shape[0], 8 * 8))

        scaler = StandardScaler()

        scaled_points = scaler.fit_transform(layer_points)
        model = linear_model.SGDClassifier(loss="squared_error", penalty="l1", alpha=0.001, max_iter=2000)
        model = model.fit(scaled_points, targets)
        scaled_validation_points = scaler.transform(val_layer_points)
        predictions = model.predict(scaled_validation_points)

        val = binary_accuracy_metric(validation_targets, predictions)
        if val > best:
            best = val
            m.set_postfix_str("Best: {}".format(best))
    return best


def perform_linear_regression(points, targets, validation_points, validation_targets):
    model = linear_model.LinearRegression()
    model = model.fit(points, targets)
    predictions = model.predict(validation_points)
    return r2_score(validation_targets, predictions)


def binary_accuracy_metric(targets, predictions):
    return 2 * (((targets == np.squeeze(predictions)).sum() / len(targets)) - 0.5)


def perform_logistic_regression_gbm(points, targets, validation_points, validation_targets):
    import xgboost
    model = xgboost.XGBClassifier(n_estimators=1000)
    model.fit(points, targets, verbose=2)

    predictions = model.predict(validation_points)
    return binary_accuracy_metric(validation_targets, predictions)
