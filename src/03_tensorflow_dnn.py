"""
03_tensorflow_dnn.py — Build, train, and evaluate TensorFlow DenseNN classifiers.

Experiments with shallow vs deep architectures + dropout.
Reads preprocessed data from outputs/models/*.npy
Writes models to outputs/models/ and training curves to outputs/figures/

Usage:
  python src/03_tensorflow_dnn.py
"""
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report

from config import MODEL_DIR, FIGURE_DIR, RESULTS_FILE, DNN_EPOCHS, DNN_PATIENCE, RANDOM_STATE

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Fix seeds
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# ── DNN Experiment grid ───────────────────────────────────────────────────────
# Each entry: (system_id, hidden_layers, dropout_rate, learning_rate, batch_size)
# System H: Shallow architecture (baseline for DNN, minimal regularization)
# System I: Medium depth with moderate regularization
# System J: Deeper architecture with strong regularization  
DNN_CONFIGS: list[tuple] = [
    ("H_dnn_shallow",     [64],              0.0,  0.001,  64),   # 1 layer, no dropout
    ("I_dnn_medium",      [128, 64],         0.2,  0.001,  64),   # 2 layers with light dropout
    ("J_dnn_deep",        [256, 128, 64],    0.3,  0.0005, 32),   # 3 layers with strong dropout
    ("K_dnn_wide",        [512, 256],        0.25, 0.001,  64),   # Wide layers for feature learning
]


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load preprocessed data arrays.

    Returns:
        Tuple of (X_train, y_train, X_test, y_test).
    """
    X_train = np.load(MODEL_DIR / "X_train.npy")
    y_train = np.load(MODEL_DIR / "y_train.npy")
    X_test = np.load(MODEL_DIR / "X_test.npy")
    y_test = np.load(MODEL_DIR / "y_test.npy")
    return X_train, y_train, X_test, y_test


def build_model(input_dim: int, num_classes: int, hidden_layers: list[int],
                dropout_rate: float, learning_rate: float) -> keras.Model:
    """Build a Dense neural network with the Functional API.

    Args:
        input_dim:    Number of input features.
        num_classes:  Number of output classes.
        hidden_layers: List of unit counts for each hidden Dense layer.
        dropout_rate: Dropout probability after each hidden layer (0 = disabled).
        learning_rate: Adam optimizer learning rate.

    Returns:
        Compiled keras Model.
    """
    inputs = layers.Input(shape=(input_dim,))
    x = inputs
    
    # Hidden layers
    for units in hidden_layers:
        x = layers.Dense(units, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)
    
    # Output layer (softmax for multi-class)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model


def plot_history(history: keras.callbacks.History, system_id: str) -> None:
    """Save training/validation loss and accuracy curves.

    Args:
        history:   Keras History object from model.fit().
        system_id: Used for the output filename.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    axes[0].plot(history.history["loss"], label="Training Loss")
    axes[0].plot(history.history["val_loss"], label="Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{system_id} - Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history.history["accuracy"], label="Training Accuracy")
    axes[1].plot(history.history["val_accuracy"], label="Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{system_id} - Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    output_path = FIGURE_DIR / f"dnn_{system_id}_history.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("History plot saved: %s", output_path)


def train_dnn(system_id: str, hidden_layers: list[int], dropout_rate: float,
              learning_rate: float, batch_size: int,
              X_train: np.ndarray, y_train: np.ndarray,
              X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Train one DNN configuration and return result dict.

    Args:
        system_id:      Key for this model.
        hidden_layers:  List of hidden layer sizes.
        dropout_rate:   Dropout rate (0 = none).
        learning_rate:  Adam LR.
        batch_size:     Mini-batch size.
        X_train, y_train: Training data.
        X_test, y_test:   Test data.

    Returns:
        Dict with keys: system, classifier, config, accuracy, f1.
    """
    input_dim   = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    # Compute class weights
    class_weights_dict = compute_class_weight(
        "balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    cw_dict = {i: w for i, w in enumerate(class_weights_dict)}
    
    # Build model
    model = build_model(input_dim, num_classes, hidden_layers, dropout_rate, learning_rate)
    
    # Callbacks
    checkpoint_path = MODEL_DIR / f"{system_id}.keras"
    callbacks = [
        keras.callbacks.EarlyStopping(patience=DNN_PATIENCE, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, verbose=0)
    ]
    
    # Train
    history = model.fit(
        X_train, y_train,
        epochs=DNN_EPOCHS,
        batch_size=batch_size,
        validation_split=0.1,
        class_weight=cw_dict,
        callbacks=callbacks,
        verbose=0
    )
    
    # Load best checkpoint and evaluate
    best_model = keras.models.load_model(checkpoint_path)
    y_pred_probs = best_model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    f1 = report.get("macro avg", {}).get("f1-score", 0.0)
    
    # Plot history
    plot_history(history, system_id)
    
    # Format config string
    config_str = f"layers={hidden_layers}, dropout={dropout_rate:.1f}, lr={learning_rate}"
    
    result = {
        "system": system_id,
        "classifier": "DenseNN",
        "config": config_str,
        "accuracy": accuracy,
        "f1": f1
    }
    
    return result


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()

    all_results: list[dict] = []

    for (system_id, hidden_layers, dropout_rate, learning_rate, batch_size) in DNN_CONFIGS:
        logger.info("=" * 60)
        logger.info("Training DNN config: %s | layers=%s | dropout=%.1f | lr=%s",
                    system_id, hidden_layers, dropout_rate, learning_rate)

        result = train_dnn(
            system_id, hidden_layers, dropout_rate, learning_rate, batch_size,
            X_train, y_train, X_test, y_test
        )
        if result:
            all_results.append(result)
            logger.info("Test Accuracy: %.4f", result["accuracy"])

    # Merge into results.json
    existing: list[dict] = []
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            existing = json.load(f)

    updated_ids = {r["system"] for r in all_results}
    merged = [r for r in existing if r["system"] not in updated_ids]
    merged.extend(all_results)

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    logger.info("DNN results saved to %s", RESULTS_FILE)
