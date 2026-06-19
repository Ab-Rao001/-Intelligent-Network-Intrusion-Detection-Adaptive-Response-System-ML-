"""
ann_model.py — Supervised Learning Module

Provides:
  - ANN (Keras Sequential) for multi-class attack classification
  - Optional SVM classifier (sklearn)
  - Training with early stopping and model checkpointing
  - Prediction and evaluation utilities
"""

import os
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from utils.logger import app_logger


# Suppress TF info logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

ATTACK_CLASSES = ["Normal", "DOS", "Probe", "R2L", "U2R"]


class ANNClassifier:
    """Deep Neural Network for multi-class intrusion detection."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 5,
        hidden_layers: list[int] | None = None,
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
    ):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_layers = hidden_layers or [128, 64, 32]
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.history = None

    def _build_model(self) -> keras.Model:
        """Build and compile a Sequential ANN."""
        model = keras.Sequential(name="IDS_ANN")
        model.add(layers.Input(shape=(self.input_dim,)))

        for i, units in enumerate(self.hidden_layers):
            model.add(layers.Dense(units, activation="relu", name=f"dense_{i}"))
            model.add(layers.BatchNormalization(name=f"bn_{i}"))
            model.add(layers.Dropout(self.dropout_rate, name=f"dropout_{i}"))

        model.add(layers.Dense(self.num_classes, activation="softmax", name="output"))

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        epochs: int = 50,
        batch_size: int = 64,
        save_dir: str = "saved_models",
        progress_callback=None,
    ):
        """Train the ANN with early stopping.

        Args:
            progress_callback: Optional callable func(epoch, logs) 
            called at the end of each epoch to report progress.

        Returns:
            keras History object
        """
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, "ann_model.keras")

        cb_list = [
            callbacks.EarlyStopping(
                monitor="val_loss" if X_val is not None else "loss",
                patience=5,
                restore_best_weights=True,
                min_delta=1e-4,
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss" if X_val is not None else "loss",
                factor=0.5,
                patience=2,
                min_lr=1e-6,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                model_path, save_best_only=True,
                monitor="val_loss" if X_val is not None else "loss",
            ),
        ]
        
        if progress_callback is not None:
            class ProgressCallback(callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    progress_callback(epoch, logs)
            cb_list.append(ProgressCallback())

        validation_data = (X_val, y_val) if X_val is not None else None
        
        # Compute dynamic class weights to handle minority classes (U2R, R2L)
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = {cls: weight for cls, weight in zip(classes, weights)}

        try:
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=cb_list,
                class_weight=class_weight_dict,
                verbose=1,
            )
            app_logger.info("ANN training completed successfully.")
            return self.history
        except ValueError as ve:
            app_logger.error(f"ValueError during ANN training (shape mismatch): {str(ve)}", exc_info=True)
            raise ValueError(f"Failed to train ANN due to data shape mismatch. Check input dimension ({self.input_dim}): {str(ve)}")
        except MemoryError as me:
            app_logger.error(f"MemoryError during ANN training: {str(me)}", exc_info=True)
            raise MemoryError("System ran out of memory during deep learning training. Try reducing batch_size.")
        except Exception as e:
            app_logger.error(f"Unexpected error during ANN training: {str(e)}", exc_info=True)
            raise RuntimeError(f"Unexpected error during ANN training: {str(e)}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class indices."""
        try:
            probs = self.model.predict(X, verbose=0)
            return np.argmax(probs, axis=1)
        except Exception as e:
            app_logger.error(f"Error during ANN prediction: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to run predictions: {str(e)}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities."""
        try:
            return self.model.predict(X, verbose=0)
        except Exception as e:
            app_logger.error(f"Error during ANN predict_proba: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to run predict_proba: {str(e)}")

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate model and return metrics dict."""
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(
            y_test, y_pred,
            target_names=ATTACK_CLASSES,
            labels=range(len(ATTACK_CLASSES)),
            output_dict=True,
            zero_division=0,
        )
        return {"accuracy": acc, "report": report, "y_pred": y_pred}

    def save(self, path: str = "saved_models/ann_model.keras"):
        """Save the Keras model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"ANN model saved to {path}")

    def load(self, path: str = "saved_models/ann_model.keras"):
        """Load a saved Keras model."""
        self.model = keras.models.load_model(path)

    def summary(self) -> str:
        """Return model summary as string."""
        lines = []
        self.model.summary(print_fn=lambda x: lines.append(x))
        return "\n".join(lines)


class SVMClassifier:
    """Optional SVM classifier for comparison."""

    def __init__(self, kernel: str = "rbf", C: float = 1.0, random_state: int = 42):
        self.model = SVC(
            kernel=kernel, C=C, random_state=random_state,
            decision_function_shape="ovr", probability=True,
        )
        self.is_trained = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the SVM. Warning: can be slow on large datasets."""
        self.model.fit(X_train, y_train)
        self.is_trained = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(
            y_test, y_pred,
            target_names=ATTACK_CLASSES,
            labels=range(len(ATTACK_CLASSES)),
            output_dict=True,
            zero_division=0,
        )
        return {"accuracy": acc, "report": report, "y_pred": y_pred}

    def save(self, path: str = "saved_models/svm_model.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"SVM model saved to {path}")

    def load(self, path: str = "saved_models/svm_model.pkl"):
        self.model = joblib.load(path)
        self.is_trained = True


if __name__ == "__main__":
    # Quick test
    n, d = 1000, 41
    X = np.random.rand(n, d).astype(np.float32)
    y = np.random.randint(0, 5, n)
    split = int(0.8 * n)

    print("=" * 50)
    print("Testing ANN Classifier")
    print("=" * 50)
    ann = ANNClassifier(input_dim=d)
    ann.train(X[:split], y[:split], X[split:], y[split:], epochs=5)
    result = ann.evaluate(X[split:], y[split:])
    print(f"ANN Accuracy: {result['accuracy']:.4f}")
    ann.save()

    print("\n" + "=" * 50)
    print("Testing SVM Classifier")
    print("=" * 50)
    svm = SVMClassifier()
    svm.train(X[:split], y[:split])
    result = svm.evaluate(X[split:], y[split:])
    print(f"SVM Accuracy: {result['accuracy']:.4f}")
    svm.save()
