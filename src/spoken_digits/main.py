import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from spoken_digits import config
from spoken_digits.data import load_fsdd, split_fsdd
from spoken_digits.preprocessing import preprocess_audio
from spoken_digits.features import mfcc
from spoken_digits.models import GMMClassifier


def extract_features(df):
    """
    Extract MFCC features from all recordings in the df.

    Returns
    -------
    dict
        Maps each digit label to a matrix containing all
        MFCC frames from that digit.
    """

    X = {}

    for label, group in df.groupby("label"):

        features = []

        for _, row in group.iterrows():

            x = preprocess_audio(
                row["x"]
            )

            mfcc_features = mfcc(
                x=x,
                fs=config.FS,
                window_length=config.WINDOW_LENGTH,
                hop_length=config.HOP_LENGTH,
                use_vad=config.USE_VAD
            )

            features.append(mfcc_features)

        X[label] = np.vstack(features)

        print(
            f"Digit {label}: "
            f"{len(features)} recordings, "
            f"{X[label].shape[0]} frames"
        )

    return X


def evaluate(classifier, test_df):
    """
    Classify the test recordings and calculate accuracy.

    Returns
    -------
    y_true : ndarray
    y_pred : ndarray
    """

    y_true = []
    y_pred = []

    for _, row in test_df.iterrows():

        x = preprocess_audio(
            row["x"]
        )

        features = mfcc(
            x=x,
            fs=config.FS,
            window_length=config.WINDOW_LENGTH,
            hop_length=config.HOP_LENGTH,
            use_vad=config.USE_VAD
        )

        prediction = classifier.predict(features)

        y_true.append(row["label"])
        y_pred.append(prediction)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    accuracy = np.mean(y_true == y_pred)

    print(f"\nAccuracy: {accuracy:.3%}")

    return y_true, y_pred


def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix."""

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=range(10),
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=range(10),
    )

    disp.plot(
        cmap="Blues",
        values_format="d",
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.show()


def main():

    # Load and split data
    df = load_fsdd(config.RECORDING_DIR)
    train_df, test_df = split_fsdd(df)

    print(f"Training recordings: {len(train_df)}")
    print(f"Test recordings:     {len(test_df)}")

    # Extract features
    X_train = extract_features(train_df)

    # Training
    classifier = GMMClassifier(
        C=config.N_COMPONENTS,
        seed=config.SEED,
    )

    classifier.fit(X_train)

    # Evaluation
    y_true, y_pred = evaluate(
        classifier,
        test_df,
    )

    # Confusion matrix
    plot_confusion_matrix(
        y_true,
        y_pred,
    )


if __name__ == "__main__":
    main()