import tensorflow as tf
from pathlib import Path

# Dataset configuration
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "data" / "classification"

TRAIN_DIR = DATASET_DIR / "Training"
TEST_DIR = DATASET_DIR / "Testing"

# Image configuration
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# Load training dataset
def load_training_data():

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=True,
        seed=SEED
    )

    return train_dataset


# Load testing dataset
def load_testing_data():

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False
    )

    return test_dataset


# Normalize images and improve data loading performance
def prepare_datasets():

    train_dataset = load_training_data()
    test_dataset = load_testing_data()

    class_names = train_dataset.class_names

    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = train_dataset.prefetch(
        buffer_size=AUTOTUNE
    )

    test_dataset = test_dataset.prefetch(
        buffer_size=AUTOTUNE
    )

    return train_dataset, test_dataset, class_names


if __name__ == "__main__":

    train_ds, test_ds, classes = prepare_datasets()

    print("\nDataset loaded successfully!")
    print("Classes:", classes)
    print("Training batches:", len(train_ds))
    print("Testing batches:", len(test_ds))
