import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from keras import layers, models, regularizers
from keras.api.utils import to_categorical
from keras.api.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, confusion_matrix
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SR = 12000
N_FFT = 512
N_MELS = 96
HOP_LEN = 256
DURA = 29.12
GENRE_DIR = "data"
GENRE_LIST = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
genre_dict = {i: genre for i, genre in enumerate(GENRE_LIST)}


def augment_audio(audio, sr):
    try:
        augmented = []
        augmented.append(audio)
        noise = np.random.randn(len(audio))
        audio_noise = audio + 0.005 * noise
        augmented.append(audio_noise)
        audio_pitch = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=2)
        augmented.append(audio_pitch)
        audio_stretch = librosa.effects.time_stretch(y=audio, rate=0.9)
        augmented.append(audio_stretch)
        shift_amount = int(sr * 0.5)  # Shift by 0.5 seconds
        audio_shift = np.roll(audio, shift_amount)
        augmented.append(audio_shift)
        audio_volume = audio * np.random.uniform(0.5, 1.5)
        augmented.append(audio_volume)
        background_noise = np.random.normal(0, 0.01, len(audio))
        audio_bg_noise = audio + background_noise
        augmented.append(audio_bg_noise)
        return augmented
    except Exception as e:
        logger.error(f"Error in audio augmentation: {str(e)}")
        return [audio]


def compute_melgram(audio, sr=SR):
    try:
        n_sample = audio.shape[0]
        n_sample_fit = int(DURA * sr)

        if n_sample < n_sample_fit:
            audio = np.hstack((audio, np.zeros((n_sample_fit - n_sample,))))
        elif n_sample > n_sample_fit:
            audio = audio[(n_sample - n_sample_fit) // 2:(n_sample + n_sample_fit) // 2]

        melgram = librosa.feature.melspectrogram(
            y=audio, sr=sr, hop_length=HOP_LEN, n_fft=N_FFT, n_mels=N_MELS
        )
        ret = librosa.power_to_db(melgram, ref=np.max)
        ret = ret[:, :, np.newaxis]
        logger.debug(f"Melgram shape: {ret.shape}")
        return ret
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        return None


def extract_melgrams(audio, sr=SR):
    try:
        n_sample = audio.shape[0]
        n_sample_fit = int(DURA * sr)
        overlap = n_sample_fit // 2
        step = n_sample_fit - overlap
        num_frames = max(1, (n_sample - n_sample_fit) // step + 1)
        melgrams = []

        for i in range(num_frames):
            start = i * step
            end = min(start + n_sample_fit, n_sample)
            segment = audio[start:end]
            melgram = compute_melgram(segment, sr)
            if melgram is not None:
                melgrams.append(melgram)

        if not melgrams:
            return None
        melgrams = np.stack(melgrams, axis=0)
        logger.debug(f"Extracted melgrams shape: {melgrams.shape}")
        return melgrams
    except Exception as e:
        logger.error(f"Error extracting melgrams: {str(e)}")
        return None


def load_data():
    features = []
    labels = []
    logger.info(f"Loading data from {GENRE_DIR}")
    if not os.path.exists(GENRE_DIR):
        logger.error(f"Directory {GENRE_DIR} does not exist")
        return np.array([]), np.array([])

    for genre_label, genre in genre_dict.items():
        genre_folder = os.path.join(GENRE_DIR, genre)
        if not os.path.exists(genre_folder):
            logger.warning(f"Genre folder {genre_folder} does not exist")
            continue

        logger.info(f"Processing genre: {genre}")
        files = os.listdir(genre_folder)
        for file in files:
            file_path = os.path.join(genre_folder, file)
            logger.info(f"Processing file: {file_path}")
            try:
                audio, sr = librosa.load(file_path, sr=SR, mono=True)
                augmented_audios = augment_audio(audio, sr)
                for idx, aug_audio in enumerate(augmented_audios):
                    logger.info(f"Processing augmented audio {idx + 1}/{len(augmented_audios)}")
                    melgrams = extract_melgrams(aug_audio, sr)
                    if melgrams is not None:
                        for melgram in melgrams:
                            features.append(melgram)
                            labels.append(genre_label)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
                continue

    features = np.array(features)
    labels = np.array(labels)
    logger.info(f"Loaded {len(features)} samples with {len(labels)} labels")
    logger.info(f"Features shape: {features.shape}")
    return features, labels


def create_model(input_shape, num_genres):
    inputs = layers.Input(shape=input_shape)

    x = layers.BatchNormalization(axis=-1)(inputs)
    x = layers.Conv2D(16, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D((2, 4))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D((2, 4))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D((2, 4))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D((3, 5))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.7)(x)
    outputs = layers.Dense(num_genres, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_")}.png')
    plt.close()


if __name__ == '__main__':
    logger.info("Starting training process")
    features, labels = load_data()
    if len(features) == 0:
        logger.error("No valid audio files processed. Check data directory and file formats.")
    else:
        logger.info("Splitting data into train and validation sets")
        X_train, X_val, y_train, y_val = train_test_split(
            features, to_categorical(labels), test_size=0.2, random_state=42
        )
        input_shape = (96, 1366, 1)
        num_genres = len(genre_dict)

        class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
        class_weight_dict = dict(enumerate(class_weights))

        logger.info("Creating model")
        model = create_model(input_shape, num_genres)

        logger.info(f"X_train shape: {X_train.shape}")
        logger.info(f"y_train shape: {y_train.shape}")
        logger.info(f"X_val shape: {X_val.shape}")
        logger.info(f"y_val shape: {y_val.shape}")

        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True),
            ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6)
        ]

        logger.info("Starting model training")
        model.fit(
            X_train, y_train, validation_data=(X_val, y_val),
            epochs=100, batch_size=32, callbacks=callbacks, class_weight=class_weight_dict,
            initial_epoch=0, verbose=1
        )

        logger.info("Evaluating model")
        y_pred_proba = model.predict(X_val)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_val, axis=1)
        auc = roc_auc_score(y_val, y_pred_proba, multi_class='ovr')
        logger.info(f"AUC-ROC: {auc:.4f}")

        cm = confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(cm, classes=GENRE_LIST, title='Confusion Matrix')
        logger.info("Confusion matrix saved as Confusion_Matrix.png")

        logger.info("Saving model")
        model.save('model.keras')