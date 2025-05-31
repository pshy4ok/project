from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import librosa
import tensorflow as tf
import sounddevice as sd
from fastapi.middleware.cors import CORSMiddleware
import io
from pydub import AudioSegment
import tempfile
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SR = 12000
N_FFT = 512
N_MELS = 96
HOP_LEN = 256
DURATION = 29.12

MODEL_PATH = "model.keras"
GENRE_LIST = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
GENRE_LIST_UA = ["блюз", "класика", "кантрі", "диско", "хіп-хоп", "джаз", "метал", "поп", "регі", "рок"]
genre_dict = {i: genre for i, genre in enumerate(GENRE_LIST)}
genre_dict_ua = {i: genre for i, genre in enumerate(GENRE_LIST_UA)}
model = tf.keras.models.load_model(MODEL_PATH)


def compute_melgram(audio, sr=SR):
    try:
        n_sample = audio.shape[0]
        n_sample_fit = int(DURATION * sr)

        if n_sample < n_sample_fit:
            audio = np.hstack((audio, np.zeros((n_sample_fit - n_sample,))))
        elif n_sample > n_sample_fit:
            audio = audio[(n_sample - n_sample_fit) // 2:(n_sample + n_sample_fit) // 2]

        melgram = librosa.feature.melspectrogram(
            y=audio, sr=sr, hop_length=HOP_LEN, n_fft=N_FFT, n_mels=N_MELS
        )
        ret = librosa.power_to_db(melgram, ref=np.max)
        ret = ret[:, :, np.newaxis]
        return ret
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Помилка обробки аудіо: {str(e)}")


def extract_melgrams(audio, sr=SR):
    try:
        n_sample = audio.shape[0]
        n_sample_fit = int(DURATION * sr)
        num_frames = max(1, n_sample // n_sample_fit)
        melgrams = []

        for i in range(num_frames):
            start = i * n_sample_fit
            end = min(start + n_sample_fit, n_sample)
            segment = audio[start:end]
            melgram = compute_melgram(segment, sr)
            melgrams.append(melgram)

        return np.stack(melgrams, axis=0)
    except Exception as e:
        print(f"Error extracting melgrams: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Помилка обробки аудіо: {str(e)}")


@app.post("/predict/")
async def predict_genre(file: UploadFile = File(...)):
    print("Receiving file:", file.filename)
    try:
        audio_bytes = await file.read()
        print(f"File size: {len(audio_bytes)} bytes")

        try:
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
            audio_segment = audio_segment.set_frame_rate(SR).set_channels(1)
            audio = np.array(audio_segment.get_array_of_samples(), dtype=np.float32) / 32768.0
            print("File converted successfully with pydub")
        except Exception as e:
            print(f"Pydub failed: {str(e)}. Falling back to librosa.")
            audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=SR, mono=True)

        print("File loaded successfully")
        if len(audio) == 0:
            raise ValueError("Аудіофайл порожній")

        melgrams = extract_melgrams(audio)
        predictions = model.predict(melgrams)
        mean_pred = np.mean(predictions, axis=0)
        predicted_idx = np.argmax(mean_pred)
        genre = genre_dict[predicted_idx]
        genre_ua = genre_dict_ua[predicted_idx]
        confidence = float(mean_pred[predicted_idx]) * 100

        all_genres = []
        for i, (genre_en, genre_ua_name) in enumerate(zip(GENRE_LIST, GENRE_LIST_UA)):
            all_genres.append({
                "genre": genre_en,
                "genre_ua": genre_ua_name,
                "confidence": float(mean_pred[i]) * 100
            })

        all_genres.sort(key=lambda x: x['confidence'], reverse=True)

        print("Prediction made:", genre, confidence)
        return {
            "genre": genre,
            "genre_ua": genre_ua,
            "confidence": confidence,
            "all_genres": all_genres
        }
    except Exception as e:
        print("Error processing file:", str(e))
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/microphone/")
async def predict_from_microphone(duration: int = 16):
    try:
        print("Запис аудіо з мікрофона...")
        audio = sd.rec(int(duration * SR), samplerate=SR, channels=1, dtype='float32')
        sd.wait()
        audio = audio.flatten()

        melgrams = extract_melgrams(audio)
        predictions = model.predict(melgrams)
        mean_pred = np.mean(predictions, axis=0)
        predicted_idx = np.argmax(mean_pred)
        genre = genre_dict[predicted_idx]
        genre_ua = genre_dict_ua[predicted_idx]
        confidence = float(mean_pred[predicted_idx]) * 100

        all_genres = []
        for i, (genre_en, genre_ua_name) in enumerate(zip(GENRE_LIST, GENRE_LIST_UA)):
            all_genres.append({
                "genre": genre_en,
                "genre_ua": genre_ua_name,
                "confidence": float(mean_pred[i]) * 100
            })

        all_genres.sort(key=lambda x: x['confidence'], reverse=True)

        print("Prediction made:", genre, confidence)
        return {
            "genre": genre,
            "genre_ua": genre_ua,
            "confidence": confidence,
            "all_genres": all_genres
        }
    except Exception as e:
        print("Error processing microphone input:", str(e))
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)