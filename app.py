import pyaudio
import wave
import whisper
import ollama
import numpy as np
import time
import os
from piper import PiperVoice

# Configurações de áudio
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
WAVE_OUTPUT_FILENAME = "input.wav"
DEVICE_INDEX = 5  # Logitech Stereo H650e

# Parâmetros para detecção de voz
SILENCE_THRESHOLD = 500
SILENCE_DURATION = 1.5
MAX_DURATION = 20

# Carrega o modelo Piper uma vez no início
voice = PiperVoice.load("voice.onnx", config_path="voice.onnx.json")


def is_speech(data, threshold):
    audio_data = np.frombuffer(data, dtype=np.int16)
    return np.abs(audio_data).mean() > threshold


def record_audio():
    print("Aguardando você falar...")
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK,
                        input_device_index=DEVICE_INDEX)

    frames = []
    recording = False
    silence_start = None

    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        if not recording and is_speech(data, SILENCE_THRESHOLD):
            print("Detectei voz! Gravando...")
            recording = True
            frames.append(data)
            silence_start = None
        elif recording:
            frames.append(data)
            if not is_speech(data, SILENCE_THRESHOLD):
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > SILENCE_DURATION:
                    print("Silêncio detectado. Parando gravação...")
                    break
            else:
                silence_start = None
            if len(frames) * CHUNK / RATE > MAX_DURATION:
                print("Tempo máximo atingido. Parando gravação...")
                break

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    print("Gravação concluída!")


def speech_to_text():
    model = whisper.load_model("tiny")
    result = model.transcribe(WAVE_OUTPUT_FILENAME)
    return result["text"]


def ask_tinyllama(question):
    response = ollama.chat(model="tinyllama", messages=[
        {"role": "user", "content": question}
    ])
    return response["message"]["content"]


def text_to_speech(text):
    audio_file = "response.wav"
    with wave.open(audio_file, "wb") as wav_file:
        voice.synthesize(text, wav_file)
    os.system("aplay " + audio_file)


def main():
    while True:
        record_audio()
        question = speech_to_text()
        print(f"Pergunta reconhecida: {question}")
        answer = ask_tinyllama(question)
        print(f"Resposta do TinyLlama: {answer}")
        text_to_speech(answer)
        print("Pronto para a próxima pergunta! (Ctrl+C para sair)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Programa encerrado pelo usuário.")