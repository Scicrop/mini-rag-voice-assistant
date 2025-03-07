import pyaudio
import wave
import whisper
import ollama
import numpy as np
import time
import os
import argparse
from piper import PiperVoice

# Configurações de áudio constantes
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
WAVE_OUTPUT_FILENAME = "input.wav"

# Parâmetros para detecção de voz
SILENCE_THRESHOLD = 500
SILENCE_DURATION = 1.5
MAX_DURATION = 20

# Carrega o modelo Piper uma vez no início
VOICE = PiperVoice.load("voice.onnx", config_path="voice.onnx.json")

def parse_arguments():
    """Parseia os argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description="Assistente de voz com STT, LLM e TTS.")
    parser.add_argument("--device-index", type=int, default=5, help="Índice do dispositivo de áudio (padrão: 5)")
    parser.add_argument("--model", type=str, default="tinyllama", help="Nome do modelo Ollama (padrão: tinyllama)")
    return parser.parse_args()

def is_speech(data, threshold=SILENCE_THRESHOLD):
    """Verifica se os dados de áudio contêm fala com base no limiar."""
    audio_data = np.frombuffer(data, dtype=np.int16)
    return np.abs(audio_data).mean() > threshold

def record_audio(device_index):
    """Grava áudio do microfone até detectar silêncio ou atingir o tempo máximo."""
    print("Aguardando você falar...")
    audio = pyaudio.PyAudio()
    try:
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True,
                            frames_per_buffer=CHUNK,
                            input_device_index=device_index)
    except OSError as e:
        print(f"Erro ao abrir dispositivo de áudio {device_index}: {e}")
        audio.terminate()
        return None

    frames = []
    recording = False
    silence_start = None

    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        if not recording and is_speech(data):
            print("Detectei voz! Gravando...")
            recording = True
            frames.append(data)
            silence_start = None
        elif recording:
            frames.append(data)
            if not is_speech(data):
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

    try:
        with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
        print("Gravação concluída!")
        return True
    except Exception as e:
        print(f"Erro ao salvar gravação: {e}")
        return None

def speech_to_text():
    """Converte áudio gravado em texto usando Whisper."""
    try:
        model = whisper.load_model("tiny")
        result = model.transcribe(WAVE_OUTPUT_FILENAME)
        return result["text"]
    except Exception as e:
        print(f"Erro na transcrição: {e}")
        return None

def ask_ollama(question, model_name):
    """Faz uma pergunta ao modelo Ollama especificado e retorna a resposta."""
    try:
        response = ollama.chat(model=model_name, messages=[
            {"role": "user", "content": question}
        ])
        return response["message"]["content"]
    except Exception as e:
        print(f"Erro ao consultar o modelo {model_name}: {e}")
        return None

def text_to_speech(text):
    """Converte texto em fala usando Piper TTS."""
    audio_file = "response.wav"
    try:
        with wave.open(audio_file, "wb") as wav_file:
            VOICE.synthesize(text, wav_file)
        os.system(f"aplay {audio_file}")
    except Exception as e:
        print(f"Erro na síntese de voz: {e}")

def main(device_index, model_name):
    """Loop principal do assistente de voz."""
    while True:
        if not record_audio(device_index):
            continue
        question = speech_to_text()
        if question:
            print(f"Pergunta reconhecida: {question}")
            answer = ask_ollama(question, model_name)
            if answer:
                print(f"Resposta do {model_name}: {answer}")
                text_to_speech(answer)
        print("Pronto para a próxima pergunta! (Ctrl+C para sair)")

if __name__ == "__main__":
    args = parse_arguments()
    try:
        main(args.device_index, args.model)
    except KeyboardInterrupt:
        print("Programa encerrado pelo usuário.")