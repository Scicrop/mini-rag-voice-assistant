import pyaudio
import wave
from faster_whisper import WhisperModel
import ollama
import numpy as np
import time
import os
import argparse
from piper import PiperVoice
import torch
import onnxruntime as ort

# Configurações de áudio otimizadas
CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 8000
WAVE_OUTPUT_FILENAME = "input.wav"
SILENCE_THRESHOLD = 400
SILENCE_DURATION = 1.5
MAX_DURATION = 10

class VoiceAssistant:
    def __init__(self, device_index, model_name):
        self.device_index = device_index
        self.model_name = model_name
        self.audio = pyaudio.PyAudio()
        
        # Faster-Whisper na CPU
        self.whisper_device = "cpu"
        print(f"Dispositivo do Faster-Whisper: {self.whisper_device}")
        
        # Piper tenta GPU via onnxruntime
        self.piper_device = "cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"
        print(f"Dispositivo esperado do Piper: {self.piper_device}")
        
        # Ollama (assume GPU se compilado com CUDA)
        self.ollama_device = "cuda" if torch.cuda.is_available() else "cpu"  # Apenas indicativo
        print(f"Dispositivo esperado do Ollama: {self.ollama_device}")
        
        # Carrega o Faster-Whisper na CPU
        self.whisper_model = WhisperModel("tiny", device=self.whisper_device, compute_type="int8")
        
        # Carrega o Piper (GPU se disponível)
        self.piper_voice = PiperVoice.load("voice.onnx", config_path="voice.onnx.json")

    def is_speech(self, data, threshold=SILENCE_THRESHOLD):
        audio_data = np.frombuffer(data, dtype=np.int16)
        return np.abs(audio_data).mean() > threshold

    def record_audio(self):
        print("Aguardando você falar...")
        try:
            stream = self.audio.open(format=FORMAT, channels=CHANNELS,
                                     rate=RATE, input=True,
                                     frames_per_buffer=CHUNK,
                                     input_device_index=self.device_index)
        except OSError as e:
            print(f"Erro ao abrir dispositivo {self.device_index}: {e}")
            return False

        frames = []
        recording = False
        silence_start = None

        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            if not recording and self.is_speech(data):
                print("Detectei voz! Gravando...")
                recording = True
                frames.append(data)
                silence_start = None
            elif recording:
                frames.append(data)
                if not self.is_speech(data):
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

        try:
            with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(self.audio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
            print("Gravação concluída!")
            return True
        except Exception as e:
            print(f"Erro ao salvar gravação: {e}")
            return False

    def speech_to_text(self):
        print(f"Dispositivo do Faster-Whisper: {self.whisper_device}")
        if not os.path.exists(WAVE_OUTPUT_FILENAME):
            print(f"Arquivo de áudio {WAVE_OUTPUT_FILENAME} não encontrado.")
            return None
        try:
            start_time = time.time()
            segments, info = self.whisper_model.transcribe(
                WAVE_OUTPUT_FILENAME,
                language="pt",
                initial_prompt="Este é um áudio em português brasileiro."
            )
            text = " ".join(segment.text for segment in segments)
            end_time = time.time()
            print(f"Tempo de transcrição: {end_time - start_time:.2f} segundos")
            return text
        except Exception as e:
            print(f"Erro na transcrição: {e}")
            return None

    def ask_ollama(self, question):
        print("Consultando Ollama (espera-se GPU)...")
        start_time = time.time()
        try:
            response = ollama.chat(model=self.model_name, messages=[
                {"role": "user", "content": question}
            ])
            end_time = time.time()
            print(f"Tempo de resposta do Ollama: {end_time - start_time:.2f} segundos")
            return response["message"]["content"]
        except Exception as e:
            print(f"Erro ao consultar o modelo {self.model_name}: {e}")
            return None

    def text_to_speech(self, text):
        print(f"Dispositivo do Piper: {self.piper_device}")
        audio_file = "response.wav"
        try:
            start_time = time.time()
            with wave.open(audio_file, "wb") as wav_file:
                self.piper_voice.synthesize(text, wav_file)
            end_time = time.time()
            print(f"Tempo de síntese Piper: {end_time - start_time:.2f} segundos")
            with wave.open(audio_file, 'rb') as wf:
                stream = self.audio.open(format=self.audio.get_format_from_width(wf.getsampwidth()),
                                         channels=wf.getnchannels(),
                                         rate=wf.getframerate(),
                                         output=True)
                data = wf.readframes(CHUNK)
                while data:
                    stream.write(data)
                    data = wf.readframes(CHUNK)
                stream.stop_stream()
                stream.close()
        except Exception as e:
            print(f"Erro na síntese ou reprodução de voz: {e}")

    def run(self):
        while True:
            if not self.record_audio():
                continue
            question = self.speech_to_text()
            if question:
                print(f"Pergunta reconhecida: {question}")
                answer = self.ask_ollama(question)
                if answer:
                    print(f"Resposta do {self.model_name}: {answer}")
                    self.text_to_speech(answer)
            if os.path.exists(WAVE_OUTPUT_FILENAME):
                os.remove(WAVE_OUTPUT_FILENAME)
            print("Pronto para a próxima pergunta! (Ctrl+C para sair)")

    def cleanup(self):
        self.audio.terminate()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Assistente de voz otimizado para Jetson Orin Nano.")
    parser.add_argument("--device-index", type=int, default=5, help="Índice do dispositivo de áudio (padrão: 5)")
    parser.add_argument("--model", type=str, default="tinyllama", help="Nome do modelo Ollama (padrão: tinyllama)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    assistant = VoiceAssistant(args.device_index, args.model)
    try:
        assistant.run()
    except KeyboardInterrupt:
        print("Programa encerrado pelo usuário.")
        assistant.cleanup()
