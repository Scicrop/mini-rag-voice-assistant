import pyaudio
import wave
import whisper
import ollama
import numpy as np
import time
import os
import argparse
from piper import PiperVoice
import torch

# Configurações de áudio otimizadas
CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 8000
WAVE_OUTPUT_FILENAME = "input.wav"
SILENCE_THRESHOLD = 400
SILENCE_DURATION = 1.5
MAX_DURATION = 20


class VoiceAssistant:
    """Classe para gerenciar o assistente de voz no Jetson Orin Nano."""

    def __init__(self, device_index, model_name):
        self.device_index = device_index
        self.model_name = model_name
        # Inicialização única de recursos
        self.audio = pyaudio.PyAudio()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Usando dispositivo: {self.device}")
        self.whisper_model = whisper.load_model("tiny", device=self.device)
        self.piper_voice = PiperVoice.load("voice.onnx", config_path="voice.onnx.json")

    def is_speech(self, data, threshold=SILENCE_THRESHOLD):
        """Verifica se os dados de áudio contêm fala."""
        audio_data = np.frombuffer(data, dtype=np.int16)
        return np.abs(audio_data).mean() > threshold

    def record_audio(self):
        """Grava áudio do microfone com detecção de silêncio."""
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
        """Converte áudio em texto usando Whisper com GPU, se disponível."""
        try:
            result = self.whisper_model.transcribe(WAVE_OUTPUT_FILENAME, language="pt")
            return result["text"]
        except Exception as e:
            print(f"Erro na transcrição: {e}")
            return None

    def ask_ollama(self, question):
        """Consulta o modelo Ollama especificado."""
        try:
            response = ollama.chat(model=self.model_name, messages=[
                {"role": "user", "content": question}
            ])
            return response["message"]["content"]
        except Exception as e:
            print(f"Erro ao consultar o modelo {self.model_name}: {e}")
            return None

    def text_to_speech(self, text):
        """Converte texto em fala usando Piper e reproduz diretamente com pyaudio."""
        audio_file = "response.wav"
        try:
            # Gera o arquivo WAV com Piper
            with wave.open(audio_file, "wb") as wav_file:
                self.piper_voice.synthesize(text, wav_file)

            # Reproduz o áudio diretamente com pyaudio
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
            # os.remove(audio_file)  # Limpeza opcional
        except Exception as e:
            print(f"Erro na síntese ou reprodução de voz: {e}")

    def run(self):
        """Executa o loop principal do assistente."""
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
        """Libera recursos ao encerrar."""
        self.audio.terminate()


def parse_arguments():
    """Parseia os argumentos da linha de comando."""
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