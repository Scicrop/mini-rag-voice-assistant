import pyaudio
import wave
import whisper
import ollama
import numpy as np
import time
import os
import argparse
from piper import PiperVoice
import onnxruntime as ort

# Configurações de áudio otimizadas
CHUNK = 512  # Menor tamanho de buffer para reduzir latência
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 8000  # Taxa de amostragem reduzida para menor carga
WAVE_OUTPUT_FILENAME = "input.wav"
SILENCE_THRESHOLD = 400  # Ajustado para sensibilidade em taxa menor
SILENCE_DURATION = 1.5
MAX_DURATION = 20


class VoiceAssistant:
    """Classe para gerenciar o assistente de voz no Jetson Orin Nano."""

    def __init__(self, device_index, model_name):
        self.device_index = device_index
        self.model_name = model_name

        # Inicialização única de recursos
        self.audio = pyaudio.PyAudio()
        self.whisper_model = whisper.load_model("tiny", device="cuda" if self.is_cuda_available() else "cpu")
        self.piper_session = ort.InferenceSession("voice.onnx",
                                                  providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.piper_voice = PiperVoice.load("voice.onnx", config_path="voice.onnx.json", session=self.piper_session)

    def is_cuda_available(self):
        """Verifica se CUDA está disponível para aceleração."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

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
        """Converte áudio em texto usando Whisper com GPU, se disponível."""
        try:
            result = self.whisper_model.transcribe(WAVE_OUTPUT_FILENAME)
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
        """Converte texto em fala usando Piper com GPU, se disponível."""
        audio_file = "response.wav"
        try:
            with wave.open(audio_file, "wb") as wav_file:
                self.piper_voice.synthesize(text, wav_file)
            os.system(f"aplay {audio_file}")
            os.remove(audio_file)  # Limpeza do arquivo temporário
        except Exception as e:
            print(f"Erro na síntese de voz: {e}")

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
                os.remove(WAVE_OUTPUT_FILENAME)  # Limpeza do arquivo de entrada
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