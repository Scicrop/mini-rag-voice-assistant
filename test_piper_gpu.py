# test_piper_gpu.py
from piper import PiperVoice
import wave
import time
import onnxruntime as ort

# Verifique os provedores disponíveis antes de carregar o Piper
print("Provedores disponíveis:", ort.get_available_providers())

# Carregue o modelo (ajuste os caminhos conforme seus arquivos)
voice = PiperVoice.load("voice.onnx", config_path="voice.onnx.json")

# Texto de teste
text = "Olá, este é um teste para verificar a GPU."

# Medir o tempo de síntese
start_time = time.time()
with wave.open("test.wav", "wb") as wav_file:
    voice.synthesize(text, wav_file)
end_time = time.time()

print(f"Tempo de síntese: {end_time - start_time:.2f} segundos")
