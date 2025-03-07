import pyaudio

def list_audio_devices():
    audio = pyaudio.PyAudio()
    print("Dispositivos de áudio disponíveis:")
    try:
        for i in range(audio.get_device_count()):
            try:
                device_info = audio.get_device_info_by_index(i)
                if device_info["maxInputChannels"] > 0:  # Apenas dispositivos de entrada
                    print(f"Índice: {i} - Nome: {device_info['name']} - "
                          f"Canais de entrada: {device_info['maxInputChannels']} - "
                          f"Padrão: {device_info['defaultSampleRate']}")
            except Exception as e:
                print(f"Erro ao acessar dispositivo no índice {i}: {e}")
    except Exception as e:
        print(f"Erro ao listar dispositivos: {e}")
    finally:
        audio.terminate()

list_audio_devices()