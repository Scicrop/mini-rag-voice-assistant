#!/usr/bin/env python3
"""
Módulo matrix_behaviors
------------------------
Fornece animações para uma matriz LED 8x8 (WS2812) via SPI.
Os behaviors disponíveis são:
  - "thinking"
  - "listening"
  - "pulse_green"
  - "pulse_blue"

As animações rodam em uma thread separada. Para controlar o behavior,
basta chamar a função set_behavior(new_behavior). Para iniciar a animação,
use start_animation(), e para parar, use stop_animation().
"""

import time
import math
import spidev
import threading

# Configurações da matriz
NUM_LEDS = 64      # 8x8
COLS = 8
ROWS = 8
MIN_BRIGHTNESS = 32        # Brilho mínimo para gradiente
MAX_PULSE_BRIGHTNESS = 128 # Brilho máximo para os behaviors de pulso

# Configuração do SPI
_spi = spidev.SpiDev()
_spi.open(0, 0)             # Usando SPI0; ajuste se necessário
_spi.max_speed_hz = 2400000   # 2.4MHz (~416 ns por SPI bit)

# Variáveis de controle de animação
current_behavior = "thinking"  # Behavior inicial
_behavior_lock = threading.Lock()
_running = False
_animation_thread = None

def encode_pixel(pixel):
    """
    Codifica um pixel (r, g, b) para WS2812 usando 3 SPI bits por bit.
    Utiliza 0b110 para '1' e 0b100 para '0'.
    WS2812 espera a ordem GRB.
    Retorna uma lista de 9 bytes.
    """
    # Reordena para GRB
    colors = [pixel[1], pixel[0], pixel[2]]
    bitstring = 0
    for color in colors:
        for i in range(7, -1, -1):
            bit = (color >> i) & 1
            encoding = 0b110 if bit else 0b100
            bitstring = (bitstring << 3) | encoding
    result = []
    for i in range(9):
        shift = 8 * (9 - i - 1)
        byte = (bitstring >> shift) & 0xFF
        result.append(byte)
    return result

def send_matrix(matrix):
    """
    Recebe uma lista de 64 tuplas (r, g, b) e envia os dados via SPI.
    Após o envio, aguarda 100µs para o sinal de reset.
    """
    data = []
    for pixel in matrix:
        data.extend(encode_pixel(pixel))
    _spi.writebytes(data)
    time.sleep(0.0001)  # 100µs para reset

# --- Behaviors com gradiente por coluna ---

def thinking_cycle():
    """
    Behavior "thinking":
    Cada coluna, da esquerda para a direita, acende gradualmente em tons de azul.
    As colunas já ativadas mantêm seu brilho alvo.
    """
    steps = 8
    for col in range(COLS):
        for step in range(steps):
            matrix = []
            for row in range(ROWS):
                for c in range(COLS):
                    if c < col:
                        target = int(MIN_BRIGHTNESS + (c / (COLS - 1)) * (255 - MIN_BRIGHTNESS))
                        brightness = target
                    elif c == col:
                        target = int(MIN_BRIGHTNESS + (c / (COLS - 1)) * (255 - MIN_BRIGHTNESS))
                        brightness = int(MIN_BRIGHTNESS + ((step + 1) / steps) * (target - MIN_BRIGHTNESS))
                    else:
                        brightness = 0
                    matrix.append((0, 0, brightness))
            send_matrix(matrix)
            time.sleep(0.05)

def listening_cycle():
    """
    Behavior "listening":
    Igual ao "thinking", mas utiliza tons de verde.
    """
    steps = 8
    for col in range(COLS):
        for step in range(steps):
            matrix = []
            for row in range(ROWS):
                for c in range(COLS):
                    if c < col:
                        target = int(MIN_BRIGHTNESS + (c / (COLS - 1)) * (255 - MIN_BRIGHTNESS))
                        brightness = target
                    elif c == col:
                        target = int(MIN_BRIGHTNESS + (c / (COLS - 1)) * (255 - MIN_BRIGHTNESS))
                        brightness = int(MIN_BRIGHTNESS + ((step + 1) / steps) * (target - MIN_BRIGHTNESS))
                    else:
                        brightness = 0
                    matrix.append((0, brightness, 0))
            send_matrix(matrix)
            time.sleep(0.05)

# --- Behaviors de pulso (respiração) para toda a matriz ---

def pulse_green_cycle(cycle_time=5):
    """
    Behavior "pulse_green":
    Efeito de respiração para toda a matriz com tons de verde.
    Durante um ciclo (cycle_time segundos), o brilho varia de 0 até MAX_PULSE_BRIGHTNESS e volta a 0.
    """
    start_time = time.time()
    while time.time() - start_time < cycle_time:
        t = time.time() - start_time
        brightness = int((math.sin(2 * math.pi * t / cycle_time) + 1) / 2 * MAX_PULSE_BRIGHTNESS)
        matrix = [(0, brightness, 0)] * NUM_LEDS
        send_matrix(matrix)
        time.sleep(0.05)

def pulse_blue_cycle(cycle_time=5):
    """
    Behavior "pulse_blue":
    Efeito de respiração para toda a matriz com tons de azul.
    Durante um ciclo (cycle_time segundos), o brilho varia de 0 até MAX_PULSE_BRIGHTNESS e volta a 0.
    """
    start_time = time.time()
    while time.time() - start_time < cycle_time:
        t = time.time() - start_time
        brightness = int((math.sin(2 * math.pi * t / cycle_time) + 1) / 2 * MAX_PULSE_BRIGHTNESS)
        matrix = [(0, 0, brightness)] * NUM_LEDS
        send_matrix(matrix)
        time.sleep(0.05)

# --- Loop de animação em thread ---

def _animation_loop():
    global current_behavior
    while _running:
        with _behavior_lock:
            b = current_behavior
        if b == "thinking":
            thinking_cycle()
        elif b == "listening":
            listening_cycle()
        elif b == "pulse_green":
            pulse_green_cycle(cycle_time=5)
        elif b == "pulse_blue":
            pulse_blue_cycle(cycle_time=5)
        else:
            # Behavior desconhecido: apaga LEDs
            send_matrix([(0, 0, 0)] * NUM_LEDS)
            time.sleep(0.1)

def start_animation():
    """
    Inicia a thread de animação que observa a variável current_behavior.
    """
    global _running, _animation_thread
    _running = True
    _animation_thread = threading.Thread(target=_animation_loop, daemon=True)
    _animation_thread.start()

def stop_animation():
    """
    Encerra a thread de animação e apaga os LEDs.
    """
    global _running
    _running = False
    send_matrix([(0, 0, 0)] * NUM_LEDS)
    _spi.close()

def set_behavior(new_behavior):
    """
    Atualiza o behavior atual.
    Parâmetro:
      new_behavior: string ("thinking", "listening", "pulse_green" ou "pulse_blue")
    """
    global current_behavior
    with _behavior_lock:
        current_behavior = new_behavior

# Inicializa o lock para a variável current_behavior
_behavior_lock = threading.Lock()

# Exemplo de uso (caso o módulo seja executado diretamente)
if __name__ == '__main__':
    start_animation()
    try:
        # Exemplo: o behavior muda a cada 10 segundos
        behaviors = ["thinking", "listening", "pulse_green", "pulse_blue"]
        idx = 0
        while True:
            set_behavior(behaviors[idx % len(behaviors)])
            print("Behavior atual:", behaviors[idx % len(behaviors)])
            idx += 1
            time.sleep(10)
    except KeyboardInterrupt:
        print("Encerrando animação.")
        stop_animation()
