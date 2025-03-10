# mini-rag-voice-assistant

## Important
Make sure that your NVIDIA Jetson Orin Nano firmware is at version >= 36.4.2, and you are running JetPack version 6.2 and your board is working with MAXN SUPER  unregulated performance.


## Dependencies
### Generic Package Dependencies
```
sudo apt install portaudio19-dev python3-pyaudio
sudo apt install python3-pip libopenblas-dev ffmpeg 
sudo apt install -y cmake g++ libsndfile1
sudo apt-get purge libespeak-ng*
sudo apt-get install nvidia-jetpack
sudo apt install git cmake build-essential python3-dev libcudnn9 libcudnn9-dev tensorrt-dev
sudo apt-get remove --purge cmake
sudo apt-get install software-properties-common lsb-release
sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc | sudo apt-key add -
sudo apt-get update
sudo apt-get install cmake

```
### Cuda and PyTorch
```
sudo -i
cd /opt
wget raw.githubusercontent.com/pytorch/pytorch/5c6af2b583709f6176898c017424dc9981023c28/.ci/docker/common/install_cusparselt.sh
export CUDA_VERSION=12.1
bash ./install_cusparselt.sh
exit
python3 -m pip install --upgrade pip; python3 -m pip install numpy=='1.26.1';
python3 -m pip install --no-cache https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
pip3 install numpy wheel setuptools packaging
```
### Ollama 
```
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull tinyllama
```
```
### STT and TTS Dependencies (ESPEAK-NG fork)
```
mkdir git
cd git
git clone https://github.com/rhasspy/espeak-ng
cd espeak-ng
./autogen.sh
./configure
make
sudo make install
cd ..
```
### STT and TTS Dependencies (Onnxruntime for piper-tts)
```
git clone --recursive -b v1.20.0 https://github.com/microsoft/onnxruntime.git
cd onnxruntime
git checkout v1.20.0
git submodule update --init --recursive
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu/tegra:$LD_LIBRARY_PATH
./build.sh \
  --config Release \
  --update \
  --build \
  --build_wheel \
  --use_cuda \
  --cuda_home /usr/local/cuda \
  --cudnn_home /usr/lib/aarch64-linux-gnu \
  --skip_tests \
  --build_shared_lib
cd build/Linux/Release
make
sudo make install
pip3 install dist/onnxruntime_gpu*.whl
sudo cp libonnxruntime.so /usr/lib
sudo ldconfig
cd ../../../../
```
### STT and TTS Dependencies (piper-phonemize)
```
git clone https://github.com/rhasspy/piper-phonemize
cd piper-phonemize/
export CPLUS_INCLUDE_PATH=/usr/local/include/onnxruntime:$CPLUS_INCLUDE_PATH
pip install -e .
cd ..
```
### STT and TTS Dependencies (piper-tts)
```
git clone https://github.com/Scicrop/mini-rag-voice-assistant
git clone https://github.com/rhasspy/piper
cp mini-rag-voice-assistant/piper-requirements.txt piper/src/python_run/requirements.txt
pip install piper/src/python_run/
cd mini-rag-voice-assistant
```
## Piper-voice Portuguese
Choose only one of the two options, Portuguese or English
```
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/edresson/low/pt_BR-edresson-low.onnx -O voice.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/edresson/low/pt_BR-edresson-low.onnx.json -O voice.onnx.json
```
## Piper-voice English
Choose only one of the two options, Portuguese or English
```
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx -O voice.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json -O voice.onnx.json
```