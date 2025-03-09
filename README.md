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
python3 -m pip install --no-cache https://developer.download.nvidia.com/compute/redist/jp/v60dp/pytorch/torch-2.3.0a0+40ec155e58.nv24.03.13384722-cp310-cp310-linux_aarch64.whl
pip3 install numpy wheel setuptools packaging
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
git clone --recursive -b v1.16.3 https://github.com/microsoft/onnxruntime.git
cd onnxruntime
git checkout v1.16.3
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
  --parallel \
  --skip_tests \
  --build_shared_lib

git clone https://github.com/rhasspy/piper-phonemize
cd piper-phonemize/
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-aarch64-1.16.3.tgz
tar -xzf onnxruntime-linux-aarch64-1.16.3.tgz
mkdir -p lib/Linux-aarch64/onnxruntime
cp -r onnxruntime-linux-aarch64-1.16.3/include lib/Linux-aarch64/onnxruntime/
cp -r onnxruntime-linux-aarch64-1.16.3/lib lib/Linux-aarch64/onnxruntime/
cd ..
git clone https://github.com/Scicrop/mini-rag-voice-assistant
git clone https://github.com/rhasspy/piper
cp mini-rag-voice-assistant/piper-requirements.txt piper/src/python_run/requirements.txt
pip install piper/src/python_run/
sudo cp piper-phonemize/onnxruntime-linux-aarch64-1.16.3/lib/libonnxruntime.so /usr/lib
sudo ldconfig
cd mini-rag-voice-assistant
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx -O voice.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json -O voice.onnx.json

```