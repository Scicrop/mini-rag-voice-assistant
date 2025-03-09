# mini-rag-voice-assistant

## Important
Make sure that your NVIDIA Jetson Orin Nano firmware is at version >= 36.4.2, and you are running JetPack version 6.2 and your board is working with MAXN SUPER  unregulated performance.


## Dependencies
```
sudo apt install portaudio19-dev python3-pyaudio
sudo apt install python3-pip libopenblas-dev ffmpeg 
sudo apt install -y cmake g++ libsndfile1
sudo apt-get purge libespeak-ng*
sudo apt-get install nvidia-jetpack
sudo -i
cd /opt
wget raw.githubusercontent.com/pytorch/pytorch/5c6af2b583709f6176898c017424dc9981023c28/.ci/docker/common/install_cusparselt.sh
export CUDA_VERSION=12.1
bash ./install_cusparselt.sh
exit
export TORCH_INSTALL=https://developer.download.nvidia.cn/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
python3 -m pip install --upgrade pip; python3 -m pip install numpy==’1.26.1’; python3 -m pip install --no-cache $TORCH_INSTALL


git clone https://github.com/rhasspy/espeak-ng
./autogen.sh
./configure
make
sudo make install
cd ..
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