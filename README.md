# mini-rag-voice-assistant

sudo apt install portaudio19-dev python3-pyaudio
sudo apt install python3-pip
sudo apt install -y cmake g++ libsndfile1
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
git clone https://github.com/rhasspy/piper