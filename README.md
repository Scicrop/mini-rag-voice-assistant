# mini-rag-voice-assistant

sudo apt install portaudio19-dev python3-pyaudio
sudo apt install python3-pip
sudo apt install -y cmake g++ libsndfile1
sudo apt-get purge libespeak-ng*
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
cp mini-rag-voice-assistant/piper-requirements.txt piper/src/python_run/requirements.txt
pip install piper/src/python_run/
sudo cp piper-phonemize/onnxruntime-linux-aarch64-1.16.3/lib/libonnxruntime.so /usr/lib
sudo ldconfig