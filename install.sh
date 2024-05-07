conda env create --name detect_znak-onnx-cpu python=3.8 -f environment.yaml
conda activate detect_znak-onnx-cpu
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cpuonly -c pytorch
pip install -U openmim
mim install mmcv==2.0.1
mim install mmdet==3.1.0
mim install mmengine

cd ..
git clone https://github.com/open-mmlab/mmocr.git
git clone https://github.com/open-mmlab/mmdeploy.git
cd mmocr
pip install -v -e .
pip install -r requirements/albu.txt
pip install -r requirements.txt
pip install git+https://github.com/fcakyon/craft-text-detector.git
pip install pytorch_lightning
pip install gevent
pip install ujson
pip install ultralytics
#download data folder with weights



cd ../mmdeploy

python3 tools/scripts/build_ubuntu_x64_ort.py $(nproc)
export PYTHONPATH=$(pwd)/build/lib:$PYTHONPATH
export LD_LIBRARY_PATH=$(pwd)/../mmdeploy-dep/onnxruntime-linux-x64-1.8.1/lib/:$LD_LIBRARY_PATH

cd ../detect_znak/
mv scripts/*  ../mmocr/configs/textrecog/abinet/
