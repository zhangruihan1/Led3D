# Led3D

## PyTorch Version

Convert to PyTorch, using the commands from this [Tutorial](https://blog.paperspace.com/convert-full-imagenet-pre-trained-model-from-mxnet-to-pytorch/).
```
pip install mxnet
pip install --upgrade mmdnn
cd Led3D/python
python -m mmdnn.conversion._script.convertToIR -f mxnet -n param/led3d-symbol.json -w param/led3d-0000.params -d param/led3d --inputShape 3,128,128
python -m mmdnn.conversion._script.IRToCode -f pytorch --IRModelPath param/led3d.pb --dstModelPath led3d.py --IRWeightPath param/led3d.npy -dw led3d.npy  
python -m mmdnn.conversion.examples.pytorch.imagenet_test --dump led3d.pth -n led3d.py -w led3d.npy
```
Usage
```python
import torch
from led3d import Led3D

model = Led3D()
model.load_state_dict(torch.load('Led3D/python/led3d.pth'))
model.eval()
```

Do not normalise inputs to (0, 1), leave them (0, 255)


## Original
This project is an implementation for "Led3D: An lightweight and efficent deep approach to recognizing low-quality 3d faces"  [ [Download](http://openaccess.thecvf.com/content_CVPR_2019/papers/Mu_Led3D_A_Lightweight_and_Efficient_Deep_Approach_to_Recognizing_Low-Quality_CVPR_2019_paper.pdf) ], which is accepted by **CVPR2019**.

Dataset: [Lock3DFace](http://irip.buaa.edu.cn/lock3dface/index.html)
<<<<<<< HEAD

![pipeline](fig/pipeline.png)

### Function

- 3D Face Preprocessing (Done.)
- 3D Face Augmentation (Done.)
- Python Inference Code (Done.)

### Citation
```latex
@InProceedings{Mu_2019_CVPR,
author = {Mu, Guodong and Huang, Di and Hu, Guosheng and Sun, Jia and Wang, Yunhong},
title = {Led3D: A Lightweight and Efficient Deep Approach to Recognizing Low-Quality 3D Faces},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```
