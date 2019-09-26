# Face-Detetion-Base-on-Mask-R-CNN
The project use Wider Face as train set. Download the training and validation data from: https://storage.cloud.google.com/maskrcnn-data/wider_face_split.zip

Or from the project home: http://shuoyang1213.me/WIDERFACE/ (Author = {Yang, Shuo and Luo, Ping and Loy, Chen Change and Tang, Xiaoou})

### Setup

```
virtualenv -p python3 env
source env/bin/activate
```

```
pip3 install requirements.txt
```

Train model with cmd shell or powershell:
```
python face_detection.py train
```

Train model with GPU at least 4G memory.
</br>Check the result for detection of face,please run Jupeter Notebook with inspect_face_data.ipynb and replace the image path.
</br>The application is my graduation design.
</br>If you have any questions, please submit a issue.
