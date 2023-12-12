'''

# TODO: These are flyingsquid model repo:

https://github.com/HazyResearch/flyingsquid?tab=readme-ov-file

there are 3 different ways to install. maybe I should try cl

1. git clone https://github.com/HazyResearch/flyingsquid.git

cd flyingsquid

conda env create -f environment.yml
conda activate flyingsquid

2. then try  lternatively, you can install the dependencies yourself:

Pgmpy
PyTorch (only necessary for the PyTorch integration)
And then install the actual package:

pip install flyingsquid



sample usage


from flyingsquid.label_model import LabelModel
import numpy as np

L_train = np.load('...')

m = L_train.shape[1]
label_model = LabelModel(m)
label_model.fit(L_train)

preds = label_model.predict(L_train)
'''