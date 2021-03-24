from Task1Loader import Task1_loader
from runner import *
from face_recognition import FaceRecog
from sklearn.metrics import accuracy_score, precision_score, recall_score
from matplotlib import pyplot
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorchcv.model_provider import get_model
import torchvision.models as models
from task1_main import *
import os

def roc_auc_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor):
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        raise RuntimeError("This contrib module requires sklearn to be installed.")

    y_true = y_targets
    y_pred = y_preds
    return roc_auc_score(y_true, y_pred)
def roc_auc_curve_compute_fn(y_targets: torch.Tensor, y_preds: torch.Tensor):
    try:
        from sklearn.metrics import roc_curve
    except ImportError:
        raise RuntimeError("This contrib module requires sklearn to be installed.")

    y_true = y_targets
    y_pred = y_preds
    return roc_curve(y_true, y_pred)

"""
model = get_model("xception", pretrained=True)
model = nn.Sequential(*list(model.children())[:-1]) # Remove original output layer

model[0].final_block.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1)) # xcep
model = FCN(model, 2048)
"""
model = torch.load("./checkpoints/89.pkl")

test_data = Task1_loader("./Task_1/test.csv", phase='test')
valid_loader = DataLoader(test_data, batch_size=55, shuffle=False, num_workers=8)

y_pred = []
test_y = []
with torch.no_grad():

    for batch in valid_loader:

        # Move the validation batch to the GPU
        inputs = Variable(batch['X'])
        labels = Variable(batch['Y'])

        labels = labels.unsqueeze(1)
        labels = labels.to(torch.float32)

        inputs, labels = inputs.to('cuda'), labels.to('cuda')

        # forward propagation
        # predictions, interm_feats = ???
        _, predictions = model(inputs)
        #print(predictions.cpu().numpy())
        y_pred.extend(predictions.cpu().numpy().flatten())
        test_y.extend(labels.cpu().numpy().flatten())

print(y_pred, test_y)
fpr, tpr, thresholds = roc_auc_curve_compute_fn(test_y, y_pred)

print(fpr, tpr, thresholds)
pyplot.figure()
pyplot.plot(fpr, tpr)

# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.title(str(roc_auc_compute_fn(y_pred,test_y)))
pyplot.show()
