from torch import nn

# https://pytorch.org/vision/stable/models.html

class BinaryClassifier(nn.Module):

    def __init__(self, originalModel, freeze=True):

        super().__init__()

        self.model = originalModel

        if freeze:
            self.model = BinaryClassifier.freezeLayers(self.model)

        self.model.fc = nn.Linear(originalModel.fc.in_features, 1)

        self.activation = nn.Sigmoid()

    def forward(self, x):

        x = self.model(x)
        x = self.activation(x)

        return x

    @staticmethod
    def freezeLayers(model):

        for p in model.parameters():
            p.requires_grad = False

        return model



if __name__ == '__main__':

    model = models.resnet18(pretrained=True)
    BinaryClassifier(model)
