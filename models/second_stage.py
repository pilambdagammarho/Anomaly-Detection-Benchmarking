import torch.nn as nn


class FeatClassifier(nn.Module):
    """
    This is the second downstream classifier working on the feature extracted
    from the up stream feature.
    """
    def __init__(self, input_dim, hidden_dim, activation_function):
        super().__init__()
        # self.encoder = encoder.eval()
        # self.encoder = self.encoder.get_features
        self.name = "FeatClassifier"
        self.hidden_dim = hidden_dim
        self.activation_function = activation_function
        self.input_dim = input_dim
        self.linear = nn.Linear(self.input_dim, self.hidden_dim)
        self.fcout = nn.Linear(self.hidden_dim, 1)

        self.loss_fn = nn.BCEWithLogitsLoss(reduce=False)

    def forward(self, x):
        x = nn.ReLU()(self.linear(x))
        x = self.fcout(x)
        return x

