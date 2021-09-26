
import torch
import torch.nn as nn

from commons.constants import Enum


class SiameseFeatureExtractor(nn.Module):
    """
    This is the feature extractor for model. The architecture replicates the best performing
    architecture. HP for the Siamese are fixed and must be avoided for replicating the original results.
    """
    def __init__(self, layers, dataset_name, feature_dimensions, siamese_activation, window_length):
        super().__init__()
        self.name = "Siamese"
        self.in_dimensions = {Enum.SMAP: 25, Enum.MSL: 55, Enum.SMD: 27}[dataset_name]
        self.rep_dim = feature_dimensions
        self.window_length = window_length
        self.conv = nn.Sequential(nn.Conv1d(self.in_dimensions, 33, 3),
                      nn.Conv1d(33, 32, 3, dilation=2),
                      nn.BatchNorm1d(32),
                      nn.ReLU(),
                      nn.Conv1d(32, 16, 3),
                      nn.BatchNorm1d(16),
                      nn.ReLU())

        try:
            self.fc1 = nn.Linear({100: 1472, 10:32, 500:7872}[self.window_length], self.rep_dim)
        except KeyError:
            print("ERROR: Please enter the dimension in the specified range")
            exit(-1)
        self.fcOut = nn.Linear(self.rep_dim, 1)

    def forward(self, x1, x2):
        """
        This is invoked during training. The output is real value depicting whether
        the two signals are similar or different based on the absolute difference in
        the feature extracted from the conv module.
        :param x1: Signal 1
        :param x2: Signal 2
        :return: Real value score
        """
        x1 = torch.transpose(x1, 1, 2)
        x1 = self.conv(x1)
        x1 = x1.reshape(x1.size(0), -1)
        x1 = self.fc1(x1)
        x1 = nn.Sigmoid()(x1)
        x2 = torch.transpose(x2, 1, 2)
        x2 = self.conv(x2)
        x2 = x2.reshape(x2.size(0), -1)
        x2 = self.fc1(x2)
        x2 = nn.Sigmoid()(x2)
        x = torch.abs(x1 - x2)
        x = self.fcOut(x)
        return x

    def get_features(self, x):
        """
        This function is invoked to provide features for the input signal. These features
        are used by the downstream feature classifier to identify if the current signal is anomalous or
        no.
        :param x: Signal
        :return: Representative dimension(Rep_dim) long feature vector.
        """
        x = torch.transpose(x, 1, 2)
        x = self.conv(x)
        x = x.reshape(x.size(0), -1).contiguous()
        x = self.fc1(x)
        return x


