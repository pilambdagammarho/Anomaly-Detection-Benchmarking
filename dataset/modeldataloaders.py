from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class SiamDataLoader(Dataset):
    """
    This is dataset class for Siamese Network. Since the Feature Extractor
    requires the data to be present in pairs, with single value depicting whether the
    input values are same or not, the preprocess method creates such labels
    """
    def __init__(self, data):
        self.signal, self.label = self.preprocess(data)
        # print(self.signal.shape, self.label.shape)

    def preprocess(self, data) -> Tuple[np.ndarray, np.ndarray]:
        """
        This method reads the signals in the form of list, converts them into pairs of anomaly and anomaly signal,
        non anomalous vs non anomalous and anomalous vs non anomalous signal pairs. The labels are then assigned 1 for
        similar signal while 0 for different signals.
        :param data: List of data and label values
        :return: Array of signal pairs and corresponding labels
        """
        original_signal, label = data
        original_signal = np.array(original_signal)
        label = np.array(label).squeeze()
        assert len(label.shape) == 1, "ERROR: Error in parsing labels"
        anomaly_indices = np.where(label == 1.)
        non_anomaly_indices = np.where(label == 0.)

        # create similar pairs
        temp_data_anomaly = np.stack((original_signal[anomaly_indices],
                                      original_signal[anomaly_indices]),
                                     axis=0)
        temp_data_nonanomaly = np.stack((original_signal[non_anomaly_indices],
                                         original_signal[non_anomaly_indices]),
                                        axis=0)

        # create different pairs
        intersection = min(original_signal[anomaly_indices].shape[0], original_signal[non_anomaly_indices].shape[0])
        temp_data_diff1 = np.stack((original_signal[anomaly_indices][:intersection],
                                    original_signal[non_anomaly_indices][:intersection]),
                                   axis=0)
        temp_data_diff2 = np.stack((original_signal[anomaly_indices][:intersection],
                                    original_signal[non_anomaly_indices][:intersection]),
                                   axis=0)
        # print(temp_data_anomaly.shape, temp_data_nonanomaly.shape, temp_data_diff1.shape, temp_data_diff2.shape)
        total_data = np.concatenate((temp_data_anomaly,
                                     temp_data_nonanomaly,
                                     temp_data_diff1,
                                     temp_data_diff2),
                                    axis=1)

        total_labels = np.ones(total_data.shape[1])
        total_labels[total_data.shape[1] // 2:] = 0.
        print(f"Total Siamese {sum(total_labels)} Similar vs {len(total_labels) - sum(total_labels)} Diff Labels")
        return total_data, total_labels

    def __getitem__(self, item):
        x = torch.from_numpy(self.signal[0, item])
        x_prime = torch.from_numpy(self.signal[1, item])
        y = torch.from_numpy(np.asarray(self.label[item]))
        return x, x_prime, y

    def __len__(self):
        return self.signal.shape[1]


class FeatDataLoader(Dataset):
    """
    Dataset class for dataloader for feature classifier. Here the input is the feature extracted
    from pre trained or just trained siamese feature extractor. The labels are original labels of anomaly
    vs non anomaly.
    """
    def __init__(self, data, model):
        self.signal, self.labels = self.preprocess(data, model)
        # extract the features for the signal from the model

    def preprocess(self, data, model) -> Tuple[np.ndarray, np.ndarray]:
        """
        This method iterates over the batch of data to extract features from the input signal.
        :param data: Data signals
        :param model: Trained Siamese Model
        :return: Tuple[np.ndarray, np.ndarray]
        """
        signal, label = data
        signal = np.array(signal)
        device = "cuda" if next(model.parameters()).is_cuda else "cpu"
        window_length = model.window_length
        final_output = list()
        for i in range(len(signal) // window_length):
            output = model.get_features(
                torch.from_numpy(
                    signal[i * window_length: (i + 1) * window_length]).to(device))
            if device == "cuda":
                output = output.detach().cpu().numpy()
            else:
                output = output.detach().numpy()
            final_output.extend(output)

        return np.array(final_output), label

    def __getitem__(self, idx):
        return torch.from_numpy(self.signal[idx]), \
               torch.from_numpy(np.asarray(self.labels[idx]))

    def __len__(self):
        return len(self.signal)
