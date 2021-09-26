import glob
import os
import random
import time
from ast import literal_eval
from typing import Tuple

import numpy as np
import pandas as pd

from commons.constants import Enum


class DataManager:
    """
    This class is responsible for reading the data from the disk based on the
    data_name specified in the constructor.
    """
    def __init__(self, data_name, window, anomaly_ratio, t_t_split, signal_name):
        self.BASE_FOLDER = Enum.DATASET_FOLDER
        self.data_name = data_name
        self.window = window
        self.anomaly_ratio = anomaly_ratio
        self.signal_name = signal_name
        self.data_split = t_t_split
        self.train_data, self.test_data = self.load_data()

    def load_data(self) -> Tuple[list, list]:
        """
        This method reads the data from the disk and process it into train and test data
        :return: Tuples of train and test data as list containing signals and labels
        """
        if self.data_name in [Enum.SMAP, Enum.MSL]:
            if os.path.isdir(os.path.join(Enum.DATASET_FOLDER, Enum.DATA_ZIP)):
                return self.process_smap_msl()
            else:
                print("ERROR: Please download and extract data.zip in ./dataset folder")
        else:
            if os.path.isdir(os.path.join(Enum.DATASET_FOLDER, Enum.SMD_DATA)):
                return self.process_smd()
            else:
                print(f"ERROR: Please download {Enum.SMD_DATA} folder from GitHub.")

    def create_train_test_split(self, window_signal, window_label, anomaly_indices) -> Tuple[list, list]:
        """
        This method divides the total signal into train and test, ensures that the initial split between
        anomalous and non anomalous signals are auto balanced. Once done, it trims off the anomalous signal based on
        the ratio.
        :param window_signal: Sampled signals with fixed window
        :param window_label: Labels for these signals
        :param anomaly_indices: Indexes of labels where anomaly exists
        :return: Tuple[List, List]
        """
        # first perform undersampling on the non anomalous data
        non_anomaly_indices = set(range(len(window_signal))) - anomaly_indices
        print(f"INFO: TOTAL {len(non_anomaly_indices)} NON ANOMALY vs {len(anomaly_indices)} ANOMALY")
        balance_ratio = len(anomaly_indices) / len(non_anomaly_indices)
        non_anomaly_indices_list = list(non_anomaly_indices)
        anomaly_indices_list = list(anomaly_indices)
        random.shuffle(non_anomaly_indices_list)
        random.shuffle(anomaly_indices_list)
        non_anomaly_indices_list = random.sample(non_anomaly_indices_list,
                                                 int(balance_ratio * len(non_anomaly_indices_list)))

        # create the train-test split
        na_data_cutoff = int(self.data_split * len(non_anomaly_indices_list))
        train_non_anomaly_indices_list = non_anomaly_indices_list[:na_data_cutoff]
        test_non_anomaly_indices_list = non_anomaly_indices_list[na_data_cutoff:]

        a_data_cutoff = int(self.anomaly_ratio * self.data_split * len(anomaly_indices_list))
        train_anomaly_indices_list = anomaly_indices_list[:a_data_cutoff]
        test_anomaly_indices_list = anomaly_indices_list[a_data_cutoff:]

        total_train_indices = train_anomaly_indices_list + train_non_anomaly_indices_list
        total_test_indices = test_anomaly_indices_list + test_non_anomaly_indices_list

        random.shuffle(total_train_indices)
        random.shuffle(total_test_indices)

        train_data = [window_signal[idx] for idx in total_train_indices]
        train_label = [window_label[idx] for idx in total_train_indices]

        test_data = [window_signal[idx] for idx in total_test_indices]
        test_label = [window_label[idx] for idx in total_test_indices]

        return [train_data, train_label], [test_data, test_label]

    def process_smap_msl(self) -> None:
        """
        This method reads labels and loads the mentioned signal and samples window length signals with corresponding
        labels.
        """

        if os.path.isfile(os.path.join(self.BASE_FOLDER, Enum.SMAP_MSL_LABEL)):
            data = pd.read_csv(os.path.join(self.BASE_FOLDER, Enum.SMAP_MSL_LABEL))
            data.anomaly_sequences = data.anomaly_sequences.apply(literal_eval)
            data = data[data.iloc[:, 1] == self.data_name]
            if self.signal_name is not None:
                data = data[data.iloc[:, 0] == self.signal_name]
            #start splitting and storing the data for MSL Data and SMAP data
            train_data, test_data, train_label, test_label = list(), list(), list(), list()
            #create the label signal for the entire data
            start_time = time.time()
            window_signal, window_label, anomaly_indices = list(), list(), set()
            a_idx = 0
            for idx, (ch_idx, an_seq, length) in enumerate(
                    zip(data["chan_id"], data["anomaly_sequences"], data["num_values"])):
                seq = np.zeros(length)
                # if idx in msl_indices:
                for an_substring in an_seq:
                    start, end = an_substring
                    seq[start:end] = 1.

                with open(os.path.join(self.BASE_FOLDER, Enum.DATA_ZIP, "test", ch_idx+".npy"), 'rb') as f:
                    input_signal = np.load(f)

                for i in range(len(input_signal) - self.window):
                    _x = input_signal[i:(i + self.window)]
                    if np.sum(seq[i:(i + self.window)]) >= 1:
                        _y = 1.
                        anomaly_indices.add(a_idx)
                        a_idx +=1
                    else:
                        _y = 0.

                    window_signal.append(_x)
                    window_label.append(_y)
                break
            print(f"INFO: Loaded and split the {self.data_name} in {int(time.time() - start_time)} secs")
            return self.create_train_test_split(window_signal, window_label, anomaly_indices)
        else:
            print(f"ERROR: {Enum.SMAP_MSL_LABEL} does not exist. Please download it and place it under ./dataset folder!")
            exit()

    def process_smd(self):

        """train_data, test_data, train_label, test_label = list(), list(), list(), list()
        for filename in glob.glob(os.path.join(self.BASE_FOLDER, Enum.SMD_DATA, "test_label", "*.txt")):
            #basename = os.path.basename(filename)[:-4]
            data = np.array(pd.read_csv(filename.replace("test_label", "test"), header=None, delimiter=","))
            labels = np.array(pd.read_csv(filename, header=None))
            cutoff = int(self.data_split * len(data))
            data_train, data_test = data[:cutoff], data[cutoff:]
            label_train, label_test = labels[:cutoff], labels[cutoff:]
            train_data.append(data_train)
            test_data.append(test_data)
            train_label.append(label_train)
            test_label.append(label_test)
        return train_data, test_data, train_label, test_label
        """
        """
        SMD not implemented
        """
        return NotImplementedError

