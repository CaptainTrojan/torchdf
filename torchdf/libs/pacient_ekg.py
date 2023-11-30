import matplotlib.pyplot as plt
import numpy as np
import xmltodict
import hashlib

from libs.EKG_muse_reader import load_data_aorta

try:
    import cPickle as pickle
except:
    import pickle


def gen_dict_extract(key, var):
    """
    Find all ocurencess of key in dict
    :param key to find:
    :param var dictonary to search in:
    :return yield need to iterate over result if multiple results expeted ITERATION ERROR if not found:
    """
    if hasattr(var, 'items'):
        for k, v in var.items():
            if k == key:
                yield v
            if isinstance(v, dict):
                for result in gen_dict_extract(key, v):
                    yield result
            elif isinstance(v, list):
                for d in v:
                    for result in gen_dict_extract(key, d):
                        yield result


class PatientEKGFile:
    """
    Basic pacient class
    """

    def __init__(self):
        self.waveforms = None
        self.ID = None
        self.age = None
        self.date_of_birth = None
        self.gender = None
        self.first_name = None
        self.last_name = None
        self.acquisition_date = None
        self.weight = None
        self.height = None
        self.ekg_mesurements = None
        self.dic = None
        self.bmi = None

    def save(self, path=None):
        """
        Save object to dict
        #TODO make more memory eficient
        :param path:
        :return:
        """
        if path == None:
            f = open(f"{self.ID}-{self.acquisition_date}.json", 'wb')
        else:
            f = open(f"{path}{hash(self.ID)}-{self.acquisition_date}.json", "wb")
        pickle.dump(self.__dict__, f, 5)
        f.close()

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.__dict__ = data

    def check_completnes(self):
        """
        Check if all data for training is available
        :return usable for traing:
        """
        if all(item is not None for item in [self.ID, self.waveforms, self.age, self.gender, self.weight, self.height,
                                             self.ekg_mesurements]):
            return True
        else:
            return False

    def gen_tensor_matrix(self, leads_12=True):
        """
        Return "waveforms" dictionary in a list format
        :return:
        """
        if leads_12:
            ekg_matrix = [None] * 12
            ekg_matrix[0] = self.waveforms["V1"]
            ekg_matrix[1] = self.waveforms["V2"]
            ekg_matrix[2] = self.waveforms["V3"]
            ekg_matrix[3] = self.waveforms["V4"]
            ekg_matrix[4] = self.waveforms["V5"]
            ekg_matrix[5] = self.waveforms["V6"]
            ekg_matrix[6] = self.waveforms["aVL"]
            ekg_matrix[7] = self.waveforms["I"]
            ekg_matrix[8] = self.waveforms["aVR"]
            ekg_matrix[9] = self.waveforms["II"]
            ekg_matrix[10] = self.waveforms["aVF"]
            ekg_matrix[11] = self.waveforms["III"]

        else:
            ekg_matrix = [None] * 8
            ekg_matrix[0] = self.waveforms["V1"]
            ekg_matrix[1] = self.waveforms["V2"]
            ekg_matrix[2] = self.waveforms["V3"]
            ekg_matrix[3] = self.waveforms["V4"]
            ekg_matrix[4] = self.waveforms["V5"]
            ekg_matrix[5] = self.waveforms["V6"]
            ekg_matrix[6] = self.waveforms["I"]
            ekg_matrix[7] = self.waveforms["II"]
        ekg_matrix = np.array(ekg_matrix, dtype=np.short)

        """data_matrix = np.zeros(12)
        data_matrix[0] = self.age
        if self.gender == "MALE":
            data_matrix[1] = 0
        elif self.gender == "FEMALE":
            data_matrix[1] = 1
        else:
            raise ValueError
        data_matrix[2] = self.weight
        data_matrix[3] = self.height
        data_matrix[4] = self.bmi"""
        # TODO finish
        return ekg_matrix

    def load_data(self, path):
        """
        Load from data from muse XML file slow operation
        :param path to xml muse file:
        :return:
        """
        self.waveforms, encoding = load_data_aorta(path)
        with open(path, 'rb') as fd:
            dic = xmltodict.parse(fd.read().decode(encoding))
            fd.close()
        self.dic = dic
        self.__parse_dict__()
        del self.dic

    def show_ekg(self):
        """
        Plots EKG of patient
        :return:
        """
        t = np.linspace(0, 10, 40000)
        fig, axs = plt.subplots(6, 2)
        fig.suptitle(f'{self.first_name} {self.last_name}, age:{self.age}', fontsize=16)
        for index, i in enumerate(self.waveforms["header"]):
            axs[index % 6, index // 6].set_title(i)
            axs[index % 6, index // 6].plot(t, self.waveforms[i])
        plt.show()

    def check_waveforms(self, shape):
        # TODO make better not eficient or follow OOP standarts
        x = len(self.waveforms["header"])
        if (x != shape[0]):
            return False
        for i in self.waveforms["header"]:
            if len(self.waveforms[i]) != shape[1]:
                return False
        return True

    def get_shape(self):
        num_waveforms = len(self.waveforms["header"])
        first_size = len(self.waveforms[self.waveforms['header'][0]])
        for waveform_name in self.waveforms["header"][1:]:
            if len(self.waveforms[waveform_name]) != first_size:
                return None  # mismatching lengths
        return num_waveforms, first_size

    def get_unique_identifier(self):
        return hashlib.sha1(f"{self.ID}{str(self.date_of_birth)}".encode('utf-8')).hexdigest()

    def get_waveforms(self):
        return self.waveforms

    def calc_bmi(self):
        try:
            self.bmi = self.height / self.height ^ 2
        except:
            self.bmi = None

    def __parse_dict__(self):
        """
        Parse XML dict
        :return:
        """
        try:
            x = gen_dict_extract('PatientID', self.dic)
            x = next(x)
            self.ID = x
        except StopIteration:
            assert "ID not found"
        try:
            x = gen_dict_extract('PatientAge', self.dic)
            x = next(x)
            self.age = x
        except StopIteration:
            assert "'PatientAge not found"
        try:
            x = gen_dict_extract('DateofBirth', self.dic)
            x = next(x)
            self.date_of_birth = x
        except StopIteration:
            assert "DateofBirth not found"
        try:
            x = gen_dict_extract('Gender', self.dic)
            x = next(x)
            self.gender = x
        except StopIteration:
            assert "Gender not found"
        try:
            x = gen_dict_extract('HeightCM', self.dic)
            x = next(x)
            x = int(x)
            if (x != 0):
                self.height = x
        except StopIteration:
            assert "Height not found"
        try:
            x = gen_dict_extract('WeightKG', self.dic)
            x = next(x)
            x = int(x)
            if (x != 0):
                self.weight = x
        except StopIteration:
            assert "Weight not found"
        try:
            x = gen_dict_extract('PatientLastName', self.dic)
            x = next(x)
            self.last_name = x
        except StopIteration:
            assert "Patient last name not found"
        try:
            x = gen_dict_extract('PatientFirstName', self.dic)
            x = next(x)
            self.first_name = x
        except StopIteration:
            assert "Patient first name not found"
        try:
            x = gen_dict_extract('AcquisitionDate', self.dic)
            x = next(x)
            self.acquisition_date = x
        except StopIteration:
            assert "Acquisition date name not found"
        try:
            x = gen_dict_extract('OriginalRestingECGMeasurements', self.dic)
            x = next(x)
            self.ekg_mesurements = x
        except StopIteration:
            assert "Ekg mesurements date name not found"
        if self.bmi == None and self.height == None and self.weight == None:
            self.calc_bmi()
