# Created by David at 23.09.2022
# Project name AortaCheck
# Created by David at 15.09.2022
# Project name AortaCheck
import matplotlib
import numpy as np
import xmltodict
import matplotlib.pyplot as plt
from libs.EKG_muse_reader import load_data_aorta
import pathlib
from pathlib import Path
import builtins
from inspect import getframeinfo, stack
import os
from multiprocessing import Pool
import statistics
from tqdm import tqdm
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
    if hasattr(var,'items'):
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


class Patient:
    def __init__(self):
        self.ID = None
        self.age = None
        self.date_of_birth = None
        self.gender = None
        self.first_name = None
        self.last_name = None
        self.ekg = []


    def load_ekg(self,path):
        if len(self.ekg==0):
            ekg = Ekg()



        ekg = Ekg()
        ekg.load_data(path)
        self.ekg.append(ekg)



class Ekg:
    def __init__(self):
        self.acquisition_date = None
        self.weight = None
        self.height = None
        self.ekg_mesurements = None
        self.dic = None
        self.bmi = None

    def load_data(self,path):
        """
        Load from data from muse XML file slow operation
        :param path to xml muse file:
        :return:
        """
        self.waveforms = load_data_aorta(path)
        with open(path, 'rb') as fd:
            dic = xmltodict.parse(fd.read().decode('utf8'))
            fd.close()
        self.dic = dic
        self.__parse_dict_ekg__()
        del self.dic
    def __parse_dict_ekg__(self):
        raise NotImplementedError

    def __parse_dict_complete__(self):
        """
        Parse XML dict
        :return:
        """
        pac_info = {}
        try:
            x = gen_dict_extract('PatientID', self.dic)
            x = next(x)
            pac_info["ID"] = x
        except StopIteration:
            assert "ID not found"
        try:
            x = gen_dict_extract('DateofBirth', self.dic)
            x = next(x)
            pac_info["date_of_birth"] = x
        except StopIteration:
            assert "DateofBirth not found"
        try:
            x = gen_dict_extract('Gender', self.dic)
            x = next(x)
            pac_info["gender"] = x
        except StopIteration:
            assert "Gender not found"

        try:
            x = gen_dict_extract('PatientLastName', self.dic)
            x = next(x)
            pac_info["last_name"] = x
        except StopIteration:
            assert "Patient last name not found"
        try:
            x = gen_dict_extract('PatientFirstName', self.dic)
            x = next(x)
            pac_info["first_name"] = x
        except StopIteration:
            assert "Patient first name not found"
        return pac_info
    def gen_tensor_matrix(self):
        """
        :return EKG_matrix and Data matrix for AI trainig:
        """
        ekg_matrix = np.zeros((12,40000))
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

        data_matrix = np.zeros(12)
        data_matrix[0] = self.age
        if self.gender == "MALE":
            data_matrix[1] = 0
        elif self.gender == "FEMALE":
            data_matrix[1] = 1
        else:
            raise ValueError
        data_matrix[2] = self.weight
        data_matrix[3] = self.height
        data_matrix[4] = self.bmi
        #TODO not sure if this is right need doctor consultation
        data_matrix[5] = self.ekg_mesurements['QRSCount']*6
        #TODO uknow value
        data_matrix[6] = self.ekg_mesurements['QRSCount']
        data_matrix[7] = self.ekg_mesurements['QTInterval']
        data_matrix[8] = self.ekg_mesurements['QRSDuration']
        data_matrix[9] = self.ekg_mesurements['QTCorrected']
        data_matrix[10] = self.ekg_mesurements['RAxis']
        data_matrix[11] = self.ekg_mesurements['TAxis']

        return ekg_matrix,data_matrix
class Patient:
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

    def save(self,path = None):
        """
        Save object to dict
        #TODO make more memory eficient
        :param path:
        :return:
        """
        if path == None:
            f = open(f"{self.ID}-{self.acquisition_date}.json", 'wb')
        else:
            f = open(f"{path}{hash(self.ID)}-{self.acquisition_date}.json","wb")
        pickle.dump(self.__dict__, f, 5)
        f.close()

    def load(self,path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.__dict__ = data



    def check_completnes(self):
        """
        Check if all data for training is available
        :return usable for traing:
        """
        if all(item is not None for item in [self.ID, self.waveforms, self.age,self.gender,self.weight,self.height,
                                             self.ekg_mesurements]):
            return True
        else:
            return False



    def load_data(self,path):
        """
        Load from data from muse XML file slow operation
        :param path to xml muse file:
        :return:
        """
        self.waveforms = load_data_aorta(path)
        with open(path, 'rb') as fd:
            dic = xmltodict.parse(fd.read().decode('utf8'))
            fd.close()
        self.dic = dic
        self.__parse_dict__()
        del self.dic

    def show_ekg(self):
        """
        Plots EKG of patient
        :return:
        """
        t = np.linspace(0,len(self.waveforms[self.waveforms["header"][0]])/500,len(self.waveforms
                                                                    [self.waveforms["header"][0]]))
        fig, axs = plt.subplots(6, 2)
        fig.suptitle(f'{self.first_name} {self.last_name}, age:{self.age}', fontsize=16)
        for index,i in enumerate(self.waveforms["header"]):
            axs[index%6,index//6].set_title(i)
            axs[index%6,index//6].plot(t, self.waveforms[i])
        plt.show()

    def calc_bmi(self):
        self.bmi = self.height/self.height^2

    def __parse_dict__(self):
        """
        Parse XML dict
        :return:
        """
        try:
            x = gen_dict_extract('PatientID',self.dic)
            x = next(x)
            self.ID = x
        except StopIteration:
            assert "ID not found"
        try:
            x = gen_dict_extract('PatientAge',self.dic)
            x = next(x)
            self.age = x
        except StopIteration:
            assert "'PatientAge not found"
        try:
            x = gen_dict_extract('DateofBirth',self.dic)
            x = next(x)
            self.date_of_birth = x
        except StopIteration:
            assert "DateofBirth not found"
        try:
            x = gen_dict_extract('Gender',self.dic)
            x = next(x)
            self.gender = x
        except StopIteration:
            assert "Gender not found"
        try:
            x = gen_dict_extract('HeightCM',self.dic)
            x = next(x)
            x = int(x)
            if (x != 0):
                self.height = x
        except StopIteration:
            assert "Height not found"
        try:
            x = gen_dict_extract('WeightKG',self.dic)
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

def load_ekgs():
    raise NotImplementedError

def counter_of_ekg(path):
    with open(path, 'rb') as handle:
        dict = pickle.load(handle)
    lst = []
    for key in dict:
        lst.append(len(dict[key]))
    print(max(lst))
    print(sum(lst)/len(lst))
    print(statistics.median(lst))
    matplotlib.use('TkAgg')
    plt.hist(lst, bins=30)
    plt.gca().set(title='Frequency Histogram', ylabel='Frequency');
    plt.show()
def load_helper(path):
    filelist = []

    dict = {}


    for root, dirs, files in os.walk(path):
        for file in files:
            # append the file name to the list
            filelist.append(os.path.join(root, file))
    for i in tqdm(range(len(filelist))):
        with open(filelist[i]) as fd:
            try:
                doc = xmltodict.parse(fd.read())
                x = gen_dict_extract('PatientID', doc)
                id = next(x)
                if id in dict:
                    dict[id].append(i)
                else:
                    dict[id] = [i]
            except:
                print(i)
    f = open("dict_lst.pkl","wb")

    pickle.dump(dict,f)


if __name__ == '__main__':
    #load_helper(Path("C:\\Users\\David\\PycharmProjects\\AortaCheck\\data\\ikem_mix"))
    counter_of_ekg("C:\\Users\\David\\PycharmProjects\\AortaCheck\\libs\\dict_lst.pkl")
    """
    matplotlib.use('TkAgg')
    path = Path("C:\\Users\\David\\PycharmProjects\\AortaCheck\\data\\process_data\\suitable_for_training\\1467883369093439-02-09-2018.json")
    pac = Patient()
    pac.load(path)
    pac.show_ekg()
    pac.gen_tensor_matrix()

    
    path = Path("")
    matplotlib.use('TkAgg')
    pac = Patient_EKG()
    pac.load_data(path)
    pac.show_ekg()
    print("done")
    """
