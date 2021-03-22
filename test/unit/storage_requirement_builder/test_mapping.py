import unittest, pytest
from luxio.common.error_codes import *
from luxio.common.enumerations import *
from luxio.common.configuration_manager import *
from luxio.mapper.models.app_io_classifier import *
import pandas as pd

class TestMapping(unittest.TestCase):

    def get_fitnesses(self, io_identifier:pd.DataFrame, ac:AppClassifier) -> pd.DataFrame:
        """
        Determine how well the I/O Identifier fits within each class of behavior
        """
        #Calculate the scores
        io_identifier = ac.standardize(io_identifier)
        #Get the distance between io_identifier and every app class
        distance = 1 - np.absolute(ac.app_classes[ac.scores] - io_identifier[ac.scores].to_numpy())
        print(distance)
        #Get the magnitude of the fitnesses
        distance.loc[:,"magnitude"] = ac.get_magnitude(distance)
        return distance

    def test_app_classifier(self):
        #traces = pd.read_csv("sample/traces/sample_ior/sample.csv")
        traces = pd.read_pickle("sample/app_classifier/app_test.pkl").iloc[0,:].to_frame().transpose()
        ac = AppClassifier.load("sample/app_classifier/app_class_model.pkl")
        print(traces)
        fitnesses = self.get_fitnesses(traces, ac)
        print(fitnesses)
        print()
        print()

if __name__ == "__main__":
    unittest.main()
