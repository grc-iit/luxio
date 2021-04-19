import unittest, pytest
from luxio.common.error_codes import *
from luxio.common.enumerations import *
from luxio.common.configuration_manager import *
from luxio.mapper.models.app_io_classifier import *
from luxio.io_requirement_extractor.trace_parser.trace_parser_factory import *
import pandas as pd
import numpy as np

class TestMapping(unittest.TestCase):

    features = ['TOTAL_BYTES_READ', 'TOTAL_BYTES_WRITTEN', 'TOTAL_READ_OPS', 'TOTAL_POSIX_OPENS']
    featurespm = features + ['magnitude']

    def get_logdiff(self, io_identifier:pd.DataFrame, ac:AppClassifier) -> pd.DataFrame:
        ioid_features = io_identifier[ac.features].to_numpy()
        app_features = ac.app_classes[ac.features].to_numpy()
        logdiff = pd.DataFrame(np.abs(np.log(ioid_features+1)/np.log(10) - np.log(app_features+1)/np.log(10)), columns=ac.features)
        return logdiff

    def test_map_workloads(self):
        features = TestMapping.features
        featurespm = TestMapping.featurespm
        trace_paths = [
            "sample/traces/ior/compute_intense.json",
            "sample/traces/ior/compute_heavy.json",
            "sample/traces/ior/balanced.json",
            "sample/traces/ior/data_intense.json",
            "sample/traces/ior/data_heavy.json"
        ]
        ac = AppClassifier.load("sample/app_classifier/app_class_model.pkl")

        df = []

        for trace_path in trace_paths:
            parser = TraceParserFactory.get_parser(TraceParserType.DARSHAN_DICT)
            io_identifier = parser.preprocess({"path": trace_path})
            io_identifier = ac.standardize(io_identifier)
            fitness = ac.get_fitnesses(io_identifier)
            mape = self.get_mape(io_identifier, ac)
            fit_mape = pd.concat([fitness[ac.scores + ['magnitude']], mape], axis=1)
            df.append(fit_mape[fit_mape.magnitude == fit_mape.magnitude.max()])

        df = pd.concat(df)
        df['avg'] = df[features].mean(axis=1)
        df['max'] = df[features].max(axis=1)
        df[ac.scores + featurespm + ['avg','max']].to_csv("sample/workload_test.csv")

    def test_app_classifier(self):
        features = TestMapping.features
        featurespm = TestMapping.featurespm
        traces = pd.read_csv("sample/traces/ior/sample.csv")
        #traces = pd.read_pickle("datasets/app_classifier/app_test.pkl").sample(n=5000)
        ac = AppClassifier.load("sample/app_classifier/app_class_model.pkl")

        """
        print()
        print("AC:")
        print(ac.app_classes[features])
        """

        count = 0
        df = []
        for io_identifier in traces.to_numpy():
            #print(count)
            io_identifier = pd.DataFrame([io_identifier], columns=traces.columns)
            io_identifier = ac.standardize(io_identifier)
            fitness = ac.get_fitnesses(io_identifier)
            #logdiff = self.get_logdiff(io_identifier, ac)
            #fit_logdiff = pd.concat([fitness[ac.scores + ['magnitude']], logdiff], axis=1)
            #best_fit = fit_logdiff[fit_logdiff.magnitude == fit_logdiff.magnitude.max()]
            best_fit = fitness[fitness.magnitude == fitness.magnitude.max()].iloc[0:1,:]
            best_fit.loc[:,ac.mandatory_features + ac.features] = io_identifier[ac.mandatory_features + ac.features].to_numpy()
            df.append(best_fit)
            """
            print("IO_ID (scores):")
            print(io_identifier[ac.scores])
            print("IO_ID (features)")
            print(io_identifier[features])
            print("ALL FITNESS (scores+mag):")
            print(fitness[ac.scores + ['magnitude']])
            print("BEST (features+mag+logdiff)")
            print(best_fit[featurespm])
            print("LOGDIFF")
            print(logdiff[features])
            """
            count += 1
        #exit(1)
        df = pd.concat(df)
        df['avg'] = df[features].mean(axis=1)
        df['max'] = df[features].max(axis=1)
        df[ac.features + ac.scores + featurespm + ['avg','max','app_id']].transpose().to_csv("datasets/results/mapping_test.csv")

if __name__ == "__main__":
    unittest.main()
