
from .interference_factor import InterferenceFactor
from .price.price_factory import PriceFactory
from .resolver_policy.resolver_policy_factory import ResolverPolicyFactory
from luxio.scheduler.client import LuxioSchedulerClient
from luxio.external_clients.json_client import JSONClient
from luxio.common.configuration_manager import ConfigurationManager
from luxio.database.database import *
from luxio.mapper.models import StorageClassifier
from luxio.common.enumerations import *
from clever.common import *
import pandas as pd
import numpy as np

class Resolver:
    """
    A class used to Query the scheduler and keep track of existing resources
    """

    def _initialize(self) -> None:
        self.conf = ConfigurationManager.get_instance()
        self.db = DataBase.get_instance()

    def _finalize(self) -> None:
        return

    def choose_deployment(self):
        id = self.conf.job_spec["deployment_id"]
        #Get the set of deployments
        self.conf.timer.resume()
        self.conf.storage_classifier = self.db.get("storage_classifier")
        self.conf.timer.pause().log("DownloadStorageModel")
        self.conf.timer.resume()
        #Get deployment "id"
        deployments = self.conf.storage_classifier.qosa_to_deployment
        deployments = deployments[deployments.deployment_id == id]
        deployments['status'] = DeploymentStatus.NEW
        return deployments

    def build_qosas(self, io_identifier:pd.DataFrame, ranked_qosas:pd.DataFrame) -> pd.DataFrame:
        #Set of candidate deployments
        deployments = pd.DataFrame()
        #Get the availability of the resources
        self.conf.timer.resume()
        scheduler = LuxioSchedulerClient()
        scheduler.download_resource_graph()
        self.conf.timer.pause().log("DownloadResourceGraph")
        #Get the set of deployments corresponding to each of the QoSAs
        self.conf.timer.resume()
        self.conf.storage_classifier = self.db.get("storage_classifier")
        self.conf.timer.pause().log("DownloadStorageModel")
        self.conf.timer.resume()
        if not self.conf.force_colocate:
            new_deployments = pd.merge(ranked_qosas['qosa_id'], self.conf.storage_classifier.qosa_to_deployment, on="qosa_id")
            new_deployments.loc[:,"status"] = DeploymentStatus.NEW
        else:
            new_deployments = pd.DataFrame(columns=list(self.conf.storage_classifier.qosa_to_deployment.columns) + ["status"])
        #Get the set of existing deployments
        if self.conf.isolate_deployments == False:
            existing_deployments = scheduler.download_existing_deployments()
            if len(existing_deployments) > 0:
                elastic_deployments = existing_deployments[existing_deployments.malleable == True]
                inelastic_deployments = existing_deployments[existing_deployments.malleable == False]
                if len(elastic_deployments) > 0:
                    #Get the resource tiers
                    tiers = list(scheduler.resource_graph.keys())
                    #Add potential to modify existing malleable deployments
                    #elastic_deployments,new_deployments = pd_merge(new_deployments, elastic_deployments, on='qosa_id', split=True)
                    elastic_deployments,new_deployments = pd_merge(new_deployments, elastic_deployments, how='cartesian', split=True)
                    elastic_deployments.loc[:,tiers] = new_deployments[tiers] - elastic_deployments[tiers]
                deployments = pd.concat([deployments, elastic_deployments, inelastic_deployments])
        deployments = pd.concat([deployments, new_deployments])
        #Verify that there are deployments
        if len(deployments) == 0:
            print("There are no deployments to satisfy for your job!")
            exit(1)
        #Get the set of user resources from the job spec
        if "resources" in self.conf.job_spec:
            scheduler.remove_resources(self.conf.job_spec["resources"])
        #Get the interference factor for each deployment (0 for new)
        interference = InterferenceFactor()
        deployments = interference.score(deployments)
        #Get the price of each deployment (0 for queued/running deployments)
        pricer = PriceFactory.get(self.conf.price_type)
        deployments = pricer.price(deployments, scheduler)
        deployments = deployments[deployments.price < np.inf] #Remove infeasible deployments
        #Get the performance of each deployment
        deployments = pd.merge(ranked_qosas[['qosa_id','satisfaction']], deployments, on = 'qosa_id')
        #Rank deployments according to some policy
        policy = ResolverPolicyFactory.get(self.conf.resolver_policy)
        deployments = policy.rank(deployments)
        self.conf.timer.pause().log("Resolver")
        return deployments

    def run(self, io_identifier:pd.DataFrame, ranked_qosas:pd.DataFrame) -> pd.DataFrame:
        """
        Query the available resources from the scheduler
        :param : dict
        :return: dict
        """
        self._initialize()
        if "deployment_id" in self.conf.job_spec:
            deployments = self.choose_deployment()
        elif "qosa_id" in self.conf.job_spec:
            ranked_qosas = pd.DataFrame({"qosa_id": self.conf.job_spec["qosa_id"], "satisfaction": 1}, index=[0])
            deployments = self.build_qosas(io_identifier, ranked_qosas)
        else:
            deployments = self.build_qosas(io_identifier, ranked_qosas)
        self._finalize()
        return deployments
