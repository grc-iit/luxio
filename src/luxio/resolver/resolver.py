
from .interference_factor import InterferenceFactor
from .price.price_factory import PriceFactory
from .resolver_policy.resolver_policy_factory import ResolverPolicyFactory
from luxio.scheduler.client import LuxioSchedulerClient
from luxio.external_clients.json_client import JSONClient
from luxio.common.configuration_manager import ConfigurationManager
from luxio.mapper.models import StorageClassifier
from luxio.common.enumerations import *
import pandas as pd
import numpy as np

class Resolver:
    """
    A class used to Query the scheduler and keep track of existing resources
    """

    def _initialize(self) -> None:
        self.conf = ConfigurationManager.get_instance()

    def _finalize(self) -> None:
        return

    def run(self, io_identifier:pd.DataFrame, ranked_qosas:pd.DataFrame) -> pd.DataFrame:
        """
        Query the available resources from the scheduler
        :param : dict
        :return: dict
        """
        self._initialize()
        #Get the set of deployments corresponding to each of the QoSAs
        self.conf.storage_classifier = StorageClassifier.load(self.conf.storage_classifier_path)
        new_deployments = pd.merge(ranked_qosas['qosa_id'], self.conf.storage_classifier.qosa_to_deployment, on="qosa_id")
        new_deployments.loc[:,"status"] = DeploymentStatus.NEW
        #Get the availability of the resources
        scheduler = LuxioSchedulerClient()
        scheduler.download_resource_graph()
        #Get the set of queued/running deployments
        """
        if io_identifier.isolate == False:
            existing_deployments = scheduler.download_existing_deployments()
            elastic_deployments = existing_deployments[
                (existing_deployments.elastic == True) |
                (
                    (existing_deployments.status == DeploymentStatus.QUEUED) &
                    (existing_deployments.isolate == False)
                )
            ]
            inelastic_deployments = existing_deployments[
            elastic_deployments,new_deployments = cartesian_product(new_deployments, elastic_deployments, split=True)
        """
        deployments = new_deployments
        #Get the set of user resources from the job spec
        if "resources" in self.conf.job_spec:
            scheduler.remove_resources(self.conf.job_spec["resources"])
        #Get the interference factor for each deployment (0 for new)
        interference = InterferenceFactor()
        deployments = interference.score(deployments)
        #Get the price of each deployment (0 for queued/running deployments)
        pricer = PriceFactory.get(self.conf.price_type)
        deployments = pricer.price(deployments, scheduler)
        deployments = deployments[deployments.price < np.inf]
        #Get the performance of each deployment
        deployments = pd.merge(ranked_qosas[['qosa_id','satisfaction']], deployments, on = 'qosa_id')
        #Rank deployments according to some policy
        policy = ResolverPolicyFactory.get(self.conf.resolver_policy)
        deployments = policy.rank(deployments)
        self._finalize()
        return deployments
