import os.path
import numpy as np
import pandas
import pickle
import requests
import ast
import typing
from json import JSONDecoder
from typing import List

from punk.feature_selection import RFFeatures
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult

from d3m import container, utils
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata import hyperparams, base as metadata_base, params
from common_primitives import dataset_to_dataframe as DatasetToDataFrame

__author__ = 'Distil'
__version__ = '3.1.1'
__contact__ = 'mailto:nklabs@newknowledge.io'

Inputs = container.pandas.DataFrame
Outputs = container.pandas.DataFrame

class Hyperparams(hyperparams.Hyperparams):
    pass

class rffeatures(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
        Perform supervised recursive feature elimination using random forests to generate an ordered
        list of features 
        Parameters
        ----------
        inputs : Input pandas frame, NOTE: Target column MUST be the last column

        Returns
        -------
        Outputs : pandas frame with ordered list of original features in first column
        """
    metadata = metadata_base.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': "ef6f3887-b253-4bfd-8b35-ada449efad0c",
        'version': __version__,
        'name': "RF Features",
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['Rank and score numeric features based on Random Forest and Recursive Feature Elimination'],
        'source': {
            'name': __author__,
            'contact': __contact__,
            'uris': [
                # Unstructured URIs.
                "https://github.com/NewKnowledge/rffeatures-d3m-wrapper",
            ],
        },
        # A list of dependencies in order. These can be Python packages, system packages, or Docker images.
        # Of course Python packages can also have their own dependencies, but sometimes it is necessary to
        # install a Python package first to be even able to run setup.py of another package. Or you have
        # a dependency which is not on PyPi.
         'installation': [{
            'type': metadata_base.PrimitiveInstallationType.PIP,
            'package_uri': 'git+https://github.com/NewKnowledge/rffeatures-d3m-wrapper.git@{git_commit}#egg=RffeaturesD3MWrapper'.format(
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),
            ),
        }],
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.feature_selection.rffeatures.Rffeatures',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.RANDOM_FOREST,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.FEATURE_SELECTION,

    })
    
    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0)-> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)
               
     
    def produce_metafeatures(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Perform supervised recursive feature elimination using random forests to generate an ordered
        list of features 
        Parameters
        ----------
        inputs : Input pandas frame, NOTE: Target column MUST be the last column

        Returns
        -------
        Outputs : pandas frame with ordered list of original features in first column
        """
        # add metadata to output dataframe
        rff_df = d3m_DataFrame(RFFeatures().rank_features(inputs = inputs.iloc[:,:-1], targets = pandas.DataFrame(inputs.iloc[:,-1])), columns=['features'])
        # first column ('features')
        col_dict = dict(rff_df.metadata.query((metadata_base.ALL_ELEMENTS, 0)))
        col_dict['structural_type'] = type("it is a string")
        col_dict['name'] = 'features'
        col_dict['semantic_types'] = ('http://schema.org/Text', 'https://metadata.datadrivendiscovery.org/types/Attribute')
        rff_df.metadata = rff_df.metadata.update((metadata_base.ALL_ELEMENTS, 0), col_dict)
        
        return CallResult(rff_df)
    
    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Perform supervised recursive feature elimination using random forests to generate an ordered
        list of features 
        Parameters
        ----------
        inputs : Input pandas frame, NOTE: Target column MUST be the last column

        Returns
        -------
        Outputs : pandas frame with ordered list of original features in first column
        """
        # generate feature ranking
        rff_features = pandas.DataFrame(RFFeatures().rank_features(inputs = inputs.iloc[:,:-1], targets = pandas.DataFrame(inputs.iloc[:,-1])), columns=['features'])
        
        # set threshold for the top five features
        bestFeatures = rff_features.iloc[0:5].values
	bestFeatures = [row[0] for row in bestFeatures]
        unique_index = pandas.Index(bestFeatures)
	bestFeatures = [unique_index.get_loc(row) for row in bestFeatures]
        # add suggested target
        bestFeatures.append(inputs.shape[1]-1) # assuming that the last column is the target column
		
        from d3m.primitives.data_transformation.extract_columns import DataFrameCommon as ExtractColumns
        extract_client = ExtractColumns(hyperparams={"columns":bestFeatures})
        result = extract_client.produce(inputs=inputs)

        return result
        
if __name__ == '__main__':
    # LOAD DATA AND PREPROCESSING
    input_dataset = container.Dataset.load('file:///home/datasets/seed_datasets_current/196_autoMpg/196_autoMpg_dataset/datasetDoc.json') 
    ds2df_client = DatasetToDataFrame.DatasetToDataFramePrimitive(hyperparams={"dataframe_resource":"learningData"})
    df = ds2df_client.produce(inputs = input_dataset)  
    client = rffeatures(hyperparams={})
    result = client.produce(inputs = df.value)
    print(result.value)
