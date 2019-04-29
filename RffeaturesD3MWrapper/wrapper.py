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
    num_features = hyperparams.UniformInt(lower=1, upper=100, default=5, semantic_types = ['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
       description = 'number of features to rank')
    only_numeric_cols = hyperparams.UniformBool(default = True, semantic_types = [
       'https://metadata.datadrivendiscovery.org/types/ControlParameter'],
       description="consider only numeric columns for feature selection")

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

        if not self.hyperparams:
            self.hyperparams['num_features'] = 5
        num_features = self.hyperparams['num_features']        

        # remove primary key and targets from feature selection
        inputs_primary_key = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/PrimaryKey')
        inputs_target = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/Target') + \
                  inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/TrueTarget')  
        if not len(inputs_target):
            inputs_target = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/SuggestedTarget')

        # extract numeric columns and suggested target
        if self.hyperparams['only_numeric_cols']:
            inputs_float = inputs.metadata.get_columns_with_semantic_type('http://schema.org/Float')
            inputs_integer = inputs.metadata.get_columns_with_semantic_type('http://schema.org/Integer')
            inputs_numeric = [*inputs_float, *inputs_integer]
            inputs_cols = [x for x in inputs_numeric if x not in inputs_primary_key]
        else:
            inputs_cols = [x for x in inputs if x not in inputs_primary_key]

        # generate feature ranking
        rff_features = pandas.DataFrame(RFFeatures().rank_features(inputs = inputs.iloc[:, inputs_cols], targets = pandas.DataFrame(inputs.iloc[:, inputs_target])), columns=['features'])
        # set threshold for the top seven features
        bestFeatures = rff_features.iloc[0:num_features].values
        bestFeatures = [row[0] for row in bestFeatures]
        bestFeatures = [inputs.columns.get_loc(row) for row in bestFeatures] # get integer location for each label
        # add suggested target
        bestFeatures = [*bestFeatures, *inputs_target]
     
        # drop all values below threshold value   
        from d3m.primitives.data_transformation.extract_columns import DataFrameCommon as ExtractColumns
        extract_client = ExtractColumns(hyperparams={"columns":bestFeatures})
        result = extract_client.produce(inputs=inputs)
        
        return result
        
if __name__ == '__main__':
    # LOAD DATA AND PREPROCESSING
    input_dataset = container.Dataset.load('file:///home/datasets/seed_datasets_current/185_baseball/185_baseball_dataset/datasetDoc.json') 
    ds2df_client = DatasetToDataFrame.DatasetToDataFramePrimitive(hyperparams={"dataframe_resource":"learningData"})
    df = d3m_DataFrame(ds2df_client.produce(inputs=input_dataset).value)
    hyperparams_class = rffeatures.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']  
    client = rffeatures(hyperparams=hyperparams_class.defaults())
    result = client.produce(inputs = df)
    print(result.value)
