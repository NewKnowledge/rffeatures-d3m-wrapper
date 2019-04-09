# RFFeatures D3M Wrapper

Wrapper of the punk rrfeatures library into D3M infrastructure. All code is written in Python 3.5 and must be run in 3.5 or greater.

The base library can be found here: https://github.com/NewKnowledge/punk.

## Install

`pip3 install -e git+https://github.com/NewKnowledge/rffeatures-d3m-wrapper.git#egg=RffeaturesD3MWrapper`

## Output

The output is a pandas frame with ordered list of original features in first column.


## Available Functions

#### produce
Produce performs supervised recursive feature elimination using random forests to generate an ordered list of features. The output is described above.
