from distutils.core import setup

setup(name='RffeaturesD3MWrapper',
    version='3.1.2',
    description='A wrapper for running the punk rffeatures functions in the d3m environment.',
    packages=['RffeaturesD3MWrapper'],
    install_requires=["numpy",
        "pandas == 0.25.2",
        "requests == 2.22.0",
        "typing",
        "punk==3.0.0"],
    entry_points = {
        'd3m.primitives': [
            'feature_selection.rffeatures.Rffeatures = RffeaturesD3MWrapper:rffeatures'
        ],
    },
)
