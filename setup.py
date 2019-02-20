from distutils.core import setup

setup(name='RffeaturesD3MWrapper',
    version='3.1.1',
    description='A wrapper for running the punk rffeatures functions in the d3m environment.',
    packages=['RffeaturesD3MWrapper'],
    install_requires=["numpy",
        "pandas == 0.23.4",
        "requests >= 2.18.4, <= 2.20.0",
        "typing",
        "punk==3.0.0"],
    dependency_links=[
    ],
    entry_points = {
        'd3m.primitives': [
            'feature_selection.rffeatures.Rffeatures = RffeaturesD3MWrapper:rffeatures'
        ],
    },
)
