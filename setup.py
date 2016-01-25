from setuptools import setup

setup(name='networkprops',
      version='0.1',
      description='Compute a bunch of network properties for a given graph',
      url='https://github.com/benmaier/Get-Network-Properties',
      author='Benjamin F. Maier',
      author_email='bfmaier@physik.hu-berlin.de',
      license='MIT',
      packages=['networkprops'],
      install_requires=[
          'numpy',
          'scipy',
          'networkx',
          'python-louvain',
          'effective-distance',
      ],
      zip_safe=False)
