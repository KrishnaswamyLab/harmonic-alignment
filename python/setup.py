import os
import sys
from setuptools import setup

install_requires = [
    'numpy>=1.14.0',
    'scipy>=1.1.0',
    'pygsp>=>=0.5.1',
    'scikit-learn>=0.20.0',
    'future',
    'tasklogger>=0.4.0',
    'graphtools>=0.2.1',
    'joblib',
]

test_requires = [
    'nose2',
    'coverage',
    'coveralls',
    'matplotlib'
]

version_py = os.path.join(os.path.dirname(
    __file__), 'harmonicalignment', 'version.py')
version = open(version_py).read().strip().split(
    '=')[-1].replace('"', '').strip()

readme = open('README.rst').read()

setup(name='harmonicalignment',
      version=version,
      description='harmonicalignment',
      author='Jay Stanley and Scott Gigante, Krishnaswamy Lab, Yale University',
      author_email='jay.stanley@yale.edu',
      packages=['harmonicalignment', ],
      include_package_data=True,
      license='GNU General Public License Version 2',
      python_requires='>=3.5',
      install_requires=install_requires,
      extras_require={'test': test_requires},
      test_suite='nose2.collector.collector',
      long_description=readme,
      url='https://github.com/KrishnaswamyLab/harmonic-alignment',
      download_url="https://github.com/KrishnaswamyLab/harmonic-alignment/archive/v{}.tar.gz".format(
          version),
      keywords=['graphs',
                'big-data',
                'signal processing',
                'manifold-learning',
                ],
      classifiers=[
          'Development Status :: 2 - Pre-Alpha',
          'Environment :: Console',
          'Framework :: Jupyter',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Natural Language :: English',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX :: Linux',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Mathematics',
      ]
      )
