from setuptools import setup, find_packages

setup(
    name='staghunt',
    version='0.0.1',
    packages=find_packages(),
    test_suite='nose.collector',
    tests_require=['nose'],
    url='https://github.com/airibarne/msthesis',
    license='MIT',
    author='Albert Iribarne',
    author_email='albert.iribarne@gmail.com',
    description='Sparse Matrix Belief Propagation on a multi-agent Stag-Hunt game', install_requires=['numpy',
                                                                                                      'matplotlib',
                                                                                                      'torch']
)
