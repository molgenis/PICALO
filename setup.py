from setuptools import setup

setup(
    name='PICALO',
    version='0.1',
    license='BSD (3-Clause)',
    author="Martijn Vochteloo",
    author_email='m.vochteloo@umcg.nl',
    package_dir={'': 'src'},
    url='https://github.com/molgenis/PICALO',
    keywords='',
    install_requires=[
        'numpy==1.19.5',
        'pandas==1.2.1',
        'scipy==1.6.0',
        'statsmodels==0.12.2',
        'matplotlib==3.3.4',
        'seaborn==0.11.1'
    ],

)