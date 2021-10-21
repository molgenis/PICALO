# Principal Interaction Component Analysis (PICA)
Project repository a novel method called PICA that allows for the identification of interaction components that influence eQTL effect size. This program uses an expectation maximization algorithm to maximize an interaction component. 

## Introduction

TODO

## Prerequisites  

This program is developed in Pycharm v2021.1.2 (Professional Edition) and written in Python v3.7.4, performance on other system / version is not guaranteed.

The program requires the following packages to be installed:  

 * matplotlib (>= v3.3.4)
 * numpy (>= v1.19.5)
 * pandas (>= v1.2.1) 
 * scipy (>= v1.7.1)
 * seaborn (>= v0.11.1)
 * statsmodels (>= v0.12.2)

See 'Installing' on how to install these packages.

### Installing  

Install the required packages. The python packager manager (pip) and the requirements file can be used to install all the necessary packages:  
```  
pip install -r requirements.txt
```  

## Input data

TODO

## Usage  

```  
./pica.py -h
```  
  

## Content

 * **[src](src)**: the source code for the PICA program
 * **[dev/preprocess_scripts](dev/preprocess_scripts)**: script used for pre-processing the input data
 * **[dev/plot_scripts](dev/plot_scripts)**: scripts used to visualise the output of PICA
 * **[dev/test_scripts](dev/test_scripts)**: scripts used to test things / visualise stuff / etc.


## Author  

Martijn Vochteloo *(1)*

1. University Medical Centre Groningen, Groningen, The Netherlands

## License  

This project is licensed under the GNU GPL v3 license - see the [LICENSE](LICENSE) file for details

