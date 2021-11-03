# Principal Interaction Component Analysis through Likelihood Optimization (PICALO)
Project repository a novel method called PICALO that allows for the identification of interaction components that influence eQTL effect size. This program uses an expectation maximization algorithm to maximize an interaction component. 

## Introduction

TODO

## Prerequisites  

This program is written in Python v3.7.4. The program requires the following packages to be installed:  

 * numpy (v1.19.5)
 * pandas (v1.2.1) 
 * scipy (v1.6.0)
 * statsmodels (v0.12.2)
 * matplotlib (v3.3.4)
 * seaborn (v0.11.1)

See 'Installing' on how to install these packages. Note that the performance / reliability of this program on other versions is not guaranteed.

### Installing  

Install the required packages. Consider using a virtual environment to ensure the right version of packages are used.
```  
python3 -m venv env
source env/bin/activate
```

The python packager manager (pip) and the requirements file can be used to install all the necessary packages. Note that the [requirements.txt](requirements.txt) file includes depedencies with their correct versions. Therefore, include the flag --no-dependencies when installing the packages to prevent unnecessary upgrading. 
```  
pip install -r requirements.txt --no-dependencies
```

You can exit the virtual environment by typing:
```  
deactivate
```

## Input data

TODO

## Usage  

```  
./picalo.py -h
```  
  

## Content

 * **[src](src)**: the source code for the PICALO program
 * **[dev/preprocess_scripts](dev/preprocess_scripts)**: script used for pre-processing the input data
 * **[dev/plot_scripts](dev/plot_scripts)**: scripts used to visualise the output of PICALO
 * **[dev/test_scripts](dev/test_scripts)**: scripts used to test things / visualise stuff / etc.


## Author  

Martijn Vochteloo *(1)*

1. University Medical Centre Groningen, Groningen, The Netherlands

## License  

This project is licensed under the GNU GPL v3 license - see the [LICENSE](LICENSE) file for details

