# Principal Interaction Component Analysis through Likelihood Optimization (PICALO)

Expression quantitative trait loci (eQTL) help explain the regulatory mechanisms of trait associated variants. eQTL effect sizes are often dependent on observed and unobserved biological contexts, such as cell type composition and environmental factors. Here, we introduce PICALO (Principal Interaction Component Analysis through Likelihood Optimization) which is an unbiased method to identify known and hidden contexts that influence eQTLs.

## How it works

PICALO is an expectation maximization (EM) based algorithm. Using an initial guess to start the optimization, PICALO identifies eQTLs that interact with this context (interaction eQTLs; ieQTLs) and subsequently optimizes this interaction. 

### PICALO applied to a single eQTL
Per sample the context value is adjusted to reduce maximize the log likelihood of the whole model (i.e. minimize the residuals) while maintaining the model parameters. The underlying assumption is that while the per sample context value might be noisy, the whole model captures the true interaction. Considering the case in which only one ieQTL is optimized, this results in all samples moving towards their genotype's regression line:

![Single ieQTL optimization example]()

### Numerous eQTLs are inlcuded to prevent overfitting
In practise however, PICALO applies this optimization over all ieQTLs simultaneously. This will result in a new vector that maximally interacts with the eQTLs. 

![Double ieQTL optimization example]()

### Process is repeated to identify Principal Interaction Components
Next, a new set eQTLs that interact with the optimized vector are identified over which the optimization step is repeated. This process repeats until convergence. The resulting vector is referred to as Principal Interaction Components (PICs), which maximally affect eQTL effect-sizes.

![EM looping example]()

Finally, multiple PICs are identified by removing the previous PICs and repeating the proces.

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
python3 -m venv <name of virtual environment>
source <name of virtual environment>/bin/activate
```

The python packager manager (pip) and the requirements file can be used to install all the necessary packages. Note that the [requirements.txt](requirements.txt) file includes depedencies with their correct versions. Therefore, include the flag --no-dependencies when installing the packages to prevent unnecessary upgrading. 
```  
pip install -r requirements.txt --no-dependencies
```

You can exit the virtual environment by typing:
```  
deactivate
```

## Usage  

```  
./picalo.py -h
```  

### Information arguments:
 * **-h**, **--help**: show this help message and exit
 * **-v**, **--version**: show program's version number and exit

### Required data arguments:
 * **-eq**, **--eqtl**: The path to the eqtl matrix. Expects to contain the columns 'Pvalue' (eQTL p-value), 'SNPName' (the eQTL SNP), and 'ProbeName' (the eQTL gene).

 * **-ge**, **--genotype**: The path to the genotype matrix. Expects to contain genotype dosages (between 0 en 2). Missing genotypes are by default -1 (can be changed used **-na** / **--genotype_na**). The rows should contain the SNPs on the same order as the **-eq** / **--eqtl** files. The columns should contain the samples on the same order as the **-ex** / **--expression** and **-co** / **--covariate** files.

 * **-ex**, **--expression**: The path to the expression matrix. The rows should contain the gene expression on the same order as the **-eq** / **--eqtl** files. The columns should contain the samples on the same order as the **-ge** / **--genotype** and **-co** / **--covariate** files.

 * **-co**, **--covariate**: The path to the covariate matrix (i.e. the matrix used as initial guess for the optimization). The rows should contain the different covariates on the same order as the **-eq** / **--eqtl** files. The columns should contain the samples on the same order as the **-ge** / **--genotype** and **-ex** / **--expression** files.

### Optinal data arguments:
 * **-na**, **--genotype_na**: The genotype value that equals a missing value. Default: -1.

 * **-tc**, **--tech_covariate**: The path to the technical covariate matrix (excluding an interaction with genotype). Default: None. The rows should contain the technical covariates to correct for. The columns should contain the samples on the same order as the **-ge** / **--genotype**, **-ex** / **--expression** file, and **-co** / **--covariate** files.

 * **-tci**, **--tech_covariate_with_inter**: The path to the technical covariate matrix(including an interaction with genotype). Default: None. The rows should contain the technical covariates to correct for including an interaction term with genotype. The columns should contain the samples on the same order as the **-ge** / **--genotype**, **-ex** / **--expression** file, and **-co** / **--covariate** files.

 * **-std**, **--sample_to_dataset**: The path to the sample-dataset link matrix. Default: None. Note that his argument is required if the input data conists of multiple datasets. The rows should contain sample - dataset links on the same order as the **-ge** / **--genotype**, **-ex** / **--expression** file, and **-co** / **--covariate** files.

### eQTL inclusion arguments:
 * **-mds**, **--min_dataset_size**: The minimal number of samples per dataset. Default: >=30.
 * **-ea**, **--eqtl_alpha**: The eQTL significance cut-off. Default: <=0.05.
 * **-cr**, **--call_rate**: The minimal call rate of a SNP (per dataset).Equals to (1 - missingness). Default: >= 0.95.
 * **-hw**, **--hardy_weinberg_pvalue**: The Hardy-Weinberg p-value threshold.Default: >= 1e-4.
 * **-maf**, **--minor_allele_frequency**: The MAF threshold. Default: >0.01.
 * **-mgs**, **--min_group_size**: The minimal number of samples per genotype group. Default: >= 2.
 * **-iea**, **--ieqtl_alpha**: The interaction eQTL significance cut-off. Default: <=0.05.

### PICALO optimization arguments:
 * **-n_components**: The number of components to extract. Default: 10.
 * **-min_iter**: The minimum number of optimization iterations to perform per component. Default: 5.
 * **-max_iter**: The maximum number of optimization iterations to perform per component. Default: 100.
 * **-tol**: The convergence threshold. The optimization will stop when the 1 - pearson correlation coefficient is below this threshold. Default: 1e-3.
 * **-force_continue**: Force to identify more components even if the previous one did not converge. Default: False.

### Output arguments:
 * **-o**, **--outdir**: The name of the output folder.
 * **-verbose**: Enable verbose output. Default: False.

## Output

PICALO generate the following output files:
 * **call_rate.txt.gz**: containing the per dataset snp call rates.
 * **components.txt.gz**: containing the identified components (also including non converged components in contrast to 'PICs.txt.gz').
 * **genotype_stats.txt.gz**: containing the gentype summary statistics like sample size, Hardy-Weinberg equilibrium p-value, minor allele frequency (MAF), etc.
 * **log.log**: log file containing all terminal output.
 * **PICs.txt.gz**: containing the identified PICs.
 * **SummaryStats.txt.gz**: containing the number of interacting eQTLs per PIC.

And per PIC the following output files:
 * **PICN/component.npy**: numpy binary file containing the converged PIC for easy loading in case of a restart.
 * **PICN/covariate_selection.txt.gz**: containing the number of interacting eQTLs per covaraite (i.e. initial guess).
 * **PICN/info.txt.gz**: containing optimization statistics per iterations like the number of interacting eQTLs, the correlation with the previous iteration etc.
 * **PICN/iteration.txt.gz**: containing the component loadings per iteration.
 * **PICN/n_ieqtls_per_sample.txt.gz**: containing the number of eQTLs used for optimization per sample per iteration.
 * **PICN/results_iterationN.txt.gz**: containing the interaction eQTL summary statistics per iteration.

Finally, the number of interacting eQTLs are determined without removal of PICs. The summary statistics of which are stored in: **PIC_interactions/PICN.txt.gz**.

## Github Content

 * **src**: the source code for the PICALO program
 * **dev**: the source code used for development
 * **dev/*/plot_scripts**: scripts used to visualise the output of PICALO
 * **dev/*/postprocess_scripts**: script used for post-processing the PICALO output (i.e. downstream analyses)
 * **dev/*/preprocess_scripts**: script used for pre-processing the input data
 * **dev/*/presentation_scripts**: script used for creating plots for presentations
 * **dev/*/test_scripts**: scripts used to test things / visualise stuff / etc.

The files are split between **general** and dataset specific scripts (**BIOS** / **MetaBrain**).

## Author  

Martijn Vochteloo (m.vochteloo@umcg.nl) *(1)*

1. University Medical Centre Groningen, Groningen, The Netherlands

## License  

This project is licensed under the GNU GPL v3 license - see the [LICENSE](LICENSE) file for details

