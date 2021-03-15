# Generalized Linear Model

The generalized linear model plugin models the data using bigglm() function in R and outputs csv file containing the summary of the analysis.The input file should be in csv format.

## Inputs:
### Input csv collection:
The input file that needs to be modeled. The file should be in csv format. This is a required parameter for the plugin.

### Predict column:
Enter the column name that needs to be predicted.

### Exclude:
Enter column names that needs to be excluded from the analysis.

### Methods:
There are 3 options:
1. Primary factors - If only the relationship between primary factors and the column to be predicted is required, then choose this option.
2. Interaction - To find how interaction of variables affect the response, choose interactions for two-way interaction.
3. Second order - To find second order effect use this option. Degree is 2.

### Modeltype:
Choose any one of the modeltype based on list of options- Binomial, Gaussian, Gamma, Poisson, Quasi, Quasibinomial, Quasipoisson, NegativeBinomial, Multinomial based on the distribution of dataset.
      
## Note:
If multiple csv files needs to be processed, the column names should be the same for all files.

## Output:
The output is a csv file containing the summary of the analysis with p-value and regression value for the factors..

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes eight input argument and one output argument:

| Name                   | Description             | I/O    | Type   |
|------------------------|-------------------------|--------|--------|
| `--inpdir` | Input csv collection| Input | csvCollection |
| `--predictcolumn` | Column needs to be predicted | Input | string |
| `--exclude` | Enter columns to be excluded| Input | string |
| `--glmmethod` | Analyse either primaryfactors or interaction or second order effects for modeling | Input | enum |
| `--modeltype` | Select the distribution to be considered for modeling| Input | enum |
| `--outdir` | Output collection | Output | csvCollection |


