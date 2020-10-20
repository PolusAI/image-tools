suppressWarnings(library("argparse"))
suppressWarnings(library("MASS"))
suppressWarnings(library("DescTools"))
suppressWarnings(library("logging"))
suppressWarnings(library("broom"))

# Initialize the logger
basicConfig()

# Setup the argument parsing
addRequiredArgs <- function(parser) {
  parser$add_argument("--inpdir",type = "character",help="Input csv file")
  invisible(NULL)
}
addPredictArgs <- function(parser) {
  parser$add_argument("--predictcolumn",type = "character",help="Column to predict")
  invisible(NULL)
}
addExcludeArgs <- function(parser) {
  parser$add_argument("--exclude",type = "character",help="Column to exclude from the analysis")
  invisible(NULL)
}
addMethodArgs <- function(parser) {
  parser$add_argument("--glmmethod",type = "character",help="Analyse only primary factors or interactions or second order effects")
  invisible(NULL)
}
addModelArgs <- function(parser) {
  parser$add_argument("--modeltype",type = "character",help="Select either binomial or gaussian or Gamma or poisson or quasi or quasibinomial or quasipoisson or negativebinomial")
  invisible(NULL)
}
addOutputArgs <- function(parser) {
  parser$add_argument("--outdir", type = "character",help="Output csv file")
  invisible(NULL)
}
getAllParser <- function() {
  parser <- ArgumentParser(description="ALL PARSER")
  addRequiredArgs(parser)
  addPredictArgs(parser)
  addExcludeArgs(parser)
  addMethodArgs(parser)
  addModelArgs(parser)
  addOutputArgs(parser)
  return(parser)
}

# Parse the arguments
parser <- getAllParser()
args <- parser$parse_args()

#Path to csvfile directory
inpfile <- args$inpdir
loginfo('inpfile = %s', inpfile)

#Column to be predicted
predictcolumn <- args$predictcolumn
loginfo('predictcolumn = %s', predictcolumn)

#Columns to be excluded
exclude_col <- args$exclude
excludes<-as.list(strsplit(exclude_col, ",")[[1]])
loginfo('exclude = %s', excludes)

#Select glmmethod-primary or interaction or secondorder
glmmethod <- args$glmmethod
loginfo('glmmethod = %s', glmmethod)

#Select modeltype based on distribution of data
modeltype <- args$modeltype
loginfo('modeltype = %s', modeltype)

#Path to save output csvfiles
csvfile <- args$outdir
loginfo('csvfile = %s', csvfile)


#Get list of .csv files in the directory including sub folders for modeling
files_to_read = list.files(
  path = inpfile,        
  pattern = ".*csv", 
  recursive = TRUE,          
  full.names = TRUE
)

#Check whether there are csv files in the directory
if(length(files_to_read) == 0) {
  tryCatch(
    error = function(e) { 
      message('No .csv files in the directory')
    }
  )
}

#Read the csv files
datalist = lapply(files_to_read, read.csv)

for (dataset in datalist) {
  for (file_csv in files_to_read) {
    #Get filename
    file_name <- SplitPath(file_csv)$filename
    
    #Check whether any column needs to be excluded
    if(length(excludes) > 0) {
      for (i in 1:length(excludes)) {
        if(!excludes[i] %in% colnames(dataset)) {
          logwarn('column to exclude from %s is not found',file_name)
        }
      }
      datasub <-dataset[ , !(names(dataset) %in% excludes)]
    }
    else if(length(excludes) == 0) {
      datasub<-dataset
    }
    
    
    #Check whether predict column is present in dataframe
    if(!(predictcolumn %in% colnames(datasub))) {
      logwarn('predict column name is not found in %s',file_name)
      next
    }
    
    #Get column names without predict variable
    drop_dep <- datasub[ , !(names(datasub) %in% predictcolumn)]
    resp_var <- colnames(drop_dep)
    
    if((modeltype == 'Gaussian') || (modeltype == 'Poisson') || (modeltype == 'Binomial') || (modeltype == 'Quasibinomial') || (modeltype == 'Quasipoisson') || (modeltype == 'Quasi')) {
      modeltype <- tolower(modeltype)
    }

    #Model data based on the options selected
    if (glmmethod == 'PrimaryFactors') {
      if((modeltype == 'gaussian') || (modeltype == 'Gamma') || (modeltype == 'poisson') || (modeltype == 'quasipoisson') || (modeltype == 'quasi')) {
        test_glm <- glm(as.formula(paste(predictcolumn,paste(resp_var,collapse= "+"),sep="~")),data = datasub, family = modeltype)
       }
      else if (modeltype == 'NegativeBinomial') {
        test_glm <- glm.nb(as.formula(paste(predictcolumn,paste(resp_var,collapse= "+"),sep="~")), data = datasub)
      }
      else if ((modeltype == 'binomial') || (modeltype == 'quasibinomial')) {
        test_glm <- glm(as.formula(paste(paste("as.factor(",predictcolumn,")"),paste(resp_var,collapse= "+"),sep="~")), data = datasub, family = modeltype)
      }
    }
    
    else if (glmmethod == 'Interaction') {
      if((modeltype == 'gaussian') || (modeltype == 'Gamma') || (modeltype == 'poisson') || (modeltype == 'quasipoisson') || (modeltype == 'quasi')) {
        test_glm <- glm(as.formula(paste(predictcolumn,paste('(',paste(resp_var,collapse= "+"),')^2'),sep="~")), data = datasub, family = modeltype)
      }
      else if (modeltype == 'NegativeBinomial') {
        test_glm <- glm.nb(as.formula(paste(predictcolumn,paste('(',paste(resp_var,collapse= "+"),')^2'),sep="~")), data = datasub)
      }
      else if ((modeltype == 'binomial') || (modeltype == 'quasibinomial')) {
        test_glm <- glm(as.formula(paste(paste("as.factor(",predictcolumn,")"),paste('(',paste(resp_var,collapse= "+"),')^2'),sep="~")), data = datasub, family = modeltype)
      }
    }
    
    else if (glmmethod == 'SecondOrder') {
      if((modeltype == 'gaussian') || (modeltype == 'Gamma') || (modeltype == 'poisson') || (modeltype == 'quasipoisson') || (modeltype == 'quasi')) {
        test_glm <- glm(as.formula(paste(predictcolumn,paste('poly(',resp_var,',2)',collapse = ' + '),sep="~")),data=datasub,family = modeltype)
      }
      else if (modeltype == 'NegativeBinomial') {
        test_glm <- glm.nb(as.formula(paste(predictcolumn,paste('poly(',resp_var,',2)',collapse = ' + '),sep="~")), data = datasub)
      }
      else if ((modeltype == 'binomial') || (modeltype == 'quasibinomial')) {
        test_glm <- glm(as.formula(paste(paste("as.factor(",predictcolumn,")"),paste('poly(',resp_var,',2)',collapse = ' + '),sep="~")),data=datasub,family = modeltype)
      }
    }
    
    #Set output directory
    setwd(csvfile)
    file_save <- paste0(file_name,".csv")
    
    #Convert summary of the analysis to a dataframe
    tidy_summary <- tidy(test_glm)
    
    #Reorder the columns
    tidy_final <- tidy_summary[c("term", "p.value", "estimate","std.error")]
    colnames(tidy_final) <- c("Factors","P-Value","Estimate","Std.Error")
    
    #Write the dataframe to csv file
    write.csv(tidy_final, file_save)
  }
}

