suppressWarnings(library("argparse"))
suppressWarnings(library("DescTools"))
suppressWarnings(library("logging"))
suppressWarnings(library("broom"))
suppressWarnings(library("biglm"))
suppressWarnings(library("corrplot"))
suppressWarnings(library("parallel"))
suppressWarnings(library("ff"))
suppressWarnings(library("nnet"))
suppressWarnings(library("MASS"))


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

#Read the csv files
datalist = lapply(files_to_read,read.csv)

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
      datasub <-dataset
    }
    # Remove columns with all values as zero
    datasub <- datasub[colSums(datasub) > 0]
    
    #Check whether predict column is present in dataframe
    if(!(predictcolumn %in% colnames(datasub))) {
      logwarn('predict column name is not found in %s',file_name)
      next
    }
    
    #Get column names without predict variable
    drop_dep <- datasub[ , !(names(datasub) %in% predictcolumn)]
    resp_var <- colnames(drop_dep)
  
    #Number of cores
    num_of_cores = detectCores()
    loginfo('Cores = %s', num_of_cores)
    
    #Chunk Size
    chunk <- floor((nrow(datasub)/ncol(datasub))*num_of_cores)
    
    #Function to determine chunks
    make.data<-function(formula,data,chunksize,...){
      n<-nrow(data)
      cursor<-0
      datafun<-function(reset=FALSE){
        if (reset){
          cursor<<-0
          return(NULL)
        }
        if (cursor>=n)
          return(NULL)
        start<-cursor+1
        cursor<<-cursor+min(chunksize, n-cursor)
        data[start:cursor,]
      }
    }
    
    #Convert to ffdf object
    datasub_ff = as.ffdf(datasub)
    
    #Chunk data
    chunk_data <-make.data(formula(paste(predictcolumn,paste(resp_var,collapse= "+"),sep="~")), datasub_ff, chunksize=chunk)

    if((modeltype == 'Gaussian') || (modeltype == 'Poisson') || (modeltype == 'Binomial') || (modeltype == 'Quasibinomial') || (modeltype == 'Quasipoisson') || (modeltype == 'Quasi')) {
      modeltype <- tolower(modeltype)
    }
    
    if (modeltype == 'NegativeBinomial') {
      fit <- glm.nb(as.formula(paste(predictcolumn,1,sep="~")), data = datasub)
      mu <- exp(coef(fit))
      val_pred<-eval(parse(text=paste('datasub',predictcolumn, sep = "$")))
      theta_val = theta.ml(val_pred, mu,nrow(datasub), limit = 22, eps = .Machine$double.eps^0.25, trace = FALSE)
    }
    
    model_list <- c('gaussian','Gamma', 'binomial', 'poisson', 'quasi', 'quasibinomial', 'quasipoisson' )
    
    model_data <- function(pred_var, data_model) {
      if((modeltype %in% model_list)) {
        reg_model <- bigglm(formula(paste(predictcolumn,paste(pred_var,collapse= "+"),sep="~")), data = data_model, family = eval(parse(text=paste(modeltype,"()", sep = ""))), chunksize = chunk)
      }
      else if(modeltype == 'NegativeBinomial') {
        reg_model <- bigglm(formula(paste(predictcolumn,paste(pred_var,collapse= "+"),sep="~")), data = data_model, family = negative.binomial(theta= theta_val), chunksize=chunk)
      }
      else if(modeltype == 'Multinomial') {
        reg_model <- multinom(formula(paste(paste("as.factor(",predictcolumn,")"),paste(pred_var,collapse= "+"),sep="~")), data = data_model, maxit=10, MaxNWts = 10000)
      }
      return(reg_model)
    }
    
    #Model data based on the options selected
    #Get only main effects of the variables
    if (glmmethod == 'PrimaryFactors') {
      if (modeltype != 'Multinomial') {
        test_glm<- model_data(resp_var,chunk_data)
      }
      else if (modeltype == 'Multinomial') {
        test_glm<- model_data(resp_var,datasub_ff)
      }
    }
    #Get interaction values
    else if (glmmethod == 'Interaction') {
      datasub_pred <- datasub[ , !(names(datasub) %in% predictcolumn)]
      #Get correlation between variables
      tmp <- cor(datasub_pred)
      tmp[upper.tri(tmp)] <- 0
      diag(tmp) <- 0
      
      #Remove variables with no interaction
      data_no_int <- which(tmp >= 0.1 | tmp < -0.1, arr.ind = TRUE)
      data_frame<-data.frame(row = rownames(data_no_int), col = colnames(tmp)[data_no_int[, "col"]],
                             value = tmp[tmp >= 0.1 | tmp < -0.1])
      colnames(data_frame)<- c("variable1","variable2","coef")
      
      #Interaction variables
      data_frame$variableint <- paste(data_frame$variable1, data_frame$variable2, sep="*")
      data_list <- as.character(data_frame$variableint)
      if (modeltype != 'Multinomial') {
        test_glm<- model_data(data_list,chunk_data)
      }
      else if (modeltype == 'Multinomial') {
        test_glm<- model_data(data_list, datasub_ff)
      }
    }
    #Get second order polynomial values
    else if (glmmethod == 'SecondOrder') {
      var_resp <- paste('poly(',resp_var,',2)')
      if (modeltype != 'Multinomial') {
        test_glm<- model_data(var_resp,chunk_data)
      }
      else if (modeltype == 'Multinomial') {
        test_glm<- model_data(var_resp,datasub_ff)
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