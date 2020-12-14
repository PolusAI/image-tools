suppressWarnings(library("argparse"))
suppressWarnings(library("DescTools"))
suppressWarnings(library("logging"))
suppressWarnings(library("broom"))
suppressWarnings(library("biglm"))
suppressWarnings(library("corrplot"))
suppressWarnings(library("parallel"))
suppressWarnings(library("ff"))
suppressWarnings(library("nnet"))


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
    
    #Scaling the data
    datasub1<- datasub[1:(length(datasub)-1)]
    datasub_scale <- scale(datasub1, center = TRUE, scale = TRUE)
    datasub$Cluster <- as.factor(datasub$Cluster)
    Cluster <- datasub$Cluster
    data_final <- cbind((as.data.frame(datasub_scale)),Cluster)
    
    #Get column names without predict variable
    drop_dep <- data_final[ , !(names(data_final) %in% predictcolumn)]
    resp_var <- colnames(drop_dep)
    
    
    if((modeltype == 'Gaussian') || (modeltype == 'Poisson') || (modeltype == 'Binomial') || (modeltype == 'Multinomial') || (modeltype == 'Quasibinomial') || (modeltype == 'Quasipoisson') || (modeltype == 'Quasi')) {
      modeltype <- tolower(modeltype)
    }
    
    memory.limit(size= 40000)
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
    datasub_ff = as.ffdf(data_final)
    
    #Chunk data
    chunk_data <-make.data(formula(paste(predictcolumn,paste(resp_var,collapse= "+"),sep="~")), datasub_ff, chunksize=chunk)

    #Model data based on the options selected
    #Get only main effects of the variables
    if (glmmethod == 'PrimaryFactors') {
      if((modeltype == 'gaussian') || (modeltype == 'Gamma') || (modeltype == 'binomial') ||  (modeltype == 'quasibinomial') || (modeltype == 'poisson') || (modeltype == 'quasipoisson') || (modeltype == 'quasi')) {
        test_glm <- bigglm(formula(paste(predictcolumn,paste(resp_var,collapse= "+"),sep="~")), data = chunk_data, family = eval(parse(text=paste(modeltype,"()", sep = ""))), chunksize = chunk)
      }
      else if (modeltype == 'NegativeBinomial') {
        #Based on glm.nb from MASS package, if theta is not given then poisson GLM is calculated and hence, have considered family as poisson
        test_glm <- bigglm(formula(paste(predictcolumn,paste(resp_var,collapse= "+"),sep="~")), data = chunk_data, family = poisson(),chunksize=chunk)
      }
      else if (modeltype == 'multinomial') {
        test_glm <- multinom(formula(paste(paste("as.factor(",predictcolumn,")"),paste(resp_var,collapse= "+"),sep="~")), data = data_final)
      }
    }
    #Get interaction values
    else if (glmmethod == 'Interaction') {
      datasub_pred <- data_final[ , !(names(data_final) %in% predictcolumn)]
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
      
      if((modeltype == 'gaussian') || (modeltype == 'Gamma') || (modeltype == 'binomial') ||  (modeltype == 'quasibinomial') || (modeltype == 'poisson') || (modeltype == 'quasipoisson') || (modeltype == 'quasi')) {
        test_glm <- bigglm(formula(paste(predictcolumn,paste(data_list,collapse= "+"),sep="~")), data = chunk_data, family = eval(parse(text=paste(modeltype,"()", sep = ""))), chunksize = chunk)
      }
      else if (modeltype == 'NegativeBinomial') {
        test_glm <- bigglm(formula(paste(predictcolumn,paste(data_list,collapse= "+"),sep="~")), data = chunk_data, family = poisson(link='log'), chunksize = chunk)
      }
      else if (modeltype == 'multinomial') {
        rm(datasub, datasub1,datasub_scale,datasub_ff,drop_dep, dataset)
        gc()
        tidy_check = NULL
        tidy_summary= tidy(tidy_check)

        for (ls in data_list) {
          test_glm1 <- multinom(formula(paste(predictcolumn,paste(ls,1,sep="-"),sep="~")), data = data_final,maxit=1000,MaxNWts = 10000, trace= FALSE)
          tidy_df <- tidy(test_glm1)
          tidy_combine <- rbind(tidy_summary, tidy_df)
          tidy_summary <- tidy_combine
        }
      }
    }
    #Get second order polynomial values
    else if (glmmethod == 'SecondOrder') {
      if((modeltype == 'gaussian') || (modeltype == 'Gamma') || (modeltype == 'binomial') ||  (modeltype == 'quasibinomial') || (modeltype == 'poisson') || (modeltype == 'quasipoisson') || (modeltype == 'quasi')) {
        test_glm <- bigglm(formula(paste(predictcolumn,paste('poly(',resp_var,',2)',collapse = ' + '),sep="~")), data = chunk_data,family = eval(parse(text=paste(modeltype,"()", sep = ""))),chunksize=chunk)
      }
      else if (modeltype == 'NegativeBinomial') {
        test_glm <- bigglm(formula(paste(predictcolumn,paste('poly(',resp_var,',2)',collapse = ' + '),sep="~")), data = chunk_data, family = poisson(), chunksize=chunk)
      }
      else if (modeltype == 'multinomial') {
        test_glm <- multinom(formula(paste(predictcolumn,paste('poly(',resp_var,',2)',collapse = ' + '),sep="~")), data = data_final)
      }
    }
    
    #Set output directory
    setwd(csvfile)
    file_save <- paste0(file_name,".csv")

    #Convert summary of the analysis to a dataframe
    if ((glmmethod != 'Interaction') && (modeltype == 'multinomial')) {

    tidy_summary <- tidy(test_glm)
    }
    
    #Reorder the columns
    tidy_final <- tidy_summary[c("y.level","term", "p.value", "estimate","std.error")]
    colnames(tidy_final) <- c("Level","Factors","P-Value","Estimate","Std.Error")
    tidy_final <- tidy_final[order(tidy_final$Level),]
    if (glmmethod == 'Interaction'){
    tidy_final<- tidy_final[grep(':',tidy_final$Factors), ]
    }
    #Write the dataframe to csv file
    write.csv(tidy_final, file_save)
  }
}


