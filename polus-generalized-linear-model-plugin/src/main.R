suppressWarnings(library("argparse"))
suppressWarnings(library("MASS"))
suppressWarnings(library("DescTools"))
suppressWarnings(library("logging"))
suppressWarnings(library("broom"))
suppressWarnings(library("dplyr"))

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
  parser$add_argument("--modeltype",type = "character",help="Select either binomial or gaussian or Gamma or inverse.gaussian or poisson or quasi or quasibinomial or quasipoisson or negativebinomial")
  invisible(NULL)
}
addOutputArgs <- function(parser) {
  parser$add_argument("--outfile", type = "character",help="Output csv file")
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
exclude<-as.list(strsplit(exclude_col, ",")[[1]])
loginfo('exclude = %s', exclude)

#Select glmmethod-primary or interaction or secondorder
glmmethod <- args$glmmethod
loginfo('glmmethod = %s', glmmethod)

#Select modeltype based on distribution of data
modeltype <- args$modeltype
loginfo('modeltype = %s', modeltype)

#Path to save output csvfiles
csvfile <- args$outfile
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
  tryCatch(error = function(e) { message('No .csv files in the directory')})
}

#Read the csv files
datalist = lapply(files_to_read, read.csv)

for (dataset in datalist){
for (file_csv in files_to_read){
#Check whether any column needs to be excluded
if(length(exclude) > 0){
datasub <-dataset[ , !(names(dataset) %in% exclude)]}
else if(length(exclude) == 0){
datasub<-dataset}
predictcolumns = eval(parse(text=paste("datasub$", predictcolumn, sep = "")))
if (!(modeltype == 'binomial')) {
  poly_check = as.formula(paste('predictcolumns ~',paste('poly(',colnames(datasub[-1]),',2,raw=TRUE)',collapse = ' + ')))
}else if (modeltype == 'binomial') {
  poly_check = as.formula(paste('as.factor(predictcolumns) ~',paste('poly(',colnames(datasub[-1]),',2,raw=TRUE)',collapse = ' + ')))
}
#Model data based on the options selected
if (glmmethod == 'PrimaryFactors') {
  if(!(modeltype == 'NegativeBinomial') && !(modeltype == 'binomial') && !(modeltype == 'quasibinomial')){
  test_glm <- glm(predictcolumns ~., data = datasub, family = modeltype)
}else if (modeltype == 'NegativeBinomial'){
  test_glm <- glm.nb(predictcolumns ~ ., data = datasub)
}else if ((modeltype == 'binomial') || (modeltype == 'quasibinomial')){
  test_glm <- glm(as.factor(predictcolumns) ~ ., data = datasub, family = modeltype)
}}else if (glmmethod == 'Interaction') {
  if(!(modeltype == 'NegativeBinomial') && !(modeltype == 'binomial') && !(modeltype == 'quasibinomial')){
  test_glm <- glm(predictcolumns ~ (.)^2, data = datasub, family = modeltype)
}else if (modeltype == 'NegativeBinomial'){
  test_glm <- glm.nb(predictcolumns ~ (.)^2, data = datasub)
}else if ((modeltype == 'binomial') || (modeltype == 'quasibinomial')){
  test_glm <- glm(as.factor(predictcolumns) ~ (.)^2, data = datasub, family = modeltype)
}}else if (glmmethod == 'SecondOrder') {
  if(!(modeltype == 'NegativeBinomial')&& !(modeltype == 'binomial') && !(modeltype == 'quasibinomial')){
  test_glm <- glm(poly_check, data = datasub, family = modeltype)
}else if (modeltype == 'NegativeBinomial'){
  test_glm <- glm.nb(poly_check, data = datasub)
}else if ((modeltype == 'binomial') || (modeltype == 'quasibinomial')){
  test_glm <- glm(poly_check, data = datasub, family = modeltype)
}}

#Set output directory
setwd(csvfile)
#Get filename
file_name <- SplitPath(file_csv)$filename
file_save <- paste0(file_name,".csv")
#COnvert summary of the analysis to a dataframe
tidy_summary <- tidy(test_glm)
#Reorder the columns
tidy_che <- tidy_summary[c("term", "p.value", "estimate","std.error")]
tidy_final <- tidy_che %>% rename(Factor = term)
#Write the dataframe to csv file
write.csv(tidy_final, file_save)}}

