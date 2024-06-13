library("sparsebn")
library("stringi")


paths <- list.files(path = "path_to_project_folder/ppi4aid/data/preprocessed/networks_changes_SDR", full.names = TRUE, recursive = FALSE)
paths <- paths[grepl('changes', paths)]

for (filep in paths){
  print(filep)
  S <- t(data.matrix(read.csv(filep, header=FALSE, sep = ",")))
  data <- sparsebnData(S, type = "continuous")
  dags.estimate <- estimate.dag(data)
  dags.param <- estimate.parameters(dags.estimate, data=data)
  selected.lambda <- select.parameter(dags.estimate, data=data)
  dags.final.net <- dags.estimate[[selected.lambda]]
  dags.final.param <- dags.param[[selected.lambda]]
  adjMatrix <- dags.final.param$coefs
  write.table(as.matrix(adjMatrix), file=paste("path_to_project_folder/ppi4aid/data/preprocessed/networks_sparse_SDR/", stri_sub(filep,85,), sep=''), row.names=FALSE, col.names=FALSE)

}

















