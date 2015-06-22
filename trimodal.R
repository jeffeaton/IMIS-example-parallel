## Compare the output of 50 replications from R implementation and C implementation to confirm it works

library(IMIS)

likelihood <- function(theta)  dmvnorm(theta, c(-1.5,-1.5), matrix(c(.3,0,0,.3),2,2)) + 
  dmvnorm(theta, c(-2,1.5), matrix(c(.2,-.15,-.15,.2),2,2)) +
  dmvnorm(theta, c(1.5,-.5), matrix(c(.4, 0.2,0.2,.4),2,2))
prior <- function(theta)        dmvnorm(theta, c(0,0), diag(c(3,2)))
sample.prior <- function(n)     rmvnorm(n, c(0,0), diag(c(3,2)))

## Fit model using IMIS package
system.time(trimodal <- replicate(50, IMIS(1e4, 1e5, 100, 0), simplify=FALSE))

## Load output from C simulation
cout <- lapply(paste("trimodal", 0:49, "Resample.txt", sep=""), read.table)

rowMeans(sapply(cout, function(x) colMeans(x[,1:2])))
rowMeans(sapply(trimodal, function(x) colMeans(x$resample)))

rowMeans(sapply(cout, function(x) cov(x[,1:2])))
rowMeans(sapply(trimodal, function(x) cov(x$resample)))
