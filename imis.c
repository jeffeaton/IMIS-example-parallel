#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_statistics.h>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>

#include <gsl/gsl_heapsort.h>
#include <gsl/gsl_sort_double.h>

#include "imis.h"
#include "likelihood.h"

// Declare IMIS functions

struct dst { double v; size_t idx; };
int cmp_dst(const void * a, const void * b){ 
  struct dst aa = *(struct dst *) a, bb = *(struct dst *) b; 
  if(aa.v < bb.v){ return -1; } 
  if(aa.v > bb.v) return 1; 
  return 0;
};

void GetMahalanobis_diag(const gsl_matrix * mat, double * center, double * invCov, size_t nrow, size_t ncol, struct dst * retDistance);
void GenerateRandMVnorm(gsl_rng * r, const size_t numSamples, const double * mu, const gsl_matrix *sigChol, const size_t NumParam, gsl_matrix * returnSamples);
void GetMVNpdf(const gsl_matrix * mat, const double * mu, const gsl_matrix * simgaInv, const gsl_matrix * simgaChol, const size_t npoints, const size_t nDim, double returnVal[]);
void covariance_weighted(const gsl_matrix * mat, double * weights, size_t n, double * center, size_t numParam, gsl_matrix * retCov);
void weighted_random_sample(const gsl_rng * r, const size_t numSample, const double * probs, const size_t N, int * returnSample);
void walker_ProbSampleReplace(const gsl_rng * r, int n, double *p, int nans, int *ans);


void fnIMIS(const size_t InitSamples, const size_t StepSamples, const size_t FinalResamples, const size_t MaxIter, const size_t NumParam, unsigned long int rng_seed, const char * runName)
{

  // Declare and configure GSL RNG
  gsl_rng * rng;
  const gsl_rng_type * T;

  gsl_rng_env_setup();
  T = gsl_rng_default;
  rng = gsl_rng_alloc (T);
  gsl_rng_set(rng, rng_seed);

  char strDiagnosticsFile[strlen(runName) + 15 +1];
  char strResampleFile[strlen(runName) + 12 +1];
  strcpy(strDiagnosticsFile, runName); strcat(strDiagnosticsFile, "Diagnostics.txt");
  strcpy(strResampleFile, runName); strcat(strResampleFile, "Resample.txt");
  FILE * diagnostics_file = fopen(strDiagnosticsFile, "w");
  fprintf(diagnostics_file, "Seeded RNG: %zu\n", rng_seed);
  fprintf(diagnostics_file, "Running IMIS. InitSamples: %zu, StepSamples: %zu, FinalResamples %zu, MaxIter %zu\n", InitSamples, StepSamples, FinalResamples, MaxIter);

  // Setup IMIS arrays
  gsl_matrix * Xmat = gsl_matrix_alloc(InitSamples + StepSamples*MaxIter, NumParam);
  double * prior_all = (double*) malloc(sizeof(double) * (InitSamples + StepSamples*MaxIter));
  double * likelihood_all = (double*) malloc(sizeof(double) * (InitSamples + StepSamples*MaxIter));
  double * imp_weight_denom = (double*) malloc(sizeof(double) * (InitSamples + StepSamples*MaxIter));  // proportional to q(k) in stage 2c of Raftery & Bao
  double * gaussian_sum = (double*) calloc(InitSamples + StepSamples*MaxIter, sizeof(double));      // sum of mixture distribution for mode
  struct dst * distance = (struct dst *) malloc(sizeof(struct dst) * (InitSamples + StepSamples*MaxIter)); // Mahalanobis distance to most recent mode
  double * imp_weights = (double*) malloc(sizeof(double) * (InitSamples + StepSamples*MaxIter));
  double * tmp_MVNpdf = (double*) malloc(sizeof(double) * (InitSamples + StepSamples*MaxIter));

  gsl_matrix * nearestX = gsl_matrix_alloc(StepSamples, NumParam);
  double center_all[MaxIter][NumParam];
  gsl_matrix * sigmaChol_all[MaxIter];
  gsl_matrix * sigmaInv_all[MaxIter];

  // Initial prior samples
  sample_prior(rng, InitSamples, Xmat);

  // Calculate prior covariance
  double prior_invCov_diag[NumParam];
  /*
    The paper describing the algorithm uses the full prior covariance matrix
    For now, this follows Le's IMIS R code and diagonalizes the prior covariance
    matrix to ensure invertibility
  */
  for(size_t i = 0; i < NumParam; i++){
    gsl_vector_view tmpCol = gsl_matrix_subcolumn(Xmat, i, 0, InitSamples);
    prior_invCov_diag[i] = gsl_stats_variance(tmpCol.vector.data, tmpCol.vector.stride, InitSamples);
    prior_invCov_diag[i] = 1.0/prior_invCov_diag[i];
  }

  // IMIS steps
  fprintf(diagnostics_file, "Step Var(w_i)  MargLik    Unique Max(w_i)     ESS     Time\n");
  time_t time1, time2;
  time(&time1);
  size_t imisStep = 0, numImisSamples; // declared outside so can use later
  for(imisStep = 0; imisStep < MaxIter; imisStep++){
    numImisSamples = (InitSamples + imisStep*StepSamples);
    
    // Evaluate prior and likelihood
    if(imisStep == 0){ // initial stage
      #pragma omp parallel for
      for(size_t i = 0; i < numImisSamples; i++){
        gsl_vector_const_view theta = gsl_matrix_const_row(Xmat, i);
        prior_all[i] = prior(&theta.vector);
        likelihood_all[i] = likelihood(&theta.vector);
      }
    } else {  // imisStep > 0
      #pragma omp parallel for
      for(size_t i = InitSamples + (imisStep-1)*StepSamples; i < numImisSamples; i++){
        gsl_vector_const_view theta = gsl_matrix_const_row(Xmat, i);
        prior_all[i] = prior(&theta.vector);
        likelihood_all[i] = likelihood(&theta.vector);
      }
    }

    // Determine importance weights, find current maximum, calculate monitoring criteria
    double sumWeights = 0.0;

    #pragma omp parallel for reduction(+: sumWeights)
    for(size_t i = 0; i < numImisSamples; i++){
      imp_weight_denom[i] = (InitSamples*prior_all[i] + StepSamples*gaussian_sum[i])/(InitSamples + StepSamples * imisStep);
      imp_weights[i] = likelihood_all[i]*prior_all[i]/imp_weight_denom[i];
      sumWeights += imp_weights[i];
    }

    double maxWeight = 0.0, varImpW = 0.0, entropy = 0.0, expectedUnique = 0.0, effSampSize = 0.0, margLik;
    size_t maxW_idx;
    #pragma omp parallel for reduction(+: varImpW, entropy, expectedUnique, effSampSize)
    for(size_t i = 0; i < numImisSamples; i++){
      imp_weights[i] /= sumWeights;
      varImpW += pow(numImisSamples * imp_weights[i] - 1.0, 2.0);
      entropy += imp_weights[i] * log(imp_weights[i]);
      expectedUnique += (1.0 - pow((1.0 - imp_weights[i]), FinalResamples));
      effSampSize += pow(imp_weights[i], 2.0);
    }

    for(size_t i = 0; i < numImisSamples; i++){
      if(imp_weights[i] > maxWeight){
        maxW_idx = i;
        maxWeight = imp_weights[i];
      }
    }
    for(size_t i = 0; i < NumParam; i++)
      center_all[imisStep][i] = gsl_matrix_get(Xmat, maxW_idx, i);

    varImpW /= numImisSamples;
    entropy = -entropy / log(numImisSamples);
    effSampSize = 1.0/effSampSize;
    margLik = log(sumWeights/numImisSamples);

    fprintf(diagnostics_file, "%4zu %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f\n", imisStep, varImpW, margLik, expectedUnique, maxWeight, effSampSize, difftime(time(&time2), time1));
    time1 = time2;

    // Check for convergence
    if(expectedUnique > FinalResamples*(1.0 - exp(-1.0))){
      fclose(diagnostics_file);
      break;
    }

    // Calculate Mahalanobis distance to current mode
    GetMahalanobis_diag(Xmat, center_all[imisStep],  prior_invCov_diag, numImisSamples, NumParam, distance);

    // Find StepSamples nearest points
    // (Note: this was a major bottleneck when InitSamples and StepResamples are large. qsort outperformed GSL sort options.)
    qsort(distance, numImisSamples, sizeof(struct dst), cmp_dst);

    #pragma omp parallel for
    for(size_t i = 0; i < StepSamples; i++){
      gsl_vector_const_view tmpX = gsl_matrix_const_row(Xmat, distance[i].idx);
      gsl_matrix_set_row(nearestX, i, &tmpX.vector);
    }

    // Calculate weighted covariance of nearestX

    // (a) Calculate weights for nearest points 1...StepSamples
    double weightsCov[StepSamples];
    #pragma omp parallel for
    for(size_t i = 0; i < StepSamples; i++){
      weightsCov[i] = 0.5*(imp_weights[distance[i].idx] + 1.0/numImisSamples); // cov_wt function will normalize the weights
    }

    // (b) Calculate weighted covariance
    sigmaChol_all[imisStep] = gsl_matrix_alloc(NumParam, NumParam);
    covariance_weighted(nearestX, weightsCov, StepSamples, center_all[imisStep], NumParam, sigmaChol_all[imisStep]);

    // (c) Do Cholesky decomposition and inverse of covariance matrix
    gsl_linalg_cholesky_decomp(sigmaChol_all[imisStep]);
    for(size_t j = 0; j < NumParam; j++) // Note: GSL outputs a symmetric matrix rather than lower tri, so have to set upper tri to zero
      for(size_t k = j+1; k < NumParam; k++)
        gsl_matrix_set(sigmaChol_all[imisStep], j, k, 0.0);

    sigmaInv_all[imisStep] = gsl_matrix_alloc(NumParam, NumParam);
    gsl_matrix_memcpy(sigmaInv_all[imisStep], sigmaChol_all[imisStep]);

    gsl_linalg_cholesky_invert(sigmaInv_all[imisStep]);

    // Sample new inputs
    gsl_matrix_view newSamples = gsl_matrix_submatrix(Xmat, numImisSamples, 0, StepSamples, NumParam);
    GenerateRandMVnorm(rng, StepSamples, center_all[imisStep], sigmaChol_all[imisStep], NumParam, &newSamples.matrix);

    // Evaluate sampling probability from mixture distribution
    // (a) For newly sampled points, sum over all previous centers
    for(size_t pastStep = 0; pastStep < imisStep; pastStep++){
      GetMVNpdf(&newSamples.matrix, center_all[pastStep], sigmaInv_all[pastStep], sigmaChol_all[pastStep], StepSamples, NumParam, tmp_MVNpdf);
      #pragma omp parallel for
      for(size_t i = 0; i < StepSamples; i++)
        gaussian_sum[numImisSamples + i] += tmp_MVNpdf[i];
    }
    // (b) For all points, add weight for most recent center
    gsl_matrix_const_view Xmat_curr = gsl_matrix_const_submatrix(Xmat, 0, 0, numImisSamples + StepSamples, NumParam);
    GetMVNpdf(&Xmat_curr.matrix, center_all[imisStep], sigmaInv_all[imisStep], sigmaChol_all[imisStep], numImisSamples + StepSamples, NumParam, tmp_MVNpdf);
    #pragma omp parallel for
    for(size_t i = 0; i < numImisSamples + StepSamples; i++)
      gaussian_sum[i] += tmp_MVNpdf[i];
  } // loop over imisStep

  //// FINISHED IMIS ROUTINE

  // Resample posterior outputs
  int resampleIdx[FinalResamples];
  // weighted_random_sample(rng, FinalResamples, imp_weights, numImisSamples, resampleIdx);
  walker_ProbSampleReplace(rng, numImisSamples, imp_weights, FinalResamples, resampleIdx); // Note: Random sampling routine used in R sample() function. 
  
  // Print results
  FILE * resample_file = fopen(strResampleFile, "w");
  for(size_t i = 0; i < FinalResamples; i++){
    for(size_t j = 0; j < NumParam; j++)
      fprintf(resample_file, "%e\t", gsl_matrix_get(Xmat, resampleIdx[i], j));
    gsl_vector_const_view theta = gsl_matrix_const_row(Xmat, resampleIdx[i]);
    fprintf(resample_file, "%e\t%e\t%e\n", prior(&theta.vector), likelihood(&theta.vector), imp_weights[resampleIdx[i]]);
  }
  fclose(resample_file);
  
  /*
  // This outputs some things to files for debugging
  FILE * Xmat_file = fopen("Xmat.txt", "w");
  for(size_t i = 0; i < numImisSamples; i++){
  for(size_t j = 0; j < NumParam; j++)
  fprintf(Xmat_file, "%f\t", gsl_matrix_get(Xmat, i, j));
  fprintf(Xmat_file, "%e\t%e\t%e\t%e\t%e\t\n", prior_all[i], likelihood_all[i], imp_weights[i], gaussian_sum[i], distance[i]);
  }
  fclose(Xmat_file);
  
  FILE * centers_file = fopen("centers.txt", "w");
  for(size_t i = 0; i < imisStep; i++){
  for(size_t j = 0; j < NumParam; j++)
  fprintf(centers_file, "%f\t", center_all[i][j]);
  fprintf(centers_file, "\n");
  }
  fclose(centers_file);

  FILE * sigmaInv_file = fopen("sigmaInv.txt", "w");
  for(size_t i = 0; i < imisStep; i++){
  for(size_t j = 0; j < NumParam; j++)
  for(size_t k = 0; k < NumParam; k++)
  fprintf(sigmaInv_file, "%f\t", gsl_matrix_get(sigmaInv_all[i], j, k));
  fprintf(sigmaInv_file, "\n");
  }
  fclose(sigmaInv_file);
  */

  // free memory allocated by IMIS
  for(size_t i = 0; i < imisStep; i++){
    gsl_matrix_free(sigmaChol_all[i]);
    gsl_matrix_free(sigmaInv_all[i]);
  }

  // release RNG
  gsl_rng_free(rng);
  gsl_matrix_free(Xmat);
  gsl_matrix_free(nearestX);

  free(prior_all);
  free(likelihood_all);
  free(imp_weight_denom);
  free(gaussian_sum);
  free(distance);
  free(imp_weights);
  free(tmp_MVNpdf);

  return;
}

void GenerateRandMVnorm(gsl_rng * r, const size_t numSamples, const double * mu, const gsl_matrix *sigChol, const size_t NumParam, gsl_matrix * returnSamples)
{

  // generate standard normal random samples
  for(size_t i = 0; i < numSamples; i++)
    for(size_t j = 0; j < NumParam; j++)
      gsl_matrix_set(returnSamples, i, j, gsl_ran_gaussian(r, 1.0));

  gsl_blas_dtrmm(CblasRight, CblasLower, CblasTrans, CblasNonUnit, 1.0, sigChol, returnSamples); // matrix multiplcation stdNormSamples %*% sigChol, for upper triangular matrix

  // add mu to each row
  gsl_vector_const_view vecMu = gsl_vector_const_view_array(mu, NumParam);
  #pragma omp parallel for
  for(size_t i = 0; i < numSamples; i++){
    gsl_vector_view tmpRow = gsl_matrix_row(returnSamples, i);
    gsl_vector_add(&tmpRow.vector, &vecMu.vector);
  }

  return;
}


void GetMahalanobis_diag(const gsl_matrix * mat, double * center, double * invCov, size_t nrow, size_t ncol, struct dst * retDistance)
// Squared Mahalanobis distance assuming covariance matrix is diagonal
{
  #pragma omp parallel for
  for(size_t i = 0; i < nrow; i++){
    retDistance[i].idx = i;
    retDistance[i].v = 0.0;
    for(size_t j = 0; j < ncol; j++){
      retDistance[i].v += pow(gsl_matrix_get(mat, i, j) - center[j], 2.0) * invCov[j];  // relies on invCov being diagonal, as in Le's code.
    }
  }
  return;
}

void covariance_weighted(const gsl_matrix * mat, double * weights, size_t nr, double * center, size_t nc, gsl_matrix * retCov)
{

  size_t i, j, k;
  double sumWeights = 0.0, sumWeights2 = 0.0, tmpCovSum;

  if(weights == NULL){
    for(i = 0; i < nr; i++)
      weights[i] = 1.0/nr;
    sumWeights2 = 1.0/nr;
  } else {
    #pragma omp parallel for reduction(+: sumWeights)
    for(i = 0; i < nr; i++)
      sumWeights += weights[i];
    #pragma omp parallel for reduction(+: sumWeights2)
    for(i = 0; i < nr; i++){
      weights[i] /= sumWeights;
      sumWeights2 += pow(weights[i], 2.0);
    }
  }

  if(center == NULL)
    for(j = 0; j < nc; j++){
      double tmpSum = 0.0;
      #pragma omp parallel for reduction(+: tmpSum)
      for(size_t i = 0; i < nr; i++){
        tmpSum += gsl_matrix_get(mat, i, j) * weights[i];
      }
      center[j] = tmpSum;
    }

  for(size_t j = 0; j < nc; j++)
    for(size_t k = j; k < nc; k++){
      double tmpCovSum = 0.0;
      #pragma omp parallel for reduction(+: tmpCovSum)
      for(size_t i = 0; i < nr; i++){
        tmpCovSum += weights[i] * (gsl_matrix_get(mat, i, j) - center[j]) * (gsl_matrix_get(mat, i, k) - center[k]);
      }
      tmpCovSum /= (1.0 - sumWeights2);
      gsl_matrix_set(retCov, j, k, tmpCovSum);
      gsl_matrix_set(retCov, k, j, tmpCovSum);
    }

  return;
}


void GetMVNpdf(const gsl_matrix * mat, const double * mu, const gsl_matrix * sigmaInv, const gsl_matrix * sigmaChol, const size_t nPoints, const size_t nDim, double * returnVal)
{

  double normConst = - log(2*M_PI)*nDim/2.0;
  for(size_t j = 0; j < nDim; j++)
    normConst -= log(gsl_matrix_get(sigmaChol, j, j));

  gsl_vector_const_view vecMu = gsl_vector_const_view_array(mu, nDim);

  gsl_vector * ones = gsl_vector_alloc(nDim);
  gsl_vector_set_all(ones, 1.0);

  #pragma omp parallel for
  for(size_t i = 0; i < nPoints; i++){
    gsl_vector * x1 = gsl_vector_alloc(nDim);  // Note: allocating and freeing these every loop is not ideal, OpenMP might have a better way to handle this
    gsl_vector * x2 = gsl_vector_alloc(nDim);
    gsl_matrix_get_row(x1, mat, i);
    gsl_vector_sub(x1, &vecMu.vector);
    gsl_blas_dsymv(CblasUpper, 1.0, sigmaInv, x1, 0.0, x2);
    gsl_blas_ddot(x1, x2, &returnVal[i]);
    returnVal[i] = exp(normConst - 0.5*returnVal[i]);
    gsl_vector_free(x1);
    gsl_vector_free(x2);
  }

  return;
}


int compare_doubles_decr(const double * a, const double * b)
{
  if(*a > *b)
    return -1;
  if(*a < *b)
    return 1;
  else
    return 0;
}

void weighted_random_sample(const gsl_rng * r, const size_t numSample, const double * probs, const size_t N, int * returnSample)
{

  size_t * perm  = (size_t *) malloc(sizeof(size_t) * N);

  gsl_heapsort_index(perm, probs, N, sizeof(double), (gsl_comparison_fn_t) compare_doubles_decr);

  double * cumProb = (double *) malloc(sizeof(double) * N);
  cumProb[0] = probs[perm[0]];
  for(size_t i = 1; i < N; i++)
    cumProb[i] = cumProb[i-1] + probs[perm[i]];

  size_t idx;
  double runif;

  for(size_t i = 0; i < numSample; i++){
    runif = gsl_rng_uniform(r);
    for(idx = 0; idx < N; idx++)
      if(runif < cumProb[idx])
        break;
    returnSample[i] = perm[idx];
  }

  free(cumProb);
  free(perm);
  return;
}


#define SMALL 10000
void walker_ProbSampleReplace(const gsl_rng * r, int n, double *p, int nans, int *ans)
// Code adapted from R source code file random.c
{

  int *a = (int *) malloc(n * sizeof(int));

  double *q, rU;
  int i, j, k;
  int *HL, *H, *L;

  /* Create the alias tables.
     The idea is that for HL[0] ... L-1 label the entries with q < 1
     and L ... H[n-1] label those >= 1.
     By rounding error we could have q[i] < 1. or > 1. for all entries.
  */
  if(n <= SMALL) {
    // R_CheckStack2(n *(sizeof(int) + sizeof(double)));
    /* might do this repeatedly, so speed matters */
    HL = (int *) alloca(n * sizeof(int));
    q = (double *) alloca(n * sizeof(double));
  } else {
    /* Slow enough anyway not to risk overflow */
    HL = (int *) calloc(n, sizeof(int));
    q = (double *) calloc(n, sizeof(double));
  }
  H = HL - 1; L = HL + n;
  for (i = 0; i < n; i++) {
    q[i] = p[i] * n;
    if (q[i] < 1.) *++H = i; else *--L = i;
  }
  if (H >= HL && L < HL + n) { /* So some q[i] are >= 1 and some < 1 */
    for (k = 0; k < n - 1; k++) {
      i = HL[k];
      j = *L;
      a[i] = j;
      q[j] += q[i] - 1;
      if (q[j] < 1.) L++;
      if(L >= HL + n) break; /* now all are >= 1 */
    }
  }
  for (i = 0; i < n; i++) q[i] += i;

  /* generate sample */
  for (i = 0; i < nans; i++) {
    rU = gsl_rng_uniform(r) * n;
    k = (int) rU;
    // ans[i] = (rU < q[k]) ? k+1 : a[k]+1;
    ans[i] = (rU < q[k]) ? k : a[k];  // R version (previously line) uses 1 based indexing, here use 0 based.
  }
  if(n > SMALL) {
    free(HL);
    free(q);
  }
  free(a);
}
