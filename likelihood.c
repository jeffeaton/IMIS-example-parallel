#include <math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

void sample_prior(gsl_rng * r, size_t numSamples, gsl_matrix * storeSamples)
{

  double x, y;
  for(size_t i = 0; i < numSamples; i++){
    gsl_ran_bivariate_gaussian(r, sqrt(3.0), sqrt(2.0), 0.0, &x, &y);
    gsl_matrix_set(storeSamples, i, 0, x);
    gsl_matrix_set(storeSamples, i, 1, y);
  }
  return;

}

double prior(const gsl_vector * theta)
{
  double x = gsl_vector_get(theta, 0);
  double y = gsl_vector_get(theta, 1);
  return gsl_ran_bivariate_gaussian_pdf(x - 0.0, y - 0.0, sqrt(3.0), sqrt(2.0),  0.0);
}

double likelihood(const gsl_vector * theta)
{

  double x = gsl_vector_get(theta, 0);
  double y = gsl_vector_get(theta, 1);

  double mu_x1 = -1.5;
  double mu_y1 = -1.5;
  double sigma_x1 = sqrt(0.3);
  double sigma_y1 = sqrt(0.3);
  double rho1 = 0.0;

  double mu_x2 = -2.0;
  double mu_y2 = 1.5;
  double sigma_x2 = sqrt(0.2);
  double sigma_y2 = sqrt(0.2);
  double rho2 = -0.15/(sigma_x2*sigma_y2);

  double mu_x3 = 1.5;
  double mu_y3 = -0.5;
  double sigma_x3 = sqrt(0.4);
  double sigma_y3 = sqrt(0.4);
  double rho3 = 0.2/(sigma_x3*sigma_y3);

  return gsl_ran_bivariate_gaussian_pdf(x - mu_x1, y - mu_y1, sigma_x1, sigma_y1,  rho1) +
    gsl_ran_bivariate_gaussian_pdf(x - mu_x2, y - mu_y2, sigma_x2, sigma_y2,  rho2) +
    gsl_ran_bivariate_gaussian_pdf(x - mu_x3, y - mu_y3, sigma_x3, sigma_y3,  rho3);

}
