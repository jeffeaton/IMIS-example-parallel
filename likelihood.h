double prior(const gsl_vector * theta);
double likelihood(const gsl_vector * theta);
void sample_prior(gsl_rng * r, size_t numSamples, gsl_matrix * storeSamples);
