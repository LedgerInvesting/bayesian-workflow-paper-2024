functions {
  int isin(int i, array[] int x) {
    for (n in x) {
      if (n == i) {
        return 1;
      }
    }
    return 0;
  }

  real mean_variance_gamma_lpdf(real obs, real obs_mean, real obs_variance) {
    real rate_param = obs_mean / obs_variance;
    real shape_param = obs_mean ^ 2 / obs_variance;
    return gamma_lpdf(obs | shape_param, rate_param);
  }

  real mean_variance_gamma_rng(real obs_mean, real obs_variance) {
    real rate_param = obs_mean / obs_variance;
    real shape_param = obs_mean ^ 2 / obs_variance;
    return gamma_rng(shape_param, rate_param);
  }
}