#include functions.stan

data {
  int<lower=0> T;
  int<lower=0> T_prime;
  int<lower=0> N;
  int<lower=0> M;
  int<lower=1, upper=M> tau;
  array[2] int<lower=1, upper=M+1> rho;
  array[T + T_prime] int<lower=1> ii;
  array[T + T_prime] int<lower=1> jj;
  array[T + T_prime] int<lower=0> B;
  array[N] vector[T + T_prime] y;
  int<lower=0, upper=1> prior_only;
  vector[N] MAX_PRED;
}

parameters {
  array[N] vector[tau - 1] alpha_star;
  array[N] real<lower=0> omega_star;
  array[N] real beta_star;
  array[N] vector[2] gamma;
  array[N] vector[2] lambda;
}

transformed parameters {
  array[N] vector[T] lp = rep_array(rep_vector(0.0, T), N);
  array[N] vector<lower=0>[tau - 1] alpha;
  array[N] real<lower=0> omega;
  array[N] real<lower=0, upper=1> beta;

  for(n in 1:N){
    alpha[n] = exp(alpha_star[n]);
    omega[n] =  exp(omega_star[n]);
    beta[n] = inv_logit(beta_star[n]);

    for(t in 1:T){
      int lag = jj[t];
      int year = ii[t];
    
      if(lag > 1){
        real mu;
        real sigma2;
        if(lag <= tau){
          mu = alpha_star[n][lag - 1] + log(y[n][B[t]]);
          sigma2 = exp(gamma[n][1] + gamma[n][2] * lag + log(y[n][B[t]]));
          lp[n][t] += lognormal_lpdf(y[n][B[t] + 1] | mu, sqrt(sigma2));
        }
        if(lag >= rho[1] && lag <= rho[2]){
          mu = omega_star[n] * pow(beta[n], lag) + log(y[n][B[t]]);
          sigma2 = exp(lambda[n][1] + lambda[n][2] * lag + log(y[n][B[t]]));
          lp[n][t] += lognormal_lpdf(y[n][B[t] + 1] | mu, sqrt(sigma2));
        }
      }
    }
  }
}

model {
  for(n in 1:N){
    alpha_star[n] ~ normal(0, 1);
    omega_star[n] ~ normal(0, 1);
    beta_star[n] ~ normal(-2, .5);
    gamma[n][1] ~ normal(-3, .25);
    gamma[n][2] ~ normal(-1, .1);
    lambda[n][1] ~ normal(-3, .25);
    lambda[n][2] ~ normal(-1, .1);
    if(!prior_only) target += sum(lp[n]);
  }
}

generated quantities {
  array[N] vector<lower=0>[T + T_prime] y_tilde;
  array[N] vector[T + T_prime] log_lik;

  for(n in 1:N){
    for(t in 1:(T + T_prime)){
      real mu;
      real sigma2;
      int lag = jj[t];
      int year = ii[t];
      real lagged_y;

      if(lag > 1) {
        if(isin(B[t], B[1:T]))
          lagged_y = y[n][B[t]];
        else 
          lagged_y = y_tilde[n][B[t]];
      }
      if(lag <= tau && lag > 1){
        mu = log(lagged_y) + alpha_star[n][lag - 1];
        sigma2 = exp(gamma[n][1] + gamma[n][2] * lag + log(lagged_y));
      }
      else if(lag > tau) {
        mu = log(lagged_y) + omega_star[n] * pow(beta[n], lag);
        sigma2 = exp(lambda[n][1] + lambda[n][2] * lag + log(lagged_y));
      }
      if(lag == 1){
        y_tilde[n][B[t] + 1] = y[n][B[t] + 1];
        log_lik[n][B[t] + 1] = 0.0;
      } else {
        y_tilde[n][B[t] + 1] = min([MAX_PRED[n], lognormal_rng(mu, sqrt(sigma2))]);
        log_lik[n][B[t] + 1] = lognormal_lpdf(y[n][B[t] + 1] | mu, sqrt(sigma2));
      }
    }
  }
}
