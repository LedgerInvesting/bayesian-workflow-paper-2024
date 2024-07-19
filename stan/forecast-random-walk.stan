#include functions.stan

data {
  int<lower=0> T;
  int<lower=0> T_train;
  int<lower=0> N;
  array[N] vector[T] y;
  array[N] vector[T_train] ulr_mean;
  array[N] vector<lower=0>[T_train] ulr_std;
  array[N] vector[T] premium;
  int<lower=0, upper=1> prior_only;
}

transformed data {
  matrix[N,T] y_ulr;
  array[N,T_train] int me;
  array[N,T_train] int me_idx;
  real mean_y_ulr;
  real sd_y_ulr;
  int T_me;
  int run_sum = 0;
  for(n in 1:N){
    y_ulr[n,] = to_row_vector(y[n] ./ premium[n]);
    for(t in 1:T_train) me[n,t] = ulr_std[n][t] > .01 ? 1 : 0; 
  }
  mean_y_ulr = mean(y_ulr[,1]);
  sd_y_ulr = sd(y_ulr[,1]);
  T_me = sum(to_array_1d(me));
  for(n in 1:N) {
    for(t in 1:T_train) me_idx[n,t] = run_sum + sum(me[n,1:t]);
    run_sum = me_idx[n,T_train];
  }
}

parameters {
  real eta_init_mu;
  real<lower=0> eta_init_sigma;
  vector[N] eta_init_star;
  
  real epsilon_mu; 
  real<lower=0> epsilon_sigma;
  vector[N] epsilon_star;
  
  array[N] vector[2] gamma;
  array[N] vector[T_train] eta_star;
  vector<lower=0>[T_me] me_ulr;
}

transformed parameters {
  array[N] vector[T_train] lp = rep_array(rep_vector(0.0, T_train), N);
  array[N] vector[T_train] eta;
  array[N] vector<lower=0>[T_train] ulr;
  vector[N] eta_init;
  vector<lower=0>[N] epsilon;

  for(n in 1:N){
    eta_init[n] = eta_init_mu + eta_init_sigma * eta_init_star[n];
    epsilon[n] = exp(epsilon_mu + epsilon_sigma * epsilon_star[n]);
    for(t in 1:T_train){
      real sigma2;
      if(me[n,t]){
        ulr[n][t] = me_ulr[me_idx[n,t]];
      }else{
        ulr[n][t] = ulr_mean[n,t];
      }
      if(t==1)
        eta[n][t] = eta_init[n] + epsilon[n] * eta_star[n][t];
      else
        eta[n][t] = eta[n][t-1] + epsilon[n] * eta_star[n][t];
      sigma2 = exp(gamma[n][1])^2 + exp(gamma[n][2])^2 / sqrt(premium[n][t]);
      lp[n][t] += lognormal_lpdf(ulr[n][t] | eta[n][t], sqrt(sigma2));
    }
  }
}

model {
  eta_init_mu ~ normal(-1, .5);
  eta_init_sigma ~ lognormal(-2, .5);
  epsilon_mu ~ normal(-2, .5);
  epsilon_sigma ~ lognormal(-2, .5);
  me_ulr ~ lognormal(
    log(mean_y_ulr^2 / sqrt(mean_y_ulr^2 + sd_y_ulr^2)), 
    sqrt(log(1 + sd_y_ulr^2 / mean_y_ulr^2))
  );
  for(n in 1:N){    
    eta_init_star[n] ~ std_normal();
    epsilon_star[n] ~ std_normal();
    eta_star[n] ~ std_normal(); 
    gamma[n] ~ normal(-2, 1);
    
    for(t in 1:T_train){
      if(me[n,t]){
        ulr_mean[n][t] ~ lognormal(
          log(ulr[n][t]^2 ./ sqrt(ulr[n][t]^2 + ulr_std[n][t]^2)), 
          sqrt(log(1 + ulr_std[n][t]^2 ./ ulr_mean[n][t]^2))
        );
      }
    }
    if(!prior_only) target += sum(lp[n]);
  }
}

generated quantities {
  array[N] vector<lower=0>[T_train] y_pred_train;
  array[N] vector<lower=0>[T-T_train] y_pred_test;
  array[N] vector[T_train] log_lik_train;
  array[N] vector[T-T_train] log_lik_test;

  for(n in 1:N){
    real eta_pred;
    real sigma2;
    for(t in 1:T){
      sigma2 = exp(gamma[n][1])^2 + exp(gamma[n][2])^2 / sqrt(premium[n][t]);
      if(t<=T_train){
        eta_pred = eta[n][t];
        y_pred_train[n][t] = lognormal_rng(eta_pred, sqrt(sigma2)) * premium[n][t];
        log_lik_train[n][t] = lognormal_lpdf(y_ulr[n,t] | eta_pred, sqrt(sigma2));
      }else{
        eta_pred = eta_pred + normal_rng(0, epsilon[n]);
        y_pred_test[n][t-T_train] = lognormal_rng(eta_pred, sqrt(sigma2)) * premium[n][t];
        log_lik_test[n][t-T_train] = lognormal_lpdf(y_ulr[n,t] | eta_pred, sqrt(sigma2));
      }
    }
  }
}