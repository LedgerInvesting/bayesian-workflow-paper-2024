#include functions.stan

data {
  int<lower=0> T_train;
  int<lower=0> N;
  array[N] vector[T_train] y;
  array[N] vector[T_train] ulr_mean;
  array[N] vector<lower=0>[T_train] ulr_std;
  array[N] vector[T_train] premium;
  int<lower=0, upper=1> prior_only;
  int n_year_ahead;
  real eta_init__loc;
  real eta_init__scale;
  real epsilon__loc;
  real epsilon__scale;
  real mu__loc;
  real mu__scale;
  real phi__loc;
  real phi__scale;
  vector[2] gamma__loc; 
  vector[2] gamma__scale;
}

transformed data {
  array[N] vector[T_train] y_ulr;
  array[N,T_train] int me;
  array[N,T_train] int me_idx;
  real mean_y_ulr;
  real sd_y_ulr;
  int T_me;
  int run_sum = 0;
  for(n in 1:N){
    y_ulr[n] = y[n] ./ premium[n];
    for(t in 1:T_train) me[n,t] = ulr_std[n][t] > .01 ? 1 : 0; 
  }
  mean_y_ulr = mean(y_ulr[,1]);
  sd_y_ulr = N > 1 ? sd(y_ulr[,1]) : 1;
  T_me = sum(to_array_1d(me));
  for(n in 1:N) {
    for(t in 1:T_train) me_idx[n,t] = run_sum + sum(me[n,1:t]);
    run_sum = me_idx[n,T_train];
  }
}

parameters {
  vector[N] eta_init_star;
  vector[N] epsilon_star;
  vector[N] mu_star;
  vector[N] phi_star;
  
  array[N] vector[2] gamma;
  
  array[N] vector[T_train] eta_star;
  vector<lower=0>[T_me] me_ulr;
}

transformed parameters {
  array[N] vector[T_train] lp = rep_array(rep_vector(0.0, T_train), N);
  array[N] vector[T_train] eta;
  array[N] vector<lower=0>[T_train] ulr;
  vector[N] eta_init;
  vector[N] epsilon;
  vector[N] mu;
  vector[N] phi;

  for(n in 1:N){
    eta_init[n] = eta_init_star[n];
    epsilon[n] = exp(epsilon_star[n]);
    mu[n] = mu_star[n];
    phi[n] = inv_logit(phi_star[n]) * 2 - 1;
    for(t in 1:T_train){
      real sigma2;
      real ep = epsilon[n] * eta_star[n][t];
      if(me[n,t]){
        ulr[n][t] = me_ulr[me_idx[n,t]];
      }else{
        ulr[n][t] = ulr_mean[n,t];
      }
      if(t==1)
        eta[n][t] = (mu[n]*(1-phi[n]) + phi[n]*eta_init[n]) + ep;
      else
        eta[n][t] = (mu[n]*(1-phi[n]) + phi[n]*eta[n][t-1]) + ep;
      sigma2 = exp(gamma[n][1])^2 + exp(gamma[n][2])^2 / sqrt(premium[n][t]);
      lp[n][t] += lognormal_lpdf(ulr[n][t] | eta[n][t], sqrt(sigma2));
    }
  }
}

model {
  eta_init_star ~ normal(eta_init__loc, eta_init__scale);
  epsilon_star ~ normal(epsilon__loc, epsilon__scale);
  mu_star ~ normal(mu__loc, mu__scale);  
  phi_star ~ normal(phi__loc, phi__scale); 
  me_ulr ~ lognormal(
    log(mean_y_ulr^2 / sqrt(mean_y_ulr^2 + sd_y_ulr^2)), 
    sqrt(log(1 + sd_y_ulr^2 / mean_y_ulr^2))
  );
  for(n in 1:N){   
    eta_star[n] ~ std_normal(); 
    gamma[n] ~ normal(gamma__loc, gamma__scale);
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
  array[N] vector<lower=0>[T_train+n_year_ahead] y_pred;
  array[N] vector[T_train+n_year_ahead] log_lik;

  for(n in 1:N){
    real eta_pred;
    real sigma2;
    for(t in 1:(T_train+n_year_ahead)){
      if(t<=T_train){
        eta_pred = eta[n][t];
        sigma2 = exp(gamma[n][1])^2 + exp(gamma[n][2])^2 / sqrt(premium[n][t]);
        y_pred[n][t] = lognormal_rng(eta_pred, sqrt(sigma2)) * premium[n][t];
        log_lik[n][t] = lognormal_lpdf(y_ulr[n,t] | eta_pred, sqrt(sigma2));
      }else{
        eta_pred = mu[n]*(1-phi[n]) + eta_pred*phi[n] + normal_rng(0, epsilon[n]);
        sigma2 = exp(gamma[n][1])^2 + exp(gamma[n][2])^2 / sqrt(premium[n][T_train]);
        y_pred[n][t] = lognormal_rng(eta_pred, sqrt(sigma2)) * premium[n][T_train];
        log_lik[n][t] = 0; // No log likelihood for future predictions
      }
    }
  }
}