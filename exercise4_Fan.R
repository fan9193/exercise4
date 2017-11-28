library(rstan)


###the model
gaussianmodel="
functions{
  // Squared exponential kernel
real se_k(real x, real y, real var, real l_sq){
return var * exp(-((x-y)^2)/l_sq);
}

// Periodic variant of SE kernel
real periodic_se_k(real x, real y, real c, real var, real l_sq){
return var * exp(-2*(sin(pi()*(x-y)/c))^2 / l_sq);
}

// Linear kernel
real lin_k(real x, real y, real var){
return var*x*y;
}
}
data{
int<lower=1> P; // number of periods
int<lower=1> N; // number of customers
int<lower=1> M; // number of data points
int<lower=1> K; // max number of purchases

int<lower=1> t[M];    // calendar time
int<lower=1> l[M];    // lifetime
int<lower=1> r[M];    // interpurchase time (recency)    
int<lower=1> pnum[M]; // purchase number

int<lower=1> id[M]; // customer id of observation m

int<lower=0,upper=1> y[M]; // outcome of observation m (purchase or not)  
}
parameters{
// non-restricted components of the GP's
vector[P] alpha_long; 
vector[P-1] free_week;  
vector[P-1] free_short;     
vector[P-1] free_rec;   
vector[P-1] free_life;
vector[K-1] free_pnum;


real mu; // baseline spending level (cal mean function)


vector[N] delta;     // heterogeneity random effects
real<lower=0> sigsq; // variance of random effects


// hyperparameters of the GP's
real<lower=0> etasq_week; 
real<lower=0> rhosq_week;

real<lower=0> etasq_short;
real<lower=0> etasq_long;
positive_ordered[2] rhosq_cal; 

real<lower=0> etasq_rec;
real<lower=0> rhosq_rec;

real<lower=0> etasq_life; 
real<lower=0> rhosq_life;

real<lower=0> etasq_pnum; 
real<lower=0> rhosq_pnum;


real lambda_rec1;
real<lower=0> lambda_rec2;
real lambda_life1;
real<lower=0> lambda_life2;
real lambda_pnum1;
real<lower=0> lambda_pnum2;
}
transformed parameters{
// full GPs (with the first period restriction)
vector[P] alpha_week;   
vector[P] alpha_short;     
vector[P] alpha_rec;   
vector[P] alpha_life;
vector[K] alpha_pnum;
vector[P] week_mean; 
vector[P] short_mean;   
vector[P] long_mean;     
vector[P] rec_mean;   
vector[P] life_mean;
vector[K] pnum_mean;
vector[1] z;



// add zero first period restriction
z[1] <- 0;
alpha_week <- append_row(z,free_week);
alpha_short <- append_row(z,free_short);
alpha_rec <- append_row(z,free_rec);
alpha_life <- append_row(z,free_life);
alpha_pnum <- append_row(z,free_pnum);	 


// set mean functions
for(p in 1:P)
{
  rec_mean[p] <- lambda_rec1*(p-1)^lambda_rec2;
  life_mean[p] <- lambda_life1*(p-1)^lambda_life2;
  short_mean[p] <- 0;
  long_mean[p] <- mu;
  week_mean[p] <- 0;
}
for(k in 1:K)
{
  pnum_mean[k] <- lambda_pnum1*(k-1)^lambda_pnum2;
}
}
model{
vector[M] theta;          
matrix[P,P] Sigma_short;            
matrix[P,P] L_short;
matrix[P,P] Sigma_long;            
matrix[P,P] L_long;
matrix[P,P] Sigma_week;
matrix[P,P] L_week;
matrix[P,P] Sigma_rec;   
matrix[P,P] L_rec;
matrix[P,P] Sigma_life;
matrix[P,P] L_life;
matrix[K,K] Sigma_pnum;
matrix[K,K] L_pnum;


// calendar time, cyclic component kernel
for(i in 1:P)
{
  for(j in 1:P)
  {
  Sigma_week[i,j] <- etasq_week * exp(-2*(sin(pi()*(i-j)/7))^2/rhosq_week);
  Sigma_week[j,i] <- Sigma_week[i,j];
  }
  Sigma_week[i,i] <- Sigma_week[i,i]+0.0001;
}

L_week <- cholesky_decompose(Sigma_week); // square root of the given matrix


// calendar time, short-run component kernel
for(i in 1:P)
{
  for(j in 1:P)
  {
  Sigma_short[i,j] <- etasq_short * exp(-((i-j)^2)/rhosq_cal[1]);
  Sigma_short[j,i] <- Sigma_short[i,j];
  }
  Sigma_short[i,i] <- Sigma_short[i,i]+0.0001;
}

L_short <- cholesky_decompose(Sigma_short);


// calendar time, long-run component kernel
for(i in 1:P)
{
  for(j in 1:P)
  {
  Sigma_long[i,j] <- etasq_long * exp(-((i-j)^2)/rhosq_cal[2]);
  Sigma_long[j,i] <- Sigma_long[i,j];
  }
  Sigma_long[i,i] <- Sigma_long[i,i]+0.0001;
}

L_long <- cholesky_decompose(Sigma_long);


// recency kernel
for(i in 1:P)
{
  for(j in 1:P)
  {
  Sigma_rec[i,j] <- etasq_rec * exp(-((i-j)^2)/rhosq_rec);
  Sigma_rec[j,i] <- Sigma_rec[i,j];
  }
  Sigma_rec[i,i] <- Sigma_rec[i,i]+0.0001;
}

L_rec <- cholesky_decompose(Sigma_rec);


// lifetime kernel
for(i in 1:P)
{
  for(j in 1:P)
  {
  Sigma_life[i,j] <- etasq_life * exp(-((i-j)^2)/rhosq_life);
  Sigma_life[j,i] <- Sigma_life[i,j];
  }
  Sigma_life[i,i] <- Sigma_life[i,i]+0.0001;
}

L_life <- cholesky_decompose(Sigma_life);


// pnum kernel
for(i in 1:K)
{
  for(j in 1:K)
  {
  Sigma_pnum[i,j] <- etasq_pnum * exp(-((i-j)^2)/rhosq_pnum);
  Sigma_pnum[j,i] <- Sigma_pnum[i,j];
  }
  Sigma_pnum[i,i] <- Sigma_pnum[i,i]+0.0001;
}

L_pnum <- cholesky_decompose(Sigma_pnum);


// kernel hyperparameter priors
etasq_week ~ normal(0,5);
rhosq_week ~ normal(P/2,P);

etasq_short ~ normal(0,5);
etasq_long ~ normal(0,5);
rhosq_cal ~ normal(P/2,P);

etasq_rec ~ normal(0,5);
rhosq_rec ~ normal(P/2,P);

etasq_life ~ normal(0,5);
rhosq_life ~ normal(P/2,P);

etasq_pnum ~ normal(0,5);
rhosq_pnum ~ normal(20,10);


// kernel mean function priors
mu ~ normal(0,5);

lambda_rec1 ~ normal(0,5);
lambda_pnum1 ~ normal(0,5);
lambda_life1 ~ normal(0,5);
lambda_rec2 ~ normal(0,5);
lambda_pnum2 ~ normal(0,5);
lambda_life2 ~ normal(0,5);


// sample GPs
alpha_week ~ multi_normal_cholesky(week_mean,L_week);
alpha_short ~ multi_normal_cholesky(short_mean,L_short);
alpha_long ~ multi_normal_cholesky(long_mean,L_long);
alpha_rec ~ multi_normal_cholesky(rec_mean,L_rec);
alpha_life ~ multi_normal_cholesky(life_mean,L_life);
alpha_pnum ~ multi_normal_cholesky(pnum_mean,L_pnum);


// sample random effects
delta ~ normal(0,sigsq);
sigsq ~ normal(0,2.5);


// gppm likelihood
for(m in 1:M)
theta[m] <- alpha_week[t[m]]+alpha_long[t[m]]+alpha_short[t[m]]+alpha_life[l[m]]+alpha_rec[r[m]]+alpha_pnum[pnum[m]]+delta[id[m]];

y ~ bernoulli_logit(theta);
}
"
####generate data: calendar time, lifetime,interpurchase time (recency)  ,purchase number
    n = 1000; # of consumers
    w = 10; # of weeks
    d = 7*w; # of days
    
    #aquire date
    Date=seq.Date(as.Date('2016-01-01'),as.Date('2016-10-02'),by='day')
    Acquired=sample(Date,size=n,replace = T)
    Consumer=data.frame(ID=1:n,Date=Acquired)
    
    #Lifetime
    Date=seq.Date(as.Date('2016-10-03'),length.out = d, by='day')
    Consumer=data.frame(ID=rep(Consumer$ID,times=rep(d,times=n)),
                               Acquired=rep(Consumer$Date,times=rep(d,times=n)),
                               Calender=rep(Date,times=n))
    Consumer$lifetime=as.numeric(difftime(Consumer$Calender,Consumer$Acquired,units='days'))
    Consumer$WeekDay=rep(1:7,times=w*n)
    Consumer$Weekend=0
    Consumer$Weekend[Consumer$WeekDay %in% c(6,7)]=1
    
    #calender time
    Consumer$Calendertime=as.numeric(difftime(Consumer$Calender,as.Date('2016-10-03'),units='days') )+1
    
    #big calendar event: Thanksgiving
    Consumer$Promotion=0;
    Consumer$Promotion[Consumer$Calender%in%c(as.Date('2016-11-21'),as.Date('2016-11-22'),as.Date('2016-11-23'),
                                                            as.Date('2016-11-24'),as.Date('2016-11-25'),as.Date('2016-11-26'),as.Date('2016-11-27'))]=1
    #historical purchase number
    Consumer$PurchaseNum=0
    Consumer$PurchaseNum[Consumer$Calender%in%c(as.Date('2016-10-03'))]=rpois(n,2)+1
    
    #random effect
    Consumer$RmError=rep(rnorm(n)*0.5,times=rep(d,n))
    
    
    ## purchase decision, recency
    Consumer$Purchase=0
    Consumer$recency=0
    
    i=1;j=1
    for(i in 1:n)
      {
      for(j in 1:d)
        {
        Weekend=Consumer$Weekend[j+(i-1)*d]
        Promotion=Consumer$Promotion[j+(i-1)*d]
        
        PurchaseNum=Consumer$PurchaseNum[j+(i-1)*d]
        recency=Consumer$recency[j+(i-1)*d]
        history=Consumer$lifetime[j+(i-1)*d]
        
        RmError=Consumer$RmError[j+(i-1)*d]
        
        prob=exp(Weekend+2*Promotion+0.5*PurchaseNum+(-0.5)*recency+(-0.01)*history+RmError)/
          (1+exp(Weekend+2*Promotion+0.5*PurchaseNum+(-0.5)*recency+(-0.01)*history+RmError))  #purchase probability
        
        Consumer$Purchase[j+(i-1)*d]=(runif(1)<=prob)
        if(j<=d-1)
          {
          Consumer$PurchaseNum[j+(i-1)*d+1]=Consumer$PurchaseNum[j+(i-1)*d]+Consumer$Purchase[j+(i-1)*d]
          if(Consumer$Purchase[j+(i-1)*d]==1)
            {
            Consumer$recency[j+(i-1)*d+1]=1
            }else
            {
            Consumer$recency[j+(i-1)*d+1]=1+Consumer$recency[j+(i-1)*d]
            }
          }
        } 
      }

#data
data_list=list(
   P=max(Consumer$lifetime), # max lifetime
   N=max(Consumer$ID), # number of customers
   M=nrow(Consumer), # number of data points
   K=max(Consumer$PurchaseNum), # max number of purchases
  
   t=Consumer$Calendertime,  # calendar time
   l= Consumer$lifetime,  # lifetime
   r=Consumer$recency+1,   # interpurchase time (recency)    
   pnum=Consumer$PurchaseNum, # purchase number
  
   id=Consumer$ID, # customer id of observation m
  
   y=Consumer$Purchase # outcome of observation m (purchase or not)  
)

correct_fit= stan(model_code = gaussianmodel, data = data_list, chains = 2, iter = 10) ## too slow!!

alpha = print(correct_fit,pars=c('alpha_long','alpha_short','alpha_week','alpha_rec','alpha_rec','alpha_life','alpha_pnum'))
para = print(correct_fit,pars=c('etasq_long','etasq_short','rhosq_cal','etasq_week','rhosq_week','etasq_rec','rhosq_rec','etasq_life','rhosq_life','etasq_pnum','rhosq_pnum','lambda_rec1','lambda_rec2','lambda_life1','lambda_life2','lambda_pnum1','lambda_pnum2','mu','sigsq'))
