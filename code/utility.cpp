#include "utility.hpp"

bool debug_var = false;
 
double Utility::norm_const(const double d ,const double c, const double m) const{
  
  double z= (m-1)/m;
  double alpha= d+c;
  double beta=1;
  double gamma= d+2;
  
  gsl_sf_result out;
  
  gsl_set_error_handler_off();
  
  int stat= gsl_sf_hyperg_2F1_e(alpha, beta, gamma, z, & out);
  
  if (stat != GSL_SUCCESS)
  {
    //Rcpp::Rcout<<"Sono qui\n";
    return R_NaN;
  }
  else{
    return pow(std::pow(m,-d-c)/(d+1)*out.val,-1.0);
  }
} 

 
double Utility::hyperg(double a, double b, double c, double x) const {
    gsl_sf_result result;
    gsl_set_error_handler_off();
    int stat = gsl_sf_hyperg_2F1_e(a, b, c, x, &result);
    if (stat != GSL_SUCCESS)
    {
      Rcpp::Rcout<<"hypergeometric non converging\n";
      return R_NaN;
    }
    else
      return result.val;
  }

// u is the argument of the density
// c and d the two real parameter of the hyper distr
// m also is a parameter but it is "fixed" by the data (m  greather than 2)
 
Rcpp::NumericVector Utility::dhyper_raf(const Rcpp::NumericVector u,const double d ,const double c, const double m, const bool log_scale=false) const{
  
  int n= u.length();
  Rcpp::NumericVector out(n);
  
  double K = norm_const(d ,c,m); 
  
  
  for(int i=0;i<n;i++){
    out[i]=log(K)+d*log(u[i])-(d+c)*log(1+u[i]*(m-1));
  }

  if(log_scale){
    return(out);
  }
  else{
    return(Rcpp::exp(out));
  }
}

///Newtown method
 
double Utility::newton_hyper(const double d,const double c,const double m,const double Omega,const  double u0=0.5) const{
    
    double hu=1;
    double u_current=u0;
    double x;
    double dens;
    Rcpp::NumericVector app(1);
    int contatore=0;
    while(std::abs(hu)>0.00001){
      x=u_current*(m-1)/(1+u_current*(m-1));
      Rcpp::Rcout<<"u_current="<<u_current<<" x="<<x<<"\n";
      app[0]=u_current;
      dens= dhyper_raf(app, d, c,m,true)[0];
      hu=-u_current/(d+1)*hyperg(1, d+c, d+2,x )+exp(log(Omega)-dens);
      Rcpp::Rcout<<"hu="<<hu<<" Omega/dens"<<exp(log(Omega)-dens)<<"\n";
      u_current += hu;
      if(u_current<0){u_current=0.01;}
      if(u_current>1){u_current=0.99;}
      contatore +=1;
      if(contatore>100){return R_NaN;}
    }
    
    
    return(u_current);
  }


double Utility::lF_conK(const double u, const double d,const double c,const double m,const double K) const{
  if(u==0){return 0;}
  if(u==1){return 1;}
  double x=u*(m-1)/(1+u*(m-1));
  double out = log(K)-log(d+1)+(d+1)*log(u)-(d+c)*log(1+u*(m-1))+log(hyperg(1, d+c, d+2,x ));
  return out;
}


///Newtown method
 
double Utility::bisec_hyper(const double d,const double c,const double m,const double Omega) const{
    
    
    double centro=0.5;
    double K = norm_const(d ,c,m); 
    double app=lF_conK(centro,d,c,m,K)-log(Omega);
    //Rcpp::Rcout<<"app="<<app<<"\n";
    
    double su;
    double giu;
    
    
    if(app<0){
      giu=0.5;
      su=1;
    }else{
      giu=0;
      su=0.5;
    }
    
    while( (su-giu)>0.0001){
      centro=(su+giu)/2;
      app=lF_conK(centro,d,c,m,K)-log(Omega);
      // Rcpp::Rcout<<"app="<<app<<"\n";
      
      if(app<0){
        giu=centro;
        su=su;
      }else{
        giu=giu;
        su=centro;
      }
      
    }
    
    
    return(centro);
  }

// n is an integer that represent the sample size
// c and d the two real parameter of the hyper distr
// m also is a parameter but it is "fixed" by the data (m>=2)
// u0 is the initial guess
 
Rcpp::NumericVector Utility::rhyper_raf(const int n,const double d ,const double c, const double m) const {
  
  Rcpp::NumericVector out(n);
  double Omega;
  // the output of a gsl special function
  for(int i=0;i<n;i++){
    Omega=R::runif(0,1);
    out[i]=bisec_hyper(d,c, m,Omega);
  }
  
  return(out);
}

// n is an integer that represent the sample size
// c and d the two real parameter of the hyper distr
// m also is a parameter but it is "fixed" by the data (m>=2)
 
Rcpp::NumericVector Utility::rhyper_sig(const int n,const double d ,const double c, const double m) const{
  
  Rcpp::NumericVector out(n);
  double Omega;
  // the output of a gsl special function
  for(int i=0;i<n;i++){
    Omega=R::runif(0,1);
    out[i]=bisec_hyper(d,c, m,Omega);
  }
  
  return(-1/Rcpp::log(out));
}

// Prior over sigma
// x is the argument of the density
// c and d the two real parameter of the hyper distr
// m also is a parameter but it is "fixed" by the data (m  greather than 2)
 
Rcpp::NumericVector Utility::dhyper_sig_raf(const Rcpp::NumericVector x,const double d ,const double c,
                                   const double m, const bool log_scale=false) const {
  
  int n= x.length();
  Rcpp::NumericVector out(n);
  
  double K = norm_const(d ,c,m); 
  
  
  for(int i=0;i<n;i++){
    out[i]=log(K)-(d+1)/x[i]-(d+c)*log(1+exp(-1/x[i])*(m-1))-2*log(x[i]);
  }
  if(log_scale){
    return(out);
  }
  else{
    return(Rcpp::exp(out));
  }
}

// Generates a list of size p for which each element is a vector of probabilities of size m_j
// - data matrix 
// - sigma is a VECTOR of size p for the scale parameter, each component of the vector is a sampled parameter for each variable
// - attrisize is a VECTOR of size p that contains the modalities for each variable
 
List Utility::Center_prob (const NumericMatrix& data, const NumericVector& sigma, const NumericVector& attrisize) const {
  
  int p = data.ncol();
  int n = data.nrow();
  
  List prob(p);
  for(int i = 0;i<p;i++){
    int m_j = attrisize[i];
    
    //Counting modalities 
    NumericVector val (m_j);
    NumericVector freq (m_j);
    
    for(int m = 0; m<m_j;m++){
      val[m]=m+1;
      freq[m] = std::count(data(_,i).begin(),data(_,i).end(),val[m]);
    }
    NumericVector prob_tmp(m_j);
    
    prob_tmp[val-1] = freq;
    
    for(int j = 0; j<m_j;j++){
      prob_tmp[j]=-(n-prob_tmp[j])/sigma[i];
    }
    prob_tmp = exp(prob_tmp-max(prob_tmp));                                                //Riga aggiunta il 14/09/21 per risolvere instabilità numerica
    prob_tmp = prob_tmp/sum(prob_tmp);
    prob[i]=prob_tmp;
  }
  return prob;
}

// Same as Center_prob, but sigma is a CONSTANT (double)a
 
List Utility::Center_prob_2 (NumericMatrix data, double sigma, NumericVector attrisize) const {
  
  int p = data.ncol();
  int n = data.nrow();
  
  List prob(p);
  for(int i = 0;i<p;i++){
    int m_j = attrisize[i];
    
    //Counting modalities 
    NumericVector val (m_j);
    NumericVector freq (m_j);
    
    for(int m = 0; m<m_j;m++){
      val[m]=m+1;
      freq[m] = std::count(data(_,i).begin(),data(_,i).end(),val[m]);
    }
    
    NumericVector prob_tmp(m_j);
    
    prob_tmp[val-1] = freq;
    
    for(int j = 0; j<m_j;j++){
      prob_tmp[j]=-(n-prob_tmp[j])/sigma;
    }
    prob_tmp = exp(prob_tmp-max(prob_tmp));                                                //Riga aggiunta il 14/09/21 per risolvere instabilità numerica
    prob_tmp = prob_tmp/sum(prob_tmp);
    prob[i]=prob_tmp;
  }
  return prob;
}

// Samples center values:
// - attriList: list of size p where each element is a vector of modalities for the jth variable
// - center_prob: output list of the function Center_prob (or Center_prob_2)
// - p: amount of variables of the data matrix
 
NumericVector Utility::Samp_Center(List attriList, List center_prob, int p) const{
  IntegerVector tmp;
  NumericVector samp(p);
  for (int i=0; i < p; i++){
    IntegerVector attributes = attriList[i];
    tmp = sample(attributes, 1, true, center_prob[i]);
    samp[i]=tmp[0];
  }
  return samp;
}

// Hamming density:
// - x: data sample of the jth variable
// - c: center value for the jth variable
// - s: sigma value for the jth variable
// - attrisize (m_j): number of modalities for the jth variable

 
double Utility::dhamming (int x, int c, double s, int attrisize, const bool log_scale=false) const {
  double out;
  double num;
  double den;
  
  if(log_scale){
    num = -(x!=c)/s;
    den=log(1+((attrisize-1)/exp(1/s)));
    out = num-den;
  }else{
    num = exp(-(x!=c)/s);

    den = 1+((attrisize-1)/exp(1/s));
    out = num/den;
  }
  return out;
}

// List of attributes per variable
// - data matrix
// - p is the number of variable
 
List Utility::Attributes_List(NumericMatrix data,int p) const {
  List attr(p);
  for(int j=0; j<p;j++){
    int m_j = max(data(_,j));
    IntegerVector elem = seq_len(m_j);
    attr[j]=elem;
  }
  return attr;
}

// List of attributes per variable
// - data matrix
// - p is the number of variable
 
List Utility::Attributes_List_manual(NumericMatrix data,int p) const {
  List attr(p);
  for(int j=0; j<p;j++){
    int m_j = 2;
    if(j == 12){
      m_j = 6;
    }
    IntegerVector elem = seq_len(m_j);
    attr[j]=elem;
  }
  return attr;
}

// Hamming distance between two equal length vectors
 
int Utility::hamming_distance(NumericVector a, NumericVector b) const {
  int out = sum(a!=b);
  return out;
}

// Allocation Matrix:
// for each observation, it computes the probability of being in one of the M component
// - cent: matrix of sampled center values
// - sigma: matrix of smapled sigma values
// - data: data matrix
// - M_curr: number of components
// - attrisize: vector of size p, each component is the number of modalities for each variable
// - Sm: matrix of sampled Sm values
 
NumericMatrix Utility::alloc_matrix( NumericMatrix cent, NumericMatrix sigma, NumericMatrix data, int M_curr, NumericVector attrisize, NumericVector Sm,const bool log_scale=false) const {
  
  int p = data.ncol();
  int n = data.nrow();
  
  double num;
  double den;
  double out_ham;
  
  NumericMatrix out(n,M_curr);
  
  for (int i = 0;i<n;i++){
    for (int j = 0; j<M_curr;j++){
      
      NumericVector c = cent(j,_);
      NumericVector s = sigma(j,_);
      NumericVector vec_dham_tmp(p);
      
      for( int e=0; e<p; e++){
        //Hamming density
        if(log_scale){
          num = (-(data(i,e)!=c[e]))*1/s[e];
          den = log(1+((attrisize[e]-1)/exp(1/s[e])));
          out_ham = num-den;
          vec_dham_tmp[e]=out_ham;
          
        }else{
          num = exp((-(data(i,e)!=c[e]))*1/s[e]);
          den = 1+((attrisize[e]-1)/exp(1/s[e]));
          out_ham = num/den ;
          vec_dham_tmp[e]=out_ham;
        }
      }
      if(log_scale){
        double summation = algorithm::sum(vec_dham_tmp.begin(),vec_dham_tmp.end());
        out(i,j)=log(Sm[j])+summation;
      }else{
        double product = algorithm::prod(vec_dham_tmp.begin(),vec_dham_tmp.end());
        out(i,j) = Sm[j]*product;
      }
    }
  }
  return out;
}


// Same as alloc_matrix, but sigma is a VECTOR of M components
 
NumericMatrix Utility::alloc_matrix2(NumericMatrix cent, NumericVector sigma, NumericMatrix data, int M_curr, NumericVector attrisize, NumericVector Sm,const bool log_scale=false) const {
  
  int p = data.ncol();
  int n = data.nrow();
  double num;
  double den; 
  double out_ham;
  NumericMatrix out(n,M_curr);
  
  for (int i = 0;i<n; i++){
    for (int j = 0; j<M_curr; j++){
      
      NumericVector c = cent(j,_);
      NumericVector vec_dham_tmp(p);
      double s = sigma[j];
      
      for( int e=0; e<p; e++){
        
        if(log_scale){
          num = -(data(i,e)!=c[e])/s;
          den = log(1+((attrisize[e]-1)/exp(1/s)));
          out_ham = num-den;
          vec_dham_tmp[e]=out_ham;
          
        }else{
          num = exp(-(data(i,e)!=c[e])/s);
          den = 1+((attrisize[e]-1)/exp(1/s));
          out_ham = num/den ;
          vec_dham_tmp[e]=out_ham;
        }
      }
      if(log_scale){
        double summation = algorithm::sum(vec_dham_tmp.begin(),vec_dham_tmp.end());
        out(i,j) =log(Sm[j])+summation;
      }else{
        double product = algorithm::prod(vec_dham_tmp.begin(),vec_dham_tmp.end());
        out(i,j) = Sm[j]*product;
      }
    }
  }
  return out;
}
