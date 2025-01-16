#include <Rcpp.h>
using namespace Rcpp;


// Generates a list of size p for which each element is a vector of probabilities of size m_j
// - data matrix 
// - sigma is a VECTOR of size p for the scale parameter, each component of the vector is a sampled parameter for each variable
// - attrisize is a VECTOR of size p that contains the modalities for each variable
// [[Rcpp::export]]
List Center_prob (NumericMatrix data, NumericVector sigma, NumericVector attrisize){
  
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
// [[Rcpp::export]]
List Center_prob_2 (NumericMatrix data, double sigma, NumericVector attrisize){
  
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
// [[Rcpp::export]]
NumericVector Samp_Center(List attriList, List center_prob, int p){
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

// [[Rcpp::export]]
double dhamming (int x, int c, double s, int attrisize, const bool log_scale=false){
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
// [[Rcpp::export]]
List Attributes_List(NumericMatrix data,int p){
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
// [[Rcpp::export]]
List Attributes_List_manual(NumericMatrix data,int p){
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
// [[Rcpp::export]]
int hamming_distance(NumericVector a, NumericVector b){
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
// [[Rcpp::export]]
NumericMatrix alloc_matrix( NumericMatrix cent, NumericMatrix sigma, NumericMatrix data, int M_curr, NumericVector attrisize, NumericVector Sm,const bool log_scale=false){
  
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
// [[Rcpp::export]]
NumericMatrix alloc_matrix2(NumericMatrix cent, NumericVector sigma, NumericMatrix data, int M_curr, NumericVector attrisize, NumericVector Sm,const bool log_scale=false){
  
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
