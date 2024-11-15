#################################################
#### HD-vector algorithm ########################
#################################################

# Hash Function for duplication detection
hash = function(x)
{
  calc = function(a,b)
    (a*1009+b) %% 1000000009
  Reduce(calc,x)
}

# Get the list of possible values for each attribute
attriList = function(mat)
{
  ans = apply(mat,2,function(x) sort(unique(x)))
  ans
}

# Extend the neighbour from the original data
extension = function(mat,attri_list)
{
  ans = list()
  l = 1
  for (i in 1:nrow(mat))
    for (j in 1:ncol(mat))
    {
      for (k in attri_list[[j]])
      {
        if (mat[i,j]!=k)
        {
          ans[[l]] = mat[i,]
          # ans[[l]][j] = 1-ans[[l]][j]
          ans[[l]][j] = k
          l = l+1
        }
      }
    }
  ans = do.call(rbind,ans)
  ans = rbind(mat,ans)
  # Calculate hashVal for each value
  hashVal = apply(ans,1,hash)
  # No further examination here
  ind = which(!duplicated(hashVal))
  ans = ans[ind,]
  ans
}

# Calculate Hamming Distance between a center and a list of samples
HammingDistance = function(S,mat)
{
  ans = S!=t(mat)
  ans = colSums(ans)
  ans
}

# Calculate the U(S) defined in the paper
UofS = function(S,mat)
{
  Dist = HammingDistance(S,mat)
  ans = tabulate(Dist,ncol(mat))
  ans = c(sum(Dist==0),ans)
  ans
}

# Uniform Distribution
UniformHD = function(attri_size,row_no)
{
  n = attri_size - 1
  p = length(n)
  uni_size = rep(1,p)
  w = rep(1,p)
  s = rep(1,p)
  for (i in 1:p)
  {
    for (j in i:p)
    {
      if (j == 1)
        w[1] = n[1]
      else
        w[j] = n[j]*s[j-1] + w[j-1]
    }
    uni_size[i] = w[p]
    s = w
    w = rep(0,p)
  }
  uni_distribution = c(1,uni_size)
  uni_distribution = uni_distribution * row_no / prod(attri_size)
  uni_distribution
}

# calculate the modified chisq statistic for each center candidate
choose.r.M = function(p,US,UHD)
{
  chi1 = 0
  chi2 = 0
  chi3 = 0
  
  cnn = which(US[-1]<UHD[-1])
  if (length(cnn)==0)
    cnn = 1
  else
    cnn = cnn[1]-1
  if (cnn==0)
    chi = 0
  else
  {
    if (cnn>1)
      chi3 = chi3 + sum((US[1:(cnn-1)]-UHD[1:(cnn-1)])^2/UHD[1:(cnn-1)])
    chi3 = chi3 + 
      (sum(US[1:(cnn-1)])-sum(UHD[1:(cnn-1)]))^2/sum(UHD[-(1:(cnn-1))])
    chi2 = chi2 + sum((US[1:cnn]-UHD[1:cnn])^2/UHD[1:cnn])
    chi1 = chi2
    chi2 = chi2 + (sum(US[1:cnn])-sum(UHD[1:cnn]))^2/sum(UHD[-(1:cnn)])
    chi1 = chi1 + (US[cnn+1]-UHD[cnn+1])^2/UHD[cnn+1]
    chi1 = chi1 + 
      (sum(US[1:(cnn+1)])-sum(UHD[1:(cnn+1)]))^2/sum(UHD[-(1:(cnn+1))])
    if (cnn==1)
      p3 = 0
    else
      p3 = dchisq(chi3,cnn-1)
    
    res = c(dchisq(chi2,cnn),dchisq(chi1,cnn+1),p3)
    ans = c(chi2,chi1,chi3)
    ind = which.max(res)
    chi = ans[ind]
  }
  return(chi)
}

# Calculate the radius according to the paper
Radius = function(C,mat)
{
  p = ncol(mat)
  UC = UofS(C,mat)
  ans = UC[2:p]<UC[1:(p-1)] & UC[2:p]<UC[3:(p+1)]
  if (sum(ans)==0)
    ans = 0
  else
    ans = which(ans)[1]
  ans
}

# Main function for this algorithm
CategorialCluster = function(mat)
{
  clusters = list()
  centers = list()
  sampleInd = 1:nrow(mat)
  numOfClusters = 1
  ori = mat
  
  attri_list = attriList(mat)
  Set = extension(mat,attri_list)
  p = ncol(mat)
  attri_size = sapply(attri_list,length)
  UHD = UniformHD(attri_size,nrow(mat))
  
  indicator = 1
  ending = FALSE
  while (!ending)
  {
    maxStat = 0
    C = NULL
    # Loop through every possible S
    for (ind in 1:nrow(Set)){
      S = Set[ind,]
      US = UofS(S,mat)
      tmp = choose.r.M(p,US,UHD)
      if (tmp[1]>maxStat){
        maxStat = tmp[1]
        C = S
      }
    }
    # If no significant left
    if (maxStat<qchisq(0.95,p)){
      ending = TRUE
    }else{
      # Calculate the radius, with some midification from the paper
      if (indicator==1){
        rad0 = Radius(C,mat)
        if(rad0 == 0){
          rad0 = 1
        }
        rad = rad0
      }else{
        UC = UofS(C,mat)
        if (UC[rad0]>=UC[rad0+1]){
          rad = rad0
        }else{
          if(rad0 != 1){
            if (UC[rad0-1]-UC[rad0] >= 2){rad = rad0-1}
            else{rad = rad0}
          }
          else{rad=rad0}
          }
      }
      
      dis = HammingDistance(C,mat)
      ind = which(dis<=rad)
      if (length(ind)>0){
        clusters[[numOfClusters]] = sampleInd[ind]
        centers[[numOfClusters]] = C
        numOfClusters = numOfClusters+1
        mat = mat[-ind,,drop=FALSE]
        sampleInd = sampleInd[-ind]
        if (nrow(mat)==0)
          ending = TRUE
      }else{
        ending = TRUE
        }
      
      # Delete the neighbours in the radius as well
      dis = HammingDistance(C,Set)
      ind = which(dis<=rad)
      if (length(ind)>0){
        Set = Set[-ind,,drop=FALSE]
        }
    }
    indicator = indicator+1
  }
  Dist = list()
  for (i in 1:length(centers))
    Dist[[i]] = HammingDistance(centers[[i]],ori)
  Dist = do.call(cbind,Dist)
  # Find the nearest center
  clusters = max.col(-Dist)
  list(clusters,centers)
  }

#####################################################
######### other functions ###########################
#####################################################
gini=function(x){
  1-sum((table(x)/length(x))^2)
}

sigma = function(g,m){ 
  a = 1/log((-g*m + sqrt(-g*m^2 + g*m + m^2 - 2*m + 1) + g + m - 1)/g)
  b = 1/log(-(g*m + sqrt(-g*m^2 + g*m + m^2 - 2*m + 1) - g - m + 1)/g)
  {if(a!="NaN")return(a)
    else return(b)}
}

moda = function(x){
  tx = table(x)
  return(as.numeric(dimnames(tx)$x[which.max(tx)]))
}

### function my image
heatmap = function (x, Rowv = NULL, Colv = if (symm) "Rowv" else NULL, 
                    distfun = dist, ax = F, ay = F, hclustfun = hclust, reorderfun = function(d, 
                                                                              w) reorder(d, w), add.expr, symm = FALSE, revC = identical(Colv, 
                                                                                                                                         "Rowv"), scale = c("row", "column", 
                                                                                                                                                            "none"), na.rm = TRUE, margins = c(5, 5), ColSideColors, 
                    RowSideColors, cexRow = 0.2 + 1/log10(nr), cexCol = 0.2 + 
                      1/log10(nc), labRow = NULL, labCol = NULL, main = NULL, 
                    xlab = NULL, ylab = NULL, keep.dendro = FALSE, verbose = getOption("verbose"), 
                    ...) 
{
  scale <- if (symm && missing(scale)) 
    "none"
  else match.arg(scale)
  if (length(di <- dim(x)) != 2 || !is.numeric(x)) 
    stop("'x' must be a numeric matrix")
  nr <- di[1L]
  nc <- di[2L]
  if (nr <= 1 || nc <= 1) 
    stop("'x' must have at least 2 rows and 2 columns")
  if (!is.numeric(margins) || length(margins) != 2L) 
    stop("'margins' must be a numeric vector of length 2")
  doRdend <- !identical(Rowv, NA)
  doCdend <- !identical(Colv, NA)
  if (!doRdend && identical(Colv, "Rowv")) 
    doCdend <- FALSE
  if (is.null(Rowv)) 
    Rowv <- rowMeans(x, na.rm = na.rm)
  if (is.null(Colv)) 
    Colv <- colMeans(x, na.rm = na.rm)
  if (doRdend) {
    if (inherits(Rowv, "dendrogram")) 
      ddr <- Rowv
    else {
      hcr <- hclustfun(distfun(x))
      ddr <- as.dendrogram(hcr)
      if (!is.logical(Rowv) || Rowv) 
        ddr <- reorderfun(ddr, Rowv)
    }
    if (nr != length(rowInd <- order.dendrogram(ddr))) 
      stop("row dendrogram ordering gave index of wrong length")
  }
  else rowInd <- 1L:nr
  if (doCdend) {
    if (inherits(Colv, "dendrogram")) 
      ddc <- Colv
    else if (identical(Colv, "Rowv")) {
      if (nr != nc) 
        stop("Colv = \"Rowv\" but nrow(x) != ncol(x)")
      ddc <- ddr
    }
    else {
      hcc <- hclustfun(distfun(if (symm) 
        x
        else t(x)))
      ddc <- as.dendrogram(hcc)
      if (!is.logical(Colv) || Colv) 
        ddc <- reorderfun(ddc, Colv)
    }
    if (nc != length(colInd <- order.dendrogram(ddc))) 
      stop("column dendrogram ordering gave index of wrong length")
  }
  else colInd <- 1L:nc
  x <- x[rowInd, colInd]
  labRow <- if (is.null(labRow)) 
    if (is.null(rownames(x))) 
      (1L:nr)[rowInd]
  else rownames(x)
  else labRow[rowInd]
  labCol <- if (is.null(labCol)) 
    if (is.null(colnames(x))) 
      (1L:nc)[colInd]
  else colnames(x)
  else labCol[colInd]
  if (scale == "row") {
    x <- sweep(x, 1L, rowMeans(x, na.rm = na.rm), check.margin = FALSE)
    sx <- apply(x, 1L, sd, na.rm = na.rm)
    x <- sweep(x, 1L, sx, "/", check.margin = FALSE)
  }
  else if (scale == "column") {
    x <- sweep(x, 2L, colMeans(x, na.rm = na.rm), check.margin = FALSE)
    sx <- apply(x, 2L, sd, na.rm = na.rm)
    x <- sweep(x, 2L, sx, "/", check.margin = FALSE)
  }
  lmat <- rbind(c(NA, 3), 2:1)
  lwid <- c(if (doRdend) 1 else 0.05, 4)
  lhei <- c((if (doCdend) 1 else 0.05) + if (!is.null(main)) 0.2 else 0, 
            4)
  if (!missing(ColSideColors)) {
    if (!is.character(ColSideColors) || length(ColSideColors) != 
        nc) 
      stop("'ColSideColors' must be a character vector of length ncol(x)")
    lmat <- rbind(lmat[1, ] + 1, c(NA, 1), lmat[2, ] + 1)
    lhei <- c(lhei[1L], 0.2, lhei[2L])
  }
  if (!missing(RowSideColors)) {
    if (!is.character(RowSideColors) || length(RowSideColors) != 
        nr) 
      stop("'RowSideColors' must be a character vector of length nrow(x)")
    lmat <- cbind(lmat[, 1] + 1, c(rep(NA, nrow(lmat) - 1), 
                                   1), lmat[, 2] + 1)
    lwid <- c(lwid[1L], 0.2, lwid[2L])
  }
  lmat[is.na(lmat)] <- 0
  if (verbose) {
    cat("layout: widths = ", lwid, ", heights = ", 
        lhei, "; lmat=\n")
    print(lmat)
  }
  dev.hold()
  on.exit(dev.flush())
  op <- par(no.readonly = TRUE)
  on.exit(par(op), add = TRUE)
  layout(lmat, widths = lwid, heights = lhei, respect = TRUE)
  if (!missing(RowSideColors)) {
    par(mar = c(margins[1L], 0, 0, 0.5))
    image(rbind(if (revC) 
      nr:1L
      else 1L:nr), col = RowSideColors[rowInd], axes = FALSE)
  }
  if (!missing(ColSideColors)) {
    par(mar = c(0.5, 0, 0, margins[2L]))
    image(cbind(1L:nc), col = ColSideColors[colInd], axes = FALSE)
  }
  par(mar = c(margins[1L], 0, 0, margins[2L]))
  if (!symm || scale != "none") 
    x <- t(x)
  if (revC) {
    iy <- nr:1
    if (doRdend) 
      ddr <- rev(ddr)
    x <- x[, iy]
  }
  else iy <- 1L:nr
  image(1L:nc, 1L:nr, x, xlim = 0.5 + c(0, nc), ylim = 0.5 + 
          c(0, nr), axes = FALSE, xlab = "", ylab = "", 
        ...)
  box()
  if(ax==T){
    axis(1, 1L:nc, labels = labCol, las = 2, line = -0.5, tick = 0,  cex.axis = cexCol)
    }
  if (!is.null(xlab)) 
    mtext(xlab, side = 1, line = margins[1L] - 1.25)
  if(ay==T){
    axis(4, iy, labels = labRow, las = 2, line = -0.5, tick = 0, cex.axis = cexRow)}
  if (!is.null(ylab)) 
    mtext(ylab, side = 4, line = margins[2L] - 1.25)
  if (!missing(add.expr)) 
    eval.parent(substitute(add.expr))
  par(mar = c(margins[1L], 0, 0, 0))
  if (doRdend) 
    plot(ddr, horiz = TRUE, axes = FALSE, yaxs = "i", 
         leaflab = "none")
  else frame()
  par(mar = c(0, 0, if (!is.null(main)) 1 else 0, margins[2L]))
  if (doCdend) 
    plot(ddc, axes = FALSE, xaxs = "i", leaflab = "none")
  else if (!is.null(main)) 
    frame()
  if (!is.null(main)) {
    par(xpd = NA)
    title(main, cex.main = 1.5 * op[["cex.main"]])
  }
  invisible(list(rowInd = rowInd, colInd = colInd, Rowv = if (keep.dendro && 
                                                              doRdend) ddr, Colv = if (keep.dendro && doCdend) ddc))
}


myplotpsm = function (psm, classes, ax = T, ay = T, cex.Row = 0.8, cex.Col = 0.8, method = "complete", ...) 
{
  library(fields)
  if (any(psm != t(psm)) | any(psm > 1) | any(psm < 0) | sum(diag(psm)) != 
      nrow(psm)) {
    stop("psm must be a symmetric matrix with entries between 0 and 1 and 1's on the diagonals")
  }
  n = nrow(psm)
  ord = order(classes)
  par(mar=c(0,0,0,0),cex.axis=0.8)
  heatmap(psm[ord,ord],ax=ax, ay=ay, Rowv=NA,Colv=NA,revC=TRUE,symm = TRUE,
          cexRow=cex.Row,cexCol=cex.Col,
          margins=c(8,3),
          col = sort(heat.colors(30),decreasing=T))
  image.plot(1:n, 1:n, psm[ord,ord], legend.only = TRUE,
             col = sort(heat.colors(30),decreasing=T),
             horizontal = TRUE, xlab = "", ylab = "",
             legend.cex=0.8,legend.mar=1,
             smallplot= c(0.02,0.9,0.1,0.15))
}

###### classification rate Zhang #####
CR_zhang = function(out,truth){
  tb = table(out,truth)
  rename = NULL
  for (i in unique(truth))
  {
    tmp = order(tb[,i],decreasing=TRUE)
    rename[i] = setdiff(tmp,rename)[1]
  }
  
  corr = sum(diag(tb[rename,]))
  return(corr/sum(tb))
}


info_gain <- function(true, cluster) {
  
  entropy <- function(p) {
    # Assuming entropy is calculated using the formula -sum(p * log2(p))
    return(-sum(p * log2(p), na.rm = TRUE))
  }
  
  sorttrue <- sort(true)
  
  # calculate the run length of the true
  rlen <- rle(sorttrue)$lengths
  norm <- length(true)
  fretrue <- rlen / norm
  
  # total entropy
  total <- entropy(fretrue)
  
  # calculate weighted subentropies
  weighted <- 0
  combin <- data.frame(cluster, true)
  sorted <- combin[order(combin$cluster),]
  truesort <- sorted$true
  sortclus <- rle(sorted$cluster)$lengths
  
  len <- diff(c(0, cumsum(sortclus)))
  addlen <- c(0, cumsum(len))
  
  ens <- numeric(length(len))
  for (i in 2:length(addlen)) {
    temclus <- truesort[(addlen[i - 1] + 1):addlen[i]]
    temsort <- sort(temclus)
    temrlen <- rle(temsort)$lengths
    temfreq <- temrlen / len[i - 1]
    ens[i - 1] <- entropy(temfreq)
  }
  
  weight <- len / norm
  weighted <- sum(weight * ens)
  
  gain <- total - weighted
  
  return(list(total = total, gain = gain))
}


