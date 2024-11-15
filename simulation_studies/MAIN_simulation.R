#=========================================================================================
# SIMULATION SCENARIO 1
#=========================================================================================
source("Simulation_1.R") # or load("output_1.RData")

# Figure S5.1 example data
res.ca_1 = CA(data_list_1[[1]]$data,graph = F)
x11()
fviz_ca_row(res.ca_1,label = '',
            col.row = factor(data_list_1[[1]]$groundTruth),
            pointsize = 2,
            title ='Scenario 1',xlim=c(-0.45,0.45),ylim=c(-0.35,0.35))+
  scale_color_manual(name = 'Cluster', labels = c('1','2','3'),
                     values=c('red','blue','green'))


# Figure S6.1 boxplot
x11()
boxplot(sim_1,ylab = 'Adjusted Rand Index',main='Scenario 1',outline=FALSE,
        ylim = c(0,1),cex.axis=0.75,border=c(1,1,1,1, "darkred",
        col=c("grey90", "grey60","grey60", "grey60", 2)))


#=========================================================================================
# SIMULATION SCENARIO 2
#=========================================================================================
source("Simulation_2.R") # or load("output_2.RData")

# Figure S5.2 example data
res.ca_2 = CA(data_list_2[[1]]$data,graph = F)
fviz_ca_row(res.ca_2,label = '',
            col.row = factor(data_list_2[[1]]$groundTruth),
            pointsize = 2,xlim=c(-0.45,0.45),ylim=c(-0.45,0.45),
            title ='Scenario 2')+
  scale_color_manual(name = 'Cluster', labels = c('1','2','3','4'),
                     values=c('red','blue','green','violet'))


# Figure S6.2 boxplot
x11()
boxplot(sim_2,ylab = 'Adjusted Rand Index',main='Scenario 2',outline=FALSE,
        ylim = c(0,1),cex.axis=0.75,border = c(1,1,1,1,"darkred"),
        col=c("grey90", "grey60","grey60", "grey60", 2))


#=========================================================================================
# SIMULATION SCENARIO 3
#=========================================================================================
source("Simulation_3.R") # or load("output_3.RData")

# Figure S5.3 example data
res.ca_3 = CA(data_list_3[[2]]$data,graph = F)
fviz_ca_row(res.ca_3,label = '',
            col.row = factor(data_list_2[[2]]$groundTruth),
            pointsize = 2,xlim=c(-0.45,0.45),ylim=c(-0.45,0.45),
            title ='Scenario 3')+
  scale_color_manual(name = 'Cluster', labels = c('1','2','3','4'),
                     values=c('red','blue','green','violet'))


# Figure S6.3 boxplot
x11()
boxplot(sim_3,ylab = 'Adjusted Rand Index',main='Scenario 3',outline=FALSE,
        ylim = c(0,1),cex.axis=0.75,border=c(1,1,1,1,"darkred"),
        col=c("grey90", "grey60","grey60", "grey60", 2))


#=========================================================================================
# SIMULATION SCENARIO 4
#=========================================================================================
source("Simulation_4.R") # or load("output_4.RData")

# Figure S5.4 example data
res.ca_4 = CA(data_list_4[[1]]$data,graph = F)
x11()
fviz_ca_row(res.ca_4,label = '',
            col.row = factor(data_list_4[[1]]$groundTruth),
            pointsize = 2,xlim=c(-0.45,0.45),ylim=c(-0.35,0.35),
            title ='Scenario 4')+
  scale_color_manual(name = 'Cluster', labels = c('1','2','3','4'),
                     values=c('red','blue','green','violet'))



# Figure S6.4 boxplot
x11()
boxplot(sim_4,ylab = 'Adjusted Rand Index',main='Scenario 4',outline=FALSE,
        ylim = c(0,1),cex.axis=0.75,border=c(1,1,1,1,"darkred"),
        col=c("grey90", "grey60","grey60", "grey60", 2))


#=========================================================================================
# SIMULATION SCENARIO 5
#=========================================================================================
source("Simulation_5.R") # or load("output_5.RData")

# Figure S5.5 data example
res.ca_5 = CA(data_list_5[[8]]$data,graph = F)
fviz_ca_row(res.ca_5,label = '',
            col.row = factor(data_list_5[[2]]$groundTruth),
            pointsize = 2,xlim=c(-0.45,0.45),ylim=c(-0.45,0.45),
            title ='Scenario 5')+
  scale_color_manual(name = 'Cluster', labels = c('1','2','3'),values=c('red','blue','green'))


# Figure S6.5 boxplot
x11()
boxplot(sim_5,ylab = 'Adjusted Rand Index',main='Scenario 5',outline=FALSE,
        ylim = c(0,1),cex.axis=0.75,
        col=c("grey90", "grey60","grey60", "grey60", 2),border=c(1,1,1,1, "darkred"))


#=========================================================================================
# SIMULATION SCENARIO 6
#=========================================================================================
source("Simulation_6.R") # or load("output_6.RData")

# Figure S5.6 data example

res.ca_6 = CA(data_list_6[[22]]$data,graph = F)
fviz_ca_row(res.ca_6,label = '',
            col.row = factor(data_list_6[[2]]$groundTruth),
            pointsize = 2,xlim=c(-0.45,0.45),ylim=c(-0.45,0.45),
            title ='Scenario 6')+
  scale_color_manual(name = 'Cluster', labels = c('1','2','3','4'),
                     values=c('red','blue','green','violet'))



# Figure S6.6 boxplot
x11()
boxplot(sim_6,ylab = 'Adjusted Rand Index',main='Scenario 6',outline=FALSE,
        ylim = c(0,1),cex.axis=0.75,
        col=c("grey90", "grey60","grey60", "grey60", 2),border=c(1,1,1,1, "darkred"))


#=========================================================================================
# SIMULATION SCENARIO 7
#=========================================================================================
source("Simulation_7.R") # or load("output_7.RData")

# Figure S5.7 data example
res.ca_7 = CA(data_list_7[[2]]$data,graph = F)
fviz_ca_row(res.ca_7,label = '',
            col.row = factor(data_list_7[[2]]$groundTruth),
            pointsize = 2,xlim=c(-0.45,0.45),ylim=c(-0.45,0.45),
            title ='Scenario 7')+
  scale_color_manual(name = 'Cluster', labels = c('1','2','3','4'),
                     values=c('red','blue','green','violet'))


# Figure S6.7 boxplot
x11()
boxplot(sim_7,ylab = 'Adjusted Rand Index',main='Scenario 7',outline=FALSE,
        ylim = c(0,1),cex.axis=0.75,
        col=c("grey90", "grey60","grey60", "grey60", 2),border=c(1,1,1,1, "darkred"))


#=========================================================================================
# SIMULATION SCENARIO 8
#=========================================================================================
source("Simulation_8.R") # or load("output_8.RData")

# Figure S5.8 data example
res.ca_8 = CA(data_list_8[[2]]$data,graph = F)
x11()
fviz_ca_row(res.ca_8,label = '',
            col.row = factor(data_list_8[[2]]$groundTruth),
            pointsize = 2,xlim=c(-0.45,0.45),ylim=c(-0.45,0.45),
            title ='Scenario 8')+
  scale_color_manual(name = 'Cluster', labels = c('1','2','3','4'),
                     values=c('red','blue','green','violet'))


# Figure S6.8 boxplot
x11()
boxplot(sim_8,ylab = 'Adjusted Rand Index',main='Scenario 8',outline=FALSE,
        ylim = c(0,1),cex.axis=0.75,
        col=c("grey90", "grey60","grey60", "grey60", 2),border=c(1,1,1,1, "darkred"))



#=========================================================================================
# SIMULATION SCENARIO 9
#=========================================================================================
source("Simulation_9.R") # or load("output_9.RData")

# Figure S5.9 data example
res.ca_9 = CA(data_list_9[[2]]$data,graph = F)
x11()
fviz_ca_row(res.ca_9,label = '',
            col.row = factor(data_list_9[[2]]$groundTruth),
            pointsize = 2,xlim=c(-0.45,0.45),ylim=c(-0.45,0.45),
            title ='Scenario 9')+
  scale_color_manual(name = 'Cluster', labels = c('1','2','3','4'),
                     values=c('red','blue','green','violet'))

# Figure S6.9 boxplot
x11()
boxplot(sim_9,ylab = 'Adjusted Rand Index',main='Scenario 9',outline=FALSE,
        ylim = c(0,1),cex.axis=0.75,
        col=c("grey90", "grey60","grey60", "grey60", 2),border=c(1,1,1,1,"darkred"))



#=========================================================================================
# SIMULATION SCENARIO 4 MISSING VALUES
#=========================================================================================
rm(list=ls())

#### percentage of missing values (10% of total elements) ####
p.missing = 0.1 
source("Simulation_4NA.R") 
rand_VI_4_na10 = unlist(rand_VI_4)


# missing data imputation 
pred = matrix(NA,length(seed_set),nrow(missing_pos))
for(seed in seed_set){
  for (i in 1:nrow(missing_pos)){
    pred[seed,i] = which.max(table(sim_4[[seed]]$samp_data[,i]))
  }
}

# Figure S8a
x11()
plot(table(c(t(pred)),unlist(true_values)),main="")

#### percentage of missing values (20% of total elements) ####
p.missing = 0.2 
source("Simulation_4NA.R") 
rand_VI_4_na20 = unlist(rand_VI_4)


# missing data imputation 
pred = matrix(NA,length(seed_set),nrow(missing_pos))
for(seed in seed_set){
  for (i in 1:nrow(missing_pos)){
    pred[seed,i] = which.max(table(sim_4[[seed]]$samp_data[,i]))
  }
}


#Figure S8b
x11()
plot(table(c(t(pred)),unlist(true_values)),main="")



#### full dataset ####
source("Simulation_4.R")
rand_VI_4 = unlist(rand_VI_4)



# Figure S7a
sim_missing = cbind(rand_VI_4, rand_VI_4_na10,rand_VI_4_na20)
colnames(sim_missing) = c("0% NA","10% NA", "20%  NA")
x11()
boxplot(sim_missing,ylab = 'Adjusted Rand Index',main='Scenario 4',outline=FALSE,
        ylim=c(0,1),
        cex.axis=0.75,
        col=c( "#DF536B" ,"#EB97A6" , "#F8DCE1"),
        border=c("darkred","darkred","darkred"))



#=========================================================================================
# SIMULATION SCENARIO 5 MISSING VALUES
#=========================================================================================
rm(list=ls())

#### percentage of missing values (10% of total elements) ####
p.missing = 0.1 
source("Simulation_5NA.R") 
rand_VI_5_na10 = unlist(rand_VI_5)

# missing data imputation 
pred = matrix(NA,length(seed_set),nrow(missing_pos))
for(seed in seed_set){
  for (i in 1:nrow(missing_pos)){
    pred[seed,i] = which.max(table(sim_5[[seed]]$samp_data[,i]))
  }
}

# Figure S8c
x11()
plot(table(c(t(pred)),unlist(true_values)),main="")

#### percentage of missing values (20% of total elements) ####
p.missing = 0.2 
source("Simulation_5NA.R") 
rand_VI_5_na20 = unlist(rand_VI_5)


# missing data imputation 
pred = matrix(NA,length(seed_set),nrow(missing_pos))
for(seed in seed_set){
  for (i in 1:nrow(missing_pos)){
    pred[seed,i] = which.max(table(sim_5[[seed]]$samp_data[,i]))
  }
}


#Figure S8d
x11()
plot(table(c(t(pred)),unlist(true_values)),main="")



#### full dataset ####
source("Simulation_5.R")
rand_VI_5 = unlist(rand_VI_5)



# Figure S7b
sim_missing = cbind(rand_VI_5, rand_VI_5_na10,rand_VI_5_na20)
colnames(sim_missing) = c("0% NA","10% NA", "20%  NA")
x11()
boxplot(sim_missing,ylab = 'Adjusted Rand Index',main='Scenario 5',outline=FALSE,
        ylim=c(0,1),
        cex.axis=0.75,
        col=c( "#DF536B" ,"#EB97A6" , "#F8DCE1"),
        border=c("darkred","darkred","darkred"))



#=========================================================================================
# SIMULATION SCENARIO NETWORK-BASED
#=========================================================================================
rm(list=ls())

########### Sparsity level 100% ##################
q = 0 #density graph
source("Simulation_network.R")
Khat100=Khat


# Figure S10a
x11()
res.ca = CA(data_list[[44]]$Y,graph = F)
fviz_ca_row(res.ca,label = '',
            col.row = factor(data_list[[40]]$groundTruth),
            pointsize = 2,
            title ='100% sparsity',xlim=c(-0.45,0.45),ylim=c(-0.35,0.35))+
  scale_color_manual(name = 'Cluster', labels = c('1','2','3'),
                     values=c(6,'blue','green'))

# Figure S11a
x11()
boxplot(sim_graph,ylab = 'Adjusted Rand Index',main='Network-based dependence 100% sparsity',outline=FALSE,
        ylim = c(0,1),cex.axis=0.75,
        col=c("grey90", "grey60","grey60", "grey60", 2),border=c(1,1,1,1,"darkred"))


########### Sparsity level 85% ##################
q = 0.15 #density graph
source("Simulation_network.R")
Khat85=Khat

# Figure S9 top panels
x11()
net1 = network(data_list[[1]]$Graph[,,1], directed = FALSE)
ggnet2(net1,label=TRUE,size = 12, label.size = 3.5,color=6,
       label.color = 'white',edge.color = rgb(212,167,48,max=255))

x11()
net2 = network(data_list[[1]]$Graph[,,2], directed = FALSE)
ggnet2(net2,label=TRUE,size = 12, label.size = 3.5,color=4,
       label.color = 'white',edge.color = rgb(212,167,48,max=255))

x11()
net3 = network(data_list[[1]]$Graph[,,3], directed = FALSE)
ggnet2(net3,label=TRUE,size = 12, label.size = 3.5,color=3,
       label.color = 'white',edge.color = rgb(212,167,48,max=255))


# Figure S10b
x11()
res.ca = CA(data_list[[44]]$Y,graph = F)
fviz_ca_row(res.ca,label = '',
            col.row = factor(data_list[[40]]$groundTruth),
            pointsize = 2,
            title ='85% sparsity',xlim=c(-0.45,0.45),ylim=c(-0.35,0.35))+
  scale_color_manual(name = 'Cluster', labels = c('1','2','3'),
                     values=c(6,'blue','green'))

# Figure S11b
x11()
boxplot(sim_graph,ylab = 'Adjusted Rand Index',main='Network-based dependence 85% sparsity',outline=FALSE,
        ylim = c(0,1),cex.axis=0.75,
        col=c("grey90", "grey60","grey60", "grey60", 2),border=c(1,1,1,1,"darkred"))


########### Sparsity level 70% ##################
q = 0.30 #density graph
source("Simulation_network.R")
Khat70=Khat

# Figure S9 middle panels
x11()
net1 = network(data_list[[1]]$Graph[,,1], directed = FALSE)
ggnet2(net1,label=TRUE,size = 12, label.size = 3.5,color=6,
       label.color = 'white',edge.color = rgb(212,167,48,max=255))

x11()
net2 = network(data_list[[1]]$Graph[,,2], directed = FALSE)
ggnet2(net2,label=TRUE,size = 12, label.size = 3.5,color=4,
       label.color = 'white',edge.color = rgb(212,167,48,max=255))

x11()
net3 = network(data_list[[1]]$Graph[,,3], directed = FALSE)
ggnet2(net3,label=TRUE,size = 12, label.size = 3.5,color=3,
       label.color = 'white',edge.color = rgb(212,167,48,max=255))


# Figure S10c
x11()
res.ca = CA(data_list[[44]]$Y,graph = F)
fviz_ca_row(res.ca,label = '',
            col.row = factor(data_list[[40]]$groundTruth),
            pointsize = 2,
            title ='70% sparsity',xlim=c(-0.45,0.45),ylim=c(-0.35,0.35))+
  scale_color_manual(name = 'Cluster', labels = c('1','2','3'),
                     values=c(6,'blue','green'))

# Figure S11c
x11()
boxplot(sim_graph,ylab = 'Adjusted Rand Index',main='Network-based dependence 70% sparsity',outline=FALSE,
        ylim = c(0,1),cex.axis=0.75,
        col=c("grey90", "grey60","grey60", "grey60", 2),border=c(1,1,1,1,"darkred"))



########### Sparsity level 50% ##################
q = 0.350 #density graph
source("Simulation_network.R")
Khat50=Khat

# Figure S9 bottom panels
x11()
net1 = network(data_list[[1]]$Graph[,,1], directed = FALSE)
ggnet2(net1,label=TRUE,size = 12, label.size = 3.5,color=6,
       label.color = 'white',edge.color = rgb(212,167,48,max=255))

x11()
net2 = network(data_list[[1]]$Graph[,,2], directed = FALSE)
ggnet2(net2,label=TRUE,size = 12, label.size = 3.5,color=4,
       label.color = 'white',edge.color = rgb(212,167,48,max=255))

x11()
net3 = network(data_list[[1]]$Graph[,,3], directed = FALSE)
ggnet2(net3,label=TRUE,size = 12, label.size = 3.5,color=3,
       label.color = 'white',edge.color = rgb(212,167,48,max=255))


# Figure S10d
x11()
res.ca = CA(data_list[[44]]$Y,graph = F)
fviz_ca_row(res.ca,label = '',
            col.row = factor(data_list[[40]]$groundTruth),
            pointsize = 2,
            title ='50% sparsity',xlim=c(-0.45,0.45),ylim=c(-0.35,0.35))+
  scale_color_manual(name = 'Cluster', labels = c('1','2','3'),
                     values=c(6,'blue','green'))

# Figure S11d
x11()
boxplot(sim_graph,ylab = 'Adjusted Rand Index',main='Network-based dependence 50% sparsity',outline=FALSE,
        ylim = c(0,1),cex.axis=0.75,
        col=c("grey90", "grey60","grey60", "grey60", 2),border=c(1,1,1,1,"darkred"))


# Figure S12
Khat = cbind(Khat100,Khat85,Khat70,Khat50)
colnames(Khat) = c("100% sparsity", "85% sparsity", "70% sparsity", "50% sparsity")
x11()
par(mgp=c(2,1,0),mar=c(3.5,3.5,1,1))
boxplot(Khat, ylab=expression(hat(K)) )
