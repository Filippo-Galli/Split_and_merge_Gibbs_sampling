###########################################################################
#         SIMULATION STUDY HMM NEAL8 SPLIT&MERGE SPLIT&MERGE+NEAL8        #
###########################################################################
source('../code/complement_functions.R', echo=TRUE)
## SCENARIO 1
# p=15 n1=25 n2=25 n3=25 k=3 sigma=0.2
load("output_1.RData")
print("Adjusted Rand Index")
sim_1
print("clustering HMM")
table(sim_1_gibbs[[1]]$C[dim(sim_1_gibbs[[1]]$C)[2],])
print("clustering Neal8")
table(neal8[[1]]$final_ass)
print("clustering Split&Merge")
table(split_merge[[1]]$final_ass)
print("clustering Split&Merge+Neal8")
table(smn[[1]]$final_ass)
myplotpsm(psm_1_gibbs[[1]], estim_1_VI_gibbs)
myplotpsm(psm_1_neal8[[1]], estim_1_VI_neal8)
myplotpsm(psm_1_split_merge[[1]], estim_1_VI_split_merge)
myplotpsm(psm_1_smn[[1]], estim_1_VI_smn)

## SCENARIO 4
# p=10 n1=75 n2=75 n3=75 n4=75 k=4 sigma=0.7
load("output_4.RData")
print("Adjusted Rand Index")
sim_4
print("clustering HMM")
table(sim_4_gibbs[[1]]$C[dim(sim_4_gibbs[[1]]$C)[2],])
print("clustering Neal8")
table(neal8[[1]]$final_ass)
print("clustering Split&Merge")
table(split_merge[[1]]$final_ass)
print("clustering Split&Merge+Neal8")
table(smn[[1]]$final_ass)
myplotpsm(psm_4_gibbs[[1]], estim_4_VI_gibbs)
myplotpsm(psm_4_neal8[[1]], estim_4_VI_neal8)
myplotpsm(psm_4_split_merge[[1]], estim_4_VI_split_merge)
myplotpsm(psm_4_smn[[1]], estim_4_VI_smn)

## SCENARIO 5
# p=15 n1=10 n2=20 n3=30 k=3 sigma=0.5
load("output_5.RData")
print("Adjusted Rand Index")
sim_5
print("clustering HMM")
table(sim_5_gibbs[[1]]$C[dim(sim_5_gibbs[[1]]$C)[2],])
print("clustering Neal8")
table(neal8[[1]]$final_ass)
print("clustering Split&Merge")
table(split_merge[[1]]$final_ass)
print("clustering Split&Merge+Neal8")
table(smn[[1]]$final_ass)
myplotpsm(psm_5_gibbs[[1]], estim_5_VI_gibbs)
myplotpsm(psm_5_neal8[[1]], estim_5_VI_neal8)
myplotpsm(psm_5_split_merge[[1]], estim_5_VI_split_merge)
myplotpsm(psm_5_smn[[1]], estim_5_VI_smn)

## SCENARIO 6
# p=15 n1=5 n2=10 n3=20 n4=30 k=4 sigma=0.5
load("output_6.RData")
print("Adjusted Rand Index")
sim_6
print("clustering HMM")
table(sim_6_gibbs[[1]]$C[dim(sim_6_gibbs[[1]]$C)[2],])
print("clustering Neal8")
table(neal8[[1]]$final_ass)
print("clustering Split&Merge")
table(split_merge[[1]]$final_ass)
print("clustering Split&Merge+Neal8")
table(smn[[1]]$final_ass)
myplotpsm(psm_6_gibbs[[1]], estim_6_VI_gibbs)
myplotpsm(psm_6_neal8[[1]], estim_6_VI_neal8)
myplotpsm(psm_6_split_merge[[1]], estim_6_VI_split_merge)
myplotpsm(psm_6_smn[[1]], estim_6_VI_smn)

