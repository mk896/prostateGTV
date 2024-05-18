library(survival)
library(prodlim)
library(riskRegression)
library(dplyr)
library(timeROC)

#################################################
# Competing risk curves
#################################################
d = read.csv('combClinMetrics.csv', header = T) # File. Each row represents one patient

# Scale d$uVolTest into units of cm^3
d$uVolTest = d$testvol/1000 # AI volume
d$uVolRef = d$refvol/1000 # Ref volume

# nVolTest uses <0.5 cm^3, 0.5-2 cm^3, and >2 cm^3
d$nVolTest = '0-0.4'
d$nVolTest[d$uVolTest>=0.5 & d$uVolTest <2.0] = '0.5-1.9'
d$nVolTest[d$uVolTest>=2.0] = '2.0-'

# Create risk variable
d$riskd = (d$risk4)
d$riskd = factor(d$riskd)
levels(d$riskd) <- c('Low', 'FIR', 'UIR', 'Hi')

## Plot bcf and met by nVolTest
ciVolbcf <- prodlim(Hist(tbcfss, bcf2) ~ nVolTest, dat = d)
ciVolmet <- prodlim(Hist(tmetss, met2) ~ nVolTest, dat = d)

#Fig. 1a
tiff('VolBCF.tif', units = "in", width = 6, height = 6, res = 300)
plot(ciVolbcf, confint = F, legend.x = 'topleft', legend.title = 'Volume (mL)', legend.cex = 1, atrisk.title = 'Volume', xlab = 'Time (years)', ylab = 'Risk of biochemical failure', atrisk.cex=1, xlim = c(0,10), axis1.at=seq(0,10,1),atrisk.at = seq(0,10,1), atrisk.labels = levels(factor(d$nVolTest)))
dev.off()

#Fig. 1b
tiff('VolMET.tif', units = "in", width = 6, height = 6, res = 300)
plot(ciVolmet, confint = F, legend.x = 'topleft', legend.title = 'Volume (mL)', legend.cex = 1, atrisk.title = 'Volume', xlab = 'Time (years)', ylab = 'Risk of metastasis', atrisk.cex=1, xlim = c(0,10), axis1.at=seq(0,10,1),atrisk.at = seq(0,10,1), atrisk.labels = levels(factor(d$nVolTest)))
dev.off()


#################################################
# Cox regression models
#################################################

# Met: Table 2.
#Univariable models
modmet1 <- coxph(Surv(tmetss,  met) ~(uVolTest) , dat = d, x = TRUE) 
modmet2 <- coxph(Surv(tmetss,  met) ~(Age) , dat = d)
modmet3 <- coxph(Surv(tmetss,  met) ~(ADTmo) , dat = d) 
modmet4 <- coxph(Surv(tmetss,  met) ~(model == 'DISCOVERY MR750w') , dat = d) 
modmet5 <- coxph(Surv(tmetss,  met) ~(risk4b), dat = d, x = TRUE) 
modmet6 <- coxph(Surv(tmetss,  met) ~(rT3) , dat = d, x = TRUE) 
modmet7 <- coxph(Surv(tmetss,  met) ~(PIRADSInd345) , dat = d) 
modmet10 <- coxph(Surv(tmetss,  met) ~(risk4), dat = d, x = TRUE) 
modmet11 <- coxph(Surv(tmetss,  met) ~(rTind), dat = d, x = TRUE) 

#Multi-variable models. Only variables on UVA with p <= 0.05 included in MVA. 
modmetmvaClin <-coxph(Surv(tmetss,  met) ~  uVolTest + ADTmo + (model == 'DISCOVERY MR750w') + risk4b, dat = d, x = TRUE)
modmetmvaClinPIRAD <-coxph(Surv(tmetss,  met) ~  uVolTest + ADTmo + (model == 'DISCOVERY MR750w') + risk4b + PIRADSInd345, dat = d)
modmetmvaClinRad <-coxph(Surv(tmetss,  met) ~  uVolTest + ADTmo + (model == 'DISCOVERY MR750w') + risk4b + rT3, dat = d, x= TRUE)


#################################################
# Time-dependent ROC analysis
#################################################

# Met: Table 4
rocNCCNmet2<-timeROC(T=d$tmetss,delta=d$met,marker=as.numeric(as.factor(d$risk4)),cause=1,weighting="marginal",  times=c(5,7), iid=TRUE)
rocVolmet2<-timeROC(T=d$tmetss,delta=d$met,marker=d$testvol,cause=1,weighting="marginal",  times=c(5,7), iid=TRUE)
rocRadmet2<-timeROC(T=d$tmetss,delta=d$met,marker=as.numeric(as.factor(d$rTind)),cause=1,weighting="marginal",  times=c(5,7), iid=TRUE)
rocVolRefmet2<-timeROC(T=d$tmetss,delta=d$met,marker=d$refvol,cause=1,weighting="marginal",  times=c(5,7), iid=TRUE)

compare(rocNCCNmet2, rocVolmet2, adjusted = TRUE)
compare(rocRadmet2, rocVolmet2, adjusted = TRUE)
compare(rocVolRefmet2, rocVolmet2, adjusted = TRUE)

SeSpPPVNPV(500, T=d$tmetss, delta=d$met, marker=d$testvol, other_markers = NULL, cause=1,
           weighting = "marginal", times=c(5,7), iid = T)
SeSpPPVNPV(2000, T=d$tmetss, delta=d$met, marker=d$testvol, other_markers = NULL, cause=1,
           weighting = "marginal", times=c(5,7), iid = T)

#################################################
# Lesion-level analysis
#################################################
dl = read.csv('combClinMetricsLesionsBxPIRADS.csv', header = T) # File. Each row represents one lesion (TP, FP, or FN). TP and FN lesions have a PI-RADS score assigned.

dl$Contrast100 = dl$Contrast*100 #Contrast multiplied by 100, in order to obtain interpretable odds ratio.

# Focus dataset only on lesions with PI-RADS scores assigned (TP or FN lesions)
dl2 = dl[!is.na(dl$L1PIRADS),]

#Table S8 # Logistic regression model for factors impacting risk of detecting FP lesion
summary(modDet <- glm((lesionTP != 'TP') ~ factor(L1PIRADS) + (L1PZ ==FALSE) + L1zpos + Contrast100 + (CohortID == 'Test') + model , dl2, family = 'binomial'))

#Table S9 # Linear regression model for factors impacting Dice coefficient
summary(modOverlap <- lm((Overlap )~ factor(L1PIRADS) + (L1PZ ==FALSE) + L1zpos + Contrast + (CohortID == 'Test') + model, dl2[dl2$lesionTP == 'TP',]))
