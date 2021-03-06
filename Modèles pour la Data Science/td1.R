library(ggplot2)
library(corrplot)
library(outliers)
library(lmPerm)

setwd(dir = "C:/Users/Armel/Desktop/Cours_ESIEA/Github/Seconde partie/Mod�les pour la Data Science")
data <- read.table("108292_ozone.txt",sep=" ",header=T)

data$vent <- as.factor(data$vent)
data$pluie <- as.factor(data$pluie)

# Donn�es num�riques du Dataframe
ozone <- data[,-13]
ozone <- ozone[,-12]

# Corr�lation
ozone.cor = cor(ozone)
corrplot(ozone.cor, type="lower")

p.mat <- cor.mtest(ozone)
p.mat
corrplot(ozone.cor, p.mat = p.mat$p, insig = 'p-value', sig.level = -1)

# Visualisation
boxplot(ozone)

# Ecriture du mod�le
modelcomp <- lm(ozone$maxO3~., data = ozone)
summary(modelcomp)
shapiro.test(residuals(modelcomp))

# Visualisation des 4 outliers 
boxplot(modelcomp$residuals, main = "residu sur jeu complet")$out

# Test pour avoir la normalit� // "type = 11" pour savoir si lowest/highest sont des outliers
grubbs.test(modelcomp$residuals, type=11)
# p_value inf�rieur � 0.05 donc test significatif

# On recr�e un dataframe en supprimant les outliers
complet_min_max <- ozone[!row.names(ozone) %in% c('20010707','20010731'),]
shapiro.test(residuals(lm(maxO3~., data=complet_min_max)))
