setwd(dir = "C:/Users/Armel/Desktop/Cours_ESIEA/Github/Seconde partie/Modèles pour la Data Science")
data <- read.table("108292_ozone.txt",sep=" ",header=T)

data$vent <- as.factor(data$vent)
data$pluie <- as.factor(data$pluie)

# Données numériques du Dataframe
ozone <- data[,-13]
ozone <- ozone[,-12]

# Corrélation
ozone.cor = cor(ozone)
corrplot(ozone.cor, type="lower")

p.mat <- cor.mtest(ozone)
p.mat
corrplot(ozone.cor, p.mat = p.mat$p, insig = 'p-value', sig.level = -1)

# Visualisation
boxplot(ozone)

# Ecriture du modèle
modelcomp <- lm(ozone$maxO3~., data = ozone)
summary(modelcomp)
shapiro.test(residuals(modelcomp))

# Visualisation des 4 outliers 
boxplot(modelcomp$residuals)$out

# Test pour avoir la normalité // "type = 11" pour savoir si lowest/highest sont des outliers
grubbs.test(modelcomp$residuals, type=11)

## A FAIRE : étudie lmperm() et voir ce qu'elle fait // voir le probleme avec le "type=20" dans la fonction grubbs.test()
