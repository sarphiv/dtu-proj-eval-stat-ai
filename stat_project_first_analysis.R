setwd("C:/Users/david/Desktop/02445 proj stat/proj/v2")
source("stat_project_preprocessing.R")



par(mfrow=c(1,1))
#length_reg <- lm(log(embedding-min(embedding)+1) ~ d*person*obstacle - d:person:obstacle, data_length)
length_reg <- lm(embedding ~ d*person*obstacle - d:person:obstacle, data_length)
anova(length_reg)
summary(length_reg)
# residual_as_function_of_curve_length
par(mfrow=c(1,1), oma = c(5,4,0,0) + 0.1, mar = c(0,0,2,2) + 0.1)
plot(length_reg, 1)
title(outer=TRUE, ylab="residual", xlab="estimated curve length")
# Q-Q of residuals 
par(mfrow=c(1,1), oma = c(5,4,0,0) + 0.1, mar = c(0,0,2,2) + 0.1)
plot(length_reg, 2)
# Hist of the curve lengths 
par(mfrow=c(1,1), oma = c(5,4,0,0) + 0.1, mar = c(0,0,1,2) + 0.1)
hist(data_length$embedding, main="Histogram of curve lengths")
title(outer=TRUE, xlab="curve length")
# Hist of residuals 
x <- seq(-20, 40, by=0.02)
hist(length_reg$residuals, freq = FALSE, nclass = 40, main="Histogram of residuals")
curve(dnorm(x, 0, sd(length_reg$residuals)), add=TRUE)
title(outer=TRUE, xlab="residual size")


#log transform : 
length_reg_log <- lm(log(embedding) ~ d*person*obstacle- d:person:obstacle-d:obstacle, data_length)
anova(length_reg_log)
# residual_as_function_of_curve_length
par(mfrow=c(1,1), oma = c(5,4,0,0) + 0.1, mar = c(0,0,2,2) + 0.1)
plot(length_reg_log, 1)
title(outer=TRUE, ylab="residual", xlab="estimated curve length")
# Q-Q of residuals 
par(mfrow=c(1,1), oma = c(5,4,0,0) + 0.1, mar = c(0,0,2,2) + 0.1)
plot(length_reg_log, 2)
# Hist of the curve lengths 
par(mfrow=c(1,1), oma = c(5,4,0,0) + 0.1, mar = c(0,0,1,2) + 0.1)
hist(log(data_length$embedding), main="Histogram of curve lengths")
title(outer=TRUE, xlab="log curve length")
# Hist of residuals 
x <- seq(-20, 40, by=0.02)
hist(length_reg_log$residuals, freq = FALSE, nclass = 40, main="Histogram of residuals")
curve(dnorm(x, 0, sd(length_reg_log$residuals)), add=TRUE)
title(outer=TRUE, xlab="residual size")



# interaction plots
colors = c("red","blue", "darkgreen", "cyan4", "deeppink", "darkviolet", "darkorange", "black", "seagreen")

interaction.plot(data_length$person, data_length$d:data_length$obstacle, data_length$embedding,col=colors, ylab="tradjectory length", xlab="people", trace.label ="experiment")
par(mfrow=c(1,2), oma = c(5,4,0,0) + 0.1, mar = c(0,0,0,1) + 0.1)
interaction.plot(data_length$person, data_length$d, data_length$embedding,col=colors, ylab="tradjectory length", xlab="people", trace.label ="position", ylim=c(60, 112))
interaction.plot(data_length$person, data_length$obstacle, data_length$embedding,col=colors, ylab="", xlab="people", trace.label ="height",ylim=c(60, 112), yaxt='n')
title(outer=TRUE, xlab ="people", ylab="trajectory length")


#boxplots 
par(mfrow=c(1,2), oma = c(5,4,0,0) + 0.1, mar = c(0,0,0,0) + 0.1)
boxplot(embedding~person, data = data_length)
boxplot(embedding~d:obstacle, data = data_length, yaxt="n")
title(outer=TRUE, xlab="people                                                                                                                         obstacle position:height", ylab="curve length")

for (i in 1:dim(nans_indices)[1]){
  # lines3d(data[[nans_indices[i,1]]][[nans_indices[i,2]]][[nans_indices[i,3]]])
  print(head(data_interpolated[[nans_indices[i,1]]][[nans_indices[i,2]]][[nans_indices[i,3]]]))
}
