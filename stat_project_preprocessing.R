setwd("C:/Users/david/Desktop/02445 proj stat/proj/v2")
data <- get(load("armdata.RData"))

library(rgl)

#function that creates 3d cylinder plot (no kidding)
create_3d_cylender_plot <- function(){
  start_cyl <- cylinder3d(cbind(0, 0, seq(0, 10, length = 10)), radius = c(3,3,3), sides = 20, closed = -2)
  target_cyl <- cylinder3d(cbind(60, 0, seq(0, 10, length = 10)), radius = c(3,3,3), sides = 20, closed = -2)
  cyl1 <- cylinder3d(cbind(0, 0, 10 + seq(0, 12.5, length = 10)), radius = c(3,3,3), sides = 20, closed = -2)
  cyl2 <- cylinder3d(cbind(60, 0, 10 + seq(0, 12.5, length = 10)), radius = c(3,3,3), sides = 20, closed = -2)
  cyl3 <- cylinder3d(cbind(30, 0, seq(0, 20, length = 10)), radius = c(3,3,3), sides = 10, closed = -2)
  shade3d(addNormals(subdivision3d(start_cyl)), col = 'darkgreen')
  shade3d(addNormals(subdivision3d(target_cyl)), col = 'darkgreen')
  shade3d(addNormals(subdivision3d(cyl1)), col = 'pink')
  shade3d(addNormals(subdivision3d(cyl2)), col = 'pink', alpha = 0.5)
  shade3d(addNormals(subdivision3d(cyl3)), col = 'lightblue')
  surface3d(c(-7, 67), c(-20, 20), matrix(0, 2, 2), col = "brown", alpha = 0.9, specular = "black")
}

#get the indices of the naans
get_nans_indices <- function(data){
  nans_indices <- c()
  for (experiment in 1:16){
    for (person in 1:10){
      for (rep in 1:10){
        n_nans <- sum(is.na(data[[experiment]][[person]][[rep]]))
        if (n_nans > 0){
          nans_indices <- c(nans_indices, experiment,person,rep,n_nans)
          # print(c(experiment,person,rep,n_nans))
        }
      }
    }
  }
  nans_indices <- matrix(nans_indices, byrow = T, ncol = 4)
}

nans_indices <- get_nans_indices(data)
 
#general function for getting the data but in the format (experient, person, embedding)
# input1: the function that computes the embedding (should take in one trajectory as a 100x3 matrix)
# input2: data - well data
# output: -- you know --
get_embedding_table <- function(embed_function, data, col_names=c("embedding")){
  ncol <- length(col_names)
  output <- rep(NaN, 16*10*10*(4+ncol))
  i <- 1
  for (experiment in 1:16){
    for (person in 1:10){
      for (rep in 1:10){
        if (experiment != 16)
          {output[i:(i + 3 + ncol)] <- c(experiment, (experiment-1)%%3 + 1, (experiment - 1)%/%3 + 1,  person, embed_function(data[[experiment]][[person]][[rep]]))}
        else
          {output[i:(i + 3 + ncol)] <- c(16, 0, 0,  person, embed_function(data[[experiment]][[person]][[rep]]))}
        i <- i+ncol + 4
      }
    }
  }

  output <- as.data.frame(matrix(output, byrow = TRUE, ncol = 4+ncol))
  colnames(output) <- c("experiment", "obstacle", "d", "person", col_names)
  output$d <- as.factor(output$d)
  output$obstacle <- as.factor(output$obstacle)
  output$person <- as.factor(output$person)

  return (output)
}



get_first_point <- function(trajectory){ 
  return(trajectory[1, ])
}


#interpolate over missing values
interpolate_nans <- function(data){
  start_points <- na.omit(get_embedding_table(get_first_point, data, col_names = c("x", "y", "z")))
  
  
  for (i in 1:dim(nans_indices)[1]){
    ex <- nans_indices[i,1]
    per <- nans_indices[i,2]
    rep <- nans_indices[i,3]
    id <- nans_indices[i,4]/3 
    first_point <- data[[ex]][[per]][[rep]][id + 1,]

    start_points_exp_per <- start_points[start_points$experiment == ex & start_points$person == per,]
    
    meanx <- mean(start_points_exp_per$x)
    meany <- mean(start_points_exp_per$y)
    meanz <- mean(start_points_exp_per$z)
    
    defined_start_point <- c(meanx, meany, meanz)
    delta <- (first_point - defined_start_point)/id
    
    for (j in 1:id){
      data[[ex]][[per]][[rep]][j,] <- defined_start_point + delta*(j-1)
    }
  }
  return (data)
}

data_interpolated <- interpolate_nans(data)

#function: compute length of the trajectory (curve)
curve_length <- function(trajectory){
  length <- 0
  for (point in 1:99){
    length <- sqrt(sum((trajectory[(point+1),] - trajectory[point,])^2)) + length
  }
  return (length)
}

dif <- function(trajectory){
  return (max(trajectory[,1]) - min(trajectory[,1]))
}

data_length <- get_embedding_table(curve_length, data_interpolated)

#data_xdif <- get_embedding_table(dif, data_interpolated)

# bugged datapoint : 
#length_reg <- lm(log(embedding-min(embedding)+1) ~ d*person*obstacle - d:person:obstacle, data_length)
length_reg <- lm(embedding ~ d*person*obstacle - d:person:obstacle, data_length)
anova(length_reg)
# residual_as_function_of_curve_length
par(mfrow=c(1,1), oma = c(5,4,0,0) + 0.1, mar = c(0,0,2,2) + 0.1)
plot(length_reg, 1)
title(outer=TRUE, ylab="residual", xlab="curve length")
# Q-Q of residuals 
par(mfrow=c(1,1), oma = c(5,4,0,0) + 0.1, mar = c(0,0,2,2) + 0.1)
plot(length_reg, 2)
# Hist of the curve lengths 
par(mfrow=c(1,1), oma = c(5,4,0,0) + 0.1, mar = c(0,0,1,2) + 0.1)
hist(data_length$embedding, main="Histogram of curve lengths")
title(outer=TRUE, xlab="curve length")
# Hist of residuals 
x <- seq(-20, 40, by=0.02)
hist(length_reg$residuals, freq = FALSE, nclass = 40)
curve(dnorm(x, 0, sd(length_reg$residuals)), add=TRUE)

# There is one messed up datapoint that has probably been dropped
weird_point <- which.max(length_reg$residuals)
create_3d_cylender_plot()
lines3d(data[[weird_point%/%100 + 1]][[weird_point%/%10%%10 + 1]][[weird_point%%10]])
points3d(data[[weird_point%/%100 + 1]][[weird_point%/%10%%10 + 1]][[weird_point%%10]])

# Replacing this point with the mean of other points in this experiment and with this person
average_point <- function(data, point){
  experiment <- point %/% 100 + 1 
  person <- point %/% 10 %% 10 + 1 
  rep <- (point-1) %% 10 + 1
  
  cumsum <- data[[experiment]][[person]][[rep]]*0 
  
  for (i in 1:10){
    if (i != rep)
      cumsum <- cumsum + data[[experiment]][[person]][[i]]
  }
  
  data[[experiment]][[person]][[rep]] <- cumsum/9
  
  return (data)
}
cleaned_data <- average_point(data_interpolated, weird_point)

create_3d_cylender_plot()
lines3d(cleaned_data[[weird_point%/%100 + 1]][[weird_point%/%10%%10 + 1]][[weird_point%%10]])
points3d(cleaned_data[[weird_point%/%100 + 1]][[weird_point%/%10%%10 + 1]][[weird_point%%10]])


data_length <- get_embedding_table(curve_length, cleaned_data)


hist(data_length$embedding)

data_all <- get_embedding_table(unlist, cleaned_data, 1:300)

write.csv(data_all,"data_trajectories_2.csv")


# Just chekcing that there are no more ugly points ;)) - nah this is fine 
length_reg <- lm(embedding ~ d*person*obstacle - d:person:obstacle, data_length)
weird_point <- which.max(length_reg$residuals)
create_3d_cylender_plot()
lines3d(data[[weird_point%/%100 + 1]][[weird_point%/%10%%10 + 1]][[weird_point%%10]])
points3d(data[[weird_point%/%100 + 1]][[weird_point%/%10%%10 + 1]][[weird_point%%10]])
