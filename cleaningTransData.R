#load libraries
library(data.table)
library(ggplot2)
library(plyr)
library(dplyr)
library(corrplot)
library(pROC)
library(glmnet)
library(caret)
library(Rtsne)
library(xgboost)
library(doMC)
# Load data and replace empty observation with na 
df <- read.csv("/users/jialichen/downloads/creditcard.csv", na.strings="")

#view dataframe
head(df,3)

#summary of dataframe
str(df)
summary(df)


#change Class from interger to factor 
df$Class <- as.factor(df$Class)

#Let’s see whether there is any missing data
apply(df, 2, function(x) sum(is.na(x)))
#Good! There are no NA values in the data


##graphing
df.true <- df[df$Class == 0, ]
df.false <- df[df$Class == 1, ]
#overlay two plots on the same graph 
ggplot()+
  geom_density(data=df.true,
               aes(x=Time), color="black",
               fill="blue", alpha=0.12) +
  geom_density(data=df.false,
               aes(x=Time), color="red",
               fill="red", alpha=0.12) 

#compare different class 
common_theme <- theme(plot.title = element_text(hjust = 0.5, face = "bold"))
p1 <- ggplot(df, aes(x = Class)) + geom_bar() + ggtitle("Number of Class Labels") + common_theme
print(p1)

#plot Transaction Amount by Class 
p <- ggplot(df, aes(x = Class, y = Amount)) + geom_boxplot() + ggtitle("Distribution of Transaction Amount by Class") + common_theme
print(p)

#compute the mean and median values for each class.
df %>% group_by(Class) %>% summarise(mean(Amount), median(Amount))

#correlation between variables and create a correlation plot
install.packages("corrplot")
df$Class <- as.numeric(df$Class)
corr_df<-cor(df)
corr_plot <- corrplot(corr_df, method = "circle", type = "upper",tl.col="black", tl.srt=45)

#normalize amount column
normalize <- function(x){
  return((x - mean(x, na.rm = TRUE))/sd(x, na.rm = TRUE))
}
df$Amount <- normalize(df$Amount)

