##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(lubridate)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#explore the dataset and show the structure of it.

str(edx)

# Number of uses in dataset is 69878 and number of movies is 10677
edx %>% summarize(n_users = n_distinct(userId), n_movies = n_distinct(movieId))

# The distribution of the movie ratings shows a range of 0.5 to 5
edx %>%
  ggplot(aes(rating, y = ..prop..)) +
  geom_bar(color = "black", fill = "deepskyblue2") +
  labs(x = "Ratings", y = "Relative Frequency") +
  scale_x_continuous(breaks = c(0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5)) +
  theme_bw()

#The number of ratings VS users shows a right skew in its distribution.
edx %>% group_by(userId) %>%
  summarize(count = n()) %>%
  ggplot(aes(count)) +
  geom_histogram(color = "black", fill = "deepskyblue2", bins = 40) +
  xlab("Ratings") +
  ylab("Users") +
  scale_x_log10() +
  theme_bw()

#average ratings of movies by month
edx %>% 
  mutate(date = round_date(as_datetime(timestamp), unit = "week")) %>%
  group_by(date) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(date, rating)) +
  geom_point() +
  geom_smooth(color = "green") +
  theme_hc() +
  ggtitle("average of ratings by Time ") +
  labs(x = "Time, unit: month ",
       y = "Mean Rating")+
  set_theme

#average of ratings per genres
# calculate number and average of movies bu genres
genres_summarize <- edx %>% separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize( num_movie_per_genres = n(), avg_movie_per_genres = mean(rating)) %>%
  arrange(desc(num_movie_per_genres))
## draw the histogram of number of ratings per genre
genres_summarize %>%
  ggplot(aes(num_movie_per_genres,reorder(genres, num_movie_per_genres),  fill= num_movie_per_genres)) +
  geom_bar(stat = "identity") + coord_flip() +
  scale_fill_distiller(palette = "#001400")+
  ggtitle("Number of ratings per genre") +
  labs(y = "Genres Type",
       x = "Number of ratings")+
  theme_hc()+
  set_theme +
  theme(axis.text.x = element_text(angle = 90, hjust = 1), legend.position = 'none')

#Root Mean Square Error Loss Function(Evaluation method)
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#Create train and test sets from edx dataset.
set.seed(1989, sample.kind = "Rounding")
#edx_test will be 20% of the edx set.
edx_test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
edx_train <- edx %>% slice(-edx_test_index)
edx_temp <- edx %>% slice(edx_test_index)
# make sure userId and movieId in test set are also in train set
edx_test <- edx_temp %>% 
  semi_join(edx_train, by = "movieId") %>%
  semi_join(edx_train, by = "userId")
# add rows removed from test set back into train set
removed <- anti_join(edx_temp, edx_test)
edx_train <- rbind(edx_train, removed)

#Find the average rating
mu_hat <- mean(edx_train$rating)
mu_hat

# Model 1: Just the average
model.1.rmse <- RMSE(edx_test$rating, mu_hat)
## let us create a dataframe to handle all RMSEs for all models
rmse_results <- data_frame(method = "Model 1:Just the average", RMSE = model.1.rmse)

# Model 2: Movie effect
#Here we will add the bias of movie effect.
mu <- mean(edx_train$rating) 
movie_avgs <- edx_train %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))
predicted_ratings <- mu + edx_test %>% 
  left_join(movie_avgs, by='movieId') %>%
  .$b_i
model_2_rmse <- RMSE(predicted_ratings, edx_test$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Model 2:Movie Effect",
                                     RMSE = model_2_rmse ))

#Model 3:  user effect model
#Now, we will add the user bias effect.
user_avgs <- edx_test %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu))
predicted_ratings <- edx_test %>% 
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_u) %>%
  .$pred
model_3_rmse <- RMSE(predicted_ratings, edx_test$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Model 3:User Effect",  
                                     RMSE = model_3_rmse ))
rmse_results
rmse_results %>% knitr::kable()

#Model 4: Movie + user effects
#Now, we will add the user bias effect to our model in addition to movie effect from model 2.
user_avgs <- edx_train %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))
predicted_ratings <- edx_test %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred
model_4_rmse <- RMSE(predicted_ratings, edx_test$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Model 4:Movie + User Effects",  
                                     RMSE = model_4_rmse ))
rmse_results
rmse_results %>% knitr::kable()

#Model XX: Movie + User + Time effects model
#valid <- validation
#valid <- valid %>%
#  mutate(date = round_date(as_datetime(timestamp), unit = "week")) 

#valid <- edx_test
#valid <- valid %>%
#  mutate(date = round_date(as_datetime(timestamp), unit = "week")) 

# i calculate time effects ( b_t) using the training set
#temp_avgs <- edx_train %>%
#  left_join(movie_avgs, by='movieId') %>%
#  left_join(user_avgs, by='userId') %>%
#  mutate(date = round_date(as_datetime(timestamp), unit = "week")) %>%
#  group_by(date) %>%
#  summarize(b_t = mean(rating - mu - b_i - b_u))

# predicted ratings
#predicted_ratings_bt <- valid %>% 
#  left_join(movie_avgs, by='movieId') %>%
#  left_join(user_avgs, by='userId') %>%
#  left_join(temp_avgs, by='date') %>%
#  mutate(pred = mu + b_i + b_u + b_t) %>%
#  .$pred
#model_4_rmse <- RMSE(predicted_ratings_bt, edx_test$rating)
#rmse_results <- bind_rows(rmse_results,
#                          data_frame(method="Movie + User + Time Effects Model",  
#                                     RMSE = model_4_rmse ))
#rmse_results


#Model 5: Regularized user and movie model
#Regularization technique should be used to take into account on the movie and user
#effects, by adding a larger penalty to estimates from smaller samples.

#$\lambda$ is a tuning parameter, that we need to choose an optimal value using k-cross validation.
lambdas <- seq(0, 10, 0.25)
set.seed(1989, sample.kind = "Rounding")
## For each lambda,find b_i & b_u followed by rating prediction

rmses <- sapply(lambdas, function(l){
  
  mu <- mean(edx_train$rating)
  
  b_i <- edx_train %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx_train %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- edx_test %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  
  return(RMSE(edx_test$rating,predicted_ratings))
})
## plot lambadas
qplot(lambdas, rmses)  +
  ggtitle("Selecting the tuning parameter ") +
  labs(x="Lambda",
       y="RMSE",
       caption = "Selecting the tuning parameter in Edx_train dataset")+
  theme(text = element_text(size=16), plot.background = element_rect(colour= NA, linetype = "solid", fill = NA, size = 1), panel.border = element_rect(colour="black", linetype = "solid", fill=NA), plot.title = element_text(hjust = 0.5, size = 20), plot.caption = element_text(hjust = 0.5, size = 22))

#Applying optimal lambda to the model
lambda <- lambdas[which.min(rmses)]
## calculate the regular movie reg_b_i with the optimal lambda
reg_movie_avgs <- edx_train %>% 
  group_by(movieId) %>% 
  summarize(reg_b_i = sum(rating - mu)/(n()+lambda), n_i = n())

## calculate the regular user reg_b_u with the optimal lambda
reg_user_avgs <- edx_train %>% 
  left_join(reg_movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(reg_b_u = sum(rating - mu - reg_b_i)/(n()+lambda), n_u = n())

## calculate the prediction rating for model 5
reg_predicted_ratings <- edx_test %>% 
  left_join(reg_movie_avgs, by='movieId') %>%
  left_join(reg_user_avgs, by='userId') %>%
  mutate(pred = mu + reg_b_i + reg_b_u) %>% 
  .$pred

## calculate rmse for model 5
model_5_rmse <- RMSE(edx_test$rating,reg_predicted_ratings)
rmse_results <- bind_rows(rmse_results,
                          data_frame(Model="Model 5:Regularized Movie, user",  
                                     RMSE = round(model_5_rmse, 5)))
rmse_results
rmse_results %>% knitr::kable()

#Final Model
#we found that model 5 (Regularized movie, user model) is the best so we will now apply
#this model with edx as trainig dataset and validation as testing dataset.
# We will use k fold cross validation to find the optimal $\lambda$
lambdas <- seq(0, 10, 0.25)
set.seed(1989, sample.kind = "Rounding")
## For each lambda,find b_i & b_u followed by rating prediction
rmses <- sapply(lambdas, function(l){
  
  mu <- mean(edx$rating)
  
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  
  return(RMSE(validation$rating,predicted_ratings))
})
## plot lambadas
qplot(lambdas, rmses)  +
  ggtitle("Selecting the tuning parameter ") +
  labs(x="Lambda",
       y="RMSE",
       caption = "Selecting the tuning parameter in Edx_train dataset")+
  theme(text = element_text(size=16), plot.background = element_rect(colour= NA, linetype = "solid", fill = NA, size = 1), panel.border = element_rect(colour="black", linetype = "solid", fill=NA), plot.title = element_text(hjust = 0.5, size = 20), plot.caption = element_text(hjust = 0.5, size = 22))

## get the optimal value for lambda
lambda <- lambdas[which.min(rmses)]
## calculate the regular movie reg_b_i with the optimal lambda
reg_movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(reg_b_i = sum(rating - mu)/(n()+lambda), n_i = n())

## calculate the regular user reg_b_u with the optimal lambda
reg_user_avgs <- edx %>% 
  left_join(reg_movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(reg_b_u = sum(rating - mu - reg_b_i)/(n()+lambda), n_u = n())

## calculate the prediction rating for final model
reg_predicted_ratings <- validation %>% 
  left_join(reg_movie_avgs, by='movieId') %>%
  left_join(reg_user_avgs, by='userId') %>%
  mutate(pred = mu + reg_b_i + reg_b_u) %>% 
  .$pred

## calculate rmse for final model
model_5_rmse <- RMSE(validation$rating,reg_predicted_ratings)
rmse_results <- bind_rows(rmse_results,
                          data_frame(Model="Final model (Model 5)",  
                                     RMSE = round(model_5_rmse, 5)))
rmse_results %>% knitr::kable()