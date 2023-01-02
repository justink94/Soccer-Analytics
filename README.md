# Soccer-Analytics

# Introduction

This was my senior capstone project to complete my bachelor's degree in Data Science in 2022. Project comprises everything from initial data import from a public source on the internet, cleaning, visualisations, machine learning and statistical modeling. 

Can we use machine learning modeling to help coaches strategize basd on available game data? Various ideas and conclusions in the project included trying to find optimal shooting and passing positions on the field of play during a soccer match, whether shooting more often regardless of position led to more goals, and finding out what factors affected goal scoring the most. These conclusions may be able to help coaches change tactics and make personnel decisions 

# Data Background
Data was imported from Wyscout's website using publicly available data from the 2017-18 season in Europe's top 5 leagues. Included goals, assists, player information, location of each event on the field based on making the field into an (x,y) coordinate plane. 

# Methods
Uses both R and Python

## Random Forests
Used shot and assist position, time of game, and body part used to shoot to determine whether a shot would lead to a goal

## Naive Bayes Model
Classification algorithm using same parameters as random forest

## K Nearest Neighbor
Clustering algorithm used to classify goals based on location on an (x,y) plane 

