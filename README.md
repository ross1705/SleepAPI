# SleepAPI

You can find the web app I have created based on a Machine Learning model I constructed and deployed as a REST API at the following link: https://sleepapi.herokuapp.com/

In preparation for todays interview, I wanted to demonstrate deploying a Machine Learning model through a REST API. My goal here was to try complete what I think may be something along the lines of a task I will encounter in Terra, and deploying ML models seems like a likely task given the job description. 


As I have never worked with wearables data, I went onto Kaggle and obtained a dataset names 'SleepQual and B.Health dataset', which is described as 'Sleep Quality and Behavioral health of 24 university students collected using their smartwatches and smartphones for 7 consecutive days and nights.' This is by no means a large or heavily detailed dataset, however given its compatability with the industry I thought it would be beneficial to work with and deploy a model based on such topical data. I have plenty of other projects available on my github which exemplify my ability to work with and analyze very large datasets. 


I firstly cleaned up and inspected the dataset to get a feel for what I am working with. This is available in the 'ML Building and Training.ipynb' file. As I have mentioned, I have carried out extensive statistical analysis on data sets through Python, but in this project I am focused on learning how to and showing my ability to deploy a REST API, so the statistical analysis isnt as rigorous as usual. I analyzed relationships between variables, and knew I wanted to model the relationship between 'night time phone usage' and 'efficiency' as it is regularly hypothesised that using your phone betfore sleeping can be detrimental to sleep quality. I then constructed a model based on this negatively correlated relationship, and produced plots to gain further insight. Saved the model as a pkl file. 


I then created the web app through Flask using VS code. This app is based on the above obtained ML model, and deployed as a REST API. I understand this is a very simple app, however I just wanted to show my appetite to learn, by deploying an ML model through a REST API. It allows the user to input the amount of time they spent on their phone before going to sleep, and will return to them their predicted sleep efficiency, described as the amount of time you spend in bed actually sleeping. Again I would like to stress my goal here is just demonstration. I used Heroku to host the web-app, and had to install Homebrew and the Heroku CLI in order to do this. 
 
