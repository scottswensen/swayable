Consulting project for Swayable: Predicting Viewer Responses to Advertised 
Messages, by Scott Swensen, September 2018

In this script, various machine learning algorithms are used to determine if 
the level of viewer agreement with a video advertisement can be predicted. 
The model is also used to identify which viewer and video features are 
important in predicting video message agreement and to identify interaction 
between model features.

Demographic information for each video viewer was provided via several MongoDB
bson files. These were pooled together into a csv file called 
"df_complete.csv". This file includes survey responses from each viewer 
including demographic information including age, race, location, income, 
gender, education level, and political leaning. The file also includes several 
survey responses that were used to calculate a user agreement score indicating
the level of viewer agreement with the advertised message. This value had to be
calculated because different survey quetions were asked to viewers who
viewed videos included in different surveys. A categorical variable for the 
survey provided was also used as a model feature. 

NOTE: This input file "df_complete.csv" is not included in the repository 
because the information is the property of Swayable.

More information about this project is included in this blog post:
https://scottswensen.wordpress.com
