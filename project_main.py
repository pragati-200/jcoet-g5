
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
import re
import emoji
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

#title
st.title('Tweet Sentiment Analysis')
#markdown
st.markdown('This application is all about tweet sentiment analysis of airlines. We can analyse reviews of the passengers using this streamlit app.')
#sidebar
st.sidebar.title('Sentiment analysis of airlines')
# sidebar markdown 
st.sidebar.markdown("🛫We can analyse passengers review from this application.🛫")
#loading the data (the csv file is in the same folder)
#if the file is stored the copy the path and paste in read_csv method.
data=pd.read_csv("D:/Tweets.csv")
#checkbox to show data 
if st.checkbox("Show Data"):
    st.write(data.head(50))
#subheader
st.sidebar.subheader('Tweets Analyser')
#radio buttons
tweets=st.sidebar.radio('Sentiment Type',('positive','negative','neutral'))
st.write(data.query('airline_sentiment==@tweets')[['text']].sample(1).iat[0,0])
st.write(data.query('airline_sentiment==@tweets')[['text']].sample(1).iat[0,0])
st.write(data.query('airline_sentiment==@tweets')[['text']].sample(1).iat[0,0])
#selectbox + visualisation
# An optional string to use as the unique key for the widget. If this is omitted, a key will be generated for the widget based on its content.
## Multiple widgets of the same type may not share the same key.
select=st.sidebar.selectbox('Visualisation Of Tweets',['Histogram','Pie Chart'],key=1)
sentiment=data['airline_sentiment'].value_counts()
sentiment=pd.DataFrame({'Sentiment':sentiment.index,'Tweets':sentiment.values})
st.markdown("###  Sentiment count")
if select == "Histogram":
        fig = px.bar(sentiment, x='Sentiment', y='Tweets', color = 'Tweets', height= 500)
        st.plotly_chart(fig)
else:
        fig = px.pie(sentiment, values='Tweets', names='Sentiment')
        st.plotly_chart(fig)

#slider
st.sidebar.markdown('Time & Location of tweets')
hr = st.sidebar.slider("Hour of the day", 0, 23)
data['Date'] = pd.to_datetime(data['tweet_created'])
hr_data = data[data['Date'].dt.hour == hr]
if not st.sidebar.checkbox("Hide", True, key='1'):
    st.markdown("### Location of the tweets based on the hour of the day")
    st.markdown("%i tweets during  %i:00 and %i:00" % (len(hr_data), hr, (hr+1)%24))
    st.map(hr_data)

#multiselect
st.sidebar.subheader("Airline tweets by sentiment")
choice = st.sidebar.multiselect("Airlines", ('US Airways', 'United', 'American', 'Southwest', 'Delta', 'Virgin America'), key = '0')  
if len(choice)>0:
    air_data=data[data.airline.isin(choice)]
    # facet_col = 'airline_sentiment'
    fig1 = px.histogram(air_data,x='airline',y='airline_sentiment',histfunc='count',color='airline_sentiment',labels={'airline_sentiment':'tweets'},height=600,width=800)
    st.plotly_chart(fig1)
data.head(20)
confidence_threshold=0.6
data.query("airline_sentiment_confidence < @confidence_threshold")
data.isnull().sum()
confidence_threshold=0.6
data.query("airline_sentiment_confidence < @confidence_threshold").index
confidence_threshold=0.6
data=data.drop(data.query("airline_sentiment_confidence < @confidence_threshold").index,axis=0).reset_index(drop=True)
tweets_df=pd.concat([data['text'],data['airline_sentiment']],axis=1)
tweets_df
tweets_df.isnull().sum()
tweets_df["airline_sentiment"].value_counts()
sentiments_ordering=['negative','neutral','positive']
sentiments_ordering
tweets_df['airline_sentiment']=tweets_df['airline_sentiment'].apply(lambda x:sentiments_ordering .index(x))
tweets_df
x=tweets_df['text']
y=tweets_df.airline_sentiment
x=tweets_df.text
y=tweets_df.airline_sentiment
print(x.shape)
print(y.shape)
from sklearn.model_selection  import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(stop_words='english')
vect.fit(x_train)
vect.vocabulary_
X_train_transformed = vect.transform(x_train)
X_test_transformed =vect.transform(x_test)
print(type(X_train_transformed))
print(X_train_transformed)
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()

# fit
mnb.fit(X_train_transformed,y_train)

# predict class
y_pred_class = mnb.predict(X_test_transformed)

# predict probabilities
y_pred_proba =mnb.predict_proba(X_test_transformed)


# printing the overall accuracy
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)

y_pred_class
metrics.confusion_matrix(y_test, y_pred_class)
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train_transformed,y_train)
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)




