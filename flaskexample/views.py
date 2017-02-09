from flask import render_template
from flaskexample import app
import pandas as pd
from flask import request
import process_reviews as jc
import re
import numpy as np
from textblob import TextBlob
from nltk import tokenize
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import WhitespaceTokenizer
from nltk.corpus import stopwords # Import the stop word list
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.stem.porter import PorterStemmer
import initial_model as initial_model
import cPickle as pickle
import os


###########    LOAD MODELS AND MATRIXES HERE.
reviews_vectorizer = pickle.load(open("data/review_classifier.pkl","rb"))
tfidf_reviews = pickle.load(open("data/review_tfidf.dat","rb"))

description_vectorizer = pickle.load(open("data/description_classifier.pkl","rb"))
tfidf_description = pickle.load(open("data/description_tfidf.dat","rb"))

hold_data = pickle.load(open("data/proc_info.pkl","rb"))
query_results = pickle.load(open("data/course_info.pkl","rb"))
#reviews = pickle.load(open("data/reviews.pkl","rb"))


@app.route('/')
@app.route('/index')
def index():
  return render_template("input.html")


@app.route('/slides')
def slide():
  return render_template("slides.html")



@app.route('/input')
def xray_input():
  return render_template("input.html")


@app.route('/output')
def xray_output():


  #some reindexing?
  documents=query_results.url
  df2=query_results.set_index('url')


  #Collect and process reviews. Now in pickle form.
  #collect_docs, collect_decs, collect_url_order,collect_course_sentiment=initial_model.initial_processing(query_results, reviews)
  collect_docs=hold_data.proc_reviews
  collect_decs=hold_data.proc_desc
  collect_url_order=hold_data.course_url
  collect_course_sentiment=hold_data.sentiment


  p_stemmer = PorterStemmer()
  keywords = request.args.get('user_keyword')
  input_string= keywords.replace(',',' ')

  input_words=[input_string]
  stemmed_input = [p_stemmer.stem(i) for i in jc.token_ws(input_string)]
  input_words=[' '.join(stemmed_input)]
  input_array= [i for i in jc.token_ws(input_string)]


  with open('data/store_keys.dat', 'wb') as outfile:
    pickle.dump(input_words, outfile, pickle.HIGHEST_PROTOCOL)

  with open('data/store_unstem.dat', 'wb') as outfile:
    pickle.dump(input_string, outfile, pickle.HIGHEST_PROTOCOL)  

  review_cosine_scores=initial_model.get_cosine_similarities(input_words, reviews_vectorizer, tfidf_reviews)
  description_cosine_scores=initial_model.get_cosine_similarities(input_words, description_vectorizer, tfidf_description)

  print len(collect_url_order), len(collect_course_sentiment), len(description_cosine_scores)

  matching_matrix= pd.DataFrame({'course_url':collect_url_order,'review_score':review_cosine_scores, 
                              'description_score':description_cosine_scores,'sentiment':collect_course_sentiment})

  matching_matrix['total_score'] = matching_matrix.apply(lambda row:row.review_score+row.description_score,axis=1)
  matching_matrix_sort=matching_matrix.sort_values(by='total_score',ascending=False).reset_index(inplace=False,drop=True)


  matching_matrix_sort['title'] = matching_matrix_sort.course_url.apply(lambda x:df2.loc[x,'format_title'])
  matching_matrix_sort['company'] = matching_matrix_sort.course_url.apply(lambda x:df2.loc[x,'company'])
  matching_matrix_sort['num_rev'] = matching_matrix_sort.course_url.apply(lambda x:df2.loc[x,'num_rev'])
  matching_matrix_sort['description'] = matching_matrix_sort.course_url.apply(lambda x:df2.loc[x,'format_desc'])
  matching_matrix_sort['short_description'] = matching_matrix_sort.course_url.apply(lambda x:df2.loc[x,'short_desc'])

  besturl=matching_matrix_sort.loc[0,'course_url']


   

  keep_r_word, keep_r_value=initial_model.get_word_power(besturl,collect_url_order,tfidf_reviews,reviews_vectorizer, stemmed_input)
  keep_d_word, keep_d_value=initial_model.get_word_power(besturl,collect_url_order,tfidf_description,description_vectorizer, stemmed_input)
  
  power_r=pd.DataFrame({'stemmed_word':keep_r_word, 'whole_word':input_array, 'review_val':keep_r_value, 'description_val':keep_d_value})

  #power_r=pd.DataFrame({'word':keep_r_word, 'review_val':keep_r_value, 'description_val':keep_d_value})
  power_r['total'] = power_r.apply(lambda row:row.review_val+row.description_val,axis=1)
  power_results = power_r.sort_values(by='total',ascending=False).reset_index(inplace=False,drop=True)


  


  courses = []
  for i in range(1,5):
      courses.append(dict(url=matching_matrix_sort.iloc[i]['course_url'], 
        coursename=matching_matrix_sort.iloc[i]['title'], description=matching_matrix_sort.iloc[i]['short_description'],
        company=matching_matrix_sort.iloc[i]['company'], number= matching_matrix_sort.iloc[i]['num_rev'] ))
      the_result = ''



  c1 = []
  
  c1.append(dict(url=matching_matrix_sort.loc[0,'course_url'], 
        coursename=matching_matrix_sort.loc[0,'title'], description=matching_matrix_sort.loc[0,'short_description'],
        company=matching_matrix_sort.loc[0,'company'], number= matching_matrix_sort.loc[0,'num_rev'] ))

  
  power = []
  for i in range(0,5):
    if i < len(power_results.whole_word):
      power.append(dict( 
        word=power_results.iloc[i]['whole_word'], 
        rev_pow=power_results.iloc[i]['review_val'], 
        des_pow=power_results.iloc[i]['description_val'] ))
    else:
        power.append(dict( 
        word='', 
        rev_pow=0, 
        des_pow=0))
  
  #d=list(collect_course_sentiment)
 
  cat_rev=[]
  take_rev=hold_data[hold_data.course_url==besturl].reviews
  for item in take_rev:
        cat_rev.append(item)

  keyword=power_results.whole_word[0]
  print keyword
  keep_polarity=[]
  sentences=tokenize.sent_tokenize(str(cat_rev).decode('utf-8'))
  for sentence in sentences:
      if keyword in sentence:
          blob=TextBlob(sentence)
          keep_polarity.append(blob.sentiment.polarity)      


  k=[x-0.01 for x in keep_polarity]

  d=list(k)


  #the_result = keywords.split(', ')
  the_result = input_array


  if len(the_result)==1:
    wordstring=( ' '.join( the_result))
  elif len(the_result)==2:
    wordstring=( ' and '.join( the_result)) 
  else:
    the_result[len(the_result)-1]='and '+str(the_result[len(the_result)-1])
    wordstring=( ", ".join( the_result)) 
  return render_template("output.html", courses = courses, the_result = wordstring,power=power,c1=c1,dhist=d,keyword=keyword)

#####################################

@app.route('/output_xray')
def xray_output2():

  old_data = pickle.load(open("data/store_keys.dat","rb"))
  old_data_pretty = pickle.load(open("data/store_unstem.dat","rb"))
  print old_data
  print old_data_pretty
 

  #some reindexing?
  documents=query_results.url
  df2=query_results.set_index('url')


  #Collect and process reviews. Now in pickle form.
  #collect_docs, collect_decs, collect_url_order,collect_course_sentiment=initial_model.initial_processing(query_results, reviews)
  collect_docs=hold_data.proc_reviews
  collect_decs=hold_data.proc_desc
  collect_url_order=hold_data.course_url
  collect_course_sentiment=hold_data.sentiment



  p_stemmer = PorterStemmer()
 
  input_string= ' '.join(old_data)#.replace(',',' ')

  input_string_pretty= old_data_pretty
  #input_words=old_data
  print input_string_pretty

  stemmed_input = [p_stemmer.stem(i) for i in jc.token_ws(input_string)]
  input_words=[' '.join(stemmed_input)]


  input_array= jc.token_ws(input_string_pretty)
  print input_array
  #with open('store_keys.dat', 'wb') as outfile:
  #  pickle.dump(input_words, outfile, pickle.HIGHEST_PROTOCOL)

  review_cosine_scores=initial_model.get_cosine_similarities(input_words, reviews_vectorizer, tfidf_reviews)
  description_cosine_scores=initial_model.get_cosine_similarities(input_words, description_vectorizer, tfidf_description)


  matching_matrix= pd.DataFrame({'course_url':collect_url_order,'review_score':review_cosine_scores, 
                              'description_score':description_cosine_scores,'sentiment':collect_course_sentiment})

  matching_matrix['total_score'] = matching_matrix.apply(lambda row:row.review_score+row.description_score,axis=1)
  matching_matrix_sort=matching_matrix.sort_values(by='total_score',ascending=False).reset_index(inplace=False,drop=True)


  matching_matrix_sort['title'] = matching_matrix_sort.course_url.apply(lambda x:df2.loc[x,'format_title'])
  matching_matrix_sort['company'] = matching_matrix_sort.course_url.apply(lambda x:df2.loc[x,'company'])
  matching_matrix_sort['num_rev'] = matching_matrix_sort.course_url.apply(lambda x:df2.loc[x,'num_rev'])
  matching_matrix_sort['description'] = matching_matrix_sort.course_url.apply(lambda x:df2.loc[x,'format_desc'])
  matching_matrix_sort['short_description'] = matching_matrix_sort.course_url.apply(lambda x:df2.loc[x,'short_desc'])

  besturl=matching_matrix_sort.loc[0,'course_url']



  keep_r_word, keep_r_value=initial_model.get_word_power(besturl,collect_url_order,tfidf_reviews,reviews_vectorizer, stemmed_input)
  keep_d_word, keep_d_value=initial_model.get_word_power(besturl,collect_url_order,tfidf_description,description_vectorizer, stemmed_input)
  
  power_r=pd.DataFrame({'stemmed_word':keep_r_word, 'whole_word':input_array, 'review_val':keep_r_value, 'description_val':keep_d_value})

#  power_r=pd.DataFrame({'word':keep_r_word, 'review_val':keep_r_value, 'description_val':keep_d_value})
  power_r['total'] = power_r.apply(lambda row:row.review_val+row.description_val,axis=1)
  power_results = power_r.sort_values(by='total',ascending=False).reset_index(inplace=False,drop=True)


  


  courses = []
  for i in range(1,5):
      courses.append(dict(url=matching_matrix_sort.iloc[i]['course_url'], 
        coursename=matching_matrix_sort.iloc[i]['title'], description=matching_matrix_sort.iloc[i]['short_description'],
        company=matching_matrix_sort.iloc[i]['company'], number= matching_matrix_sort.iloc[i]['num_rev'],
        sentiment=matching_matrix_sort.iloc[i]['sentiment'] ))
      the_result = ''



  c1 = []
  
  c1.append(dict(url=matching_matrix_sort.loc[0,'course_url'], 
        coursename=matching_matrix_sort.loc[0,'title'], description=matching_matrix_sort.loc[0,'short_description'],
        company=matching_matrix_sort.loc[0,'company'], number= matching_matrix_sort.loc[0,'num_rev'],
        sentiment=matching_matrix_sort.iloc[i]['sentiment'] ))

  
  power = []
  for i in range(0,5):
    if i < len(power_results.whole_word):
      power.append(dict( 
        word=power_results.iloc[i]['whole_word'], 
        rev_pow=power_results.iloc[i]['review_val'], 
        des_pow=power_results.iloc[i]['description_val'] ))
    else:
        power.append(dict( 
        word='', 
        rev_pow=0, 
        des_pow=0))
  
  #d=list(collect_course_sentiment)
  keywords2 = request.args.get('user_keyword2')
  print keywords2
  

 


  cat_rev=[]
  take_rev=hold_data[hold_data.course_url==besturl].reviews
  for item in take_rev:
        cat_rev.append(item)

  keyword=keywords2
  print keyword
  keep_polarity=[]
  sentences=tokenize.sent_tokenize(str(cat_rev).decode('utf-8'))
  for sentence in sentences:
      if keyword in sentence:
          blob=TextBlob(sentence)
          keep_polarity.append(blob.sentiment.polarity)      








  d=list(keep_polarity)


  k=[x-0.01 for x in keep_polarity]
  k2=[round(x,2) for x in k]


  #the_result = keywords2.split(', ')
  the_result = input_array


  if len(the_result)==1:
    wordstring=( ' '.join( the_result))
  elif len(the_result)==2:
    wordstring=( ' and '.join( the_result)) 
  else:
    the_result[len(the_result)-1]='and '+str(the_result[len(the_result)-1])
    wordstring=( ", ".join( the_result)) 
  return render_template("output_xray.html", courses = courses, the_result = wordstring,power=power,c1=c1,dhist=k2,keyword=keywords2)







