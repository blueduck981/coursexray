from flask import render_template
from flaskexample import app
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2
from flask import request
from a_Model import ModelIt
import process_reviews as jc
import re
import numpy as np
import matplotlib.pyplot as plt
import datetime
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


plt.rc('axes',edgecolor='#262626')
plt.rc('font', weight='normal')    # bold fonts are easier to see
#plt.rc["figure.dpi"] = 200
#plt.rc["savefig.dpi"] = plt.rc["figure.dpi"]
#plt.rc["fontsize"] = 10.8
#plt.rc["pdf.fonttype"] = 42
#plt.rc['font.family'] ='fantasy'

user = 'jcolucci' #add your username here (same as previous postgreSQL)            
host = 'localhost'
dbname = 'course_info_db'
pswd='morley'
#pswd = 'morley'
db = create_engine('postgres://%s:%s@%s/%s'%(user,pswd,host,dbname))
con = None
con = psycopg2.connect(database = dbname, user = user)
###########    LOAD MODELS AND MATRIXES HERE.
reviews_vectorizer = pickle.load(open("/Users/jcolucci/Dropbox/Insight/moocxray/review_classifier.pkl","rb"))
tfidf_reviews = pickle.load(open("/Users/jcolucci/Dropbox/Insight/moocxray/review_tfidf.dat","rb"))

description_vectorizer = pickle.load(open("/Users/jcolucci/Dropbox/Insight/moocxray/description_classifier.pkl","rb"))
tfidf_description = pickle.load(open("/Users/jcolucci/Dropbox/Insight/moocxray/description_tfidf.dat","rb"))

hold_data = pickle.load(open("/Users/jcolucci/Dropbox/Insight/moocxray/proc_info.pkl","rb"))


@app.route('/')
@app.route('/index')
def index():
  return render_template("input.html")




@app.route('/nicer')
def nicer_input():

  d=[0.41584049479166685, 0.2784651593386533, 0.3264728563143893, 0.0, 0.28277777777777785, 0.36346153846153845, 0.0, 0.37973002691752694, 0.15469696969696972, 0.0, 0.31002166057122965, 0.3456116364187436, 0.1904761904761905, 0.37193456758614524, 0.2952678571428571, 0.0, 0.0, 0.2988143590769641, 0.0, 0.2945484995888222, 0.18194444444444444, 0.2969047619047619, 0.3084687222569234, 0.0, 0.20826060606060603, 0.31883333333333336, 0.0, 0.33589873987398755, 0.3962373737373737, 0.19287725779967155, 0.2499872627372627, 0.2833321784131137, 0.3725878099173554, 0.27756939287441273, 0.3468251563251563, 0.34435422774026764, 0.3594791666666667, 0.2717353951890036, 0.406791984142036, 0.3764475440224962, 0.3201181283523281, 0.25566637587764346, 0.26836020591535276, 0.2998583617030218, 0.19654913187171247, 0.22306604953662545, 0.2603198655658762, 0.37087855979505757, 0.18694331065759637, 0.75, 0.0, 0.2886022149371665, 0.2474606284520079, 0.27805249464340365, 0.0, 0.29087881165682333, 0.23857218536671657, 0.3278499278499278, 0.625, 0.29344696969696965, 0.6639999999999999, 0.28642197088625665, 0.3472531328320802, 0.2515077110389611, 0.26078818638224777, 0.3876094276094277, 0.0, 0.25940632697279237, 0.2625633383010432, 0.16296296296296298, 0.33333333333333337, 0.27409382683427064, 0.26249222999223, 0.39031126781922965, 0.18951552562663673, 0.1749358974358974, 0.155571642504782, 0.24057709208085146, 0.3139835294913419, -0.044444444444444446, 0.28769527435247133, 0.3314945302445302, 0.25253572360867305, 0.2999824658545588, 0.3468999518999519, 0.2993142986095287, 0.2745714508858838, 0.3144652461775224, 0.0, 0.20624672141583908, 0.3093066403834261, 0.0, 0.0, 0.37263546328999075, 0.6333333333333333, 0.35, 0.19474194070027406, 0.0, 0.0, 0.2525699137463843, 0.0, 0.3602083333333333, 0.29314318463254635, 0.045682153095946205, 0.3902771880736583, 0.3161368398586503, 0.31722982696244073, 0.27731133979015327, 0.247208886618999, 0.3131523215346744, 0.30431943460395344, 0.3018512368936782, 0.3327241373976818, 0.0, 0.3493430762456497, 0.0, 0.23042469078540506, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.20197598697598698, 0.5599999999999999, 0.0, 0.23269360269360273, 0.2804624733599733, 0.2965847242080038, 0.22852412213274806, 0.16988972573156255, 0.15783001720501713, 0.14062501878908126, 0.24444444444444444, 0.13621756356670145, 0.35688653480320165, 0.45230862086708645, 0.3313372156485389, 0.19600192542500233, 0.1856905283254951, 0.3659700929412473, 0.37774434036252474, 0.33890338827838834, 0.377421875, 0.21891654443278227, 0.27932156352963794, 0.31500000000000006, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.22252665245202566, 0.2856674187394526, 0.1501860465116279, 0.19540816326530613, 0.18306171842984525, 0.14765708556149731, 0.18289529914529917, 0.2945070422535211, 0.35053421424952036, 0.3535143707607478, 0.39436186174606536, 0.37367234385858245, 0.0, 0.0, 0.0, 0.2763210111793232, 0.0, 0.0, 0.29993421052631586, 0.0, 0.32246160197545887, 0.2886901335338836, 0.2539179782082324, 0.3658739837398373, 0.33849937343358394, 0.0, 0.15800134104453198]
 # content={'tfidf_reviews':tfidf_reviews,'reviews_features':reviews_features,'reviews_vectorizer':reviews_vectorizer}
  return render_template("nicer.html",img1="/static/x-ray-clipart-Xrayspec.jpg",dhist=d)


@app.route('/barchart')
def barchart():


 # content={'tfidf_reviews':tfidf_reviews,'reviews_features':reviews_features,'reviews_vectorizer':reviews_vectorizer}
  return render_template("barchart.html")


@app.route('/input')
def moocxray_input():


 # content={'tfidf_reviews':tfidf_reviews,'reviews_features':reviews_features,'reviews_vectorizer':reviews_vectorizer}
  return render_template("input.html",img1="/static/x-ray-clipart-Xrayspec.jpg")

@app.route('/output')
def moocxray_output():



  #Load course info DB
  query = "SELECT * FROM course_info_corrected "
  query_reviews = """
      SELECT * FROM reviews_ratings_trimmed;
          """
  reviews = pd.read_sql_query(query_reviews,con)

  #Load reviews and ratings DB
  unformat_query_results=pd.read_sql_query(query,con)
  new_titles={}
  for ctitle in unformat_query_results['coursename']:
    new_titles[ctitle]=ctitle.decode('utf-8')

  new_des={}
  for cdes in unformat_query_results['description']:
    new_des[cdes]=cdes.decode('utf-8')

  holdd={}
  for ii,url in enumerate(unformat_query_results.url):
    dsentences=tokenize.sent_tokenize(unformat_query_results.description[ii].decode('utf-8'))
    keep_desc=dsentences[:2] 
    holdd[url]=' '.join(keep_desc)  

  unformat_query_results["format_title"]=unformat_query_results.coursename.apply(lambda x:new_titles[x])
  unformat_query_results["format_desc"]=unformat_query_results.description.apply(lambda x:new_des[x])
  unformat_query_results["short_desc"]=unformat_query_results.url.apply(lambda x:holdd[x])


  
  query_results=unformat_query_results
  
  #some reindexing?
  documents=query_results.url
  df2=query_results.set_index('url')


  #Collect and process reviews. this was too slow so i did it in advance and pickled it.
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

  with open('store_keys.dat', 'wb') as outfile:
    pickle.dump(input_words, outfile, pickle.HIGHEST_PROTOCOL)

  with open('store_unstem.dat', 'wb') as outfile:
    pickle.dump(input_string, outfile, pickle.HIGHEST_PROTOCOL)  

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
  power_r=pd.DataFrame({'word':keep_r_word, 'review_val':keep_r_value, 'description_val':keep_d_value})
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
    if i < len(power_results.word):
      power.append(dict( 
        word=power_results.iloc[i]['word'], 
        rev_pow=power_results.iloc[i]['review_val'], 
        des_pow=power_results.iloc[i]['description_val'] ))
    else:
        power.append(dict( 
        word='', 
        rev_pow=0, 
        des_pow=0))
  
  #d=list(collect_course_sentiment)
 
  cat_rev=[]
  take_rev=reviews[reviews.url==besturl].review
  for item in take_rev:
        cat_rev.append(item)

  keyword=power_results.word[0]
  print keyword
  keep_polarity=[]
  for review in take_rev:
    sentences=tokenize.sent_tokenize(review.decode('utf-8'))
    for sentence in sentences:
        if keyword in sentence:
            blob=TextBlob(sentence)
            keep_polarity.append(blob.sentiment.polarity)      

  d=list(keep_polarity)

  the_result = ModelIt(keywords,reviews,query_results)
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
def moocxray_output2():

  old_data = pickle.load(open("/Users/jcolucci/Dropbox/Insight/moocxray/store_keys.dat","rb"))
  old_data_pretty = pickle.load(open("/Users/jcolucci/Dropbox/Insight/moocxray/store_unstem.dat","rb"))

  print old_data
  #Load course info DB
  query = "SELECT * FROM course_info_corrected "
  query_reviews = """
      SELECT * FROM reviews_ratings_trimmed;
          """
  reviews = pd.read_sql_query(query_reviews,con)

  #Load reviews and ratings DB
  unformat_query_results=pd.read_sql_query(query,con)
  new_titles={}
  for ctitle in unformat_query_results['coursename']:
    new_titles[ctitle]=ctitle.decode('utf-8')

  new_des={}
  for cdes in unformat_query_results['description']:
    new_des[cdes]=cdes.decode('utf-8')

  holdd={}
  for ii,url in enumerate(unformat_query_results.url):
    dsentences=tokenize.sent_tokenize(unformat_query_results.description[ii].decode('utf-8'))
    keep_desc=dsentences[:2] 
    holdd[url]=' '.join(keep_desc)  

  unformat_query_results["format_title"]=unformat_query_results.coursename.apply(lambda x:new_titles[x])
  unformat_query_results["format_desc"]=unformat_query_results.description.apply(lambda x:new_des[x])
  unformat_query_results["short_desc"]=unformat_query_results.url.apply(lambda x:holdd[x])


  
  query_results=unformat_query_results
  
  #some reindexing?
  documents=query_results.url
  df2=query_results.set_index('url')


  #Collect and process reviews. this was too slow so i did it in advance and pickled it.
  #collect_docs, collect_decs, collect_url_order,collect_course_sentiment=initial_model.initial_processing(query_results, reviews)
  collect_docs=hold_data.proc_reviews
  collect_decs=hold_data.proc_desc
  collect_url_order=hold_data.course_url
  collect_course_sentiment=hold_data.sentiment



  p_stemmer = PorterStemmer()
  


  input_string= ' '.join(old_data)#.replace(',',' ')

  #input_words=old_data

  stemmed_input = [p_stemmer.stem(i) for i in jc.token_ws(input_string)]
  input_words=[' '.join(stemmed_input)]

  with open('store_keys.dat', 'wb') as outfile:
    pickle.dump(input_words, outfile, pickle.HIGHEST_PROTOCOL)

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
  power_r=pd.DataFrame({'word':keep_r_word, 'review_val':keep_r_value, 'description_val':keep_d_value})
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
    if i < len(power_results.word):
      power.append(dict( 
        word=power_results.iloc[i]['word'], 
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
  take_rev=reviews[reviews.url==besturl].review
  for item in take_rev:
        cat_rev.append(item)

  keyword=keywords2
  print keyword
  keep_polarity=[]
  for review in take_rev:
    sentences=tokenize.sent_tokenize(review.decode('utf-8'))
    for sentence in sentences:
        if keyword in sentence:
            blob=TextBlob(sentence)
            keep_polarity.append(blob.sentiment.polarity)      

  d=list(keep_polarity)

  the_result = ModelIt(old_data_pretty,reviews,query_results)
  if len(the_result)==1:
    wordstring=( ' '.join( the_result))
  elif len(the_result)==2:
    wordstring=( ' and '.join( the_result)) 
  else:
    the_result[len(the_result)-1]='and '+str(the_result[len(the_result)-1])
    wordstring=( ", ".join( the_result)) 
  return render_template("output_xray.html", courses = courses, the_result = wordstring,power=power,c1=c1,dhist=d,keyword=keywords2)







