from flask import render_template
from flaskexample import app
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2
from flask import request
from a_Model import ModelIt
import process_reviews as jc

user = 'jcolucci' #add your username here (same as previous postgreSQL)            
host = 'localhost'
dbname = 'course_info_db'
pswd='morley'
#pswd = 'morley'
db = create_engine('postgres://%s:%s@%s/%s'%(user,pswd,host,dbname))
con = None
con = psycopg2.connect(database = dbname, user = user)



@app.route('/input')
def cesareans_input():
    return render_template("input.html")

@app.route('/output')
def cesareans_output():

  keywords = request.args.get('user_keyword')
  keywords_split = keywords.split(', ')

  query = "SELECT * FROM course_info_partial "
  query_reviews = """
      SELECT * FROM reviews_ratings;
          """
  reviews = pd.read_sql_query(query_reviews,con)

  
  unformat_query_results=pd.read_sql_query(query,con)
  new_titles={}
  for ctitle in unformat_query_results['coursename']:
    new_titles[ctitle]=ctitle.decode('utf-8')
  unformat_query_results["format_title"]=unformat_query_results.coursename.apply(lambda x:new_titles[x])
  query_results=unformat_query_results
  documents=query_results.url
  df2=query_results.set_index('url')


  collect_docs=[]
  collect_freqs={}
  review_text_dict={}
  course_title_dict={}
  course_site_dict={}

  for course in documents:
    folded_set=jc.create_documents(course,reviews)    
    review_text_dict[course]=folded_set
    course_title_dict[course]=query_results[query_results.url==course].coursename
    course_site_dict[course]=query_results[query_results.url==course].company

    raw_doc='-'.join(folded_set)    
    course_re_tokens=jc.token_re(raw_doc)
    cleaned_tokens=jc.clean_tokens(course_re_tokens)
    recombined_doc=( " ".join( cleaned_tokens ))   
    removed_mywords=jc.remove_my_stopwords(recombined_doc)
    collect_docs.append(removed_mywords)
    if len(removed_mywords)>0:
        sorted_freq_list=jc.bagofwords(removed_mywords)
        #print sorted_freq_list[0]

        sorted_freq_dict=dict(sorted_freq_list)
        #print jc.listN(sorted_freq_dict,N=1)
        
        collect_freqs[course]=sorted_freq_dict

  #print jc.listN(collect_freqs,N=1)

  rec_list=jc.calculate_sum_keywords(collect_freqs,keywords_split)
  summary_df_sort=jc.summarize_results(rec_list,review_text_dict,course_title_dict,course_site_dict)
  summary_df_sort['title'] = summary_df_sort.url.apply(lambda x:df2.loc[x,'coursename'])
  summary_df_sort['company'] = summary_df_sort.url.apply(lambda x:df2.loc[x,'company'])
  summary_df_sort['description'] = summary_df_sort.url.apply(lambda x:df2.loc[x,'description'])

  print summary_df_sort






  courses = []
  for i in range(0,5):
      courses.append(dict(url=summary_df_sort.iloc[i]['url'], 
        #coursename=query_results.iloc[i]['coursename'], company=query_results.iloc[i]['company'], review_source=query_results.iloc[i]['review_source']))
        coursename=summary_df_sort.iloc[i]['title'], description=summary_df_sort.iloc[i]['description'],
        company=summary_df_sort.iloc[i]['company']))#, review_source=summary_df_sort.iloc[i]['review_source']))

     #     company=query_results.iloc[i]['company'], review_source=query_results.iloc[i]['review_source']))

      the_result = ''

  #the_result = ModelIt(patient,births)    
  the_result = ModelIt(keywords,reviews,query_results)
  if len(the_result)==1:
    wordstring=( ' '.join( the_result))
  elif len(the_result)==2:
    wordstring=( ' and '.join( the_result)) 
  else:
    the_result[len(the_result)-1]='and '+str(the_result[len(the_result)-1])
    wordstring=( ", ".join( the_result)) 
  return render_template("output.html", courses = courses, the_result = wordstring)


