# coding: utf-8


import pandas as pd
import re
from nltk import tokenize
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import WhitespaceTokenizer
from nltk.corpus import stopwords # Import the stop word list
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import process_reviews as jc
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def initial_processing(course_from_sql, reviews):
	p_stemmer = PorterStemmer()

	documents=course_from_sql.url
	df2=course_from_sql.set_index('url')
	descriptions=course_from_sql.description

	review_text_dict={}
	rating_text_dict={}
	course_title_dict={}
	course_site_dict={}
	course_desc_dict={}
	test_re_dict={}

	collect_decs=[]
	collect_docs=[]
	collect_url_order=[]
	collect_course_sentiment=[]

#documents=[course_from_sql.url[0]]
	for course in documents:
    	#hold all reviews as individual words
    		folded_set, folded_rating=jc.find_and_combine_reviews(course,reviews)
    		folded_desc=jc.create_desc(course_from_sql[course_from_sql.url==course].description)

    	#print folded_set
    
    	#save this block of unedited reviews for later
    		review_text_dict[course]=folded_set
    		rating_text_dict[course]=folded_rating
    
 
    		course_sentiment_text='-'.join(folded_set).replace('\n', ' ').replace('-', ' ').replace('=', ' ')
   

    
    		blob = TextBlob(course_sentiment_text.decode('utf-8'))
    		collect_course_sentiment.append(blob.sentiment.polarity)
    
    
    
    
    	#for this course save the useful information
    		course_title_dict[course]=course_from_sql[course_from_sql.url==course].coursename
    		course_site_dict[course]=course_from_sql[course_from_sql.url==course].company
    		course_desc_dict[course]=course_from_sql[course_from_sql.url==course].description
    
    
    	#PROCESS ALL THE TEXT HERE
    		normal_stopped_tokens, stopped_desc_tokens=jc.preprocess_text(folded_set,folded_desc)

    	#COMBINE TOKENS BACK INTO ONE DOCUMENT    
    		recombined_doc=( " ".join( normal_stopped_tokens ))   
    		recombined_desc=( " ".join( stopped_desc_tokens ))   

    
    
    	#OUTPUT IS LIST OF COURSES, FOR EACH COURSE IT INCLUDES COMBINED TEXT OF ALL REVIEWS
    		collect_docs.append(recombined_doc)
    		collect_decs.append(recombined_desc)
    		collect_url_order.append(course)
	return collect_docs, collect_decs, collect_url_order,collect_course_sentiment


def create_tfidf_matrix(collect_docs):
    tf_vectorizer = TfidfVectorizer(stop_words='english')#,ngram_range=(1,3))
    tfidf_matrix = tf_vectorizer.fit_transform(collect_docs)
    feature_names=tf_vectorizer.get_feature_names()
    
    return tfidf_matrix, feature_names, tf_vectorizer



def get_cosine_similarities(input_words, invectorizer, inmatrix):
    inputmatrix=invectorizer.transform(input_words)
    cosine_similarities = linear_kernel(inputmatrix, inmatrix).flatten()
    return cosine_similarities



def initial_model(collect_docs,collect_decs):
	tfidf_reviews, reviews_features, reviews_vectorizer=create_tfidf_matrix(collect_docs)
	tfidf_description, description_features, description_vectorizer=create_tfidf_matrix(collect_decs)
	return tfidf_reviews, reviews_features, reviews_vectorizer, tfidf_description, description_features, description_vectorizer


def get_word_power(besturl,collect_url_order,tfidf_reviews,reviews_vectorizer, stemmed_input):

	for ii, item in enumerate(collect_url_order):
    		if item ==besturl:
 				matching_index=ii
	collect_url_order[matching_index]


	reviews_features=reviews_vectorizer.get_feature_names()

	dense = tfidf_reviews.todense()
	dense_course = dense[matching_index].tolist()[0]
	phrase_scores = [pair for pair in zip(range(0, len(dense_course)), dense_course) ]#if pair[1] > 0]
	keep_word=[]
	keep_value=[]

	for phrase, score in [(reviews_features[word_id], score) for (word_id, score) in phrase_scores]:#[:20]:

    		if phrase in stemmed_input:
        	#print phrase, score
        		keep_word.append(phrase)
        		keep_value.append(score)
        
        d = dict(zip(keep_word,keep_value))
        x = [(k,d[k]) for k in stemmed_input]
        return [item[0] for item in x],[item[1] for item in x]          	
    	#return keep_word, keep_value


def check_input(stemmed_input,r_vect,d_vect,good_input):
    r_features=r_vect.get_feature_names()
    d_features=d_vect.get_feature_names()
    error_list=[]
    keep_error=[]

    for ii, word in enumerate(stemmed_input):
        if word not in r_features:
            #print word
            stemmed_input.remove(word)
            error_list.append(word)
            #print good_input[ii]
            keep_error.append(good_input[ii])
           # print keep_error
            del good_input[ii]
    

    for ii, word in enumerate(stemmed_input):
        if word not in d_features:
            #print word
            #print good_input[ii]
            stemmed_input.remove(word)
            error_list.append(word)
            keep_error.append(good_input[ii])

            del good_input[ii]
    bad_output= ", ".join(keep_error)
    return stemmed_input, bad_output, good_input, error_list



if __name__ == '__main__':
    test=[]