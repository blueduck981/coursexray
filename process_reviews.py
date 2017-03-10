
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



p_stemmer = PorterStemmer()


def listN(indict,N=10,rand=False):
   """
   Print first N items from a dictionary.  Can use 'rand=True' to
   look at a random selection of dictionary elements.
   """
   if rand:
       samp=random.sample(range(len(indict)),min([N,len(indict)]))
   else:
       samp=range(N)
   for i in samp:
       print str(list(indict)[i])+':'+str(indict[list(indict)[i]])


def find_and_combine_reviews(review_set,reviews):
    #This function will locate all of the individual course reviews and
    # their associated ratings. Each individual review has been stored
    # as one row, they can be associated to a particular course by the
    # unique url.  This function will collect all individual reviews
    # for a course 
    folded_set=[]
    folded_rating=[]

    relevant_reviews=reviews[reviews.url==review_set].review
    relevant_ratings=reviews[reviews.url==review_set].rating

    for item in relevant_reviews:
        folded_set.append(item)
    for item in relevant_ratings:
        folded_rating.append(item)    
               
    return folded_set, folded_rating


def create_desc(description):
    #some bookkeeping to grab appropriate course description
    folded_desc=[]
    for item in description:
        folded_desc.append(item)
    return folded_desc    


def token_re(doc):
    # this function tokenizes course document:
    # removes punctuation, converts to lower case
    re_tokenizer = RegexpTokenizer(r'\w+')
    course_re_tokens = re_tokenizer.tokenize(doc.decode('utf-8').lower())    
    return course_re_tokens


def token_ws(doc):
    #this function only tokenizes on white space
    ws_tokenizer = WhitespaceTokenizer()
    course_ws_tokens = ws_tokenizer.tokenize(doc.lower())    
    return course_ws_tokens



def clean_tokens(doc):
    # this function removes stop words from a course document
    cleaned_tokens = []
    stop_words = set(stopwords.words('english'))
    for token in doc:
        if token not in stop_words:
            cleaned_tokens.append(token)
    return cleaned_tokens

def find_duration_and_number_recent_reviews(documents):
    #this function calculates the total time period the reviews span
    durations=[]
    cut_num_revs=[]
    for ii,course in enumerate(documents):
        relevant_times=reviews[reviews.url==course].rev_time
        if len(relevant_times)>3:
            durations.append(max(relevant_times)-min(relevant_times))
            cut_num_revs.append(course_from_sql.num_rev[ii])
    return durations, cut_num_revs


def find_combine_partial_documents(course,reviews):
    #this function finds only the most recent course reviews (latest 1/3)
    folded_set=[]
    folded_rating        
    relevant_times=reviews[reviews.url==course].rev_time
        
    if len(relevant_times) > 2:    
        a= max(relevant_times)-min(relevant_times)
        endtime=max(relevant_times)
        subset=reviews[reviews.url==course]
        smaller_subset=subset[subset.rev_time > (endtime-a/3)].review
        smaller_rating=subset[subset.rev_time > (endtime-a/3)].rating

        for item in smaller_subset:
            folded_set.append(item)
        for item in smaller_rating:
            folded_rating.append(item)

        else:
            folded_set.append(' ')   
    return folded_set,folded_ratings



def preprocess_text(folded_set,folded_desc): 
    # this function does all the text preprocessing.
    # First it concatenates all the reviews for a particular course 
    # into one "document."  
    #
    #hold all reviews as big paragraph
    raw_desc='-'.join(folded_desc)    
    raw_doc='-'.join(folded_set)
    
    #all words in all reviews tokenized.
    course_re_tokens=token_re(raw_doc)    
    desc_re_tokens=token_re(raw_desc)
    
    #any hand picked stop words can be listed here
    check=[ "coursera","mooc","udacity","edx","udemy"]
    stop_words = set(stopwords.words('english'))

    # remove stop words
    mystopped_tokens = [i for i in course_re_tokens if not i in check]
    normal_stopped_tokens = [i for i in mystopped_tokens if not i in stop_words]
    stopped_desc_tokens=[i for i in desc_re_tokens if not i in stop_words]    
     
     #perform word stemming
    stemmed_reviews = [p_stemmer.stem(i) for i in normal_stopped_tokens]
    stemmed_desc = [p_stemmer.stem(i) for i in stopped_desc_tokens]

    return stemmed_reviews, stemmed_desc





def summarize_results(rec_list,review_text_dict,course_title_dict,course_site_dict):
    #this creates a dataframe to summarize results. It counts the total number
    # of reviews  
    summary_df=pd.DataFrame(rec_list,columns=['url','match_word_count'])
    
    summary_df["review_count"]=summary_df.url.apply(lambda x:len(review_text_dict[x]))
    summary_df['mc_div_rc'] = summary_df.apply(lambda row:row.
                                               match_word_count/float(row.review_count),axis=1)
    summary_df_sort=summary_df.sort_values(by='mc_div_rc',ascending=False).reset_index(inplace=False,drop=True)
    return summary_df_sort

if __name__ == '__main__':
    test=[]


