
# coding: utf-8


import pandas as pd
#import pickle
import re
#from sqlalchemy import create_engine
#from sqlalchemy_utils import database_exists, create_database
#import psycopg2




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


# In[11]:

def create_documents(review_set,reviews):
    folded_set=[]
    relevant_reviews=reviews[reviews.url==review_set].review
        #print relevant_reviews
    for item in relevant_reviews:
            #print item
        folded_set.append(item)
               
    return folded_set


# In[12]:

def token_re(doc):
    from nltk.tokenize import RegexpTokenizer
    re_tokenizer = RegexpTokenizer(r'\w+')
    course_re_tokens = re_tokenizer.tokenize(doc.decode('utf-8').lower())
    
    return course_re_tokens


# In[13]:

def token_ws(doc):
    from nltk.tokenize import WhitespaceTokenizer
    ws_tokenizer = WhitespaceTokenizer()
    course_ws_tokens = ws_tokenizer.tokenize(doc.decode('utf-8').lower())
    
    return course_ws_tokens


# In[14]:

def clean_tokens(doc):
    from nltk.corpus import stopwords # Import the stop word list
    cleaned_tokens = []
    stop_words = set(stopwords.words('english'))
    for token in doc:
        if token not in stop_words:
            cleaned_tokens.append(token)
    return cleaned_tokens


# In[15]:

def remove_my_stopwords(recombined_doc):
    check=["andrew"," ng ", "course", "professor","coursera","mooc","udacity","edx","udemy"]
    r_check=[""," ", "","","","","","",""]
    # print len(check)
    #print len(r_check)
    for a, b in zip(check,r_check):
        recombined_doc=recombined_doc.replace(a,b)         
    # print len(recombined_doc)
    #print len(removed_doc)
    return recombined_doc


# In[16]:

def bagofwords(article):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()    
    article_vect = vectorizer.fit_transform([article])
    freqs = [(word, article_vect.getcol(idx).sum()) for word, idx in vectorizer.vocabulary_.items()]
    sorted_results=sorted (freqs, key = lambda x: -x[1])
    return sorted_results


# In[17]:

def calculate_sum_keywords(collect_freqs,input_words):

    rec_list=[]
    for course in collect_freqs:
        sum_words=0.
        for match_word in input_words:
            if match_word in collect_freqs[course]:
                #print match_word
                sum_words=sum_words+collect_freqs[course][match_word]
        rec_list.append((course,sum_words))
    #print rec_list[3]
    return rec_list


# In[37]:

def summarize_results(rec_list,review_text_dict,course_title_dict,course_site_dict):
    summary_df=pd.DataFrame(rec_list,columns=['url','match_word_count'])
    #summary_df["title"]=summary_df.url.apply(lambda x:len(course_title_dict[x]))
    #summary_df["company"]=summary_df.url.apply(lambda x:len(course_site_dict[x]))
    summary_df["review_count"]=summary_df.url.apply(lambda x:len(review_text_dict[x]))
    summary_df['mc_div_rc'] = summary_df.apply(lambda row:row.
                                               match_word_count/float(row.review_count),axis=1)
    summary_df_sort=summary_df.sort_values(by='mc_div_rc',ascending=False).reset_index(inplace=False,drop=True)
    return summary_df_sort

if __name__ == '__main__':
    test=[]
# In[38]:


