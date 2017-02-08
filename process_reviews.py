
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


# In[11]:
#REMOVE ME
def create_documents(review_set,reviews):
    folded_set=[]
    relevant_reviews=reviews[reviews.url==review_set].review
        #print relevant_reviews
    for item in relevant_reviews:
            #print item
        folded_set.append(item)
               
    return folded_set


def find_and_combine_reviews(review_set,reviews):
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
    folded_desc=[]
    for item in description:
        folded_desc.append(item)
    return folded_desc    

# In[12]:
#lowercase, take out punctuation, decode
def token_re(doc):
    re_tokenizer = RegexpTokenizer(r'\w+')
    course_re_tokens = re_tokenizer.tokenize(doc.decode('utf-8').lower())    
    return course_re_tokens

# In[13]:
def token_ws(doc):
    ws_tokenizer = WhitespaceTokenizer()
    course_ws_tokens = ws_tokenizer.tokenize(doc.lower())    
    return course_ws_tokens

# In[14]:

def clean_tokens(doc):
    #from nltk.corpus import stopwords # Import the stop word list
    cleaned_tokens = []
    stop_words = set(stopwords.words('english'))
    for token in doc:
        if token not in stop_words:
            cleaned_tokens.append(token)
    return cleaned_tokens

def find_duration_and_number_recent_reviews(documents):
    durations=[]
    cut_num_revs=[]
    for ii,course in enumerate(documents):
        relevant_times=reviews[reviews.url==course].rev_time
        if len(relevant_times)>3:
            durations.append(max(relevant_times)-min(relevant_times))
            cut_num_revs.append(course_from_sql.num_rev[ii])
    return durations, cut_num_revs


def find_combine_partial_documents(course,reviews):
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

# In[15]:
#REMOVE ME
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


def preprocess_text(folded_set,folded_desc): 

    #hold all reviews as big paragraph
    raw_desc='-'.join(folded_desc)    
    raw_doc='-'.join(folded_set)
    #print raw_doc
    
    #all words in all reviews tokenized.
    course_re_tokens=token_re(raw_doc)    
    desc_re_tokens=token_re(raw_desc)
    
    check=["andrew"," ng ", "professor","coursera","mooc","udacity","edx","udemy"]
    stop_words = set(stopwords.words('english'))

    mystopped_tokens = [i for i in course_re_tokens if not i in check]
    normal_stopped_tokens = [i for i in mystopped_tokens if not i in stop_words]
    stopped_desc_tokens=[i for i in desc_re_tokens if not i in stop_words]    
    #print normal_stopped_tokens
    #   STEM??    
    stemmed_reviews = [p_stemmer.stem(i) for i in normal_stopped_tokens]
    stemmed_desc = [p_stemmer.stem(i) for i in stopped_desc_tokens]

    return stemmed_reviews, stemmed_desc


# In[16]:
#REMOVE ME
def bagofwords(article):
    vectorizer = CountVectorizer()    
    article_vect = vectorizer.fit_transform([article])
    freqs = [(word, article_vect.getcol(idx).sum()) for word, idx in vectorizer.vocabulary_.items()]
    sorted_results=sorted (freqs, key = lambda x: -x[1])
    return sorted_results


# In[17]:
#REMOVE ME
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


