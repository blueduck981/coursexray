

collect_docs=[]
collect_freqs={}
review_text_dict={}
#course_title_dict={}
#course_site_dict={}

for course in documents:
    folded_set=create_documents(course,reviews)
    #print len(folded_set)
    
    review_text_dict[course]=folded_set
    #course_title_dict[course]=course_from_sql[course_from_sql.url==course].coursename
    #course_site_dict[course]=course_from_sql[course_from_sql.url==course].company

    raw_doc='-'.join(folded_set)
    
    course_re_tokens=token_re(raw_doc)
    cleaned_tokens=clean_tokens(course_re_tokens)
    #print len(cleaned_tokens)
    recombined_doc=( " ".join( cleaned_tokens ))   
    
    removed_mywords=remove_my_stopwords(recombined_doc)
    
    collect_docs.append(removed_mywords)
    if len(removed_mywords)>0:
        sorted_freq_list=bagofwords(removed_mywords)
        sorted_freq_dict=dict(sorted_freq_list)
        
        collect_freqs[course]=sorted_freq_dict
        
#print len(collect_docs)
    


# In[29]:

#len(review_text_dict[list(review_text_dict)[1]])


# In[30]:

#listN(collect_freqs)
#print type(collect_freqs[20])


# In[39]:

keywords=["programming","python","introductory","excellent","difficult"]


# In[40]:

rec_list=calculate_sum_keywords(collect_freqs,keywords)


# In[41]:

summary_df_sort=summarize_results(rec_list,review_text_dict,course_title_dict,course_site_dict)


# In[53]:

#summary_df_sort#.company[0]
summary_df_sort['title'] = summary_df_sort.url.apply(lambda x:df2.loc[x,'coursename'])
summary_df_sort['company'] = summary_df_sort.url.apply(lambda x:df2.loc[x,'company'])

summary_df_sort


# In[52]:

print summary_df_sort[0:1]


# In[ ]:

#summary_df_sort.url[len(summary_df_sort.url)-1]


# In[ ]:

#sorted(rec_list,key=lambda x:x[1],reverse=True)


# In[ ]:

