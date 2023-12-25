#!/usr/bin/env python
# coding: utf-8

# In[52]:


# function to save a specific question
def save_question(q,path,q_n_sn):
    import pickle
    fpath = path+q_n_sn+'\\answer.pkl'
    with open(fpath,'wb') as f:
        pickle.dump(q,f)


# In[53]:


paper_path = 'J:\\work_station\\paper_meta\\subject-1\\paper-1\\' # current paper files path. all files like 
                                                                  # meta, question.<dict, answer,idf,total, etc will be accessed from here>
classified_sw = ['a', 'about', 'again', 'am', 'an', 'and', 'are', 'as', 'at', 'be', 'because', 'been', 'being', 'between', 'both', 'but', 'by', 'can', 'did', 'do', 'does', 'doing', 'during', 'each', 'for', 'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'into', 'is', 'it', "it's", 'its', 'itself', 'just', 'me', 'my', 'myself', 'of', 'only', 'or', 'our', 'ours', 'ourselves', 'over', 'own', 'she', "she's", 'should', "should've", 'so', 'such', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 'until', 'very', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves']
negative = ['no','not','none','nobody','nothing','neither','nowhere','never','hardly','scarcely','barely','doesn\'t','isn\'t','wasn\'t',"shouldn't","wouldn't","couldn't","won't",'can\'t','don\'t']


# In[54]:


def set_paper_path(path):
    global paper_path
    paper_path = path


# In[55]:


# function to load a specific question
def load_question(path,q_n_sn):
    import pickle
    fpath = path+q_n_sn+'\\answer.pkl'
    q =None
    with open(fpath,'rb') as f:
        q = pickle.load(f)
    return q


# In[56]:


# load an answer sheet
def get_meta(path):
    fpath = path + "meta_info.txt"
    type_list = {}
    file1 = open(fpath, 'r')
    count = 0
    while True:
        # Get next line from file
        line = file1.readline()
        if not line:
            break
        count += 1
        line=line.replace('\n','')
        arr = line.split('#')
        type_list['-'.join(arr[0:3])] = arr[3:]
    file1.close()
    return type_list


# In[57]:


# save a question paper
def save_meta(q_n_list,meta_list,meta_path):
    # meta list is list of q<>[0]
    # q_n_list is question number list as '1',A or or '1',B or 2',A
    m = ''
    for i in range(len(meta_list)):
        m += 'A#'+'#'.join(q_n_list[i])+ "#" +meta_list[i][0]['type']+'#' +meta_list[i][0]['question']+'#' +meta_list[i][0]['source']+'#' +meta_list[i][0]['subject']+'#' +meta_list[i][0]['background']+'#' +meta_list[i][0]['marks']
        if('diagram' in meta_list[i][0].keys()):
            m += '#' + meta_list[i][0]['diagram']
            if(meta_list[i][0]['diagram'] == 'True'):
                m+="#" +meta_list[i][0]['count']
        else:
            m += '#False'
        m += '\n'
    
    f = open(meta_path+"\\meta_info.txt",'w')
    f.write(m)
    f.close()


# In[58]:


# save the tf_idf table


# In[59]:


def make_dictionary(path,question_name,q):
    f_name = path+question_name+"\\dict.txt"
    file1 = open(f_name, 'w')
    answer_word_list = make_answer_word_list(q)
    for i in answer_word_list:
        file1.write(i+"\n")
    file1.close()
    return


# In[60]:


def make_answer_word_list(q):
    answer_word_list = []
    qtype = q[0]['type']
    if (qtype == "exp"):
        for i in q[1:]:
            for j in i[1:]:
                temp = j.lower()
                temp = temp.replace("."," ")
                temp = temp.replace(","," ")
                temp = temp.replace(":"," ")
                temp = temp.replace("("," ")
                temp = temp.replace(")"," ")
                temp = temp.replace(";"," ")
                temp = temp.replace("/"," ")
                answer_word_list += temp.split(' ')
                
    elif(qtype == "dif"):
        heads = list(q2[1][1].keys())
        for j in heads:
            for i in q2[1:]:
                temp = i[1][j].lower()
                temp = temp.replace("."," ")
                temp = temp.replace(","," ")
                temp = temp.replace(":"," ")
                temp = temp.replace("("," ")
                temp = temp.replace(")"," ")
                temp = temp.replace(";"," ")
                temp = temp.replace("/"," ")           
                answer_word_list += temp.split(' ')
                
    answer_word_list = list(set(list(filter(lambda x: x != '', answer_word_list))))
    #print(len(answer_word_list))
    return set(answer_word_list)


# In[61]:


# get dicionary for a question
def get_dictionary(path,question_name):
    f_name = path+question_name+'\\dict.txt'
    file1 = open(f_name, 'r')
    dictionary = []
    count = 0
    while True:
        count += 1
        # Get next line from file
        line = file1.readline()
        if not line:
            break
        line=line.replace(' ','')
        line=line.replace('\n','')
        dictionary.append(line.lower())
    file1.close()
    return dictionary


# In[62]:


# get lematized and non-"stop" words only 
def lem_words_only(ls):
    punc_marks = ['.',';',':',',','"',"'",')','(',"&",'-']
    import nltk
    global classified_sw
    stopwords = classified_sw.copy()
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    lem_ls = [lemmatizer.lemmatize(w.lower()) for w in ls if (not w.lower() in punc_marks and not w.lower() in stopwords)]
    return lem_ls


# In[63]:


# remove stopwords and lemmatize the sentence
def get_purified_sentence(s):
    punc_marks = ['.',';',':',',','"',"'",')','(',"&",'-']
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    global classified_sw
    stopwords = classified_sw.copy()
    lemmatizer = WordNetLemmatizer()
    word_tokens = word_tokenize(s)
    s = ' '.join([lemmatizer.lemmatize(w.lower()) for w in word_tokens if not w.lower() in stopwords and not w.lower() in punc_marks])
    return s


# In[64]:


def get_purified_answer(q):
    punc_marks = ['.',';',':',',','"',"'",')','(',"&",'-']
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    global classified_sw
    stopwords = classified_sw.copy()
    lemmatizer = WordNetLemmatizer()
    #print(get_q_type(q))
    new_q = q.copy()
    if q[0]['type'] == 'exp':
        for i in range(len(new_q[1:])):
            for j in range(len(new_q[1+i][1:])):
                word_tokens = word_tokenize(new_q[1+i][1+j])
                new_q[1+i][1+j] = ' '.join([lemmatizer.lemmatize(w.lower()) for w in word_tokens if not w.lower() in stopwords and not w.lower() in punc_marks])
    
    if q[0]['type'] == 'dif':
        for i in range(len(new_q[1:])):
            for j in new_q[1+i][1].keys():
                word_tokens = word_tokenize(new_q[1+i][1][j])
                new_q[1+i][1][j] = ' '.join([lemmatizer.lemmatize(w.lower()) for w in word_tokens if not w.lower() in stopwords and not w.lower() in punc_marks])
   
    return new_q


# In[65]:


def make_total_dict(answer,total):
    total_dict = []
    for i in answer[1:]:
        if answer[0]['type'] == 'exp':
            for j in i[1:]:
                total_dict.append(dict.fromkeys(total, 0))
                for word in tok_lem_and_get_words_only(j):
                    total_dict[-1][word] +=1
                    
        if answer[0]['type'] == 'dif':
            for j in i[1].keys():
                total_dict.append(dict.fromkeys(total, 0))
                for word in tok_lem_and_get_words_only(i[1][j]):
                    total_dict[-1][word] +=1
    return total_dict


# In[66]:


def computeTF(wordDict, doc):
    tfDict = {}
    corpusCount = len(doc)
    for word, count in wordDict.items():
        tfDict[word] = count/float(corpusCount)
    return(tfDict)


# In[67]:


def computeTF_for_answer(total_dict, answer):
    import pandas as pd
    tf_total_dict = []

    # to go through word dictionaries sentence wise
    counter = 0
    for i in answer[1:]:
        if answer[0]['type'] == 'exp':
            for j in i[1:]:
                tf_total_dict.append(computeTF(total_dict[counter],tok_lem_and_get_words_only(j)))
                counter += 1
                
        if answer[0]['type'] == 'dif':
            for j in i[1].keys():
                tf_total_dict.append(computeTF(total_dict[counter],tok_lem_and_get_words_only(i[1][j])))
                counter += 1

    #Converting to dataframe for visualization
    tf = pd.DataFrame(tf_total_dict)
    return tf


# In[68]:


def tok_lem_and_get_words_only(s):
    punc_marks = ['.',';',':',',','"',"'",')','(',"&",'-']
    import nltk
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    from nltk.tokenize import word_tokenize
    global classified_sw
    stopwords = classified_sw.copy()
    word_tokens = word_tokenize(s)
    lem_ls = [lemmatizer.lemmatize(w.lower()) for w in word_tokens if (not w.lower() in punc_marks and not w.lower() in stopwords)]
    return lem_ls


# In[69]:


def computeIDF(docList):
    import math
    idfDict = {}
    N = len(docList)
    
    idfDict = dict.fromkeys(docList[0].keys(), 0)
    for word, val in idfDict.items():
        idfDict[word] = math.log10(N / (float(val) + 1))
        
    return(idfDict)


# In[70]:


def computeTFIDF(tfBow, idfs):
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = val*idfs[word]
    return(tfidf)


# In[71]:


def cosine(tf1, tf2):
    
    denom1 = 0
    for i in tf1.keys():
        denom1 += tf1[i]**2
        
    denom2 = 0
    for i in tf2.keys():
        denom2 += tf2[i]**2
    
    numer = 0
    for i in tf1.keys():
       # if tf1[i]!=0 and tf2[i]!=0:
        numer += tf1[i]*tf2[i]
            
    denom = (denom1**0.5)*(denom2**0.5)
    
    return numer/denom


# In[72]:


# find similarity between given sentence and existing sentence and 
def sim_of_sentences(s1, s2_index, total, idfs, idf_table):

    in_words = tok_lem_and_get_words_only(s1)
    if (len(in_words) == 0):
        return 0
    word_dict_for_tf = dict.fromkeys(total,0)

    for word in in_words:
        if word in total:
            word_dict_for_tf[word] += 1

    tf_cur = computeTF(word_dict_for_tf,in_words)
    idf_cur = computeTFIDF(tf_cur,idfs)
    
    score = cosine(idf_cur,idf_table.iloc[s2_index])
    return score


# In[73]:


# find similarity between given sentence and existing sentence and 
def sim_of_sentences_od(s1, s2, total, idfs):
    in_words1 = tok_lem_and_get_words_only(s1)
    if (len(in_words1) == 0):
        return 0
    word_dict_for_tf1 = dict.fromkeys(total,0)

    for word in in_words1:
        if word in total:
            word_dict_for_tf1[word] += 1

    tf_cur1 = computeTF(word_dict_for_tf1,in_words1)
    idf_cur1 = computeTFIDF(tf_cur1,idfs)
    
    in_words2 = tok_lem_and_get_words_only(s2)
    word_dict_for_tf2 = dict.fromkeys(total,0)

    for word in in_words2:
        if word in total:
            word_dict_for_tf2[word] += 1

    tf_cur2 = computeTF(word_dict_for_tf2,in_words2)
    idf_cur2 = computeTFIDF(tf_cur2,idfs)
    
    score = cosine(idf_cur1,idf_cur2)
    return score


# In[74]:


def make_tf_idf_model_for_question(q_n,q):
    import pandas as pd
    import math 
    
    #global current_idf_table
    #global current_total_table
    global paper_path
    # get bag of all words (temporary: without synonym exchange, with lemmatization and *with* stop-words)
    un_lem = get_dictionary(paper_path, q_n)
    total = set(lem_words_only(un_lem)) # store total somewhere
    # create an array of word dicts:
    # get all sentences before to make per-question vectorizer
    answer = q.copy()
    # make the total_dict (per sentence dictionary)
    total_dict = make_total_dict(answer,total)
    tf = computeTF_for_answer(total_dict,answer)
    idfs = computeIDF(total_dict)
    all_tfIdf_list = []
    
    for i in range(len(tf)):
        all_tfIdf_list.append(computeTFIDF(tf.iloc[i],idfs))
        
    idf_data= pd.DataFrame(all_tfIdf_list)
    
    #current_idf_table = total
    #current_idf_table = idf_data
    save_tfidf_models(paper_path,q_n, total,idf_data, idfs)
    return


# In[75]:


def save_tfidf_models(path, qn, total_dict, idftable, idfs):
    import pickle
    fpath = path+qn+'\\idftable.pkl'
    with open(fpath,'wb') as f:
        pickle.dump(idftable,f)
        
    fpath = path+qn+'\\totaldict.pkl'
    with open(fpath,'wb') as f:
        pickle.dump(total_dict,f)
        
    fpath = path+qn+'\\idfs.pkl'
    with open(fpath,'wb') as f:
        pickle.dump(idfs,f)


# In[76]:


def load_tf_idf_modules(path, q_n):
    import pickle
    fpath = path+q_n+'\\idftable.pkl'
    idfdata =None
    with open(fpath,'rb') as f:
        idfdata = pickle.load(f)
    
    fpath = path+q_n+'\\totaldict.pkl'
    totaldict =None
    with open(fpath,'rb') as f:
        totaldict = pickle.load(f)

    fpath = path+q_n+'\\idfs.pkl'
    idfs =None
    with open(fpath,'rb') as f:
        idfs = pickle.load(f)
        
    return(totaldict, idfdata, idfs)


# In[77]:


def get_scaled_coverage(score):
    if(score>0.9):
        return 1
    elif(score>0.7):
        return 0.8
    elif(score>0.5):
        return 0.6
    elif(score>0.3):
        return 0.4
    elif(score>0.1):
        return 0.2
    return 0.0


# In[78]:


def get_scale_general(marks, obn, intervals):
    while(marks>0):
        if(marks-intervals<obn):
            return marks
        else:
            marks = marks-intervals
    return 0


# In[79]:


def calc_result_for_exp(q,q_in,q_n):
    # score dictionary structure: cluster<point-name>:
    # list([[<sentenceid>, similarity_score (of multiple types), keyword score, alien word score, pol_sc_ref, pol_sc_in]
    #, sequence analysis, combined_similarity])
    
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    global paper_path
    
    #print("log : now calculating result for <q-number> "+q[0]['question'])
    
    # get this answer's word dictionary
    word_dict = lem_words_only(get_dictionary(paper_path,q_n))
    
    # first, perform id-ing of each sentence in q_in to check sentence coverage 
    # structure: dict({" is<count>": "<sentence>", coverage flag, [keywords covered], [alien-words]})
    
    input_dict = {}
    counter = 1
    for i in q_in:
        i = get_purified_sentence(i)
        temp_list = i.split(' ')
        alien_words = set(temp_list) - set(word_dict)
        keywords = set(temp_list) - set(alien_words)
        input_dict["is"+str(counter)] = [i,0,keywords,alien_words]
        counter += 1
    
    #for i in input_dict.keys():
    #    print(i,input_dict[i])
    # getting question type    
    # get new list of reference answer points
    q_ref = q.copy()
    
    # for each sentence in points, remove stopwords and perform lemmatization
    q_ref = get_purified_answer(q_ref)
    
    # get tf-idf modules
    (total_dict, idf_table, idfs) = load_tf_idf_modules(paper_path,q_n)
    
    
    # to main track of point-wise score:
    score_dictionary = {}
    #cluster count
    cc = 1
    # for traversing idf table
    sen_coun = 0
    if (q_ref[0]['type'] == "exp"):
        # for each cluster point:
        for i in q_ref[1:]:
            point = i[0][0]
            #create title of that point
            score_dictionary[point]=[[]]
            # sentence counter to keep track of particular sentence in a cluster point
            sc = 1
            for j in i[1:]:
                
                # 1) Access input answer
                # 2) Perform analysis of each sentence in the answer for this particular sentence
                # 3) Store values of analysis of each sentence according to strucuture
                # 4) Mark sentence coverage using best match / added match on stressed vectors
                
                max_sim_score = 0
                temp_sen_token_list = j.split(' ')
                s_o_s = len(temp_sen_token_list)
                num_k_w_matched=0
                alien = 100
                status_list=[]
                temp_l_o_m_s = []
                for k in input_dict.keys():
                    temp = sim_of_sentences(input_dict[k][0],sen_coun,total_dict, idfs, idf_table)
                    if(temp > max_sim_score):
                        max_sim_score = temp
                        temp_l_o_m_s = [k]
                    elif(temp == max_sim_score and temp!=0):
                        temp_l_o_m_s.append(k)
                        
                
                # now you have obtained the list of id(s) of sentence(s) which have max keyword match score temp_l_o_m_s
                best_id = None
                
                if len(temp_l_o_m_s)>0:
                    #print(cc, sen_coun, temp_l_o_m_s)
                    for k in temp_l_o_m_s:
                        # find similarity for each sentence here use input_dict[i] and q[cc][sc]

                        # temporary : find sentence with least alien word count
                        if len(input_dict[k][3])<alien :
                            alien = len(input_dict[k][3])
                            best_id = k
                            
                    if(input_dict[k][1]<get_scaled_coverage(max_sim_score)):
                        input_dict[k][1] = get_scaled_coverage(max_sim_score)
                    
                    #alien = alien/(len(input_dict[k][2])+len(input_dict[k][3]))
                    alien = [a for a in ' '.split(input_dict[best_id][0]) if a not in temp_sen_token_list]
                    alien = len(alien)/(len(input_dict[best_id][2])+len(input_dict[best_id][3]))
                    num_k_w_matched = s_o_s - len((set(temp_sen_token_list)-input_dict[best_id][2]))
                    kw_match_score = num_k_w_matched/s_o_s
                    if(kw_match_score>1):
                        kw_match_score = 1
                    score_dictionary[point][0].append(["S"+str(sc),best_id,max_sim_score,kw_match_score,alien,sid.polarity_scores(q[cc][sc]),sid.polarity_scores(input_dict[best_id][0])])
                else:
                    score_dictionary[point][0].append(["S"+str(sc),best_id,0,0,0,sid.polarity_scores(q[cc][sc]),None])
                sc += 1
                sen_coun +=1
            cc += 1
            # 1) Feed sequence status of the cluster
            # 2) Feed other similarity model scores
            score_dictionary[point].append(0.5)
            score_dictionary[point].append(0.5)

    return (score_dictionary,input_dict)


# In[80]:


def calc_result_for_dif(q,q_in,q_n):
    # score dictionary structure: cluster<point-name>:
    # list([[<sentenceid>, similarity_score (of multiple types), keyword score, alien word score, pol_sc_ref, pol_sc_in]
    #, sequence analysis, combined_similarity])
    
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    global paper_path
    
    #print("log : now calculating result for <q-number> "+q[0]['question'])
    
    # get this answer's word dictionary
    word_dict = lem_words_only(get_dictionary(paper_path,q_n))
    
    # first, perform id-ing of each sentence in q_in to check sentence coverage 
    # structure:  "ip<count>": [[lhs- sentence, coverage flag, [keywords covered], [alien-words]],
    # structure: [rhs-sentence, coverage flag, [keywords covered], [alien-words]]]
    
    input_dict = {}
    counter = 1
    keys = list(q_in.keys())
    for i in range(len(q_in[keys[0]])):
        j = q_in[keys[0]][i]
        j = get_purified_sentence(j)
        temp_list = j.split(' ')
        alien_words = set(temp_list) - set(word_dict)
        keywords = set(temp_list) - set(alien_words)
        input_dict["ip"+str(counter)] = [[j,0,keywords,alien_words]]
        
        j = q_in[keys[1]][i]
        j = get_purified_sentence(j)
        temp_list = j.split(' ')
        alien_words = set(temp_list) - set(word_dict)
        keywords = set(temp_list) - set(alien_words)
        input_dict["ip"+str(counter)].append([j,0,keywords,alien_words])
        counter += 1
        
    # print current input dict:
    #for i in input_dict.keys():
    #    print(i, input_dict[i])
        
    #return input_dict

    #for i in input_dict.keys():
    #    print(i,input_dict[i])
    # getting question type    
    # get new list of reference answer points
    # now access each point by this manner: input_dict['ip#'][0][0], input_dict['ip#'][1][0],
    # where indices represent: [point][side][data]
    #q_ref = q.copy()
    q_ref = list(q)
    # for each sentence in points, remove stopwords and perform lemmatization
    q_ref = get_purified_answer(q_ref)
    
    #print(q)
    # get tf-idf modules
    (total_dict, idf_table, idfs) =load_tf_idf_modules(paper_path,q_n)
    
    
    # to main track of point-wise score:
    score_dictionary = {}
    #cluster count
    pc = 1
    # for traversing idf table
    sen_coun = 0
    if (q_ref[0]['type'] == "dif"):
        # for each cluster point:
        for i in q_ref[1:]:
            point = i[0][0]
            #create title of that point
            score_dictionary[point]=[[]]
            # sentence counter to keep track of particular sentence in a cluster point
            sc = 1
            key1 = list(i[1].keys())[0]
            key2 = list(i[1].keys())[1]
            
            j = i[1][key1]
            #for j in i[1]:
                
                # 1) Access input answer
                # 2) Perform analysis of each sentence in the answer for this particular sentence
                # 3) Store values of analysis of each sentence according to strucuture
                # 4) Mark sentence coverage using best match / added match on stressed vectors
                
            max_sim_score = 0
            temp_sen_token_list = j.split(' ')
            s_o_s = len(temp_sen_token_list)
            num_k_w_matched=0
            alien = 100
            temp_l_o_m_s = []
            
            for k in input_dict.keys():
                temp = sim_of_sentences(input_dict[k][0][0],sen_coun,total_dict, idfs, idf_table)
                if(temp > max_sim_score):
                    max_sim_score = temp
                    temp_l_o_m_s = [k]
                elif(temp == max_sim_score and temp!=0):
                    temp_l_o_m_s.append(k)


            # now you have obtained the list of id(s) of sentence(s) which have max keyword match score temp_l_o_m_s
            best_id = None

            if len(temp_l_o_m_s)>0:
                #print(cc, sen_coun, temp_l_o_m_s)
                for k in temp_l_o_m_s:
                    # find similarity for each sentence here use input_dict[i] and q[cc][sc]

                    # temporary : find sentence with least alien word count
                    if len(input_dict[k][0][3])<alien :
                        alien = len(input_dict[k][0][3])
                        best_id = k

                if(input_dict[k][0][1]<get_scaled_coverage(max_sim_score)):
                    input_dict[k][0][1] = get_scaled_coverage(max_sim_score)

                oppo_sim = sim_of_sentences(input_dict[best_id][1][0],sen_coun+1,total_dict, idfs, idf_table)
                
                if(input_dict[k][1][1]<get_scaled_coverage(oppo_sim)):
                    input_dict[k][1][1] = get_scaled_coverage(oppo_sim)
                    
                ref_pol_score = error('compound_linear_difference',sid.polarity_scores(q[pc][1][key1]),sid.polarity_scores(q[pc][1][key2]))
                # key1 is lhs, key2 is rhs
                in_pol_score = error('compound_linear_difference',sid.polarity_scores(input_dict[best_id][1][0]),sid.polarity_scores(input_dict[best_id][0][0]))

                e_term = 0
                if(ref_pol_score>in_pol_score):
                    e_term = ref_pol_score - in_pol_score
                else:
                    e_term = -1*(ref_pol_score - in_pol_score)

               
                
                dif_of_ref = sim_of_sentences(j,sen_coun+1,total_dict, idfs, idf_table)
                dif_of_in = sim_of_sentences_od(input_dict[best_id][0][0],input_dict[best_id][1][0],total_dict,idfs)
                #dif_term = ((dif_of_in - dif_of_ref)**2)**0.25
                dif_term = dif_of_in - dif_of_ref
                #print("ref",j,i[1][key2])
                #print("in",input_dict[best_id][0][0],input_dict[best_id][1][0])
                #print("dif terms (ref,in,term) = ",dif_of_ref,dif_of_in,dif_term)
                point_score = (oppo_sim+max_sim_score)/2 - e_term - dif_term
                
                point_score = point_score if point_score>0 else 0
                # averaging alien-word score
                alien1 = [a for a in tok_lem_and_get_words_only(input_dict[best_id][0][0]) if a not in temp_sen_token_list]
                #print(alien1)
                alien1 = len(alien1)/(len(input_dict[best_id][0][2])+len(input_dict[best_id][0][3]))

                alien2 = [a for a in tok_lem_and_get_words_only(input_dict[best_id][1][0]) if a not in q[pc][1][key2].split(' ')]
                #print(alien2)
                alien2 = len(alien2)/(len(input_dict[best_id][1][2])+len(input_dict[best_id][1][3]))

                alien = (alien1 + alien2)/2

                # averaging key-word score
                num_k_w_matched1 = s_o_s - len((set(temp_sen_token_list)-input_dict[best_id][0][2]))
                kw_match_score = num_k_w_matched1/s_o_s

                num_k_w_matched2 = s_o_s - len((set(q[pc][1][key2].split(' '))-input_dict[best_id][1][2]))
                kw_match_score = (kw_match_score+num_k_w_matched1/s_o_s)/2

                if(kw_match_score>1):
                    kw_match_score = 1
                score_dictionary[point][0] = ["P"+str(pc),best_id,point_score,kw_match_score,alien,ref_pol_score, in_pol_score, e_term]#sid.polarity_scores(q[cc][sc]),sid.polarity_scores(input_dict[best_id][0])])
            else:
                score_dictionary[point][0] = ["P"+str(pc),best_id,0,0,0]
            sen_coun +=2
            pc += 1
            # 1) Feed sequence status of the cluster
            # 2) Feed other similarity model scores
            score_dictionary[point].append(0.5)
            score_dictionary[point].append(0.5)
            
    return (score_dictionary,input_dict)


# In[81]:


paper_path = "J:\\work_station\\paper_meta\\subject-1\\paper-1\\"
q1 = [ { "source" : "https://byjus.com/questions/describe-functions-of-heart/", #3
       "subject" : "Science/Biology",#4
       "background" : "Heart",#5
       "question" : "Describe the function of heart", #2
       "marks" : "6",#6
       "type" : "exp",#1
       "diagram" : "False" #7
      },
      [["point-type : introduction"],
    "Heart is the vital organ of the human body.","Heart plays a major role in our body.",
    "The heart is responsible for major functions and transportation in our body."
      ],
      [["point-type : definition"],
    "The heart is an organ that pumps blood throughout the body.",
    "The human circulatory system is responsible for the transport of materials inside the human body.",
    "The organs of the circulatory system are the heart, arteries, veins and capillaries.",
    "It comprises four chambers : Atria (upper chambers), Ventricles (lower chambers), Layers of heart."
     ],
      [["point-type : parts/layers of heart"],
    "The wall of the heart is made up of three layers: Epicardium, Myocardium, Endocardium",
    "Epicardium - This is a protective layer made of connective tissues.",
    "Myocardium - This layer forms the heart muscles.",
    "Endocardium - This is the innermost layer and protects the valves and the heart.",
    "A muscular wall known as the septum separates the two sides of the heart.",
    "The heart is encased in the pericardium that protects the heart and fastens it inside the chest.",
    "The outer and inner layers of the pericardium are filled with pericardial fluid that lubricates the heart during the contractions of the lungs and the diaphragm."
      ],
    [["point-type : functions of heart"],
    "Function of Heart",
    "The heart plays a key role in the circulation of blood and maintaining the mechanism of the whole body.",
    "It is the most vital organ of the human body.",
    "The heart performs the following important functions : ",
    "The primary function of the heart is to pump blood throughout the body.",
    "It supplies oxygen and nutrients to the tissues and removes carbon dioxide and wastes from the blood.",
    "It also helps to maintain adequate blood pressure throughout the body."
    ],
    [["point-type : functioning of heart"],
    "The heart functions in the following ways : ",
    "The arteries receive oxygenated blood from the heart and supply it throughout the body.",
    "Whereas, the veins carry the deoxygenated blood from all the body parts to the heart for oxygenation.",
    "The right atrium receives blood from the veins and pumps it to the right ventricle.",
    "The right ventricle pumps the blood received from the right atrium to the lungs.",
    "The left atrium receives oxygenated blood from the lungs and pumps it to the left ventricle.",
    "The left ventricle pumps the oxygenated blood throughout the body."
    ],
    [["point-type : improper functioning of heart"],
    "Improper functioning of the heart results in heart diseases such as angina pectoris, congestive heart failure, cardiomyopathy, cardiac arrest, etc."
    ]
]
q2 = [{ "source" : "https://byjus.com/biology/difference-between-cross-pollination-and-self-pollination/",
       "subject" : "Science/Biology",
       "background" : "Pollination",
       "question" : "Differenciate between self-pollination and cross-pollination",
       "marks" : "3",
       "type" : "dif",
      },
      [["POD : base"],
       {"Self-Pollination" : "Transfer pollen grains from the anther to the stigma of the same flower.",
       "Cross-Pollination" : "Transfer pollen grains from the anther to the stigma of a different flower."}
      ],
      [["POD : process in same/different flower"],
       {"Self-Pollination" : "This process can take place in the same flower or a different flower of the same plant.",
       "Cross-Pollination" : "This process can take place between two flowers present on different plants."}
      ],
      [["POD : genetically indentical/not"],
       {"Self-Pollination" : "It occurs in the flowers which are genetically identical.",
       "Cross-Pollination" : "It occurs between flowers which are genetically different."}
      ],
      [["POD : examples"],
       {"Self-Pollination" : "Paphiopedilum parishii, Arabidopsis thaliana",
       "Cross-Pollination" : "apples, daffodils, pumpkins and grasses"}
      ],
      [["POD : conditions in progenies"],
       {"Self-Pollination" : "Causes homogenous conditions in progenies.",
       "Cross-Pollination" : "Causes heterozygous condition in progenies."}
      ],
      [["POD : pollinator requirements"],
       {"Self-Pollination" : "No need for pollinators to transfer pollen grains.",
       "Cross-Pollination" : "Requires pollinators to transfer pollen grains."}
      ],
      [["POD : stigma and anther maturity"],
       {"Self-Pollination" : "In self-pollination, both the stigma and anther simultaneously mature",
       "Cross-Pollination" : "In cross-pollination, both the stigma and anther mature at different times."}
      ]     
     ]
q3 = [{"source" : "https://byjus.com/biology/reflex-action/",
       "subject" : "Science/Biology",
       "background" : "Reflex-Actions",
       "question" : "Explain reflex actions",
       "marks" : "4",
       "type" : "exp"
      },
    [["point-type : definition"],
    "Reflex is an involuntary and sudden response to stimuli.",
    "It happens to be an integral component of the famed survival instinct."
    ],    
    [["point-type : introduction/example"],
    "Most of the common reflexes are a response to all the well-trained, accumulated knowledge of caution that we have internalized.",
    " It could be anything and ranges from the reflex action of abruptly withdrawing the hand as it comes in contact with an extremely cold or hot object.",
    " This action is termed as the reflex action. It has a subtle relation to instinct."
    ],
    [["point-type : action of neuron and events"],
    "Two neurons dominate the pathway, afferent nerves (receptor) and the efferent nerves (effector or excitor).",
    "Below is a brief description of the events that take place : ",
    "Firstly, it begins with receptor detecting the stimulus or a sudden change in the environment, where the instinct again has a role to play.",
    "The stimulus is received from a sensory organ.",
    "Then, the sensory neuron sends a signal to the relay neuron.",
    "This is followed with the relay neuron sending the signal to the motor neuron.",
    "Further, the motor neuron sends a signal to the effector.",
    "The effector produces an instantaneous response, for example, pulling away of the hand or a knee-jerk reaction.",
    "From the above explanations, it can be clearly summarized that the moment the afferent neuron receives a signal from the sensory organ; it transmits the impulse via a dorsal nerve root into the Central Nervous System.",
    "The efferent neuron then carries the signal from the CNS to the effector.",
    "The stimulus thus forms a reflex arc.",
    "In a reflex action, the signals do not route to the brain – instead, it is directed into the synapse in the spinal cord, hence the reaction is almost instantaneous."
    ]]


# In[82]:


def dif_to_sentence_list(dif_q):
    lis = [dif_q[0]]
    for i in dif_q[1:]:
        temp = [i[0]]
        for j in i[1].keys():
            temp.append(i[1][j])
        lis.append(temp)
    return lis


# In[83]:


def get_result_sheet(method,score_dict, params, input_answer, q_ref, q_in):
    # should obtain following things from the checking
    # 1) the cluster-wise grade (cluster : grade) !
    # 2) the overall grade !
    # 3) the bag of alien words in the answer !
    # 4) the sentences in the reference answer which were not covered properly 
    # 5) the sentences in the input answer which were not applicable
    #  0.8 or 1 -> well covered. 0.6 -> average coverage. 0.4 -> missing on keywords. <0.4 -> not so well covered
    (cluster_wise,total)=get_grade(method,score_dict, params)
    bag_alien_words = []
    for i in input_answer:
        bag_alien_words += list(input_answer[i][3])
    
    sentence_undercoverd_in_ref = {}
    cc = 1
    for i in score_dict.keys():
        sentence_undercoverd_in_ref[i] = []
        for j in score_dict[i][0]:
            if (j[2]<0.5): # resetable parameter
                sentence_undercoverd_in_ref[i].append(q_ref[cc][int(j[0][1:])])
        cc +=1

    sentence_extra_in_inp = []
    for i in input_answer.keys():
        if (input_answer[i][1]<0.5): # resetable parameter
            sentence_extra_in_inp.append(q_in[int(i[2:])-1])
    
    return (cluster_wise, total, bag_alien_words, sentence_undercoverd_in_ref, sentence_extra_in_inp)


# In[84]:


#(cluster_wise, total, bag_alien_words, sentence_undercoverd_in_ref, sentence_extra_in_inp) = get_result_sheet('sen_count_average',d, params, inp, q1, s['answer'])


# In[85]:


"""for i in cluster_wise.keys():
    print("You covered : ",i," at",cluster_wise[i]*100,"% closeness to reference")
print()
print("Following alien words were observed:\n",bag_alien_words[0],end="")

for i in bag_alien_words[1:]:
    print(",",i,end="")
print()

print("Following sentences were well uncovered from p.o.v of reference ")
for i in sentence_undercoverd_in_ref:
    print(sentence_undercoverd_in_ref[i])

print()
print("Following sentences were not well accepted from input answer")
for i in sentence_extra_in_inp:
    print(i)

"""


# In[86]:


def get_grade(method,score_dict, params):
    # methods are: basic average, weighted average, functional/ function based, regression
    # params shall contains parameters like marks in params[0] and for methods other than basic average
    
    if (method == 'weighted_average'):
        cluster_wise = {}
        marks = 0
        base = params[1]
        #print(base)
        # this is amount of marks for all clusters
        # for each cluster, do this:
        # line[val_score] = (sim + kw) / (sim + kw + al) - err(polarity) # given that sim + kw + al != 0 else 0
        # for line in cluster, add this score, multiply by base, store in cluster_wise
        # finally add all marks, scale them and return
        
        for point in score_dict.keys():
            cluster_wise[point] = 0
            t_val_score = 0
            for line in score_dict[point][0]:
                if(line[2]+line[3]+line[4] != 0):
                    e = error('compound_linear_difference',line[5],line[6])
                    t = line[2]*(line[3])/(line[3]+line[4]) - e
                   # print("error =",e,"sentence-score =",t)
                    if(t>0):
                        t_val_score += t
            t_val_score /= len(score_dict[point][0])
            cluster_wise[point] = base[point] * t_val_score
            marks +=cluster_wise[point]
        #print(marks)
        return cluster_wise, get_scale_general(params[0],marks,0.5)

    if (method == 'basic_average2'):
        cluster_wise = {}
        marks = 0
        base = params[0]/len(score_dict.keys())
        # this is amount of marks for all clusters
        # for each cluster, do this:
        # line[val_score] = sim * (kw) / (kw + al) - err(polarity) # given that  kw + al != 0 else 0
        # for line in cluster, add this score, multiply by base, store in cluster_wise
        # finally add all marks, scale them and return
        
        for point in score_dict.keys():
            cluster_wise[point] = 0
            t_val_score = 0
            for line in score_dict[point][0]:
                if(line[3]+line[4] != 0):
                    e = error('compound_linear_difference',line[5],line[6])
                    t = line[2]*(line[3])/(line[3]+line[4]) - e
                 #   print("error =",e,"sentence-score =",t)
                    if(t>0):
                        t_val_score += t
                    #print("for",line[0],"in",point,t_val_score)
            t_val_score /= len(score_dict[point][0])
           # print("(point t-val score)",point,t_val_score)
            cluster_wise[point] = base * t_val_score
            marks +=cluster_wise[point]
        #print(marks)
        return cluster_wise, get_scale_general(params[0],marks,0.5)
    
    if (method == 'sen_count_average'):
        cluster_wise = {}
        marks = 0
        base = {}
        t_len = 0
        for point in score_dict.keys():
            t_len +=len(score_dict[point][0]) 
            base[point]=params[0]*len(score_dict[point][0])
            
        for point in base.keys():
            base[point] /=t_len
            
        #print("here",base)
        # this is amount of marks for all clusters
        # for each cluster, do this:
        # line[val_score] = sim * (kw) / (kw + al) - err(polarity) # given that  kw + al != 0 else 0
        # for line in cluster, add this score, multiply by base, store in cluster_wise
        # finally add all marks, scale them and return
        
        for point in score_dict.keys():
            cluster_wise[point] = 0
            t_val_score = 0
            for line in score_dict[point][0]:
                if(line[3]+line[4] != 0):
                    e = error('compound_linear_difference',line[5],line[6])
                    t = line[2]*(line[3])/(line[3]+line[4]) - e
                    #print("error =",e,"sentence-score =",t)
                    if(t>0):
                        t_val_score += t
                    #print("for",line[0],"in",point,t_val_score)
            t_val_score /= len(score_dict[point][0])
            #print("(point t-val score)",point,t_val_score)
            cluster_wise[point] = base[point] * t_val_score
            marks +=cluster_wise[point]
        #print(marks)
        return cluster_wise, get_scale_general(params[0],marks,0.5)
    
    if (method == 'sen_count_average_lenient'):
        cluster_wise = {}
        marks = 0
        base = {}
        t_len = 0
        for point in score_dict.keys():
            t_len +=len(score_dict[point][0]) 
            base[point]=params[0]*len(score_dict[point][0])
            
        for point in base.keys():
            base[point] /=t_len
            
        #print("here",base)
        # this is amount of marks for all clusters
        # for each cluster, do this:
        # line[val_score] = max(sim, (kw) / (kw + al))
        # for line in cluster, add this score, multiply by base, store in cluster_wise
        # finally add all marks, scale them and return
        
        for point in score_dict.keys():
            cluster_wise[point] = 0
            t_val_score = 0
            for line in score_dict[point][0]:
                if(line[3]+line[4] != 0):
                    e = error('compound_linear_difference',line[5],line[6])
                    t = max(line[2],(line[3])/(line[3]+line[4])) - e
                    #print("error =",e,"sentence-score =",t)
                    if(t>0):
                        t_val_score += t
                    #print("for",line[0],"in",point,t_val_score)
            t_val_score /= len(score_dict[point][0])
            #print("(point t-val score)",point,t_val_score)
            cluster_wise[point] = t_val_score
            marks +=cluster_wise[point]*base[point]
        #print(marks)
        return cluster_wise, get_scale_general(params[0],marks,0.5)
    


# In[87]:


"""
mark_list = [0.5,1,1,1,2,0.5]
t = 0
mark_dict = {}
for i in d.keys():
    mark_dict[i] = mark_list[t]
    t += 1
params = [6,mark_dict]
#get_result_sheet('basic_average2',d,params,inp,q1,s)
"""


# In[88]:


s = {'question': 'A-1-A',
 'qtype': 'exp',
 'answer': ['The heart is an organ which beats continuously a ',
  'to act as a for the transport OF pump blood , which carries other substance i with it ',
  '6 ',
  'The human heart is divided into four chambers ',
  'The upper two chambers are called right and left atrium and the lower two chambers are called the right e left ventricles ',
  'Right atrium ',
  'receives carbon dioxide i rich ',
  'blood from the body ',
  'blood from right atrium enters the right ventricle which contracts and pumps ',
  'blood to the lungs ',
  ') oxygen i rich blood fram the lungs returns to the left atrium ',
  'From the left atrium , blood enters left ventricle ',
  'left ventricle contracts and pumps blood to ball parks DE body ',
  ') In this way , the rhythmic contraction and expansion of various chambers of the heart main trains ',
  'the transport of oxygen to all parts of the body '],
 'un_id': [[['PAGE_SWITCH',
    1,
    'J:\\work_station\\cache\\pg_1_DVFZJ6J1622209304611489100.jpg']]]}
t = [{'question': 'A-1-B', 'qtype': 'dif', 
      'answer': {'Self i Pollination': 
                 ['Self pollination occurs when the pollen from the anther full on the sigma of the same flower , or another flower of the same plant ', 
                  'Self pollination does not lead to any genetic diversity '], 
                 'Cross i Pollination': 
                 ['Cross pollination occurs when the pollen from the anther falls on the stigma of a different flower of CA a different species ', 
                  'Cross pollination leads to genetic diversity ']}, 
      'un_id': []}]


# In[89]:


#from copy import deepcopy as clone
#qq = clone(q2)


# In[90]:


#(d,i) = calc_result_for_dif(qq, t[0]['answer'], 'A-1-B')
#d


# In[91]:


#get_result_sheet('sen_count_average_lenient',d, [6], i, qq , s['answer'])
#get_result_sheet_dif(d, i, qq, 3)


# In[92]:


def error(type_of_e, pol_in, pol_ref):
    # type_of_e determines type of error to be addressed
    if(type_of_e == 'compound_linear_difference'):
        dif = pol_in['compound'] - pol_ref['compound']
        if dif >0:
            return dif/2
        elif dif<0:
            return -1*dif/2
        else: 
            return 0


# In[93]:


#dp = {'question':'A-1-B','type':'dif','answer':{'lhs':['Transfer pollen grains from the anther to the stigma of the same flower.','This process can take place between two flowers present on different plants.'],'rhs':['Transfer pollen grains from the anther to the stigma of the different flower.','anther']}}
#calc_result_for_dif(q2, inp, 'A-1-B')


# In[94]:


def get_result_sheet_dif(score_dict, input_answer, q_ref, marks):
    
    # get point wise t_val score
    point_wise = {}
    m = 0
    not_covered_list = []
    d = score_dict
    for i in score_dict.keys():
        #print(i)
        #print(score_dict[i][0][0][1:])
    #print(d[i])
        tval = 0
        if(d[i][0][3]+d[i][0][4] != 0):
            #tval = d[i][0][2]*d[i][0][3]/(d[i][0][3]+d[i][0][4])
            tval = max(d[i][0][2],d[i][0][3]/(d[i][0][3]+d[i][0][4]))
        point_wise[i] = tval
        if(tval<0.5):
            not_covered_list.append(' XOX '.join(list(q_ref[int(score_dict[i][0][0][1:])][1].values())))
        m += tval
    #print("tval",tval)
    #print()
    m = m*3/len(score_dict)
    m = get_scale_general(marks,m,0.5)
    
    bag_alien_words = []
    for i in input_answer:
        bag_alien_words += list(input_answer[i][0][3]) + list(input_answer[i][1][3])
        
    return(point_wise, m, bag_alien_words, not_covered_list)
    
    


# In[95]:


def start_engine(params):
    # param 0 should contain path to paper
    
    set_paper_path(params[0])
    global paper_path
    
    meta_list = get_meta(paper_path)
    # metalist is a dict with keys as q_nums, and values as list with indices 
    #0 = type, 1 = Q name, 2 = source, 3 = subject, 4 = Background, 5 = marks, 6 = diagram ...
    
    # inflate the quesitons dict. q_name : question
    # use meta_list and load question for this task
    model_answer = {}
    for i in meta_list.keys():
        model_answer[i]=load_question(paper_path,i)
        
    model_answer_data = {}
    for i in meta_list.keys():
        (totaldict, idfdata, idfs) = load_tf_idf_modules(paper_path,i)
        model_answer_data[i] = {}
        model_answer_data[i]['total-dict'] = totaldict
        model_answer_data[i]['idf-data'] = idfdata
        model_answer_data[i]['idfs'] = idfs
    
    global meta_A
    global model_A
    global model_A_data
    
    meta_A = meta_list
    model_A = model_answer
    model_A_data = model_answer_data


# In[103]:


#params=["J:\\work_station\\paper_meta\\subject-1\\paper-1\\"]


# In[97]:


def get_complete_grading(i_dif_list, i_exp_list, params, marking_scheme):
    type_of_scheme = ['basic_average2','weighted_average','sen_count_average']
    from copy import deepcopy as clone
    global meta_A
    global model_A
    global model_answer_data
    # first compute differentiate answers
    dif_result_list = {}
    for i in i_dif_list:
        q_cur = clone(model_A[i['question']])
        (d, inp) = calc_result_for_dif(q_cur, i['answer'], 'A-1-B')
        (point_wise, m, bag_alien_words, not_covered_list) =  get_result_sheet_dif(d, inp,q_cur,params[i['question']][0])
        temp = {}
        temp['point_wise'] = point_wise
        temp['obtained_marks'] = m
        temp['alien_words'] = bag_alien_words
        temp['failed_to_address_properly'] = not_covered_list
        dif_result_list[i['question']] = temp
    
    # second, calculate expand answers
    exp_result_list = {}
    for i in i_exp_list:
        q_cur = clone(model_A[i['question']])
        d, inp = calc_result_for_exp(q_cur,i['answer'],i['question'])
        (cw, marks, alien_ws, sen_uc_in_ref, sen_ex_in_inp) = get_result_sheet(marking_scheme,d, params[i['question']], inp, q_cur ,i['answer'])
        temp = {}
        temp['cluster_wise'] = cw
        temp['obtained_marks'] = marks
        temp['alien_words'] = alien_ws
        temp['failed_to_address_properly'] = sen_uc_in_ref
        temp['failed_to_present_properly'] = sen_ex_in_inp
        exp_result_list[i['question']] = temp
        
    return (dif_result_list,exp_result_list)
        


# In[98]:


#get_complete_grading([dp],[s],params,'sen_count_average')


# In[99]:


"""
params = {
    'A-1-A':[6,[0.5,1,1,1,2,0.5]],
    'A-1-B':[3],
    'A-1-C':[4,[0.5,1,2.5]]
}
"""


# In[108]:


#bm = [[{'question': 'A-1-B', 'qtype': 'dif', 'answer': {'Self i Pollination': ['Self pollination occurs when the pollen from the anther full on the sigma of the same flower , or another flower of the same plant ', 'Self pollination does not lead to any genetic diversity '], 'Cross i Pollination': ['Cross pollination occurs when the pollen from the anther falls on the stigma of a different flower of CA a different species ', 'Cross pollination leads to genetic diversity ']}, 'un_id': []}], [{'question': 'A-1-A', 'qtype': 'exp', 'answer': ['a ', 'The human heart is a pump which collect i impure blood from the rest of the body and purified it for refusing ', ') The process of purification in wolves filling the blood with oxygen from the lungs ', 'The veins in the body bring the impure blood back to the heart ', 'The right side of the heart pumps it into the lungs through the pulmonary arteries ', 'There the blood receives oxygen anew ', ') The oxygenated blood travels back to the left side of the heart through the pulmonary vein and the heart pumps it out to the rest of the body through the Aorta , i either chief artery , ', 'The whole body is a network of arteries and wins which are all branch es of these chief less els to ensure that blood reaches all parts of the body ', 'The small est of these vessels are called capillaries '], 'un_id': [[['Page', {'x': 3362.0, 'y': 147.0}, [[3260, 107], [3463, 103], [3464, 187], [3261, 191]]], ['No', {'x': 3568.5, 'y': 143.5}, [[3508, 115], [3628, 113], [3629, 172], [3509, 174]]]]]}, {'question': 'A-1-C', 'qtype': 'exp', 'answer': ['Human reflexes are a result of the neural network of the brain and the news system ', 'Every part of the body is connected to the Spinal cord through news , and the spinal cord is connected to the brain ', 'every ime any sensation is felt , it is sent to the brain to be in interpreted in electrical signals passed through the neural network ', "The brain's response is sent bock the same way at the point so that the appropriate action can be taken ", 'However , though this process is extremely it is shill too slow to react to dangerous situ actions ', 'Hence if danger quick , is perceived , the spinal cord itself puts the appropriate muses in motion to prevent any damage ', 'This is reflex action '], 'un_id': [[['PAGE_SWITCH', 1, 'J:\\work_station\\cache\\pg_1_90THF0K1622973162857069100.jpg']]]}], []]

