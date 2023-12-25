#!/usr/bin/env python
# coding: utf-8

# In[1]:


# all imports should be added here
from __future__ import print_function
from google.cloud import vision
import io
import time


# In[2]:


# defining some runtime global variables:
dict_path = ''
img_paths = []                        # path of images of the current paper being checked
a_w_s = 0                             # average word size
a_w_s_cache = []                      # utility variable for average word size
english_vocab = []                    # dictionary of "all" english words in vocab for flagging
common_words = []                     # list of common words in english
dict_answers = []                     # list of words to particular answers
log_ws = 'J:\\work_station\\logs'     #
diagram_paths = []                    # paths to diagrams wherever detected
cur_roll = ''                         # hold current roll_number
# make a function which gets list of image paths / pdf path
# (1) from pdf path open pdf and distribute it to images OR
# (2) from images path, open all images


# In[3]:


def kw_filtered(kw):
    return ''.join([c for c in kw if c.isalpha()])


# In[69]:


def write_lw_log(line_wise,roll,path):
    log_l = ''
    f_path = path + roll + '.txt'
    for i in line_wise:
        for j in i:
            log_l += j[0].encode('ascii','ignore').decode() +' '
        log_l += '\n'
        
    f = open(f_path, "w")
    f.write(log_l)
    f.close()
    


# In[5]:


def log(date,fn_name,statement, time):
    # make a new file or get existing file
    import csv
    import os
    from datetime import datetime
    global log_ws
    fname = datetime.today().strftime('%Y-%m-%d') + '.csv'
    path = log_ws+'\\'+ fname
    if not os.path.exists(path):
        with open(path, 'w', newline='') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(['Time-stamp', 'Function', 'Statement','Time'])
            filewriter.writerow([date,  fn_name, statement, time])
    else:
        with open(path, 'a+', newline='') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow([date,  fn_name, statement, time])


# In[6]:


# get time in micro seconds
def get_time():
    import time
    return round(time.time()*1000000)


# In[7]:


# get current time-stamp
def get_timeStamp(): 
    from datetime import datetime
    return datetime.today().strftime('%Y-%m-%d %H:%M:%S')


# In[8]:


# load a dictionary of a particular question
def load_dict(path):
    s_time = get_time()
    global a_w_s_cache
    dictionary=[]
    file1 = open(path, 'r')
    count = 0
    while True:
        count += 1
        # Get next line from file
        line = file1.readline()
        if not line:
            break
        line=line.replace(' ','')
        line=line.replace('\n','')
        dictionary.append(line)
        if len(line)>2 :
            a_w_s_cache.append(len(line))
    file1.close()
    e_time = get_time()
    log(get_timeStamp(),"load_dict: path = "+path," loaded dictionary successfully ",e_time-s_time)
    return dictionary


# In[9]:


def load_dict_general(path):
    s_time = get_time()
    dictionary=[]
    file1 = open(path, 'r')
    while True:
        # Get next line from file
        line = file1.readline()
        if not line:
            break
        line=line.replace(' ','')
        line=line.replace('\n','')
        dictionary.append(line)
    file1.close()
    e_time = get_time()
    log(get_timeStamp(),"load_dict: path = "+path," loaded dictionary successfully ",e_time-s_time)
    return dictionary


# In[10]:


def round(x):
    y = int(x)
    z = x - y 
    if z>0.499999999:
        return y + 1
    else:
        return y


# In[11]:


# get answer wise dictionary
def complete_dictionary(ls_of_fl_nms):
    global a_w_s_cache
    global a_w_s
    s_time = get_time()
    total_dict_for_answers = {}
    for i in ls_of_fl_nms.keys():
        # get name of file for that answer name = get_name(<name__of_answer>)
        temp = load_dict(ls_of_fl_nms[i])
        total_dict_for_answers[i] = temp
    a_w_s = round(float(sum(a_w_s_cache)/len(a_w_s_cache)))
    
    e_time = get_time()
    log(get_timeStamp(),"complete_dictionary: list of file names = "+str(' # '.join([i for i in ls_of_fl_nms.values()]))," loaded complete dictionary successfully with avg(word_size) "+str(a_w_s),e_time-s_time)
    return total_dict_for_answers


# In[12]:


# resizes the image to given dimension
def resize(img_paths, dim):
    import cv2
    for path in img_paths:
        image = cv2.imread(path, 1)
        bigger = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        cv2.imwrite(path, bigger)


# In[13]:


# generate a random_file name such that file_name = joiner + <rand-string>
def random_file_name(joiner):
    ranstr = ''
    import time
    ranstr = str(time.time_ns())
    import string    
    import random # define the random module  
    S = 7  # number of characters in the string.  
    ranstr = joiner+''.join(random.choices(string.ascii_uppercase + string.digits, k = S)) + ranstr  
    return ranstr


# In[14]:


# converts pdf to separate image
def convert(path, activity_area):
    # path = path of pdf file
    # activity area = directory from where we'll obtain on exam page images 
    s_time = get_time()
    from pdf2image import convert_from_path
    images = convert_from_path(path)
    global img_paths
    for i in range(len(images)):
          # Save pages as images in the pdf
        path = activity_area +"\\" + random_file_name('pg_'+str(i+1)+"_") +'.jpg'
        images[i].save(path, 'JPEG')
        img_paths.append(path)
    
    #resize(img_paths,(1482,int(1482/0.75)))
    e_time = get_time()
    log(get_timeStamp(),"convert: path = "+path," distributed pdf into images successfully with file names "+' # '.join([i for i in img_paths]),e_time-s_time)
    return img_paths


# In[15]:


# real OCR operation
def detect_text(path):
    s_time = get_time()
    """Detects text in the file."""
    from google.cloud import vision
    import io
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    count = 0
    
    for text in texts:
        #print('\n"{}"'.format(text.description))

        vertices = (['({},{})'.format(vertex.x, vertex.y)
                    for vertex in text.bounding_poly.vertices])

        #print('bounds: {}'.format(','.join(vertices)))
        count += 1
    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    e_time = get_time()
    log(get_timeStamp(),"detect_text: path = "+path," recieved texts from google cloud-vision ",e_time-s_time)
    return texts


# In[16]:


"""
FUNCTION DESCRIPTION:
INPUT: TEXTS -> LIST OF TYPE RESPONSE ANNOTATIONS (AS PROVIDED BY GOOGLE CLOUD)
OUTPUT: TUPLE OF FOLLOWING VALUES:
    1) MINIMUM OF TEXT BOX HEIGHT (USEFUL FOR ARRANGING AND SORTING TEXTS) FLOAT
    2) MAXIMUM OF TEXT BOX HEIGHT --------------||------------- FLOAT
    3) AVERAGE OF TEXT BOX HEIGHT (USEFUL FOR FONT-SIZE OF WORDS) FLOAT
    4) AVERAGE CHAR WIDTH (TO DETERMINE SPACE PRESENT IN UNCONDITIONALLY JOINED WORDS) FLOAT
    5) MARGIN START (TO SEPARATE MARGIN FROM ANSWER - PAGE) FLOAT
    6) LINES (LIST OF LINES SORTED AS PROVIDED BY 'TEXTS') DICTIONARY OF STRINGS
    7) WORD ID ON LINE (LIST OF WORD IDS ON LINE) DICTIONARY OF LIST OF IDS
    8) WORD TO CENTROID (WORD TO CENTROID MAPPTING) USED FOR CENTROIDAL SORTING
    
"""
def get_metric_vals(texts,pg_num):
    s_time = get_time()
    min_text_box_height = 99999
    max_text_box_height = 0
    avg_text_box_height = 0
    average_char_width = 0
    total_chars = 0
    count = 0
    margin = 99999
    Lines = {}
    word_to_centroid = []
    word_id_on_line={}
    total_words = len(texts) - 1
    line_number = 1
    accumulate = ''
    ids = []
    for text in texts:
        if count == 0 : 
            #skip
            pass
        else:
            # get bounding box first
            bb = [[vertex.x, vertex.y] for vertex in text.bounding_poly.vertices]
            
            # text box height
            # bb -> 0, 1, 2, 3 are topleft topright bottomright bottomleft and [0] -> x [0] -> y
            #calculating height:
            h = (bb[2][1]+bb[3][1])/2 - (bb[1][1]+bb[0][1])/2
            min_text_box_height = min(min_text_box_height, h)
            max_text_box_height = max(h, max_text_box_height)
            avg_text_box_height += h
            #calculating char_width:
            total_chars += len(text.description)
            average_char_width += ((bb[1][0]+bb[2][0])/2 - (bb[0][0]+bb[3][0])/2)
            #calculating left margin
            margin = min(margin,(bb[0][0]+bb[3][0])/2)
            
            # Word to centroid (for proper sort) 
            centroid_x = (bb[0][0]+bb[1][0]+bb[2][0]+bb[3][0])/4
            centroid_y = (bb[0][1]+bb[1][1]+bb[2][1]+bb[3][1])/4
            word_to_centroid.append([text.description,{"x":centroid_x,"y":centroid_y},bb])
            # words to line conversion
            if(count<total_words):
                word_base_y = (bb[2][1]+bb[3][1])/2
                next_bb = [[vertex.x, vertex.y] for vertex in texts[count+1].bounding_poly.vertices]
                next_word_base_y = (next_bb[2][1]+next_bb[3][1])/2
                if((word_base_y+10)>next_word_base_y):
                    accumulate = accumulate + text.description +' '
                    ids.append(count)
                   #print("in new word", accumulate)
                else:
                    accumulate += text.description
                    ids.append(count)
                    Lines[str('line'+str(line_number))] = accumulate
                    accumulate = ''
                    word_id_on_line[str('line'+str(line_number))] = ids
                    ids = []
                    #print("in new line")
                    line_number +=1
            if(count==total_words):
                accumulate += text.description
                ids.append(count)
                Lines[str('line'+str(line_number))] = accumulate
                accumulate = ''
                word_id_on_line[str('line'+str(line_number))] = ids
                ids = []
        count += 1
    e_time = get_time()
    log(get_timeStamp(),"get_metrics_vals: pg_num = "+str(pg_num)," received centroids and avg word height successfully, avg word height="+str(avg_text_box_height/total_words),e_time-s_time)
    return (min_text_box_height, max_text_box_height, avg_text_box_height/total_words, average_char_width/total_chars, margin-3, Lines, word_id_on_line, word_to_centroid)


# In[17]:


def search(ls, key):
    #print('search called')
    count = 0
    for i in ls:
        if i[0] == key:
            return count 
        count += 1
    return -1


# In[18]:


# arrange each word in the OCR output
def arrange_by_centroid(h, c):
    # height wise sorting
    s_time = get_time()
    for i in range(len(h)):
        for j in range(i+1, len(h)):
            if h[i][1]['y']>h[j][1]['y']:
                temp = h[i]
                h[i] = h[j]
                h[j] = temp
                
    # line wise partitioning
    line_wise = []
    temp_list = []
    bbx = []
    current_mark = h[0][1]['y']
    i = 0
    count = 1
    while i<len(h):
        temp_list.append(h[i])
        if(i<len(h)-1):
            if(h[i+1][1]['y']>(current_mark+c*15/23)):
                line_wise.append(temp_list)
                temp_list=[]
                current_mark=h[i+1][1]['y']
        i += 1
    line_wise.append(temp_list)
    # word wise sorting
    for i in range(len(line_wise)):
        for j in range(len(line_wise[i])):
            for k in range(j+1,len(line_wise[i])):
                if line_wise[i][j][1]['x']>line_wise[i][k][1]['x']:
                    temp = line_wise[i][j]
                    line_wise[i][j] = line_wise[i][k]
                    line_wise[i][k] = temp
    e_time = get_time()
    log(get_timeStamp(),"arrange_by_centroid: word_height(c) = "+str(c)," sorted successfully, received "+str(len(line_wise))+" lines",e_time-s_time)
    return line_wise


# In[19]:


def parse_expansion(line_wise,type_dict):
    import cv2
    global diagram_paths
    global cur_roll
    diagram_paths = []
    s_time = get_time()
    # keywords are <Identifier> <d> <a>, START, CONT, END, DIG
    Id = 'A'
    dict_of_answers = []
    # before each type, check dif or exp
    stack = []       # to add and cancel expansion rules
    chase = [Id]     # start with chasing a question number
    start = False    # set 'start'to false before parsing a question
    type_ = None     
    take = False     # take flag indicates take next line if not keyword
    blob = []        # blob holds current answer until start-end satisfied
    un_id = []       # hold unidentified strings
    cur_ques = ''    # current question-number
    # for diagram
    cur_t_l_x = None
    cur_t_l_y = None
    cur_b_r_y = None
    # cur b_r_x will be common
    cur_b_r_x = 1000
    img = ''
    im_count = 0
    di_count = 0
    global img_paths
    current_page = img_paths[0] 
    kws = [Id, 'START', 'CONT', 'END', 'DIG']
    for ln in line_wise:
        # ln is a list of all words on that line
        # access word as ln[i][0] or wrd[0]
        # for the first time, you're searching for Id
        # what is first time? stack empty, start is false.
        
        cur_key = ln[0][0]
        if kw_filtered(cur_key) in chase:
            # current key is one of keywords
            cur_key = kw_filtered(cur_key)
            if not start and cur_key in chase: # indicating no answer in feed yet plus no answer recording and id detected
                stack.append(cur_key) 
                chase = ['START','CONT']
                cur_key = kw_filtered(cur_key)
                start = True
                #q_num = ln[1][0]
                #q_s_n = ln[2][0]
                cur_ques = '-'.join([ln[i][0] for i in range(len(ln))])
                #cur_ques = Id+'-'+q_num+'-'+q_s_n
                # according to the logic, one iteration of the loop shall end here
            elif start and cur_key in chase:
                stack.append(cur_key)
                if stack[-1] in ['CONT','START']: # if answer starts on next line
                    chase = ['DIG','END']
                    take = True
                elif stack[-1] in ['DIG']: # if diagram starts on next line
                    img = cv2.imread(current_page)
                    cur_b_r_x = img.shape[1] - 10
                    cur_t_l_x = ln[0][2][0][0] # left  most x value for DIG bb
                    cur_t_l_y = ln[0][2][2][1] # bottom most y value for DIG bb
                    chase = ['END']
                    take = False
                elif stack[-1] in ['END']: # if answer or diagram ends at next line
                    if stack[-2] in ['DIG']: # if diagram 
                        stack.pop()
                        stack.pop()
                        cur_b_r_y = ln[0][2][0][1] # top most y value for END bb
                        crop_img = img[cur_t_l_y:cur_b_r_y, cur_t_l_x:cur_b_r_x]
                        crop_img = cv2.bitwise_not(crop_img)
                        #cv2.imshow('crop',crop_img)
                        #cv2.waitKey(0)
                        #cv2.destroyAllWindows()
                        import os
                        if not os.path.isdir("J:\\work_station\\cache\\"+cur_ques):
                            os.mkdir("J:\\work_station\\cache\\"+cur_ques)
                        f_name = "J:\\work_station\\cache\\"+cur_ques+"\\"+str(di_count)+'_'+cur_roll+'.'+current_page.split('.')[1]
                        di_count += 1
                        diagram_paths.append(f_name)
                        cv2.imwrite(f_name, crop_img)
                        blob.append([["IMAGEFLAG"]])
                        # image = crop_current_page(cur_t_l_x,cur_t_l_y, cur_b_r_x, cur_b_r_y)
                        # save(image, random_file_name())
                        # dig_list.append(random_file_name())
                        chase = ['END','DIG']
                        take = True
                    elif stack[-2] in ['CONT', 'START']: # if answer
                        stack.pop()
                        stack.pop()
                        stack.pop()
                        chase = [Id]
                        take = False
                        start = False
                        qtype = None
                        if cur_ques in type_dict.keys():
                            qtype = type_dict[cur_ques]
                        dict_of_answers.append({'question':cur_ques,'qtype':qtype,'answer':blob,'un_id':un_id})
                        blob = []
                        un_id = []
                        
        else: # useful line:
            if take and ln[0][0]!='PAGE_SWITCH':
                blob.append(ln)
            else:
                if ln[0][0]=='PAGE_SWITCH':
                    if(im_count+1<len(img_paths)):
                        im_count += 1
                    current_page = img_paths[im_count]
                un_id.append(ln)
    e_time = get_time()
    log(get_timeStamp(),"parse_expansion: number of lines = "+str(len(line_wise))," converted to dictionary of anwers successfully with num(answers) = "+str(len(dict_of_answers)),e_time-s_time)
    return dict_of_answers


# In[20]:


# lastest parse dif version
def parse_dif_after_exp(line_wise):
    s_time = get_time()
    # keywords are <Identifier> <d> <a>, START, CONT, END, NC
    # before each type, check dif or exp
    stack = []       # to add and cancel expansion rules
    chase = ['NC']     # start with chasing a question number
    type_ = None     
    take = True     # take flag indicates take next line if not keyword
    rhs_blob = []    # blob holds rhs column until start-end satisfied
    lhs_blob = []    # blob holds lhs column until start-end satisfied
    num_blob = []    # blob holds num column until start-end satisfied
    un_id = []       # hold unidentified strings
    cur_ques = ''    # current question-number
    kws = ['NC']
    rb = 0
    lb = 0
    
    kw_index = 0
    for ln in line_wise:
        # ln is a list of all words on that line
        # access word as ln[i][0] or wrd[0]
        # for the first time, you're searching for Id
        # what is first time? stack empty, start is false.

        kw_index = search(ln, 'NC')
        cur_key = ln[kw_index][0]
        #print("current_key = "+cur_key)
        #print("keyword index = ",kw_index)
        if cur_key in kws:
            # current key is one of keywords
            if cur_key in chase:
                take = True
                rb = ln[kw_index][2][1][0] + 45 
                lb = ln[kw_index][2][0][0] - 45 
                # get right words and left words of NC, add them as title
                temp_r = []
                temp_l = []
                temp_n = []
                for word in ln[0:kw_index]:
                    temp_l.append(word)
                for word in ln[kw_index+1:]:
                    temp_r.append(word)
                temp_n.append(ln[kw_index])
                rhs_blob.append(temp_r)
                lhs_blob.append(temp_l)
                num_blob.append(temp_n)
                        
        else: # useful line:
            if take:
                temp_r = []
                temp_l = []
                temp_n = []
                for word in ln:
                    if word[2][1][0]<lb:
                        temp_l.append(word)
                    elif word[2][0][0]>rb:
                        temp_r.append(word)
                    else:
                        temp_n.append(word)
                rhs_blob.append(temp_r)
                lhs_blob.append(temp_l)
                num_blob.append(temp_n)
                # rhs_blob.append(all words in ln whose left-most x > rb )
                # lhs_blob.append(all words in ln whose right-most x < lb )
                # num_blob.append(all entities in ln whose left-most x > lb and right-most x < rb )
            else:
                un_id.append(ln)
    e_time = get_time()
    log(get_timeStamp(),"parse_dif_after_exp","parsed differenciation successfully",e_time-s_time)
    return [lhs_blob,num_blob,rhs_blob, un_id]


# In[21]:


def remove_noise_and_smooth(file_name):
    s_time = get_time()
    import cv2
    import numpy as np
    img = cv2.imread(file_name, 0)
    filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 41)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    #img = image_smoothening(img)
    or_image = cv2.bitwise_or(img, closing)
    #
    #kernel = np.zeros((5,5),np.uint8)
    kernel = np.ones((2,2),np.uint8)
    erosion = cv2.erode(filtered,kernel,iterations =1)
    cv2.imwrite("J:\\cache\\1s.jpg", erosion)
    e_time = get_time()
    log(get_timeStamp(),"remove_noise_and_smooth: from image "+file_name," smoothened the image successfully ",e_time-s_time)
    #cv2.imshow('get',erosion)
    #cv2.waitKey()
    #return or_image


# In[22]:


def denoise_and_invert(path):
    s_time = get_time()
    import cv2
    import numpy as np
    im = cv2.imread(path)
    morph = im.copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    # split the gradient image into channels
    image_channels = np.split(np.asarray(morph), 3, axis=2)

    channel_height, channel_width, _ = image_channels[0].shape

    # apply Otsu threshold to each channel
    for i in range(0, 3):
        _, image_channels[i] = cv2.threshold(~image_channels[i], 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
        image_channels[i] = np.reshape(image_channels[i], newshape=(channel_height, channel_width, 1))

    image_channels = np.concatenate((image_channels[0], image_channels[1], image_channels[2]), axis=2)

    cv2.imwrite(path, image_channels)
    e_time = get_time()
    log(get_timeStamp(),"denoise_and_invert: from image "+path," Smoothened and inverted the image successfully ",e_time-s_time)


# In[23]:


# join two wprds and return appropriate match(es) for them from dictionary 
def joinifcorrect_Re(base,current,dictionary):
    import re
    search = '^'
    for word in base:
        if word != '':
            search +=  word.lower() + '+[a-z]*'
    if current.lower()!='':
        search += current.lower() + '+[a-z]*'
    #print(search)
    filter_= []
    for word in dictionary:
        if re.search(search, word):
            filter_.append(word)
    return filter_


# In[24]:


def removebad(string):
    import re
    res = "".join(re.findall("[a-zA-Z0-9\'\"().,:&;]+", string))
    return res


# In[25]:


def seperate(word, avg_s, dictionary):
    # list of word set, likely ness score
    word_set = []
    # left to right breakage, check in dictionary
    for i in range(1,len(word)):
        score = 0
        if word[i:] in dictionary:
            score += 1
        if word[:i] in dictionary:
            score += 2
        if score == 3:
            word_set.append([word[:i],word[i:],2])
        elif score == 2 or score == 1:
            word_set.append([word[:i],word[i:],1])
    # sort by score and least ratio
    
    # score
    
    score1word_set = [a for a in word_set if a[2]==1]
    score2word_set = [a for a in word_set if a[2]==2]
    
    # for simplicity, select score 1 word set only if score 2 word set is empty
    wordset = score1word_set if len(score2word_set) == 0 else score2word_set
    scoreset = []
    
    # get scores of word ratio, select ratio closest to one
    for i in range(len(wordset)):
        scoreset.append(float(min(len(wordset[i][0])/len(wordset[i][1]),len(wordset[i][1])/len(wordset[i][0]))))
    
    if len(scoreset)>0:
        ind = scoreset.index(max(scoreset))
        return word_set[ind]
    else:
        return [word, 0]


# In[26]:


def get_total_english_vocab():
    import nltk
    global english_vocab
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())


# In[27]:


def flag_generator_1_filter(dictionary,words_line_by_line):
    s_time = get_time()
    incor_flags = []    
    punctuation = [',','"',"'", '.',':','?','-', '&',';','/',')','(']
    global english_vocab
    exclude = ['b','c','d','e','f','g','h','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    counter = 0
    for line in words_line_by_line:
        temp = []
        for word in line:
            if (word.lower() not in english_vocab or word.lower() in exclude) and word not in punctuation and word.lower() not in dictionary:
                temp.append(1)
                counter += 1
            else:
                temp.append(0)
        incor_flags.append(temp)
    e_time = get_time()
    log(get_timeStamp(),"flag_generator_1_filter: "," generated flags, with "+str(counter)+" words incorrect ",e_time-s_time)
    return incor_flags


# In[28]:


def flag_generator_1_filter_m(dictionary,words_line_by_line):
    s_time = get_time()
    incor_flags = []    
    punctuation = [',','"',"'", '.',':','?','-', '&',';','/',')','(','IMAGEFLAG']
    global english_vocab
    exclude = ['b','c','d','e','f','g','h','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    counter = 0
    for line in words_line_by_line:
        temp = []
        for word in line:
            if (word.lower() in english_vocab and word.lower() not in exclude) or word in punctuation or word.lower() in dictionary:
                temp.append(0)
                counter += 0
            else:
                temp.append(1)
                counter += 1
        incor_flags.append(temp)
    e_time = get_time()
    log(get_timeStamp(),"flag_generator_1_filter: "," generated flags, with "+str(counter)+" words incorrect ",e_time-s_time)
    return incor_flags


# In[29]:


# pass 2 and 4 of 2
# find best replacement of a word and fix it on spot
# better version can be implemented using spell.candidates() rather than # spell.correct
def corrector_of_2_filter(words_line_by_line, dictionary, incor_flags):
    s_time = get_time()
    from spellchecker import SpellChecker
    spellchecker = SpellChecker()
    from autocorrect import Speller
    spell = Speller('en')
    incor = incor_flags.copy()
    new_answer = words_line_by_line.copy()
    correction = 0
    for i in range(len(new_answer)):
        for j in range(len(new_answer[i])):
            cur = new_answer[i][j]
            score = [0,0]
            temp_possible = [cur, cur]
            if incor[i][j] == 1 and ")" not in cur:
                misspelled = spellchecker.unknown([cur])
                if len(misspelled) == 1:
                    temp_possible[0] = spellchecker.correction(list(misspelled)[0])
                    if temp_possible[0] in dictionary:
                        score[0] = 2
                    else:
                        score[0] = 1
                auto = spell(cur)
                if auto != cur:
                    temp_possible[1] = auto
                    if temp_possible[1] in dictionary:
                        score[1] = 2
                    else:
                        score[1] = 1
                        
                # case 1: words not corrected or corrected i.e. scores are same and possibles are same
                if temp_possible[0]==temp_possible[1] and score[0]==score[1]:
                    new_answer[i][j] = temp_possible[0]
                    if new_answer[i][j] != cur:
                        incor[i][j] = 0
                        correction += 1
                
                # case 2: words are not same,scores are not same i.e. one of the word has higher correctness
                elif score[0]!=score[1]:
                    a = 1 if score[1]>score[0] else 0
                    new_answer[i][j] = temp_possible[a]
                    incor[i][j] = 0
                    correction += 1
                
                # case 3: scores are same, words are not same
                elif temp_possible[0]!=temp_possible[1]:
                    dif0 = abs(len(cur)-len(temp_possible[0]))
                    dif1 = abs(len(cur)-len(temp_possible[1]))
                    rep = ''
                    if dif1>=dif0: # give priority to SpellChecker
                        rep = temp_possible[0]
                    else:
                        rep = temp_possible[1]
                    new_answer[i][j] = rep
                    incor[i][j] = 0
                    correction += 1
    #print(str(correction)+" words corrected")
    e_time = get_time()
    log(get_timeStamp(),"corrector_of_2_filter: ","  corrected "+str(correction)+" word(s) ",e_time-s_time)
    return new_answer


# In[30]:


# pass 1 of 2 
def separator_of_2_filter(words_line_by_line, dictionary, incor_flags,aws):
    incor = incor_flags.copy()
    reference = words_line_by_line.copy()
    new_answer = []
    correction = 0
    for i in range(len(reference)):
        temp = []
        for j in range(len(reference[i])):
            cur = reference[i][j]
            if incor[i][j] == 1 and ")" not in cur and len(cur)>aws:
                #print('Now correcting: ',cur)
                correction = seperate(cur,aws,dictionary)[:-1]
                temp += correction
                #print('Corrected to: ',correction,'\n')
            else: 
                temp += [cur]
        new_answer.append(temp)
    return new_answer


# In[31]:


# pass 3 of 2
def joiner_of_2_filter(words_line_by_line, dictionary, incor_flags):
    s_time = get_time()
    incor = incor_flags.copy()
    reference = words_line_by_line.copy()
    new_answer = []
    correction = 0
    for i in range(len(reference)):
        temp = []
        base = ['']
        flt_list = []
        sack = []
        get = False
        
        for j in range(len(reference[i])):
            
            cur = removebad(reference[i][j])
            if (incor[i][j] == 1 or get) and ")" not in cur and cur!='' :
                #print('Now correcting: ',cur)
                
                # if this is not the first part-of-word in joining use filtered dictionary, else use dictionary
                if get:
                    flt_list = joinifcorrect_Re(base,cur,flt_list)
                else :
                    flt_list = joinifcorrect_Re(base,cur,dictionary)
                
                # if the filtered list has some element which satisfies all the parts, then keep searching until next flt_list is empty
                if len(flt_list)>0:
                    sack = flt_list
                    base += [cur.lower()]
                    get = True
                else:
                # we now least possible matches for all parts in base, they can even be zero. 
                    # find the sum of length of all part in base
                    # length could also be zero
                    
                    coverage = [len(a) for a in base]
                    coverage = sum(coverage)
                    
                    # get is True i.e flt_list is not empty, then instantiate max_coverage_word with base ('','<part1>','<part2>'...)
                    # else set m_c_word to current word
                    
                    if get:
                        m_c_word = base
                    else:
                        m_c_word = [cur]
                    # temporary score% storage variable
                    weight = 0
                    
                    # if the sack has any words, then choose the best one, set m_c_word to [best_word, cur]
                    for word in sack:
                        score = float(coverage/len(word))
                        if weight < score and score >= 0.75 :
                            weight = score
                            #if incor[i][j] == 1:
                            #    m_c_word = [word]
                            #else:
                            m_c_word = [word]
                    
                    # remove '' is there's any
                    m_c_word = [a for a in m_c_word if a != '']
                    
                    # if the current word is correct and it is not included to be added rn, append it
                    if incor[i][j] == 0 and cur not in m_c_word:
                        temp += m_c_word + [cur]
                    else:
                        temp += m_c_word
                    
                    # if there was joining, joining has ended now. joining is possible with next word too, so keep cur in base
                    # else base is empty
                    if get and incor[i][j] == 1:
                        base = [cur.lower()]
                    else:
                        base = ['']
                    
                    # if current word is incorrect, and get is False (m_c_word is cur itself) else, initiate flt_list with possibilities of cur
                    if incor[i][j] == 1:
                        if m_c_word == [cur]:
                            get = False
                        else:
                            get = True
                            flt_list = joinifcorrect_Re([''],cur,dictionary)
                    else:
                        get = False
                        flt_list = []
                    sack = []
            else: 
                temp += [cur]
                base = ['']

        if (len(base)>0):
            final_append = [a for a in base if a!=""]
            if len(sack)==0:
                temp+=final_append
            else:
                coverage = [len(a) for a in final_append]
                coverage = sum(coverage)
                weight = 0
                m_c_word = final_append
                for word in sack:
                    score = float(coverage/len(word))
                    if weight < score and score >= 0.75 :
                        weight = score
                        m_c_word = [word]
                temp+=m_c_word
        new_answer.append(temp)
        
    e_time = get_time()
    log(get_timeStamp(),"joiner_of_2_filter: ","  joined words wherever possible ",e_time-s_time)
    return new_answer


# In[32]:


def get_sentences_exp(new_answer):
    sentences = []
    temp = ''
    for i in new_answer:
        for j in i:
            if (j in ['.',':',";"] or ')' in j) and len(temp)>0:
                sentences.append(temp)
                temp = ''
            else:
                temp += j + ' '
    return sentences


# In[33]:


def to_list_of_tokens(elements):
    List_of_all_lines_words=[]
    for i in elements:
        temp=[]
        for j in i:
            temp.append(j[0])
        List_of_all_lines_words.append(temp)

    final_classified_list=[]
    for i in range(len(List_of_all_lines_words)):
        for j in range(len(List_of_all_lines_words[i])):
            if(':' in List_of_all_lines_words[i][j] and len(List_of_all_lines_words[i][j])>1):
                temp= List_of_all_lines_words[i][j].replace(':',' : ')
                List_of_all_lines_words[i][j] = temp
                
            if(':' in List_of_all_lines_words[i][j] and len(List_of_all_lines_words[i][j])>1):
                temp= List_of_all_lines_words[i][j].replace(';',' ; ')
                List_of_all_lines_words[i][j] = temp

            if('&' in List_of_all_lines_words[i][j] and len(List_of_all_lines_words[i][j])>1):
                temp= List_of_all_lines_words[i][j].replace(' & ',' and ')
                List_of_all_lines_words[i][j] = temp

            if('_' in List_of_all_lines_words[i][j] and len(List_of_all_lines_words[i][j])>1):
                temp= List_of_all_lines_words[i][j].replace('_',' ')
                List_of_all_lines_words[i][j] = temp

            if('-' in List_of_all_lines_words[i][j] and len(List_of_all_lines_words[i][j])>1):
                temp= List_of_all_lines_words[i][j].replace('-',' - ')
                List_of_all_lines_words[i][j] = temp

            if('.' in List_of_all_lines_words[i][j] and len(List_of_all_lines_words[i][j])>1):
                temp= List_of_all_lines_words[i][j].replace('.',' . ')
                List_of_all_lines_words[i][j] = temp

            if(',' in List_of_all_lines_words[i][j] and len(List_of_all_lines_words[i][j])>1):
                temp= List_of_all_lines_words[i][j].replace(',',' , ')
                List_of_all_lines_words[i][j] = temp
            
            if('"' in List_of_all_lines_words[i][j] and len(List_of_all_lines_words[i][j])>1):
                temp= List_of_all_lines_words[i][j].replace('"',' " ')
                List_of_all_lines_words[i][j] = temp
                
            if('/' in List_of_all_lines_words[i][j] and len(List_of_all_lines_words[i][j])>1):
                temp= List_of_all_lines_words[i][j].replace('/', ' / ')
                List_of_all_lines_words[i][j] = temp
            
            if('(' in List_of_all_lines_words[i][j] and len(List_of_all_lines_words[i][j])>1):
                temp= List_of_all_lines_words[i][j].replace('(',' ( ')
                List_of_all_lines_words[i][j] = temp
            
            if(')' in List_of_all_lines_words[i][j] and len(List_of_all_lines_words[i][j])>1):
                temp= List_of_all_lines_words[i][j].replace(')',' ) ')
                List_of_all_lines_words[i][j] = temp
                
        one_liner=' '.join(List_of_all_lines_words[i])
        List_of_all_lines_words[i] =[a for a in one_liner.split(' ') if a!='']
    return List_of_all_lines_words


# In[34]:


# convert images to line_wise
def images_to_line_wise(img_paths):
    pg_wise_text = []
    for path in img_paths:
        pg_wise_text.append(detect_text(path))
        
    ctrd_text = [] # centroid text 
    fo_size = []   # font size
    counter = 0    # to keep page count
    for text in pg_wise_text:
        counter += 1
        (a,b,c,d,e,f,g,h) = get_metric_vals(text,counter)
        page_flag = [['PAGE_SWITCH',counter,img_paths[counter-1]]]
        h += page_flag
        ctrd_text.append(h)
        fo_size.append(c)
    
    line_wise = []
    for i in range(len(ctrd_text)):
        page_flag = ctrd_text[i][-1]
        line_wise += arrange_by_centroid(ctrd_text[i][:-1],fo_size[i])
        line_wise += [[page_flag]]

    return line_wise


# In[35]:


def filteration(tokens,dictionary):
    global english_vocab
    global common_words
    dictionary = list(set(dictionary + common_words))
    incor_flags = flag_generator_1_filter_m(dictionary,tokens)
    """
    print('before all')
    for i in range(len(tokens)):
        print(tokens[i])
        print(incor_flags[i])  
    """
    new_answer = joiner_of_2_filter(tokens, dictionary, incor_flags)
    incor_flags = flag_generator_1_filter_m(dictionary,new_answer)

    new_answer = corrector_of_2_filter(new_answer, dictionary, incor_flags)
    incor_flags = flag_generator_1_filter_m(dictionary,new_answer)

    new_answer = separator_of_2_filter(new_answer,dictionary,incor_flags,a_w_s)
    incor_flags = flag_generator_1_filter_m(dictionary,new_answer)

    new_answer = corrector_of_2_filter(new_answer, dictionary, incor_flags)
    return new_answer


# In[36]:


def load_all_dict(com_path,base_p_path,type_list):
    global english_vocab
    global common_words
    global dict_answers
    get_total_english_vocab()
    common_words = []
    dict_answers = []
    common_words = load_dict_general(com_path)
    new_dict = []
    allowed = 'am an as at be by do go he if in is it me my no of on or so to up us we'.split(' ')
    for word in common_words:
        if(len(word)<=2):
            if(word.lower() in allowed or word.lower() in ['a','i'] ):
                new_dict.append(word.lower())
        else:
            new_dict.append(word.lower())
    common_words = new_dict
    # generate file names for particular dicts:
    ls_of_fnames = {}
    for i in type_list.keys():
        ls_of_fnames[i] = base_p_path+str(i)+'\\dict.txt'
    dict_answers = complete_dictionary(ls_of_fnames)


# In[37]:


def get_sentences_exp(new_answer):
    sentences = []
    temp = ''
    for i in new_answer:
        for j in i:
            if(j == 'IMAGEFLAG'):
                if(len(temp)>0):
                    sentences.append(temp)
                sentences.append('IMAGEFLAG')
                temp = ''
            elif (j in ['.',':',";",] or ')' in j) and len(temp)>0:
                sentences.append(temp)
                temp = ''
            else:
                temp += j + ' '
    if temp!='':
        sentences.append(temp)
    return sentences


# In[38]:


def get_sentences_dif(new_answer): # method using full_stops
    answer = {}
    lhs = ' '.join(new_answer[0][0])
    rhs = ' '.join(new_answer[2][0])
    answer[lhs]=[]
    answer[rhs]=[]    
    temp = ''
    for i in new_answer[0][1:]:
        for j in i:
            if (j in ['.'] and len(temp)>0):
                answer[lhs].append(temp)
                temp = ''
            else:
                temp += j + ' '
        if j  == [] and temp!='':
            answer[lhs].append(temp)
            temp = ''
    if temp!='':
        answer[lhs].append(temp)
        
    temp= ''
    for i in new_answer[2][1:]:
        for j in i:
            if (j in ['.'] and len(temp)>0):
                answer[rhs].append(temp)
                temp = ''
            else:
                temp += j + ' '
        if j  == [] and temp!='':
            answer[rhs].append(temp)
            temp = '' 
    if temp!='':
        answer[rhs].append(temp)
        
    return answer


# In[39]:


def get_sentences_dif_method_2(new_answer): # method using NC column
    answer = {}
    lhs = ' '.join(new_answer[0])
    rhs = ' '.join(new_answer[2])
    answer[lhs]=[]
    answer[rhs]=[]    
    temp_lhs = ''
    temp_rhs = ''
    for i in range(1,len(new_answer[1])):
        if len(new_answer[1][i])>0 and temp_lhs != '':
            answer[lhs].append(temp_lhs)
            answer[rhs].append(temp_rhs)
            temp_lhs = ' '.join(new_answer[0][i])
            temp_rhs = ' '.join(new_answer[2][i])
        else:
            temp_lhs += ' '.join(new_answer[0][i])
            temp_rhs += ' '.join(new_answer[2][i])
    
    if temp_lhs!= '':
        answer[lhs].append(temp_lhs)
        answer[rhs].append(temp_rhs)
    return answer


# In[72]:


# process of getting answer:
# function that takes parameter as pdf !
# convert pdf to images, store images in cache !
# refine images i.e remove noise wherever possible 
# send images to google_cloud_vision, get texts for all n pages !
# get metrics for all n pages !
# get line_wise conversion for all n pages and "Give output to a huge text box in an html before parsing answers" (Ignore diagrams) 
# Parse line_wise line by line !
# Parse diffs !
# Get punctuation separated text for each answer !
# perform filtering for each answer ! 
# get answer points (sentence separated)
# store answers in a list
# generate pdf with diagrams
def get_final_document(path, params):
    # let params[0] have path to activity area (cache)
    # let params[1] have path to question paper meta_info
    # let params[2] have base paper path
    # let params[3] have path to common_words list
    # let params[4] have roll number of student
    
    s_time = get_time()
    global img_paths 
    global dict_answers
    global cur_roll 
    cur_roll = params[4]
    img_paths = []
    img_paths = convert(path, params[0])
    #for path in img_paths:
    #    denoise_and_invert(path)
    base_paper_path = params[2]
    """
    may perform noise removal as required
    """
    print(img_paths,flush = True)
    line_wise =  images_to_line_wise(img_paths)
    write_lw_log(line_wise,params[4],"J:\\work_station\\student_papers\\")
    typeList = get_q_meta_and_generate_typeList(params[1])

    dict_of_answers = parse_expansion(line_wise,typeList)
    dict_of_exps = [a for a in dict_of_answers if a['qtype'] == 'exp' ]
    dict_of_difs = [a for a in dict_of_answers if a['qtype'] == 'dif' ]
    dict_of_None = [a for a in dict_of_answers if a['qtype'] ==  None ]

    sizes = "( "+str(len(dict_of_None))+" , "+str(len(dict_of_difs))+" , "+str(len(dict_of_exps))+" )"
    # parse difs to their separate arrays
    for i in range(len(dict_of_difs)):
        dict_of_difs[i]['answer'] = parse_dif_after_exp(dict_of_difs[i]['answer'])

    # convert to punctuation separated sentences
    for i in range(len(dict_of_exps)):
        dict_of_exps[i]['answer'] = to_list_of_tokens(dict_of_exps[i]['answer'])
    
    for i in range(len(dict_of_difs)):
        for j in range(len(dict_of_difs[i]['answer'])):
            if j!=1:
                dict_of_difs[i]['answer'][j] = to_list_of_tokens(dict_of_difs[i]['answer'][j])
                
    # generate dictionaries
    load_all_dict(params[3],params[2],typeList)

    # filter each expansion 
    for i in range(len(dict_of_exps)):
        q = dict_of_exps[i]['question']
        dict_of_exps[i]['answer'] = filteration(dict_of_exps[i]['answer'],dict_answers[q])
 
    # filter each differenciation 
    for i in range(len(dict_of_difs)):
        q = dict_of_difs[i]['question']
        for j in range(len(dict_of_difs[i]['answer'])):
            if j!=1:
                dict_of_difs[i]['answer'][j] = filteration(dict_of_difs[i]['answer'][j],dict_answers[q])
    # get sentences of each expansion
    for i in range(len(dict_of_exps)):
        dict_of_exps[i]['answer'] = get_sentences_exp(dict_of_exps[i]['answer'])
    
    # get sentence for each differenciation method 1
    for i in range(len(dict_of_difs)):
        dict_of_difs[i]['answer'] = get_sentences_dif(dict_of_difs[i]['answer'])
    
    e_time = get_time()
    log(get_timeStamp(),"In main function... scanned paper for "+params[4],  "Total "+str(len(img_paths))+" page(s)",e_time-s_time)
    
    return [dict_of_difs,dict_of_exps,dict_of_None]


# In[72]:


# process of getting answer:
# function that takes parameter as pdf !
# convert pdf to images, store images in cache !
# refine images i.e remove noise wherever possible 
# send images to google_cloud_vision, get texts for all n pages !
# get metrics for all n pages !
# get line_wise conversion for all n pages and "Give output to a huge text box in an html before parsing answers" (Ignore diagrams) 
# Parse line_wise line by line !
# Parse diffs !
# Get punctuation separated text for each answer !
# perform filtering for each answer ! 
# get answer points (sentence separated)
# store answers in a list
# generate pdf with diagrams
def get_final_document_w_filter(path, params):
    # let params[0] have path to activity area (cache)
    # let params[1] have path to question paper meta_info
    # let params[2] have base paper path
    # let params[3] have path to common_words list
    # let params[4] have roll number of student
    
    s_time = get_time()
    global img_paths 
    global dict_answers
    global cur_roll 
    cur_roll = params[4]
    img_paths = []
    img_paths = convert(path, params[0])
    for path in img_paths:
        denoise_and_invert(path)
    base_paper_path = params[2]
    """
    may perform noise removal as required
    """
    print(img_paths,flush = True)
    line_wise =  images_to_line_wise(img_paths)
    write_lw_log(line_wise,params[4]+"_wf","J:\\work_station\\student_papers\\")
    typeList = get_q_meta_and_generate_typeList(params[1])

    dict_of_answers = parse_expansion(line_wise,typeList)
    dict_of_exps = [a for a in dict_of_answers if a['qtype'] == 'exp' ]
    dict_of_difs = [a for a in dict_of_answers if a['qtype'] == 'dif' ]
    dict_of_None = [a for a in dict_of_answers if a['qtype'] ==  None ]

    sizes = "( "+str(len(dict_of_None))+" , "+str(len(dict_of_difs))+" , "+str(len(dict_of_exps))+" )"

    # parse difs to their separate arrays
    for i in range(len(dict_of_difs)):
        dict_of_difs[i]['answer'] = parse_dif_after_exp(dict_of_difs[i]['answer'])

    # convert to punctuation separated sentences
    for i in range(len(dict_of_exps)):
        dict_of_exps[i]['answer'] = to_list_of_tokens(dict_of_exps[i]['answer'])
    
    for i in range(len(dict_of_difs)):
        for j in range(len(dict_of_difs[i]['answer'])):
            if j!=1:
                dict_of_difs[i]['answer'][j] = to_list_of_tokens(dict_of_difs[i]['answer'][j])
                
    # generate dictionaries
    load_all_dict(params[3],params[2],typeList)

    # filter each expansion 
    for i in range(len(dict_of_exps)):
        q = dict_of_exps[i]['question']
        dict_of_exps[i]['answer'] = filteration(dict_of_exps[i]['answer'],dict_answers[q])
 
    # filter each differenciation 
    for i in range(len(dict_of_difs)):
        q = dict_of_difs[i]['question']
        for j in range(len(dict_of_difs[i]['answer'])):
            if j!=1:
                dict_of_difs[i]['answer'][j] = filteration(dict_of_difs[i]['answer'][j],dict_answers[q])
    # get sentences of each expansion
    for i in range(len(dict_of_exps)):
        dict_of_exps[i]['answer'] = get_sentences_exp(dict_of_exps[i]['answer'])
    
    # get sentence for each differenciation method 1
    for i in range(len(dict_of_difs)):
        dict_of_difs[i]['answer'] = get_sentences_dif(dict_of_difs[i]['answer'])
    
    e_time = get_time()
    log(get_timeStamp(),"In main function with filter... scanned paper for "+params[4],  "Total "+str(len(img_paths))+" page(s)",e_time-s_time)
    return [dict_of_difs,dict_of_exps,dict_of_None]


# In[45]:


param = [
'J:\\work_station\\cache',
'J:\\work_station\\paper_meta\\subject-1\\paper-1\\meta_info.txt',
'J:\\work_station\\paper_meta\\subject-1\\paper-1\\',
'J:\\work_station\\files\\common_dictionaries\\1000_common_words.txt',
'12345'
]


# In[73]:


#get_final_document('J:\\work_station\\student_papers\\12345.pdf',params)


# In[1]:


def get_q_meta_and_generate_typeList(path):
    type_list = {}
    file1 = open(path, 'r')
    count = 0
    while True:
        # Get next line from file
        line = file1.readline()
        if not line:
            break
        count += 1
        line=line.replace('\n','')
        arr = line.split('#')
        type_list['-'.join(arr[0:3])] = arr[3]
    file1.close()
    return type_list


# In[ ]:




