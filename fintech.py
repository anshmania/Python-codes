 # -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 10:46:47 2016

@author: anshmania
"""

import nltk
import re
from nltk.tokenize import word_tokenize
#from nltk.book import *
from pymongo import MongoClient
from datetime import date
from collections import defaultdict, namedtuple
from nltk import tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from elasticsearch import Elasticsearch
import geoip2.database
from user_agents import parse
from criticality_compliance import sentiment as senti
import criticality_compliance as cnc
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder,TrigramCollocationFinder
from nltk.metrics import TrigramAssocMeasures ,BigramAssocMeasures
import matplotlib.pyplot as plt
import math
from scipy.special import expit
from nltk import timex

#################################################################
################### Mongo DB creation ###########################
#################################################################

#Mongo Database creation.
client=MongoClient()
#client.test.drop_collection('chatSessions')
#
##Create a dummy chat database
#i=0
#j=0
#text=[]
#for i in range(250,115000,250):
#        for j in range(j,i):
#            text.append(str(text4[j]))
#        j=i
#        client.test.chatSessions.insert_one(
#                                                {"conversationId":j/250,
#                                                "date":str(date.today()),
#                                                "text":text})
#        text=[]

# Read from the GeoIp database
#city_db=geoip2.database.Reader('C:/Users/anshmania/codes/python/GeoLite2-City.mmdb')
#country_db=geoip2.database.Reader('C:/Users/anshmania/codes/python/GeoLite2-Country.mmdb')

keyWords=['sale','username','password','usr','pass','buy','price','try', 'policy','would', 'new','will','like','purchase','insurance','id']
keywords1=['haywire','problematic','problem','doesnt work', 'work']

#####################################################################
################# Information Extraction ############################
#####################################################################

# Long words extraction
def extractLongWords(conversationId):
    longWords=defaultdict(list)
    conversation= client.test.chatSessions.find({'conversationId':conversationId})  #cursor object
    for data in conversation:
        text=set(data['text'])
        longWords=[word for word in text if len(word)>11]
    return longWords

# High frequency words of conversation
def highFrequencyWords(conversationId):
    highFreq=defaultdict(list)
    conversation= client.test.chatSessions.find({'conversationId':conversationId})  #cursor object
    for data in conversation:
        textFreq=nltk.FreqDist(data['text'])
        highFreq=[word for word in textFreq.keys() if textFreq[word]>2 and len(word) >4]
    return highFreq

# Chunk the noun phrases
def chunkedNNP(conversationId):
    #chunkNNP=defaultdict(list)
    conversation= client.test.chatSessions.find({'conversationId':conversationId})  #cursor object
    for data in conversation:
        text=' '.join(data['text'])
        token_text=nltk.word_tokenize(text)
        tagged_text=nltk.pos_tag(token_text)
        grammar="NP: {<DT><JJ><NN>}"
        #Loca="LOCATION: {<IN>?<JJ>*<NN>}"
        parser=nltk.RegexpParser(grammar)
        #loc_parser=nltk.RegexpParser(Loca)
        tree = parser.parse(tagged_text)
        return tree

def chunker(conversationId):
    conversation= client.test.chatSessions.find({'conversationId':conversationId})  #cursor object
    for data in conversation:
        text=' '.join(data['text'])
        token_text=nltk.word_tokenize(text)
        tagged_text=nltk.pos_tag(token_text)
        return nltk.ne_chunk(tagged_text)

# Extract named entities from unstructured text
labelss=defaultdict(list)
leafs=[]
def namedEntities(conversationId):
    for item in chunker(conversationId):
        if isinstance(item,nltk.tree.Tree):
            for tuples in item.leaves():
                labelss.setdefault(item.label(),[]).append(tuples[0])
    return labelss

# Time-Date information.
timeData=defaultdict(list)
def dateTimeEntities(conversationId):
    conversation= client.test.chatSessions.find({'conversationId':conversationId})  #cursor object
    for data in conversation:
        tag_data=timex.tag(' '.join(data['text']))
    return tag_data


#########################################################################
################# Criticality Index Determination #######################
#########################################################################

# Frequency distribution of keywords in conversations
def keywordFreqAnalyser(conversationId):
    freqData=defaultdict(list)
    conversation= client.test.chatSessions.find({'conversationId':conversationId})
    for data in conversation:
        freqDist=nltk.FreqDist(data['text'])
        for i,word  in enumerate(freqDist.iterkeys()):
            if word in keyWords:
                freqData.setdefault(data['conversationId'],[]).append({word:freqDist[word]})
    return freqData[conversationId]

#########################################################################
# Sentiment Analysis of chat sessions
def sentimentAnalyser(conversationId):
    sentences=[]
    sentimentData=defaultdict(list)
    conversation= client.test.chatSessions.find({'conversationId':conversationId})
    sid=SentimentIntensityAnalyzer(lexicon_file='C:/Users/anshmania/Anaconda2/Lib/site-packages/nltk/sentiment/vader_lexicon.txt')
    for conversationId,data in enumerate(conversation):
        text=' '.join(data['text'])
        lines_list= tokenize.sent_tokenize(text)
        sentences.append(lines_list)
        for i,sentence in enumerate(sentences):
            sentence=' '.join(sentence)
            ss = sid.polarity_scores(sentence)
            for k in sorted(ss):
                sentimentData.setdefault(conversationId,[]).append({k:ss[k]})
        sentences=[]
    return sentimentData[conversationId]

#########################################################################
english_stops=set(stopwords.words('english')) #using nltk english stopwords
# Remove stop words and return list of tokenized words
def removeStops(conversationId):
    try:
        conversation= client.test.chatSessions.find({'conversationId':conversationId})
        for data in conversation:
            text=' '.join(data['text'])
            word_list= tokenize.word_tokenize(text)
            stop_free_text=[word for word in word_list if word not in english_stops]
            #♥return ' '.join(stop_free_text)
            return stop_free_text
    except:
        None

#########################################################################
# Collocations should be found for the whole text, once they have been collected for every individual conversation
def biCollocationFinder(conversationId):
    try:
        filter_stops=lambda w: len(w)<3
        bcf = BigramCollocationFinder.from_words(removeStops(conversationId))
        bcf.apply_word_filter(filter_stops)
        bcf.apply_freq_filter(1) #filter to find collocations appearing atleast twice
        bi_likelihood=bcf.nbest(BigramAssocMeasures.likelihood_ratio, 3)
    #bi_chi=bcf.nbest(BigramAssocMeasures.chi_sq, 3)
    #bi_fish=bcf.nbest(BigramAssocMeasures.fisher, 3)
        return bi_likelihood
    except:
        None
# trigram collocations
def triCollocationFinder(conversationId):
    filter_stops=lambda w: len(w)<3
    tcf = TrigramCollocationFinder.from_words(removeStops(conversationId))
    tcf.apply_word_filter(filter_stops)
    tcf.apply_freq_filter(2)
    triGrams=tcf.nbest(TrigramAssocMeasures.likelihood_ratio, 3)
    return triGrams#,bi_chi,bi_fish

#########################################################################
# finding the degree of non-compliance in a conversation
def nonCompliance(conversationId):
   text=' '.join(removeStops(conversationId))
   sentiment=senti(text)
   return sentiment['non-compliance']

#########################################################################
# calculating the sentiment score of collocations from a text
def biCollocationScore(conversationId):
    bi=biCollocationFinder(conversationId)
    [bi_list.append(senti(word)['non-compliance']) for item in bi for word in item]
    sum_b=math.sqrt(sum(bi_list))                                                                       # a way to normalize these scores???
    return sum_b

def triCollocationScore(conversationId):
    tri=triCollocationFinder(conversationId)
    [tri_list.append(senti(word)['non-compliance']) for item in tri for word in item]
    sum_t=math.sqrt(sum(tri_list))
    return sum_t

#########################################################################
## Take the long words and high frequency words of the text into account
f = 'C:/Users/anshmania/Anaconda2/Lib/site-packages/nltk/sentiment/qumram.txt'
WORD_VALENCE_DICT = cnc.make_lex_dict(f)

def valenceScore(keyword):
    return WORD_VALENCE_DICT[keyword]

def checkWordExistence(keyword):
    if keyword in WORD_VALENCE_DICT.keys():
        return True
    else:
        return False

def longWordScore(conversationId):
    long_word_score=0
    for word in extractLongWords(conversationId):
        if checkWordExistence(word)==True:
            long_word_score += valenceScore(word)
        else:
            continue
        return long_word_score
#########################################################################
########################## Analysis #####################################
#########################################################################
#
#bi_list=[]
#tri_list=[]
## putting all the features together and creating a criticality index
#def allTogether(conversationId):
#        keywordsInText=len(keywordFreqAnalyser(conversationId))/5
#        sentimentAna=(sentimentAnalyser(conversationId)[0]['compound'])/5
#        sentimentAna1=((sentimentAna*sentimentAnalyser(conversationId)[1]['neg']))/2
#        sum_feature= 0.5*(keywordsInText)+0.3*(sentimentAna1)+0.2*(biCollocationScore(conversationId)+triCollocationScore(conversationId))
##    return logistic._pdf(sum_feature)
#        if sum_feature<0.5:
#            sum_feature -= 1
#            return expit(sum_feature)
#        else:
#            return expit(sum_feature)
#########################################################################
######################### Analysis ######################################
#########################################################################

bi_list=[]
tri_list=[]
# putting all the features together and creating a criticality index
def featureExtractor(conversationId):
        keywordsInText=len(keywordFreqAnalyser(conversationId))
        #sum_feature= 0.5*(keywordsInText)+0.3*(non_comp)+0.2*(biCollocationScore(conversationId)+triCollocationScore(conversationId))
        sum_feature= 0.2*(keywordsInText)+0.6*nonCompliance(conversationId)+0.2*(biCollocationScore(conversationId)+triCollocationScore(conversationId))
#    return logistic._pdf(sum_feature)
        if sum_feature<0.5:
            sum_feature -= 1
            return expit(sum_feature)
        else:
            return expit(sum_feature)

######################################################################
## Declare some variables and Prepare for a plot
#x=[]
#y=[]
#for i in range(5):
##,data in enumerate(client.test.chatSessions.find()):
#    x.append(featureExtractor(i+1))
#    y.append(sentimentAnalyser(i+1)[0]['compound'])
#
#######################################################################
### Plot
#fig=plt.figure()
#plt.scatter(x, y, alpha=0.5)
#fig.suptitle('Conversation analysis', fontsize=20)
#plt.xlabel('Criticality', fontsize=18)
#plt.ylabel('Sentiment', fontsize=16)
#plt.show()

#####################################################################
################# Elasticsearch Indexing ############################
#####################################################################

#metaData=defaultdict(list)
#for data in client.anshmania.metaData.find():
#    metaData.setdefault(data['archivedObjectId'],[]).append({data['name']:data['value']})
#
## Retrieve metadata associated with a archiveId
#def metaDataRetrieval(archObjId):
#    return metaData[archObjId]
#
#def languageExceptionHandler(archObj):
#    try:
#        lang_doc=client.anshmania.archivedObjects.find({"_id":archObj})
#        for document in lang_doc:
#            return document['request']['headers']['Accept-Language']
#    except KeyError:
#        return None
#
## Parse user-agent
#def userAgentBrowser(archObj):
#    try:
#        lang_doc=client.anshmania.archivedObjects.find({"_id":archObj})
#        for document in lang_doc:
#            usrAgnt = document['request']['headers']['User-Agent']
#            parsedUserAgent=parse(usrAgnt)
#            browser=namedtuple('Browser',['family','version_string'])
#            browser.family=parsedUserAgent.browser.family
#            browser.version_string=parsedUserAgent.browser.version_string
#            return browser.family
#    except KeyError:
#        return None
#
#def userAgentOS(archObj):
#    try:
#        lang_doc=client.anshmania.archivedObjects.find({"_id":archObj})
#        for document in lang_doc:
#            usrAgnt = document['request']['headers']['User-Agent']
#            parsedUserAgent=parse(usrAgnt)
#            os=namedtuple('OS',['family','version_string'])
#            os.family=parsedUserAgent.os.family
#            os.version_string=parsedUserAgent.os.version_string
#            return os.family
#    except KeyError:
#        return None
#
#def userAgentDevice(archObj):
#    try:
#        lang_doc=client.anshmania.archivedObjects.find({"_id":archObj})
#        for document in lang_doc:
#            usrAgnt = document['request']['headers']['User-Agent']
#            parsedUserAgent=parse(usrAgnt)
#            device=namedtuple('Device',['family','model'])
#            device.family=parsedUserAgent.device.family
#            device.model=parsedUserAgent.device.model
#            return device.model
#    except KeyError:
#        return None
#
#def locationExceptionHandler(archObj):
#    try:
#        lang_doc=client.anshmania.archivedObjects.find({"_id":archObj})
#        for document in lang_doc:
#            return document['request']['headers']['Location']
#    except KeyError:
#        return None
#
#def contentLengthExceptionHandler(archObj):
#    try:
#        lang_doc=client.anshmania.archivedObjects.find({"_id":archObj})
#        for document in lang_doc:
#            return document['request']['headers']['Content-Length']
#    except KeyError:
#        return None
#
#
## Initialize Elasticsearch
#es=Elasticsearch()
#es.indices.delete(index='mongo_nlp')
#es.indices.create(index='mongo_nlp')
#
##Index mapping
#es.indices.put_mapping(index='mongo_nlp',doc_type='archivedObject',body={
#                                                                        "properties":{
#                                                                            "country":{"type":"string",
#                                                                                       "index":"not_analyzed"},
#                                                                            "city":{"type":"string",
#                                                                                    "index":"not_analyzed"},
#                                                                            "loc":{"type":"geo_point"},
#                                                                            "method":{"type":"string",
#                                                                                      "index":"not_analyzed"},
#                                                                            "Accept-Language":{"type":"string",
#                                                                                               "index":"not_analyzed"},
#                                                                             "url":{"type":"string",
#                                                                                    "index":"not_analyzed"},
#                                                                             "User-Agent":{"type":"string",
#                                                                                           "index":"not_analyzed"},
#                                                                             "remoteAddress":{"type":"ip"},
#                                                                             "OS":{"type":"string",
#                                                                                   "index":"not_analyzed"},
#                                                                             "Device":{"type":"string",
#                                                                                   "index":"not_analyzed"},
#                                                                             "Browser":{"type":"string",
#                                                                                   "index":"not_analyzed"}}})
#
## Index data
#for i,archObj in enumerate(client.anshmania.archivedObjects.find()):
#    ip=archObj['remoteAddress']
#    response_city=city_db.city(ip)
#    es.index(index='mongo_nlp',doc_type='archivedObject',id=archObj['_id'],body={"country":response_city.country.name,
#                                                                                  "city":response_city.city.name,
#                                                                                  "sessionId":archObj['sessionId'],
#                                                                                  "loc":{
#                                                                                      "lat":response_city.location.latitude,
#                                                                                      "lon":response_city.location.longitude},
#                                                                                  "remoteAddress":archObj['remoteAddress'],
#                                                                                  "sessionId":archObj['sessionId'],
#                                                                                  "captureTime":archObj['captureTime'],
#                                                                                  "mode":archObj['mode'],
#                                                                                  "tenantId":archObj['tenantId'],
#                                                                                  "metadata":metaDataRetrieval(archObj['_id']),
#                                                                                  "bodyContentId":archObj['request']['bodyContentId'],
#                                                                                  "Accept-Language":languageExceptionHandler(archObj['_id']),
#                                                                                  "OS":userAgentOS(archObj['_id']),
#                                                                                  "Device":userAgentDevice(archObj['_id']),
#                                                                                  "Browser":userAgentBrowser(archObj['_id']),
#                                                                                  "method":archObj['request']['method'],
#                                                                                  "url":archObj['request']['url'],
#                                                                                  "waId":archObj['request']['waId'],
#                                                                                  "bodyContentId":archObj['response']['bodyContentId'],
#                                                                                  "Content-Length":contentLengthExceptionHandler(archObj['_id']),
#                                                                                  "Content-Type":archObj['response']['headers']['Content-Type'],
#                                                                                  "Location":locationExceptionHandler(archObj['_id']),
#                                                                                  "Server":archObj['response']['headers']['Server'],
#                                                                                  "statusCode":archObj['response']['statusCode'],
#                                                                                  "title":archObj['response']['title'],
#                                                                                  "criticality":allTogether(i+1),
#                                                                                  "nonCompliance":nonCompliance(i+1),
#                                                                                  "collocations":biCollocationFinder(i+1)})
#
#
#
##send email
#def send_email(user, pwd, recipient, subject, body):
#    import smtplib
#
#    gmail_user = user
#    gmail_pwd = pwd
#    FROM = user
#    TO = recipient if type(recipient) is list else [recipient]
#    SUBJECT = subject
#    TEXT = body
#
#    # Prepare actual message
#    message = """\From: %s\nTo: %s\nSubject: %s\n\n%s
#    """ % (FROM, ", ".join(TO), SUBJECT, TEXT)
#    try:
#        server = smtplib.SMTP("smtp.gmail.com", 587)
#        server.ehlo()
#        server.starttls()
#        server.login(gmail_user, gmail_pwd)
#        server.sendmail(FROM, TO, message)
#        server.close()
#        print 'successfully sent the mail'
#    except:
#        print server.getreply



 # Create instances
   ## Feature based on freqData and sentimentData
#def featureExtractor(conversationId):

#    features={}
#    conversation= client.test.chatSessions.find({'conversationId':conversationId})
#    for data in conversation:
#        text=' '.join(data['text'])
#        tokenized=nltk.word_tokenize(text)
#        for word in keyWords:
#            if word in tokenized:
#                features['contains({})'.format(word)] = True

#def keywordFeatureExtractor(conversationId):

# #Build a classifier
#def classifier(conversationId):
#    senti=sentimentAnalyser(conversationId)
#    pos=senti[3]['pos']
#    neg=senti[1]['neg']
#    neu=senti[2]['neu']
#    if pos<neg:
#        return 'Negative'
#    elif neu>neg & neu>pos:
#        return 'Neutral'
#    else:
#        return 'Positive'






































#for data in client.test.chatSessions.find():
#    text=data['text']
#    subj_docs=


#sentences=[]
#sid=SentimentIntensityAnalyzer(lexicon_file='C:/Users/anshmania/Anaconda2/Lib/site-packages/nltk/sentiment/vader_lexicon.txt')
#for conversationId,data in enumerate(client.test.chatSessions.find()):
#    text=data['text']
#    lines_list= tokenize.sent_tokenize(str(text))
#    sentences.append(lines_list)
#    for sentence in sentences:
#        sentence=str(sentence)
#        print sentence,
#        ss = sid.polarity_scores(sentence)
#        for k in sorted(ss):
#            print '{0}: {1}, '.format(k, ss[k])


### add synonyms of keywords to the list
##
### Import text
###path = nltk.data.find('C:/Users/anshmania/codes/python/corpus/user.txt')
###raw = open(path, 'rU').read()
##for data in client.test.chatSessions.find():
##
##    fdist1=FreqDist(data['text'])
#
## Load data into NLTK
##data_freq=defaultdict(list)
#for data in client.test.chatSessions.find_one():
#   print (data['text'])
   #tokens=nltk.word_tokenize(raw)
   #print tokens
   #words = [word.lower() for word in tokens]
   #[body.append(word) for word in keyWords if word in words]
#
#
#
#if len(body) != 0:
#    print "critical"
#        if word in keyWords:
#            print freqDist1[word]#data_freq.setdefault(sessionId,[]).append({freqDist:freqDist1[word]})
#    [body.append(FreqDist(word),word) for word in raw if word in keyWords]

#    tokenize=nltk.word_tokenize(raw)
#    for word in tokenize:
#        if word in keyWords:
#           print word

#    raw = nltk.word_tokenize(raw)
#    freqDist1=FreqDist([w for w in raw])
#for i in range(sessionId):
#    if
#       print fdist2(sessionId,)

#for i in set(body):
#    print (i,fdist2[i])
#fdist1.plot(10)

#if len(body) != 0:
#        print (set(body)



#    tokens=nltk.word_tokenize(raw)
 #   words = [word.lower() for word in tokens]




#



