# -*- coding: utf-8 -*-
import nltk
from nltk.stem.snowball import SnowballStemmer
import re
import os
import pandas as pd
from bs4 import BeautifulSoup
from libleipzig import *
import unidecode
from independent_from_libleipzig_wortschatz_14_03_2017 import get_synsets


# first in the folder c:\users\*your_name* create a new folder - in my case 'experiment' - and put there all the other folders:
# Care1, Control1 etc.
path = 'C:\Users\Simon Bartke\Arbeit\Text_mining\Code\Essays\Experiment_umlaut'
file_names = [] #create the list of files' names
for f in [f for f in os.listdir(path) ]:
    # os.listdir(path) - names of directories:'Care1', 'Control1'...
    # so we proceed folder by folder
    for (dirpath, dirnames, filenames) in os.walk(path+'\\'+f):
        # dirpath-directory path: c:\users\*your_name*\experiment
        # add each file's name to file_names list
        file_names.extend(filenames)


# now i create another list
# for each file we have a name in the following form: 'Anger_1_ID_11_1_insult/frust.txt'
# i want to separate
                #   treatment: Anger, Care or Control;
                #   session: 1, 2, 3 ...
                #   id numer: 1, 2, 3 ...
                #   number of the text 1 or 2
# in order to use them later as columns in the DataFrame
parse_file_names = [] #create a new list
for i in range(len(file_names)):
    from itertools import groupby
    groupby(file_names[i], lambda x: x == "_") #split by underscore each file's name
    k = [list(group) for k, group in groupby(file_names[i], lambda x: x == "_") if not k] #create a list for each file's name
    #which consists of 5 components: treatment, session, 'ID', id_number, all the rest
    parse_file_names.append(k)


# now I want to get separate lists for
# treatment (string)
# session (transform to integer)
# idnum (transform to integer)
# textnum (transform to integer)
# essay (string)
treatment=[]
session=[]
idnum=[]
textnum=[]
essay=[]
for i in range(len(file_names)): #for each file's name
    treatment.append(''.join(parse_file_names[i][0])) #extract list with treatment
    session.append(''.join(parse_file_names[i][1])) #extract list with the number of session (string)
    idnum.append(''.join(parse_file_names[i][2]))  #extract list with id number (string)
    textnum.append(parse_file_names[i][3][0]) #extract list with number of the text (string)
    essay.append(''.join(parse_file_names[i][4][:-4])) #extract list with number of the text (string)
#idnum = map(int, idnum) #from string to integer
textnum = map(int, textnum) #from string to integer
session=map(int, session) #from string to integer


path = 'C:\Users\Simon Bartke\Arbeit\Text_mining\Code\Essays\Experiment_umlaut'
texts = []
for f in [f for f in os.listdir(path) ]: #for each folder in the experiment folder
    for k in [k for k in os.listdir(path + '\\' + f) ]: #for each file in a particular folder
        with open(path + '\\' + f + '\\' + k, "r") as myfile: texts.append(myfile.read()) #create a list with texts


texts_clean = []
for text in texts:
    text = BeautifulSoup(text, 'html.parser').getText()
    #strips html formatting and converts to unicode
    texts_clean.append(text)

texts = texts_clean



## From here on: Remove stopwords from texts. Prepare them afterwards to be stemmed by function stem_only


texts = [x.lower() for x in texts]

# load nltk's German stopwords as variable called 'stopwords'
stopwords = nltk.corpus.stopwords.words('german')

## Also important for main script: When stopwords are loaded they come in unicode,
## this has to be changed to match the umlaut representation of the other files.


#unidecode is a very handy package; removes unicode encoding and makes directly readable.
stopwords_new = []
for i in stopwords:
    i = unidecode(i)
    stopwords_new.append(i)

#Since unidecode makes texts readable in a strange way, umlaute have to be replaced by logic from rest of the data.

stopwords_new = [w.replace('fur', 'fuer') for w in stopwords_new]
stopwords_new = [w.replace('konnen', 'koennen') for w in stopwords_new]
stopwords_new = [w.replace('konnte', 'koennte') for w in stopwords_new]
stopwords_new = [w.replace('uber', 'ueber') for w in stopwords_new]
stopwords_new = [w.replace('wahrend', 'waehrend') for w in stopwords_new]
stopwords_new = [w.replace('wurde', 'wuerde') for w in stopwords_new]
stopwords_new = [w.replace('wurden', 'wuerden') for w in stopwords_new]

#When umlaute have been changed to match style of rest of input data, stopwords have to be re-converted to unicode
stopwords_new = [unidecode.unicode(i) for i in stopwords_new]




## When loaded, texts is a list with n unicode entries. Each of the n unicode entries is an essay however.
## Issue: Not the individual entries need to be accessed, but the single words in the entries, which are
## not yet list-accessible.
## Example for what happens below:
new_list = texts[:1]
s = new_list[0].split()
l = list(s)


##Create list of lists texts_listed, in which every single word in every essay is list-accessible
texts_listed = []
for i in texts:
    lol = i.split()
    texts_listed.append(lol)

##Since some sentences end with a stopword that is directly followed by a . , these actual stopwords do not get
## recognized and therefore the .'s need to be removed from the essays:
for i in range(len(idnum)):
    texts_listed[i] = [s.strip('.') for s in texts_listed[i]]

##Remove modified list of stopwords from modified list of texts:
texts_no_stopwords = []
for i in range(len(idnum)):
    unique_list=[]
    for word in texts_listed[i]:
        if word not in stopwords_new:
            unique_list += [word]
    texts_no_stopwords.append(unique_list)


##After stopwords have been removed, bring texts_no_stopwords back into old format of texts such that the
## functions from the scripts can be used on them.

texts_no_stopwords_texts_again = []
for i in texts_no_stopwords:
    lol = " ".join(i)
    texts_no_stopwords_texts_again.append(lol)


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

#let's tokenize every text and save lists of tokens for each text in a column
allwords_tokenized=[]
for i in texts_no_stopwords_texts_again:
    #tokenize each text from the table (but first transform it to unicode)
    tokenized = tokenize_only(i)
    allwords_tokenized.append(tokenized)

#number of tokens in each text
num_allwords_tokenized=[]
for i in allwords_tokenized:
    num_tokenized=len(i)
    num_allwords_tokenized.append(num_tokenized)

num_unique_tokenized_words = []
for i in range(len(idnum)):
    unique_list_experiment=[]
    for word in allwords_tokenized[i]:
        if word not in unique_list_experiment:
            unique_list_experiment += [word]
    num_unique_tokenized_words.append(unique_list_experiment)

#number of unique words in each text
num_unique_tokenized_words_experiment=[]
for i in num_unique_tokenized_words:
    num_unique_experiment=len(i)
    num_unique_tokenized_words_experiment.append(num_unique_experiment)


#Get German stemmer from NLTK
stemmer = SnowballStemmer("german")

#Tokenize and stem function
def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

allwords_stemmed=[]
for i in texts_no_stopwords_texts_again:
    stemmed = tokenize_and_stem(i)
    allwords_stemmed.append(stemmed)


#number of tokens in each text
num_allwords_tokenized=[]
for i in allwords_tokenized:
    num_tokenized=len(i)
    num_allwords_tokenized.append(num_tokenized)

#number of tokens in each text
num_texts=[]
for i in texts_listed:
    numt=len(i)
    num_texts.append(numt)

experiment_data = pd.DataFrame({'idnum' : idnum,
 'texts' : texts,
 'session' : session,
 'textnum' : textnum,
 'treatment' : treatment,
 'essay' : essay,
 'file_names': file_names,
 'allwords_tokenized': allwords_tokenized,
 'allwords_stemmed': allwords_stemmed,
 'num_allwords_tokenized': num_allwords_tokenized,
 'num_unique_tokenized_words_experiment': num_unique_tokenized_words_experiment,
 'num_texts': num_texts})


# load nltk's German stopwords as variable called 'stopwords'
stopwords = nltk.corpus.stopwords.words('german')
#print stopwords




# The following creates a dictionary that maps LIWC category numbers to the names of these categories
id_to_topic = {}
topics = []
with open("LIWC_German_no_anger_anxiety_achievement.txt") as input_data:
    # Skips text before the beginning of the interesting block:
    for line in input_data:
        if line.strip() == '%':
            break
    #Reads text until the end of the block:
    for line in input_data: #This keeps reading the file
        if line.strip()== '%':
            break
        topics.append(line)

for s in topics:
    #split each element (number of topic + topic)
    l=s.split()
    topic_id = int(l[0])
    topic_name = l[1]
    #mapping
    id_to_topic[topic_id] = topic_name

#Get words out of LIWC and their category numbers (in same line)
id_to_words = {}
words=[]
with open("LIWC_German_no_anger_anxiety_achievement.txt") as input_data:
    # Skips text before the beginning of the interesting block:
    for line in input_data:
        if line.strip() == '%':
            break
    for line in input_data: #This keeps reading the file
        if line.strip()== '%':
            break
    for line in input_data: #This keeps reading the file
        #read in all the words and corresponding topics
        words.append(line)
#print words

# Creates first half of a dictionary. On the left side are the 68 topic numbers that wait to be filled each with a list of topic words.
for m in range(1, 73):
    topic_id = m
    id_to_words.setdefault(topic_id, [])

for s in words:
    #remove * from each string
    s = s.translate(None, '*')
    #split each element (word + topics)
    l=s.split()
    word_name = l[0]

    for j in range(1, len(l)):
        topic_id = int(l[j])
        id_to_words[topic_id].append(word_name)

##Achievement
achievement_words = []
achievement_words = get_synsets('achievement_synonyms_significant_merge.txt')
id_to_words[int(66)] = achievement_words

##Affiliation
affiliation_words = []
affiliation_words = get_synsets('affiliation_synonyms_significant.txt')
id_to_words[int(67)] = affiliation_words

##Anger
anger_words = []
anger_words = get_synsets('anger_synonyms_significant_merge.txt')
id_to_words[int(68)] = anger_words

##Care
care_words = []
care_words = get_synsets('care_synonyms_significant.txt')
id_to_words[int(69)] = care_words

##Consumption
consumption_words = []
consumption_words = get_synsets('consumption_synonyms_significant.txt')
id_to_words[int(70)] = consumption_words

##Fear
fear_words = []
fear_words = get_synsets('fear_synonyms_significant_merge.txt')
id_to_words[int(71)] = fear_words

##Power
power_words = []
power_words = get_synsets('power_synonyms_significant.txt')
id_to_words[int(72)] = power_words

#print id_to_words

#Stem LIWC content
id_to_words_stem = {}

#initialization
for k in range(1,73):
    topic_id = k
    id_to_words_stem.setdefault(topic_id, [])

for i in range(1, 73):
    for j in id_to_words[i]:
        #unicode for all the words
        j = BeautifulSoup(j, 'html.parser').getText()
        #stem all the words
        word_stem = tokenize_and_stem(j)
        #create a new dictionary with stemmed words
        id_to_words_stem[i].extend(word_stem)



#number of words for each topic
for number, topic in id_to_topic.items():
    #initialize list for number of topic's words in each text
    #list [0, 1, ..., idnum-1]
    num_topic=[i for i in range(len(idnum))]
    #set all the values in the list to 0 (first we have 0 words from each topic)
    for p in range (len(idnum)):
        num_topic[p]=0
    #go through each text
    for i in range(len(idnum)):
        #for each stemmed word of the text
        for word in allwords_stemmed[i]:
            #determine whether the word is from the dictionary for a particular topic
            if word in id_to_words_stem[number]:
                #if yes - +1 in the list num_topic for this text
                num_topic[i]=num_topic[i]+1
#add a column to DataFrame
    experiment_data[topic]=num_topic


#save as a separate table the part where treatment=anger; control; care
anger_experiment_data_significant_merge=experiment_data[experiment_data['treatment'] == 'Anger']
control_experiment_data_significant_merge=experiment_data[experiment_data['treatment'] == 'Control']
care_experiment_data_significant_merge=experiment_data[experiment_data['treatment'] == 'Care']

#save as a separate table the essay types of the treatments
anger_experiment_data_insult_significant_merge=experiment_data[(experiment_data.treatment == 'Anger') & (experiment_data.essay == 'insult')]
anger_experiment_data_frust_significant_merge=experiment_data[(experiment_data.treatment == 'Anger') & (experiment_data.essay == 'frust')]
control_experiment_data_everyday_significant_merge=experiment_data[(experiment_data.treatment == 'Control') & (experiment_data.essay == 'everyday')]
control_experiment_data_yesterday_significant_merge=experiment_data[(experiment_data.treatment == 'Control') & (experiment_data.essay == 'yesterday')]
care_experiment_data_compassion_significant_merge=experiment_data[(experiment_data.treatment == 'Care') & (experiment_data.essay == 'compassion')]
care_experiment_data_help_significant_merge=experiment_data[(experiment_data.treatment == 'Care') & (experiment_data.essay == 'help')]

##Save Excel tables
header = ['file_names', 'idnum', 'session', 'textnum', 'treatment', 'essay', 'num_texts', 'num_allwords_tokenized',
          'num_unique_tokenized_words_experiment', 'num_allwords_stemmed', 'Pronoun', 'I', 'We', 'Self', 'You', 'Other', 'Negate', 'Assent', 'Article',
		  'Preps', 'Numbers', 'Affect', 'Positiveemotion', 'Positivefeeling', 'Optimism', 'Negativeemotion', 'Sad',
		  'Cognitivemechanism', 'Cause', 'Insight',  'Discrepancy',  'Inhibition', 'Tentative', 'Certain', 'Social', 'Communication',
		  'Otherreference', 'Friends', 'Family', 'Humans', 'Time', 'Past', 'Present', 'Future', 'Space', 'Up', 'Down',  'Incl', 'Excl', 'Motion',
		  'Occup',  'School',  'Job', 'Leisure', 'Home', 'Sports', 'TV', 'Music', 'Money',  'Metaph', 'Relig', 'Death', 'Physical',
		  'Body', 'Sex', 'Eat', 'Sleep',  'Grooming',  'Swear', 'Nonfluency', 'Fillers', 'Achievement', 'Affiliation', 'Threatappr', 'Care',
		  'Consumption', 'Fear', 'Power']
# Create a Pandas Excel writer using XlsxWriter as the engine.
anger_writer = pd.ExcelWriter('anger_experiment_significant_merge_words_per_text.xlsx', engine='xlsxwriter')
anger_writer_insult = pd.ExcelWriter('anger_insult_experiment_significant_merge_words_per_text.xlsx', engine='xlsxwriter')
anger_writer_frust = pd.ExcelWriter('anger_frust_experiment_significant_merge_words_per_text.xlsx', engine='xlsxwriter')
control_writer = pd.ExcelWriter('control_experiment_significant_merge_words_per_text.xlsx', engine='xlsxwriter')
control_writer_everyday = pd.ExcelWriter('control_everyday_experiment_significant_merge_words_per_text.xlsx', engine='xlsxwriter')
control_writer_yesterday = pd.ExcelWriter('control_yesterday_experiment_significant_merge_words_per_text.xlsx', engine='xlsxwriter')
care_writer = pd.ExcelWriter('care_experiment_significant_merge_words_per_text.xlsx', engine='xlsxwriter')
care_writer_compassion = pd.ExcelWriter('care_compassion_experiment_significant_merge_words_per_text.xlsx', engine='xlsxwriter')
care_writer_help = pd.ExcelWriter('care_help_experiment_significant_merge_words_per_text.xlsx', engine='xlsxwriter')

anger_experiment_data_significant_merge.to_excel(anger_writer, columns = header)
anger_experiment_data_insult_significant_merge.to_excel(anger_writer_insult, columns = header)
anger_experiment_data_frust_significant_merge.to_excel(anger_writer_frust, columns = header)
control_experiment_data_significant_merge.to_excel(control_writer, columns = header)
control_experiment_data_everyday_significant_merge.to_excel(control_writer_everyday, columns = header)
control_experiment_data_yesterday_significant_merge.to_excel(control_writer_yesterday, columns = header)
care_experiment_data_significant_merge.to_excel(care_writer, columns = header)
care_experiment_data_compassion_significant_merge.to_excel(care_writer_compassion, columns = header)
care_experiment_data_help_significant_merge.to_excel(care_writer_help, columns = header)

# Close the Pandas Excel writer and output the Excel file.
anger_writer.save()
anger_writer_insult.save()
anger_writer_frust.save()
control_writer.save()
control_writer_everyday.save()
control_writer_yesterday.save()
care_writer.save()
care_writer_compassion.save()
care_writer_help.save()
