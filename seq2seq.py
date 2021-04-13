


import pandas as pd
from time import time

#  this data was gathered from Kaggle database
data = pd.read_csv('data.csv', encoding='iso-8859-1')



#  The data has two columns, "headlines" and "text".
#  The goal is to use seq2seq model to summarize the text further in a way that has the same length as the headline. 
# This summarization helps the matching model a lot in order to find the best sentiments matching of the text with targeted headline



# the text_cleaning funcion helps to remove the non-alphabetic characters
# from the data set and 
import re
def text_cleaning(column):
    for row in column:
        
        row=re.sub("(\\t)", ' ', str(row)).lower() 
        row=re.sub("(\\r)", ' ', str(row)).lower() 
        row=re.sub("(\\n)", ' ', str(row)).lower()
        
        row=re.sub("(__+)", ' ', str(row)).lower()  
        row=re.sub("(--+)", ' ', str(row)).lower() 
        row=re.sub("(~~+)", ' ', str(row)).lower()   
        row=re.sub("(\+\++)", ' ', str(row)).lower()  
        row=re.sub("(\.\.+)", ' ', str(row)).lower()  
        
        row=re.sub(r"[<>()|&©ø\[\]\'\",;?~*!]", ' ', str(row)).lower() 
        
        row=re.sub("(mailto:)", ' ', str(row)).lower() 
        row=re.sub(r"(\\x9\d)", ' ', str(row)).lower() 
        row=re.sub("([iI][nN][cC]\d+)", 'INC_NUM', str(row)).lower() 
        row=re.sub("([cC][mM]\d+)|([cC][hH][gG]\d+)", 'CM_NUM', str(row)).lower() 
        
        row=re.sub("(\.\s+)", ' ', str(row)).lower() 
        row=re.sub("(\-\s+)", ' ', str(row)).lower()
        row=re.sub("(\:\s+)", ' ', str(row)).lower() 
        row=re.sub("(\s+.\s+)", ' ', str(row)).lower() 
        
        #Replace https://abc.xyz.net/site/site ====> abc.xyz.net
        try:
            url = re.search(r'((https*:\/*)([^\/\s]+))(.[^\s]+)', str(row))
            repl_url = url.group(3)
            row = re.sub(r'((https*:\/*)([^\/\s]+))(.[^\s]+)',repl_url, str(row))
        except:
            pass 
        
        row = re.sub("(\s+)",' ',str(row)).lower() #remove multiple spaces
        row=re.sub("(\s+.\s+)", ' ', str(row)).lower() #remove any single charecters hanging between 2 spaces
        
        yield row




text = text_cleaning(data['text'])
headline = text_cleaning(data['headlines'])




import spacy
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser']) # disabling Named Entity Recognition for speed

t = time()
#Batch the data points into 5000 
text = [str(doc) for doc in nlp.pipe(text, batch_size=5000)]
#On my mac it takes 9-10 mins
print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))




t = time()
#Batch the data points into 5000
headline = ['_START_ '+ str(doc) + ' _END_' for doc in nlp.pipe(headline, batch_size=5000)]
#On my mac, it takes 2-3 mins
print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))




data['cleaned_text'] = pd.Series(text)
data['cleaned_headline'] = pd.Series(headline)


text_count = []
headline_count = []



for sent in data['cleaned_text']:
    text_count.append(len(sent.split()))
for sent in data['cleaned_headline']:
    headline_count.append(len(sent.split()))


graph_df= pd.DataFrame()
graph_df['text']=text_count
graph_df['headline']=headline_count


# the length of the text is between 40 and 70 words
import matplotlib.pyplot as plt
graph_df['text'].hist()
plt.show()




#the headline is between 6 and 16 words
graph_df['headline'].hist()
plt.show()



#Check how much percent of headline have a length 0-14 words
cnt=0
for i in data['cleaned_headline']:
    if(len(i.split())<=14):
        cnt=cnt+1
print(cnt/len(data['cleaned_headline']))


#Check how much percet of text have words between 0-62 words
cnt=0
for i in data['cleaned_text']:
    if(len(i.split())<=62):
        cnt=cnt+1
print(cnt/len(data['cleaned_text']))


# We just understood that most of the text has length between 8-15 words for headline and 40-100 words for text, therefore, for the model we can set the maximim length of each to the following
max_text_len=62
max_headline_len=14


# Following the last argument, here, we cut down the length of text/headline in order to make sure all of the data has less than the pre-determined maximim words

import numpy as np
cleaned_text =np.array(data['cleaned_text'])
cleaned_headline=np.array(data['cleaned_headline'])

short_text=[]
short_headline=[]

for i in range(len(cleaned_text)):
    if(len(cleaned_headline[i].split())<=max_headline_len and len(cleaned_text[i].split())<=max_text_len):
        short_text.append(cleaned_text[i])
        short_headline.append(cleaned_headline[i])

# put the new data in a new dataframe        
df=pd.DataFrame({'text':short_text,'headline':short_headline})


# For a better performace of the model and in order to make the decoding/encoding easier, here, I added "sabbr" and "konnn" to first and end of the headline
df['headline'] = df['headline'].apply(lambda x : 'sabbr '+ x + ' konnn')


# Spliting the data to train and validation sets
from sklearn.model_selection import train_test_split
x_train,x_validation,y_train,y_validation=train_test_split(np.array(df['text']),np.array(df['headline']),test_size=0.15,random_state=10,shuffle=True)


# here I want to get a sense of the size of data
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences

#prepare a tokenizer on training data
x_tokenizer = Tokenizer() 
x_tokenizer.fit_on_texts(list(x_train))
x_voc_all   =  len(x_tokenizer.word_counts) + 1
print("Number of all words in x_train = {}".format(x_voc_all))


# The number of words is very large, here I should focus on the common words because the rare words do not improve the performace of the model and just makes the calculation heavier
# number of times a word repeated is used to identify the common words
repeat=3

rare_words=0
all_words=0


for word,number in x_tokenizer.word_counts.items():
    all_words+=1
    if(number<repeat):
        rare_words+=1
    
print("percent of rare words in x:",(rare_words/all_words)*100)


# As see from before over 50 percent of data are rare words, let's limit the data to only the common words in order to speed up the calculation
x_tokenizer = Tokenizer(num_words=all_words-rare_words) 
x_tokenizer.fit_on_texts(list(x_train))

# one of the steps for feeding the data into the model is to
# convert the text to integer sequences and padding zero of the length of data is less than 
# the maximim value, in this way all the data would have the same length

#convert text sequences into integer sequences 
x_train_seq    =   x_tokenizer.texts_to_sequences(x_train) 
x_validation_seq   =   x_tokenizer.texts_to_sequences(x_validation)

#padding zero up to maximum length
x_train   =   pad_sequences(x_train_seq,  maxlen=max_text_len, padding='post')
x_validation   =   pad_sequences(x_validation_seq, maxlen=max_text_len, padding='post')

#size of common vocabulary ( +1 for padding token)
x_voc   =  x_tokenizer.num_words + 1
print("Number of common words in x_train = {}".format(x_voc))


# the same as before but for Y
y_tokenizer = Tokenizer() 
y_tokenizer.fit_on_texts(list(y_train))

y_voc_all   =  len(y_tokenizer.word_counts) + 1

print("Number of all words in y_train = {}".format(y_voc_all))


# the same as before but for Y
# the number of words is very huge, here I should focus on the 
# common words because the rare words does not improve the performace and just
# make the calculation heavier
# number of times a word repeated to count as common word
repeat=3

rare_words=0
all_words=0


for word,number in y_tokenizer.word_counts.items():
    all_words+=1
    if(number<repeat):
        rare_words+=1
    
print("percent of rare words in y:",(rare_words/all_words)*100)


# the same as before but for Y
y_tokenizer = Tokenizer(num_words=all_words-rare_words) 
y_tokenizer.fit_on_texts(list(y_train))

#convert text sequences into integer sequences (i.e one hot encode the text in Y)
y_train_seq    =   y_tokenizer.texts_to_sequences(y_train) 
y_validation_seq   =   y_tokenizer.texts_to_sequences(y_validation) 

#padding zero upto maximum length
y_train    =   pad_sequences(y_train_seq, maxlen=max_headline_len, padding='post')
y_validation   =   pad_sequences(y_validation_seq, maxlen=max_headline_len, padding='post')

#size of vocabulary ( +1 for padding token)
y_voc   =  y_tokenizer.num_words + 1

print("Number of common words in x_train= {}".format(y_voc))


# It turns out that some of the headlines are empty, here we remove those ones
# those ome with "START" and "END" has only two words
ind=[]
for i in range(len(y_train)):
    cnt=0
    for j in y_train[i]:
        if j!=0:
            cnt=cnt+1
    if(cnt==2):
        ind.append(i)

y_train=np.delete(y_train,ind, axis=0)
x_train=np.delete(x_train,ind, axis=0)


ind=[]
for i in range(len(y_validation)):
    cnt=0
    for j in y_validation[i]:
        if j!=0:
            cnt=cnt+1
    if(cnt==2):
        ind.append(i)

y_validation=np.delete(y_validation,ind, axis=0)
x_validation=np.delete(x_validation,ind, axis=0)


# Until now, we perpared and processed the data for feeding to the model, from here, the modeling part starts
# here in each step, the encode, embeding layer, encoders and dense layers are defined based on the size of healines and text as discussed before
from keras import backend 
import gensim
from numpy import *
import numpy as np
import pandas as pd 
import re
from bs4 import BeautifulSoup
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import warnings
pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore")


backend.clear_session()

latent_dim = 300
embedding_dim=200

# Encoder
encoder_inputs = Input(shape=(max_text_len,))

#embedding 
enc_emb =  Embedding(x_voc, embedding_dim,trainable=True)(encoder_inputs)

#encoder 1
encoder_lstm1 = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

#encoder 2
encoder_lstm2 = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

#encoder 3
encoder_lstm3=LSTM(latent_dim, return_state=True, return_sequences=True,dropout=0.4,recurrent_dropout=0.4)
encoder_outputs, state_h, state_c= encoder_lstm3(encoder_output2)

# Set up the decoder
decoder_inputs = Input(shape=(None,))

#embedding 
dec_emb_layer = Embedding(y_voc, embedding_dim,trainable=True)
dec_emb = dec_emb_layer(decoder_inputs)

decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True,dropout=0.4,recurrent_dropout=0.2)
decoder_outputs,decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb,initial_state=[state_h, state_c])

#dense 
decoder_dense =  TimeDistributed(Dense(y_voc, activation='softmax'))
decoder_outputs = decoder_dense(decoder_outputs)

# model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
# summary of the model artichecture
model.summary()


# compile the model with "rmsprop" optimizer which seems to works good here
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

# this useful to avoid overfiting and reduce the computational cost a lot
earlystoping = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=2)

# here the model is fitted to the train data with early stoping condition on the loss of validation set.
output=model.fit([x_train,y_train[:,:-1]], y_train.reshape(y_train.shape[0],y_train.shape[1], 1)[:,1:] ,epochs=100,callbacks=[earlystoping],batch_size=50, validation_data=([x_validation,y_validation[:,:-1]], y_validation.reshape(y_validation.shape[0],y_validation.shape[1], 1)[:,1:]))


# let's visualizethe the performace of the model for both validation and training
from matplotlib import pyplot
pyplot.plot(output.history['loss'], label='training')
pyplot.plot(output.history['val_loss'], label='validation')
pyplot.legend()
pyplot.show()


# for more convenience of transforming index to word and vice versa
# here, three dictionaries are defined
reverse_target_word_index=y_tokenizer.index_word
reverse_source_word_index=x_tokenizer.index_word
target_word_index=y_tokenizer.word_index


# Encode the input sequence 
encoder_model = Model(inputs=encoder_inputs,outputs=[encoder_outputs, state_h, state_c])

# Decoder setup
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_hidden_state_input = Input(shape=(max_text_len,latent_dim))

# Embeddings of the decoder sequence
dec_emb2= dec_emb_layer(decoder_inputs) 
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])

# A dense softmax layer 
decoder_outputs2 = decoder_dense(decoder_outputs2) 

# Decoder model
decoder_model = Model(
    [decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],
    [decoder_outputs2] + [state_h2, state_c2])


# I used this function for the inference processes purposes
def decode_sequence(input_seq):
    # Encode the input 
    e_out, e_h, e_c = encoder_model.predict(input_seq)
    
    # Empty target 
    target_seq = np.zeros((1,1))
    
    # Populate with the start word.
    target_seq[0, 0] = target_word_index['sabbr']

    stop = False
    decoded_sentence = ''
    while not stop:
      
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # find the most probable tocken
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]
        
        if(sampled_token!='konnn'):
            decoded_sentence += ' '+sampled_token

        # test whether it is time to finish the process
        if (sampled_token == 'konnn'  or len(decoded_sentence.split()) >= (max_headline_len-1)):
            stop = True

        # Update the target sequence
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence


# A function to convert an integer sequence to a word sequence for headline as well as the text
def seq2headline(input_seq):
    newString=''
    for i in input_seq:
        if(i!=0 and i!=target_word_index['sabbr']) & (i!=target_word_index['konnn']):
            newString=newString+reverse_target_word_index[i]+' '
    return newString

def seq2text(input_seq):
    newString=''
    for i in input_seq:
        if(i!=0):
            newString=newString+reverse_source_word_index[i]+' '
    return newString


# Now it is time to take one step further and define a new dataframe that has an additional column named "predicted_headline" which has the same length as the "original headline" column. 
# "predicted_headline" column helps the matching model significantly to find and rank the news article headline 
df=pd.DataFrame(columns=["original_headline","text","predicted_headline"])
for i in range(0,x_train.shape[0]):
  df.loc[i,"text"]=seq2text(x_train[i])
  df.loc[i,"original_headline"]=seq2headline(y_train[i]).replace("start"," ",1).replace("end"," ",1)
  df.loc[i,"predicted_headline"]=decode_sequence(x_train[i].reshape(1,max_text_len)).replace("start"," ",1).replace("end"," ",1)
df.head(1)


#### Text sentiment matching 

# Now, we are very close to the final results.
# In this step, we just want to find the semantic similarity between the predcited_headline and the original headlines 
# and rank them according to their relevance.
#I used the fuzzywuzzy library and rank the relevance of the headline according to the Levenshtein Distance
k=3
# for string matching I used fuzzywuzzy library
from fuzzywuzzy import fuzz

# a function that ranks the "k" best matches of "original_headline"s to the "predicted_headline"s
def ranking(predicted_headline):
    matches= df.apply(lambda row: fuzz.token_sort_ratio(row["original_headline"],predicted_headline), axis=1)
    # gives the index of k best matches
    return sorted(range(len(matches)), key= lambda i: matches[i], reverse=True )[:k]

# here, based on the ranking function, the k best matching headlines 
# are stored in the database
for i, predicted_headline in enumerate(df["predicted_headline"]):
    for rank, rank_index in enumerate(ranking(predicted_headline)):
        df.loc[i,"matching rank of "+ str(rank+1) ]= df.loc[rank_index,"original_headline"]

