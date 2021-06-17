# Natural Language Processing 
  Bottom Top Approach Of Learning
  1. Text Preprocessing Level 1- Tokenization,Lemmatization,StopWords,POS
  2. Text Preprocessing Level 2- Bag Of  Words, TFIDF, Unigrams,Bigrams,n-grams
  3. Text Preprocessing- Gensim,Word2vec,AvgWord2vec
  4. Solve Machine Learning Usecases
  5. Get the Understanding Of Artificial Neural Network
  6. Understanding Recurrent Neural Networks, LSTM,GRU
  7. Text Preprocessing Level 3- Word Embeddings, Word2vec
  8. Bidirectional LSTM RNN, Encoders And Decoders, Attention Models
  9. Transformers 
  10. BERT 
  ***
  
 ## Text Processing Level - 1 
 
 ### Tokenization
   Tokenization is the process of converting the whole data or a paragraph into sentences and words.
  
   Library used :
   
   NLTK 
   
   **Paragraph -> Sentences**
   
    sentences = nltk.sent_tokenize('paragraph')
    
   **Paragraph -> Words**
   
    words = nltk.word_tokenize('paragraph')
 ***
    
 ### Stemming
  *"Stemming is the process of reducing inflection in words to their root forms such as mapping a group of words to the same stem even if the stem itself is not a valid word in the Language"* - **Wikipedia**
  
  From the defintion we came to know that we are deriving a root word from the words though it's not a legal word.
  
  Example :
  
  Trouble , Troubled , Troubling 

  When we stem the above words the output will be *troubl*. Sounds strange isn't it?.
  
  Stemming just removes the common prefix , suffix and give us the stem word which may or  may not be legal one. 
  
  **Libraries used**
  
  NLTK , PorterStemmer
  
  ### Stopwords
  
   *Stopwords are the English words which does not add much meaning to a sentence. They can safely be ignored without sacrificing the meaning of the sentence. For example, the words like the, he, have etc.*
    
  **Libraries used**
  
  NLTK , stopwords
  
  With the knowledge we so far gained we can tokenize a parangraph into sentences. Then convert those sentences into words then we can perfrom the Stemming process.
  
  Look at the [Stemming note book](https://github.com/AlbusDracoSam/NLP/blob/main/Stemming_.ipynb)
  
  ***
  
  ### Lemmatization
  
  *"Lemmatisation (or lemmatization) in linguistics is the process of grouping together the inflected forms of a word so they can be analysed as a single item, identified by the word's lemma, or dictionary form"* - **Wikipedia**
  
  In simpler words it's an extension of Stemming. The output of the lemmatization will be a meaningful legal word.
  
  **Libraries used**
  
  NLTK , WordNetLemmatizer
  
  Check the implementation [Lemmatization note book](https://github.com/AlbusDracoSam/NLP/blob/main/Lemmatization.ipynb)
  
 ***

### Bag of Words

  Suppose we have 3 sentences :
  
  Sentence I   :  good boy
  
  Sentence II  : good girl
  
  Sentence III : boy girl
  
  Considering the above 3 sentences we count the frequency of each of the word and then converting those into a vector.
  

| word       |   frequency  |                        
| ------------- | ------------- |                    
| good      | 2 |                    
| boy       | 2 |                                   
| girl      | 2 |     

**Converting to Vector**
 

|     | good | boy | girl |
| --- | ---- | --- | ---- | 
| Sentence I | 1 | 1 | 0 |
| Sentence II | 1 | 0 | 1 |
| Sentence III | 0 | 1 | 1 |

**Libraries used**

  NLTK , stopwords , re , sklearn
  
 **Process**
 1. Get the input
 2. tokenize the word into sentences
 3. lemmatize the sentences 
 4. Use Regular Expression to eliminate the symbols and numbers
 5. Using the **CountVectorizer** pass the processed data
 6. The final output will be an array like we have seen above.

Check the implementation [Bag of Words NoteBook](https://github.com/AlbusDracoSam/NLP/blob/main/Bag_of_Words.ipynb)

***

### TF - IDF

TF - Term Frequency

IDF - Inverse Document Frequency

**TF**

Refer the above example in Bag of Words.

The main disadvantange of the BOW is that it won't give us which word is the vital word. It will give us simply an vector yet powerful.

TF = No.of. frequency of the word / total no of words

**IDF**

IDF = log(No.of.sentences / No.of.Sentences contains the word)

Final table will be an product of TF * IDF

**Libraries used**

  NLTK , stopwords , re , sklearn
  
 **Process**
 1. Get the input
 2. tokenize the word into sentences
 3. lemmatize the sentences 
 4. Use Regular Expression to eliminate the symbols and numbers
 5. Using the **TfidfVectorizer** pass the processed data
 6. The final output will be an array like we have seen above.
 
 Check the implementation [TFIDF NoteBook](https://github.com/AlbusDracoSam/NLP/blob/main/TF_IDF.ipynb)
 
 ***
 
 ### Word2Vec
 
 The main problem with the BOW and TF-IDF is that they doesn't give us any semantic information. To overcome this we use Word2Vec.
 
 In BOW and TF-IDF the words are represented in an array while in Word2Vec each word in the sentence is represented as a vector of 32 or more dimension. 
 
 Using the dimensions it keeps the semantic information with the other words. Eg. Man and Woman will be near in terms of dimensions. 
 
 In simpler words related words are stored as near as possible using the vector dimensions. 
 
 It's a brief topic if you want to learn more please refer google.
 
 **Libraries used**

  NLTK , stopwords , re , sklearn, gensim
 
 **Process**
 
 1. Cleaning the data
 2. Tokenizing 
 3. Building the model
 4. Comparing the vocabularies

 Check the implementation [Word2Vec NoteBook](https://github.com/AlbusDracoSam/NLP/blob/main/Word2Vec.ipynb)
 
 ***
 
 





  

  
  
  
