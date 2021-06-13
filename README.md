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


  

  
  
  
