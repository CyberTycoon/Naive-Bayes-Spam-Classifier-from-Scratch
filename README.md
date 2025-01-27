# SpamShield  

## 🛡️ Overview  
A spam classifier built from scratch using **Naive Bayes** and **Laplace smoothing**. Distinguishes spam (financial scams, product offers, adult content) from ham (professional/personal emails).  

## 🧠 Theory  
- **Naive Bayes**: Uses Bayes’ Theorem to calculate probabilities of spam/ham given email content.  
- **Laplace Smoothing**: Handles unseen words to avoid zero probabilities.  
- **Bag-of-Words**: Represents emails as word frequency vectors.  

## 🚀 Features  
- **Custom Dataset**: Includes 50+ examples (spam: scams, offers, adult content; ham: professional/personal).  
- **Pure Python/NumPy**: No external libraries (e.g., scikit-learn).  
- **Interactive**: Input messages to classify them in real-time.  


Install dependencies:

## bash

pip install numpy  


Example Input/Output:

Enter a message: Win $1,000,000 now!  
The message 'Win $1,000,000 now!' is classified as: Spam  


🔧 Customization
Expand the Dataset: Add more examples to dataset (follow the existing format).

Adjust Smoothing: Modify the +1 in P_word_given_spam and P_word_given_ham.

📚 References
Naive Bayes theory from "Artificial Intelligence: A Modern Approach".

Laplace smoothing for handling unseen words.#   N a i v e - B a y e s - S p a m - C l a s s i f i e r - f r o m - S c r a t c h 
 
 
