import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import string

def process_email(email):
    """
    Input:
        email: Eine einzige Email als String
    Output:
        email_clean: Eine bereinigte Liste mit Wörtern, die in diesem Email vorkommen

    """
    stemmer = PorterStemmer()
    stopwords_german = stopwords.words('german')

    # tokenize email
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    email_tokens = tokenizer.tokenize(email)

    email_clean = []
    for word in email_tokens:
        if (word not in stopwords_german and
                word not in string.punctuation):
            stem_word = stemmer.stem(word)
            email_clean.append(stem_word)

    return email_clean


def haeufigkeiten_berechnen(emails, personen):
    """
    Input:
        emails: Eine Liste mit Emails
        person_zahlen: Eine Liste mit den zu den jeweiligen Emails zugeteilten Personen
    Output:
        haeufigkeiten: Ein Dictionary, das jedem (wort, person) Paar seine Häufigkeit zuteilt
    """

    # Start with an empty dictionary and populate it by looping over all tweets
    # and over all processed words in each tweet.
    haeufigkeiten = {}
    for person, email in zip(personen, emails):
        for wort in process_email(email):
            paar = (wort, person)
            if paar in haeufigkeiten:
                haeufigkeiten[paar] += 1
            else:
                haeufigkeiten[paar] = 1

    return haeufigkeiten


def extract_features(email, freqs, namen):
    '''
    Input: 
        email: Eine Liste von Wörtern, die in der E-Mail vorkommen
        freqs: Ein Dictionary mit den Häufigkeiten jedes Wortes für jede Person
        namen: Eine Liste mit den Namen der Personen
    Output: 
        x: Ein Vektor der Dimension (1,4). Jede der vier Komponenten stellt die durchschnittliche
        Häufigkeit der Worte für die vier Personen dar
    '''
    wort_l = process_email(email)
    x = np.zeros((1, 4)) 
    
    # loopen durch jedes Wort in der Liste von Wörtern (d.h. in der weiterverarbeiteten E-Mail)
    for wort in wort_l:
        
        x[0,0] += freqs.get((wort, namen[0]), 0.0)
        x[0,1] += freqs.get((wort, namen[1]), 0.0)
        x[0,2] += freqs.get((wort, namen[2]), 0.0)
        x[0,3] += freqs.get((wort, namen[3]), 0.0)
    
    x[0,0] = np.divide(x[0,0], len(wort_l))
    x[0,1] = np.divide(x[0,1], len(wort_l))
    x[0,2] = np.divide(x[0,2], len(wort_l))
    x[0,3] = np.divide(x[0,3], len(wort_l))

    return x