# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 00:46:45 2021

@author: 54112
"""
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import sentiment
from nltk import word_tokenize
from nltk.tokenize.toktok import ToktokTokenizer
from bs4 import BeautifulSoup
import urllib.request
from nltk.corpus import treebank

'''def leer_archivo(ruta):
   
    frases_archivo= open(ruta,'r',enconding='utf8').read()
    return frases_archivo
if __name__=="__main":
    frases_pos = leer_archivo(./pos.txt)
    frases_neg = leer_archivo(./neg.txt) '''
    
    
    
    
    

nltk.download('vader_lexicon')

nltk.download('punkt')

tokenizer = nltk.data.load('tokenizers/punkt/spanish.pickle')
#sentences = tokenizer.tokenize("Un padre que da consejos más que padre es un amigo. Ansí como tal les digo que vivan con precaución naides sabe en qué rincón se oculta el que es su enemigo. Yo nunca tuve otra escuela que una vida desgraciada: no extrañen si en la jugada alguna vez me equivoco, pues debe saber muy poco aquel que no aprendió nada. Hay hombres que de su cencia tienen la cabeza llena, hay sabios de todas menas, más digo sin ser muy ducho: es mejor que aprender mucho, el aprender cosas buenas. Al que es amigo, jamás lo dejen en la estocada, pero no le pidan nada ni lo aguarden todo de él. siempre el amigo más fiel es una conducta honrada. Debe trabajar el hombre para ganarse su pan; pues la miseria en su afán de perseguir de mil modos, llama en la puerta de todos, y entra en la del haragán.") 
sentences = tokenizer.tokenize("Excelente vídeo. Felicitaciones por este gran aporte, espero me recomiende el siguiente  con su respectiva dirección. Nuevamente, gracias por la maravillosa oportunidad.Muy buen día y muchas gracias por esta excelente información")
analizador = SentimentIntensityAnalyzer()

for sentence in sentences:
    print(sentence)
    scores = analizador.polarity_scores(sentence)
    for key in scores:
        print(key, ': ', scores[key])
        print() 
