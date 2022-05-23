#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import requests
from bs4 import BeautifulSoup
import nltk
from collections import Counter

# nltk.download('stopwords')


r = requests.get('https://www.gutenberg.org/files/16/16-h/16-h.htm')
r.encoding = 'utf-8'
html = r.text

text = BeautifulSoup(html, features='lxml').get_text()
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
tokens = tokenizer.tokenize(text)
words = [string.lower() for string in tokens]
sw = nltk.corpus.stopwords.words('english')
words_ns = [string for string in words if string not in sw]
count = Counter(words_ns)
top_ten = count.most_common(10)
print(top_ten)