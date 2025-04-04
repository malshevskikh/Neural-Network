from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import re
import string
import pymorphy2
import pandas as pd
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from nltk.stem.wordnet import WordNetLemmatizer
from word_forms.lemmatizer import lemmatize
from nltk.corpus import wordnet
import seaborn as sns
from matplotlib import pyplot as plt
import lemminflect


import nltk
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].lower()
    tag_dict = {"a": wordnet.ADJ,
                "n": wordnet.NOUN,
                "v": wordnet.VERB,
                "r": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)



morph = pymorphy2.MorphAnalyzer(lang='uk')
morph2 = pymorphy2.MorphAnalyzer(lang='ru')
stopWords = stopwords.words("english")

word = "called"
b = "корабли"
#p = morph.parse(a)
#print(p)
l = morph2.parse(b)
print(l)
print(lemmatize(word))

words = ['indigenous', 'to', 'the', 'forests', 'of', 'eastern', 'russia', 'these', 'endangered', 'giants', 'can', 'be', '10', 'feet', '3', 'meters', 'long', 'not', 'including', 'their', 'tail', 'and', 'weigh', 'up', 'to', '600', 'pounds', '300', 'kilograms']

for items in words:
    print(f"{items:<6}--> {lemmatizer.lemmatize(items, pos=get_wordnet_pos(items))}", end='\n')

for word in words:
    print("WordNet Lemmatizer=>", WordNetLemmatizer().lemmatize(word))



# Preprocessing and tokenizing
def preprocessing1(line):
    line = line.lower()
    #print("@")
    #print(line)
    line = re.sub(r"[{}]".format(string.punctuation), " ", line)
    #print(line)
    return line

def preprocessing2(line):
    line = line.lower()
    #print(line)
    line = re.sub(r"[{}]".format(string.punctuation), " ", line)
    sss = line.split(' ')
    qqq = []
    #print(sss)
    for i in sss:
        if i != '':
            qqq.append(lemmatizer.lemmatize(i, pos=get_wordnet_pos(i)))
        else:
            qqq.append(i)
    #print(qqq)
    new_line = ' '.join(qqq)
    #print(new_line)
    #print("+")
    line = new_line
    return line


all_text  =  """""
 Russia, the largest country in the world, occupies one-tenth of all the land on Earth.
It spans 11 time zones across two continents (Europe and Asia) and has coasts on three oceans (the Atlantic, Pacific, and Arctic).
The Russian landscape varies from desert to frozen coastline, tall mountains to giant marshes.
Much of Russia is made up of rolling, treeless plains called steppes.
Siberia, which occupies three-quarters of Russia, is dominated by sprawling pine forests called taigas.
Russia has about 100,000 rivers, including some of the longest and most powerful in the world.
There are about 120 ethnic groups in Russia who speak more than a hundred languages.
As big as Russia is, it's no surprise that it is home to a large number of ecosystems and species.
Russia's first national parks were set up in the 19th century, but decades of unregulated pollution have taken a toll on many of the country's wild places.
Currently, about one percent of Russia's land area is protected in preserves, known as zapovedniks.
Russia's most famous animal species is the Siberian tiger, the largest cat in the world.
Indigenous to the forests of eastern Russia, these endangered giants can be 10 feet (3 meters) long, not including their tail, and weigh up to 600 pounds (300 kilograms).
Russia's history as a democracy is short.
Russia is a federation of 86 republics, provinces, territories, and districts, all controlled by the government in Moscow.
The earliest human settlements in Russia arrived around A.D. 500, as Scandinavians (what is now Norway, Denmark, and Sweden) moved south to areas around the upper Volga River.
In the 1550s, Muscovite ruler Ivan IV became Russia's first tsar, or emperor, after driving the Mongols out of Kiev and unifying the region.
Laptop is a portable computer (which can be brought anywhere) and it is integrated in a casing.
In general , this laptop has a weight ranging from 1 to 6 kilograms , depending on the size , material and specifications of the laptop itself.
Laptops are designed to be used in mobile with its small size and light weight enough to be put on one’s lap while in use.
A laptop has integrated with most of the typical components such as desktop computers.
The screen that used in laptop is the LCD ( Liquid Crystal Display) which typically measuring 10 inches to 17 inches depending on the size of a laptop.
The battery that can charge as this laptop is usually storing energy to be used for two to three hours in its initial state.
Most laptops have the same types of ports found on desktop computers (such as USB), although they usually have fewer ports to save space.
Generally speaking, laptops tend to be more expensive than a desktop computer with the same internal components.
While you may find that some basic laptops cost less than desktop computers, these are usually much less powerful machines.
Typically, laptop random access memory (RAM) is 4, 8 or 16 GB.
Laptops vary in durability; some are designed for use in rugged conditions.
The first "laptop-sized notebook computer" was the Epson HX-20, released in July 1982.
The first computer to use the "clamshell" design which is used in almost all modern laptop designs, was the GRiD Systems Corporation's GriD Compass, released in April 1982.
The development of laptops is continuing, with various upgrades and additional functions added, including variouspointing devices and disk drives, colour screens and touch screens.
On average, cats sleep 16-18 hours a day.
Although there is a common perception that cats love milk, most of them are lactose intolerant and should not drink cow's milk.
While this is hard to notice in a closed environment, cats can actually run up to 30 miles per hour.
They can also jump seven times their height.
Cat's whiskers are not just for show.
Cat's whiskers are mood indicators. If they hang down relaxed, the cat is satisfied. If they are facing straight ahead, they may be angry.
Cat's serve to help cats navigate in the dark and detect objects.
Touching each other noses in cat language means greeting, like when humans shake hands.
Cats love catnip, especially the real catnip (Nepeta Cataria). It has an intoxicating effect on the animals.
Cats mark their owners as possessions. They have scent glands in their cheeks, paws and flanks.
The cat's tongue is equipped with tiny spikes (papillae).
Red cats are mostly male. Tricolored cats are almost always female.
Cats primarily only meow for humans. Communication with other cats takes place via body language.
The oldest cat was 38 years old.
Cats are believed to be the only mammals who don’t taste sweetness.
Cats can jump up to six times their length.
Cats have 230 bones, while humans only have 206.
Some cats can swim.
""".split("\n")[1:-1]

all_text2 = all_text


print("----------------Тексты-----------------")
print(all_text)
print("---------------------------------------")
tfidf_vectorizer = TfidfVectorizer(preprocessor=preprocessing1)
tfidf_vectorizer2 = TfidfVectorizer(preprocessor=preprocessing2)


tfidf = tfidf_vectorizer.fit_transform(all_text)
tfidf2 = tfidf_vectorizer2.fit_transform(all_text2)

print(tfidf_vectorizer.vocabulary_)
print("//////////////////////////")
print(tfidf_vectorizer2.vocabulary_)
print("---------------Результат векторизации без морфол. анализа--------------")
print(tfidf)
print("---------------Результат векторизации с морфол. анализом--------------")
print(tfidf2)
print("---------------------------------------")

feature_names = tfidf_vectorizer.get_feature_names_out()
feature_names2 = tfidf_vectorizer2.get_feature_names_out()
all_text_index = [n for n in all_text]
all_text_index2 = [n for n in all_text2]
df = pd.DataFrame(tfidf.todense(), index=all_text_index, columns=feature_names)
df2 = pd.DataFrame(tfidf2.todense(), index=all_text_index2, columns=feature_names2)
print(df)
print(df2)

kmeans = KMeans(n_init=10, n_clusters=3).fit(tfidf)
kmeans2 = KMeans(n_init=10, n_clusters=3).fit(tfidf2)

print("----------------Кластеризация-всех-предложений-без-морф.-анализа-------------")
print(kmeans.labels_)
print("-------------------------------------------------------")

lines_for_predicting = ["RUSSIA!",
                        "laptops - multifunctional",
                        "Сat's are awesome"]

predicted = kmeans.predict(tfidf_vectorizer.transform(lines_for_predicting))

print("----------------Кластеризация-пробных-предложений-без-морф.-анализа-------------")
print(predicted)
print("-----------------------------------------------------------------------------------")
# array([0, 1], dtype=int32)

print("----------------Кластеризация-всех-предложений-с-морф.-анализом-------------")
print(kmeans2.labels_)
print("-------------------------------------------------------")
predicted2 = kmeans2.predict(tfidf_vectorizer2.transform(lines_for_predicting))
print("----------------Кластеризация-пробных-предложений-с-морф.-анализом-------------")
print(predicted2)
print("-----------------------------------------------------------------------------------")


# word2vec

stop = set(stopwords.words("english"))
data = pd.DataFrame(all_text, columns =['title'])
data2 = data
print("----------------Стоп-слова--------------")
print(stop)
print("----------------Текст--------------")
print(data)
#print(data.title)
#print(data2)
#print(data2.title)
#print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

def preprocess(text):
    text_input = re.sub('[^a-zA-Z1-9]+', ' ', str(text))
    #print("yyyyyyyyyyyyyyyyyyyyyyyyyyy")
    #print(text_input)
    output = re.sub(r'\d+', '',text_input)
    #print("nnnnnnnnnnnnnnnnnnnnnnnnnnn")
    #print(output.lower().strip())
    return output.lower().strip()

def remove_stopwords(text):
    filtered_words = [word.lower() for word in text.split() if word.lower() not in stop]
    #print(filtered_words)
    #print(filtered_words)
    return " ".join(filtered_words)


def remove_stopwords2(text):
    filtered_words = [word.lower() for word in text.split() if word.lower() not in stop]
    #print(filtered_words)
    qqq = []
    # print(sss)
    for i in filtered_words:
        qqq.append(lemmatizer.lemmatize(i, pos=get_wordnet_pos(i)))
    #print(qqq)
    filtered_words = qqq
    #print(qqq)
    return " ".join(filtered_words)


data['title'] = data.title.map(preprocess)
data2['title'] = data2.title.map(preprocess)

a = data.title.map(remove_stopwords)
b = data2.title.map(remove_stopwords2)
#print(a)
#print("{{{{{{{{{{{{{{{{{{{{{{{{")
#print(b)

data['title'] = data.title.map(remove_stopwords)
#print("-==-=-----=-=-=-=--=-=--===-=--=-=-")
data2['title'] = data2.title.map(remove_stopwords2)



def build_corpus(data):
    corpus = []
    for sentence in data.items():
        word_list = sentence[1].split(" ")
        corpus.append(word_list)
        #print(corpus)
    return corpus

#print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
#print(data['title'])
#print(a)

corpus3 = build_corpus(data['title'])

corpus4 = build_corpus(data2['title'])

corpus = build_corpus(a)

corpus2 = build_corpus(b)

print("++++++++++++++Без+морф+ анализа+++++++++++++++++")
print(corpus)
print("++++++++++++++С+морф.+анализом++++++++++++++++++")
print(corpus2)
print("++++++++++++++++++++++++++++++++++++++++++++++++")
#print(corpus3)
#print("++++++++++++++++++++++++++++++++")
#print(corpus4)

model = Word2Vec(corpus, vector_size=100, min_count=1)
model2 = Word2Vec(corpus2, vector_size=100, min_count=1)



# fit a 2d PCA model to the vectors
vectors = model.wv[model.wv.key_to_index]
print(vectors)
words = list(model.wv.key_to_index)
pca = PCA(n_components=2)
PCA_result = pca.fit_transform(vectors)


vectors2 = model2.wv[model2.wv.key_to_index]
print(vectors2)
words2 = list(model2.wv.key_to_index)
pca = PCA(n_components=2)
PCA_result2 = pca.fit_transform(vectors2)


# prepare a dataframe
words = pd.DataFrame(words)
PCA_result = pd.DataFrame(PCA_result)
PCA_result['x_values'] =PCA_result.iloc[0:, 0]
PCA_result['y_values'] =PCA_result.iloc[0:, 1]
PCA_final = pd.merge(words, PCA_result, left_index=True, right_index=True)
PCA_final['word'] =PCA_final.iloc[0:, 0]
PCA_data_complet =PCA_final[['word','x_values','y_values']]

print(PCA_data_complet)
PCA_to_cluster = PCA_data_complet[['x_values','y_values']]
print(PCA_to_cluster)


# prepare a dataframe С МОРФ, АНАЛИЗОМ
words2 = pd.DataFrame(words2)
PCA_result2 = pd.DataFrame(PCA_result2)
PCA_result2['x_values'] =PCA_result2.iloc[0:, 0]
PCA_result2['y_values'] =PCA_result2.iloc[0:, 1]
PCA_final2 = pd.merge(words2, PCA_result2, left_index=True, right_index=True)
PCA_final2['word'] =PCA_final2.iloc[0:, 0]
PCA_data_complet2 =PCA_final2[['word','x_values','y_values']]

print(PCA_data_complet2)
PCA_to_cluster2 = PCA_data_complet2[['x_values','y_values']]
print(PCA_to_cluster2)



import plotly.graph_objects as go
import numpy as np

N = 1000000

fig = go.Figure(data=go.Scattergl(
   x = PCA_data_complet['x_values'],
   y = PCA_data_complet['y_values'],
   mode='markers',
   marker=dict(
       color=np.random.randn(N),
       colorscale='Viridis',
       line_width=1
   ),
   text=PCA_data_complet['word'],
   textposition="bottom center"
))
fig.show()

N = 1000000

fig = go.Figure(data=go.Scattergl(
   x = PCA_data_complet2['x_values'],
   y = PCA_data_complet2['y_values'],
   mode='markers',
   marker=dict(
       color=np.random.randn(N),
       colorscale='Viridis',
       line_width=1
   ),
   text=PCA_data_complet2['word'],
   textposition="bottom center"
))

fig.show()


kmeans_word2vec = KMeans(n_init=10, n_clusters=3).fit(PCA_to_cluster)
kmeans_word2vec2 = KMeans(n_init=10, n_clusters=3).fit(PCA_to_cluster2)
print("----------------Кластеризация-всех-предложений-без-морф.-анализа-------------")
print(kmeans_word2vec.labels_)
print("----------------Кластеризация-всех-предложений-c-морф.-анализом--------------")
print(kmeans_word2vec2.labels_)


'''
sns.set_style('ticks')
# word frequencies calculation
from collections import Counter
# count unique words

def counter_word(text_col):
    count = Counter()
    for text in text_col.values:
        for word in text.split():
            count[word] += 1
    return count


counter_all = counter_word(data.title)
words_all = counter_all.most_common(100)

words_all= pd.DataFrame(words_all)
words_all['word'] = words_all.iloc[0:, 0]
words_all['count'] = words_all.iloc[0:, 1]
words_all = words_all[['word','count']]
words_all.to_csv('top100_words.csv', columns=['word','count'], index=False)



sns.set_style('ticks')
fig = sns.lmplot(x='x_values', y='y_values', data = PCA_to_cluster2, fit_reg=False, legend=True, hue='Cluster')
fig = plt.gcf()
fig.set_size_inches(8, 6)

plt.savefig('word2vec_clustering.png')

plt.show()
'''