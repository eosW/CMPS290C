from iacorpus import load_dataset
from gensim import corpora,models
import numpy as np

dataset = load_dataset('fourforums', host='localhost',port='3306',username='root',password='symwrm')
query = dataset.connection.Base.classes.Quote
mturk = dataset.connection.Base.classes.Mturk_2010QrEntry
# joined = mturk.join(query, query.discussion_id == mturk.discussion_id and
#     query.post_id == mturk.post_id and
#     query.quote_index == mturk.quote_index)
turks = dataset.connection.session.query(mturk).all()
turk_scores = [(turk,turk.mturk_2010_qr_task1_average_responses[0]) for turk in turks
               if len(turk.mturk_2010_qr_task1_average_responses)==1]

presented_quotes = [turk.presented_quote for turk,_ in turk_scores]
presented_response = [turk.presented_response for turk,_ in turk_scores]
scores = [(score.disagree_agree,score.attacking_respectful,score.emotion_fact,score.nasty_nice,score.sarcasm_yes)
           for _,score in turk_scores]

combined = presented_quotes+presented_response
splited = [sentence.lower().split() for sentence in combined]
vocab = corpora.Dictionary(splited)
vocab.filter_extremes(no_below=5,no_above=0.5,keep_n=None)
corpus = [vocab.doc2bow(text) for text in splited]
tfidf = models.TfidfModel(corpus)
corpus = [tfidf[doc] for doc in corpus]

corpora.SvmLightCorpus.serialize("corpus.txt",corpus)

scores = np.array(scores)
np.save("scores.npy",scores)