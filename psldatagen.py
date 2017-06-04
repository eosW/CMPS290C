import csv

from iacorpus import load_dataset
from gensim import corpora, models
import numpy as np
import pandas as pd
from sqlalchemy import Table, Column, Integer, sql
from sklearn.model_selection import train_test_split

# read data
dataset = load_dataset('fourforums', host='localhost', port='3306', username='root', password='symwrm')

view = Table('psl', dataset.connection.metadata, Column('id', Integer, primary_key=True), autoload=True,
             autoload_with=dataset.connection.engine)

query = dataset.connection.session.query(view)

qrs = pd.read_sql(query.statement, query.session.bind)

qrs_train, qrs_test = train_test_split(qrs, test_size=0.2)

# relations
quote = qrs[['id']]
quote.to_csv('PSL/Quote.txt', sep="\t", index=False, header=False)

quote_discussion = qrs[['id', 'discussion_id']].dropna()
quote_discussion.to_csv('PSL/Quote_Discussion.txt', sep="\t", index=False, header=False)
quote_discussion.discussion_id.drop_duplicates().to_csv('PSL/Discussion.txt', sep="\t", index=False, header=False)

quote_post = qrs[['id', 'discussion_id', 'post_id']].dropna()
quote_post['post_uid'] = quote_post.discussion_id.apply(str) + "_" + quote_post.post_id.apply(str)
quote_post = quote_post[['id', 'post_uid']]
quote_post.to_csv('PSL/Quote_Post.txt', sep="\t", index=False, header=False)
quote_post.post_uid.drop_duplicates().to_csv('PSL/Post.txt', sep="\t", index=False, header=False)

quote_author = qrs[['id', 'author_id']].dropna()
quote_author.to_csv('PSL/Quote_Author.txt', sep="\t", index=False, header=False)
quote_author.author_id.drop_duplicates().to_csv('PSL/Author.txt', sep="\t", index=False, header=False)

quote_topic = qrs[['id', 'topic']].dropna()
quote_topic.topic = quote_topic.topic.apply(lambda x: x.replace(" ", "_"))
quote_topic.to_csv('PSL/Quote_Topic.txt', sep="\t", index=False, header=False)
quote_topic.topic.drop_duplicates().to_csv('PSL/Topic.txt', sep="\t", index=False, header=False)

quote_response = qrs[['id', 'discussion_id', 'source_post_id']].dropna()
quote_response['source_post_uid'] = quote_response.discussion_id.apply(str) + "_" + quote_response.source_post_id.apply(
    str)
quote_response = quote_response[~quote_response.source_post_uid.isin(quote_post.post_uid)]
quote_response.to_csv('PSL/Quote_Response.txt', sep="\t", index=False, header=False)

# labels
quote_stance = qrs[['id', 'topic_stance_votes_1', 'topic_stance_votes_2']].dropna()
quote_stance['stance'] = [1 if x >= 2 else (0 if x <= .5 else np.nan) for x in
                          quote_stance.topic_stance_votes_1 / quote_stance.topic_stance_votes_2]
quote_stance = quote_stance[['id', 'stance']].dropna()
quote_stance.to_csv('PSL/Quote_Stance.txt', sep="\t", index=False, header=False)
quote_stance.stance.drop_duplicates().to_csv('PSL/Stance.txt', sep="\t", index=False, header=False)

quote_tag = qrs[['id', 'disagree_agree', 'emotion_fact', 'nasty_nice', 'attacking_respectful', 'sarcasm_yes']].copy()
quote_tag['agree'] = quote_tag.disagree_agree >= 1
quote_tag['disagree'] = quote_tag.disagree_agree <= -1
quote_tag['fact'] = quote_tag.emotion_fact >= 1
quote_tag['emotion'] = quote_tag.emotion_fact <= -1
quote_tag['nice'] = quote_tag.nasty_nice >= 1
quote_tag['nasty'] = quote_tag.nasty_nice <= -1
quote_tag['respectful'] = quote_tag.attacking_respectful >= 1
quote_tag['attacking'] = quote_tag.attacking_respectful <= -1
quote_tag['sarcasm'] = quote_tag.sarcasm_yes >= 0.5
quote_tag[['id', 'disagree', 'agree', 'emotion', 'fact', 'nasty', 'nice', 'attacking', 'respectful', 'sarcasm']].to_csv(
    'rawtag.txt', index=False, )
quote_tag = quote_tag.melt(id_vars=['id'],
                           value_vars=['disagree', 'agree', 'emotion', 'fact', 'nasty', 'nice', 'attacking',
                                       'respectful', 'sarcasm'])
quote_tag[quote_tag.id.isin(qrs_test.id)][['id', 'variable']].to_csv('PSL/Tagging_targets.txt', sep="\t", index=False,
                                                                     header=False)
quote_tag = quote_tag[quote_tag.value][['id', 'variable']]
quote_tag.variable.drop_duplicates().to_csv('PSL/Tag.txt', sep="\t", index=False, header=False)

quote_tag[quote_tag.id.isin(qrs_train.id)].to_csv('PSL/Tagging.txt', sep="\t", index=False, header=False)
quote_tag[quote_tag.id.isin(qrs_test.id)].to_csv('PSL/Tagging_truth.txt', sep="\t", index=False, header=False)

# SVM corpora
presented_quote = qrs.presented_quote.values
presented_response = qrs.presented_response.values

combined = np.concatenate((presented_quote, presented_response))
splited = [sentence.lower().split() for sentence in combined]
vocab = corpora.Dictionary(splited)
vocab.filter_extremes(no_below=5, no_above=0.5, keep_n=None)
corpus = [vocab.doc2bow(text) for text in splited]
tfidf = models.TfidfModel(corpus)
corpus = [tfidf[doc] for doc in corpus]

corpora.SvmLightCorpus.serialize("corpus.txt", corpus)

ids = qrs[['id']].copy()
ids['train'] = ids.id.isin(qrs_train.id)
ids.to_csv('ids.txt', index=False)
