import sys
import os
import codecs
from sys import *
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
#from rank_bm25 import BM25Okapi

ATTR_TO_SPECIAL_TOKEN_BART = {'eos_token': '<eos>', 'pad_token': '<pad>', 'sep_token': '<sep>', 'cls_token': '<bos>', 'additional_special_tokens': ['<agent>', '<user>']}
SPECIAL_TOKENS_BART = ["<eos>", "<cls>", "<sep>", "<pad>", '<agent>', '<user>']


def load_answer(file):
	print("load_answer")
	answer = []
	with codecs.open(file, encoding='utf-8') as f:
		for line in f:
			temp = line.strip('\n').strip('\r').split('\t', 3)

			assert len(temp) == 4,"all_previous_query_id;all_previous_query_id;all_previous_query_id	current_query_id	background_id;background_id 	response_content"
			if len(temp[0]) < 1:
				temp[0] = []
			else:
				temp[0] = temp[0].split(';')
			temp[2] = temp[2].split(';')
			answer.append(temp)
	return answer


def load_passage(file, pool):  # background_id	background_content
	print("load_passage")
	poolset = set()
	for k in pool:
		poolset.update(pool[k])

	passage = dict()
	with codecs.open(file, encoding='utf-8') as f:
		for line in f:
			temp = line.strip('\n').strip('\r').split('\t', 1)
			assert len(temp) == 2, "load_passage"
			if temp[0] in poolset:
				#passage[temp[0]] = ' [SEP] '.join([' '.join(tokenizer(sent)) for sent in nltk.sent_tokenize(temp[1])]).split(' ')  # list的形式
				passage[temp[0]] = temp[1] #' [SEP] '.join([' '.join(tokenizer(sent)) for sent in nltk.sent_tokenize(temp[1])]).split(' ')  # list的形式
			
	print("passage:{}, poolset:{}".format(len(passage), len(poolset)))
	return passage  # {background_id1:background_content, background_id2:background_content}


def load_pool(file, topk=None):  # current_query_id Q0 background_id rank relevance_score model_name
	print("load_pool")
	pool = {}
	with codecs.open(file, encoding='utf-8') as f:
		for line in f:
			temp = line.strip('\n').strip('\r').split(' ')
			assert len(temp) == 6, "load_pool"
			if temp[0] not in pool:
				pool[temp[0]] = [temp[2]]  # {“current_query_id”:[background_id1]}
			else:
				pool[temp[0]].append(temp[2])  # {“current_query_id”:[background_id1,background_id2,background_id3...]}
	return pool


def load_qrel(file):
	print("load_qrel")
	qrel = dict()
	with codecs.open(file, encoding='utf-8') as f:
		for line in f:
			temp = line.strip('\n').strip('\r').split(' ')
			assert len(temp) == 4, "load_qrel"
			if int(temp[3]) > 0:
				qrel[temp[0]] = temp[2]  # {current_query_id:background_id1, current_query_id2:background_id2........}
	return qrel


def load_query(file):  # query_id	query_content
	print("load_query")
	query = dict()
	with codecs.open(file, encoding='utf-8') as f:
		for line in f:
			temp = line.strip('\n').strip('\r').split('\t',1)
			assert len(temp) == 2, "load_query"
			query[temp[0]] = temp[1]  # {1_1:[query_tokens],}
	return query


def load_split(dataset, file):
	train = set()
	dev_seen = set()
	dev_unseen = set()
	if dataset == "wizard_of_wikipedia":
		test_seen = set()
		test_unseen = set()
		with codecs.open(file, encoding='utf-8') as f:
			for line in f:
				temp = line.strip('\n').strip('\r').split('\t')
				assert len(temp) == 2, "query_id train/dev/test_seen/test_unseen"
				if temp[1] == 'train':
					train.add(temp[0])
				elif temp[1] == 'dev_seen':
					dev_seen.add(temp[0])
				elif temp[1] == 'dev_unseen':
					dev_unseen.add(temp[0])	
				elif temp[1] == 'test_seen':
					test_seen.add(temp[0])
				elif temp[1] == 'test_unseen':
					test_unseen.add(temp[0])

		return train, dev_seen, dev_unseen, test_seen, test_unseen

	elif dataset == "holl_e":
		test = set()
		dev = set()
		with codecs.open(file, encoding='utf-8') as f:
			for line in f:
				temp = line.strip('\n').strip('\r').split('\t')
				assert len(temp) == 2, "query_id train/dev/test"
				if temp[1] == 'train':
					train.add(temp[0])
				elif temp[1] == 'dev':
					dev.add(temp[0])
				elif temp[1] == 'test':
					test.add(temp[0])
		return train, dev, test


def split_data(dataset, split_file, samples):
	print("split_data:", dataset)
	train_samples = list()
	dev_samples = list()

	if dataset == "wizard_of_wikipedia":
		train, dev_seen, dev_unseen, test_seen, test_unseen = load_split(dataset, split_file)
		dev_seen_samples = list()
		dev_unseen_samples = list()
		test_seen_samples = list()
		test_unseen_samples = list()
		for sample in samples:
			if sample['query_id'] in train:
				train_samples.append(sample)
			elif sample['query_id'] in dev_seen:
				dev_seen_samples.append(sample)
			elif sample['query_id'] in dev_unseen:
				dev_unseen_samples.append(sample)				
			elif sample['query_id'] in test_seen:
				test_seen_samples.append(sample)
			elif sample['query_id'] in test_unseen:
				test_unseen_samples.append(sample)
		return train_samples, dev_seen_samples, dev_unseen_samples, test_seen_samples, test_unseen_samples

	elif dataset == "holl_e":
		train, dev, test = load_split(dataset, split_file)
		test_samples = list()
		for sample in samples:
			if sample['query_id'] in train:
				train_samples.append(sample)
			elif sample['query_id'] in dev:
				dev_samples.append(sample)
			elif sample['query_id'] in test:
				test_samples.append(sample)
		return train_samples, dev_samples, test_samples


def load_default(answer_file, passage_file, pool_file, qrel_file, query_file):
	random.seed(1)
	answer = load_answer(answer_file)  # [[all_previous_query_ids],current_query_id,[background_ids],[response_tokens]]
	pool = load_pool(pool_file, None)  # {“current_query_id1”:[background_id1,background_id2,background_id3...]，“current_query_id2”:[background_id1,background_id2,background_id3...]}
	query = load_query(query_file)  # {current_query_id_1:[query_tokens],current_query_id_2:[query_tokens]}
	passage = load_passage(passage_file, pool)  # {background_id1:[background_tokens], [background_id2:[background_tokens]}
	average_pool = 0
	samples = []
	for i in tqdm(range(len(answer))):
		c_id, q_id, knowledge_id, response = answer[i]  # c_id is a lis，q_id is string，p_id is a list，ans is a list
		knowledge_pool = pool[q_id]
		average_pool += len(knowledge_pool)

		for p in knowledge_id:  # label knowledge sentence id
			if p not in knowledge_pool:
				raise Exception("label shifting knowledge is not in knowledge shifting pool")

		# we want the correct knowledge to always be in index 0
		i = knowledge_pool.index(knowledge_id[0])
		if i == 0:
			pass
		else:
			knowledge_pool[0], knowledge_pool[i] = knowledge_pool[i], knowledge_pool[0]

		sample = dict()
		sample['context_id'] = c_id  # list ：[previous utterance]
		sample['query_id'] = q_id  # string ：current query
		sample['response'] = response  # string
		sample['knowledge_pool'] = knowledge_pool  # list
		sample['knowledge_label'] = knowledge_id
		samples.append(sample)  # [{example1},{example2},{example3}...]

	print("average knowledge pool:", average_pool/len(samples))
	print('total eamples:', len(samples))

	return samples, query, passage



def get_wowdk_data(args):
	dataset = 'wizard_of_wikipedia'
	if os.path.exists(args.data_path + 'train_DukeNet.pkl'):
		query = torch.load(args.data_path + 'query_DukeNet.pkl')
		passage = torch.load(args.data_path + 'passage_DukeNet.pkl')
		train_samples = torch.load(args.data_path + 'train_DukeNet.pkl')
		dev_seen_samples = torch.load(args.data_path + 'dev_seen_DukeNet.pkl')
		dev_unseen_samples = torch.load(args.data_path + 'dev_unseen_DukeNet.pkl')
		test_seen_samples = torch.load(args.data_path + 'test_seen_DukeNet.pkl')
		test_unseen_samples = torch.load(args.data_path + 'test_unseen_DukeNet.pkl')

	else:
		samples, query, passage = load_default(args.data_path + dataset + '.answer',
											args.data_path + dataset + '.passage',
											args.data_path + dataset + '.pool',
											args.data_path + dataset + '.qrel',
											args.data_path + dataset + '.query')

		train_samples, dev_seen_samples, dev_unseen_samples, test_seen_samples, test_unseen_samples = split_data(dataset, args.data_path + dataset + '.split', samples)
		torch.save(query, args.data_path + 'query_DukeNet.pkl')
		torch.save(passage, args.data_path + 'passage_DukeNet.pkl')
		torch.save(train_samples, args.data_path + 'train_DukeNet.pkl')
		torch.save(dev_seen_samples, args.data_path + 'dev_seen_DukeNet.pkl')
		torch.save(dev_unseen_samples, args.data_path + 'dev_unseen_DukeNet.pkl')	
		torch.save(test_seen_samples, args.data_path + 'test_seen_DukeNet.pkl')
		torch.save(test_unseen_samples, args.data_path + 'test_unseen_DukeNet.pkl')

	return train_samples, dev_seen_samples, dev_unseen_samples, test_seen_samples, test_unseen_samples, query, passage
	













