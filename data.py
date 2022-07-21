from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import json, random, os
from tqdm import tqdm
import logging
from torch.utils.data.distributed import DistributedSampler
from torch.nn import functional as F
from wow_duke import get_wowdk_data


ATTR_TO_SPECIAL_TOKEN_BART = {'eos_token': '<eos>', 'pad_token': '<pad>', 'sep_token': '<sep>', 'cls_token': '<bos>', 'additional_special_tokens': ['<agent>', '<user>']}
SPECIAL_TOKENS_BART = ["<eos>", "<cls>", "<sep>", "<pad>", '<agent>', '<user>']




class WOWDUKE(Dataset):

	def __init__(self, args, samples, query, passage, training, num_samples=None):  # 1e10=1E10
		super(Dataset, self).__init__()

		self.args = args
		self.training = training
		self.num_samples= num_samples
		self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
		self.samples = samples
		self.query = query
		self.passage = passage
		self.add_new_tokens()
		self.conv_samples = self.build()


	
	def add_new_tokens(self):
		self.tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN_BART) 


	def flatten(self, ls):
		return [i for j in ls for i in j]


	def build(self):
		num_samples = self.num_samples if self.num_samples is not None else 1000000
		conv_samples = []
		args = self.args
		sp_id = ['<user>', '<agent>']
		for idx, sample in tqdm(enumerate(self.samples)):
			if self.is_valid(sample):
				history = (sample['context_id'] + [sample['query_id']])[-args.num_history:]
				history = [self.query[q] for q in history]
				history.reverse()
				history = [f"{sp_id[i%2]} {h}" for i,h in enumerate(history)]
				history.reverse()
				knowledge_pool = self.create_pool(sample, idx)
				assert len(knowledge_pool) == len(set(knowledge_pool))
				labels = [1] + [0]*(len(knowledge_pool) -1)

				zipped = list(zip(knowledge_pool, labels))
				random.shuffle(zipped)
				knowledge_pool, labels = zip(*zipped)

				knowledge_pool = [self.passage[k].replace('__knowledge__', ':') for k in knowledge_pool]
				response = f"<agent> {sample['response']}"
				conv_samples.append({'history':history, 'pool':knowledge_pool, 'response':response, 'label':labels.index(1)})
			if len(conv_samples) >= num_samples: break
		return conv_samples



	def is_valid(self, sample):
		if self.args.data_mode == 'all':
			return True
		elif self.args.data_mode == 'only_with_kn':
			if sample['knowledge_pool'][0] != 'K_0':
				return True
			else: return False



	def create_pool(self, sample, idx):
		args = self.args
		pool = sample['knowledge_pool'][:1]
		for k in sample['knowledge_pool']:
			if k not in pool and ((k !='K_0') or (k== 'K_0' and args.data_mode == 'all')):
				pool.append(k)

		if self.training:
			while len(pool) < args.num_candidates:
				idd = (idx + 1) % len(self.samples)
				for kn in self.samples[idd]['knowledge_pool']:
					if kn not in pool and kn != 'K_0': pool.append(kn)
				idx = idd
			pool = pool[:args.num_candidates]
		return pool




	def __getitem__(self, index):
		cls, eos, pad = self.tokenizer.cls_token_id, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id
		sample = self.conv_samples[index]
		history, candidates, response, label = sample['history'], sample['pool'], sample['response'], sample['label']

		history_ids = [self.tokenizer.encode(h, add_special_tokens=False, max_length= self.args.utterance_max_length, truncation=True) for h in history]
		response_ids = self.tokenizer.encode(response, add_special_tokens=False) + [eos]
		candid_ids = [torch.tensor([cls] + self.tokenizer.encode(c, add_special_tokens=False, 
																	max_length= self.args.candidate_max_length - 1, 
																	truncation=True, padding='max_length')) for c in candidates]
		candid_ids = pad_sequence(candid_ids, batch_first=True, padding_value= pad)

		last_utter_index = self.args.candidate_max_length + sum([len(h) for h in history_ids[:-1]])
		last_utter_index = [last_utter_index, last_utter_index + len(history_ids[-1])]
		input_ids =  torch.tensor(self.flatten(history_ids)).repeat(candid_ids.shape[0],1) 
		input_ids = torch.cat([candid_ids, input_ids], 1)
		mask = (input_ids != pad).long()
		lm_labels = response_ids[1:] + [-100]
				
		output = (input_ids, mask, label, last_utter_index, torch.tensor(response_ids), torch.tensor(lm_labels))
		return output


	def __len__(self):
		return len(self.conv_samples)



	def pad_batch(self, batch, p):
		max_length = max([b.shape[1] for b in batch])
		batch = [F.pad(b, (0, max_length - b.shape[1]), "constant", p) for b in batch]
		return torch.stack(batch)



	def collate(self, batch):
		pad = self.tokenizer.pad_token_id
		input_ids, mask, label, lu_index, response_ids, lm_labels = zip(*batch)
		input_ids = self.pad_batch(input_ids, pad)
		mask = self.pad_batch(mask, 0)
		response_ids = pad_sequence(response_ids, batch_first=True, padding_value=pad)
		response_masks = (response_ids!=pad).long()
		lm_labels = pad_sequence(lm_labels, batch_first=True, padding_value=-100)
		labels = torch.tensor(label)
		output = {"input_ids":input_ids, 
				  "attention_mask":mask, 
				  "decoder_input_ids":response_ids, 
				  "decoder_attention_mask":response_masks, 
				  "labels":lm_labels,
				  "candidate_labels":labels, 
				  "last_utterance_index":lu_index
				  }
		return output




def get_wow_loaders(args):
	if args.is_master: print(f"Creating wow datasets...")
	train_samples, dev_seen_samples, dev_unseen_samples, test_seen_samples, test_unseen_samples, query, passage = get_wowdk_data(args)

	train_dataset = WOWDUKE(args, train_samples, query, passage, True, args.max_train_samples) if args.do_train else []
	dev_seen_dataset = WOWDUKE(args, dev_seen_samples, query, passage, True, args.max_eval_samples) if args.do_evaluate else []
	dev_unseen_dataset = WOWDUKE(args, dev_unseen_samples, query, passage, True, args.max_eval_samples) if args.do_evaluate else []
	test_seen_dataset = WOWDUKE(args, test_seen_samples, query, passage, False, args.max_infer_samples) if args.do_inference else []
	test_unseen_dataset = WOWDUKE(args, test_unseen_samples, query, passage, False, args.max_infer_samples) if args.do_inference else []

	if args.is_master:
		print("Number of train_samples:", len(train_dataset))
		print("Number of dev_seen_samples:", len(dev_seen_dataset))
		print("Number of dev_unseen_samples:", len(dev_unseen_dataset))
		print("Number of test_seen_samples:", len(test_seen_dataset))
		print("Number of test_unseen_samples:", len(test_unseen_dataset))

	sampler = DistributedSampler(train_dataset) if args.distributed else None
	train_loader = DataLoader(train_dataset, sampler=sampler, collate_fn=train_dataset.collate, batch_size=args.train_batch_size, shuffle= not args.distributed, num_workers=args.num_workers) if args.do_train else None
	test_seen_loader = DataLoader(test_seen_dataset, collate_fn=test_seen_dataset.collate, batch_size=args.infer_batch_size, shuffle=False, num_workers=args.num_workers) if args.do_inference else None
	test_unseen_loader = DataLoader(test_unseen_dataset, collate_fn=test_unseen_dataset.collate, batch_size=args.infer_batch_size, shuffle=False, num_workers=args.num_workers) if args.do_inference else None
	test_loaders =  {'Test_seen':test_seen_loader, 'Test_unseen':test_unseen_loader}

	if args.do_evaluate:
		if args.single_valid:
			dev_dataset = torch.utils.data.ConcatDataset([dev_seen_dataset, dev_unseen_dataset])
			dev_loader = DataLoader(dev_dataset, collate_fn=dev_seen_dataset.collate, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
			valid_loaders =  {'Valid':dev_loader}
		else:
			dev_seen_loader = DataLoader(dev_seen_dataset, collate_fn=dev_seen_dataset.collate, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
			dev_unseen_loader = DataLoader(dev_unseen_dataset, collate_fn=dev_unseen_dataset.collate, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
			valid_loaders =  {'Valid_seen':dev_seen_loader, 'Valid_unseen':dev_unseen_loader}
	else: valid_loaders = None


	return train_loader, valid_loaders, test_loaders








