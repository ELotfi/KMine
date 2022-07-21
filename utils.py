import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import json, math,re
from collections import Counter
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F


def top_filtering(logits, top_k=0, top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):

	assert logits.dim() == 1  
	top_k = min(top_k, logits.size(-1))
	if top_k > 0:
		indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
		logits[indices_to_remove] = filter_value

	if top_p > 0.0:
		sorted_logits, sorted_indices = torch.sort(logits, descending=True)
		cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

		sorted_indices_to_remove = cumulative_probabilities > top_p
		sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
		sorted_indices_to_remove[..., 0] = 0

		indices_to_remove = sorted_indices[sorted_indices_to_remove]
		logits[indices_to_remove] = filter_value

	indices_to_remove = logits < threshold
	logits[indices_to_remove] = filter_value

	return logits


##############################################################
#### Trackers  ###############################################
##############################################################


class LossTracker(object):
	def __init__(self, args):
		self.args = args
		self.reset()

	def reset(self):
		self.losses = {"lm_loss":[]}
		if self.args.with_kn_loss: self.losses.update({"total_loss":[], "kn_loss":[]})

	def update(self, outputs):
		for k in self.losses:
			self.losses[k].append(outputs[k].item())


	def log(self, logger, step, split):
		for k,v in self.losses.items():
			logger.add_scalar(f'{split}/ {k}', sum(v)/len(v), step)
			if k=='lm_loss': logger.add_scalar(f'{split}/ PPL', math.exp(sum(v)/len(v)), step)



class RecallTracker(object):
	def __init__(self, args):
		self.args = args
		self.reset()

	def reset(self):
		self.ratns = {f"R@{n}":[] for n in self.args.ratn}

	
	def update(self, probs, label):
		for k in self.args.ratn:
			top_k = torch.topk(probs, min(k, probs.shape[1]), dim=1)[1]
			ratk = [int(label[i] in top_k[i,:]) for i in range(probs.shape[0])]
			self.ratns[f"R@{k}"] += ratk

	
	def log(self, logger, step, split):
		for k,v in self.ratns.items():
			logger.add_scalar(f'{split}/ {k}', sum(v)/len(v), step)



class Trackers(object):
	def __init__(self, args):
		self.args = args
		self.loss_trk = LossTracker(args)
		self.recl_trk = RecallTracker(args)
		self.logger = SummaryWriter(comment=args.logger_id)


	def reset(self):
		self.loss_trk.reset()
		self.recl_trk.reset()

	def update(self, outputs, labels):
		self.loss_trk.update(outputs)
		self.recl_trk.update(outputs['kn_probs'], labels)

	def log(self, step, split):
		self.loss_trk.log(self.logger, step, split)
		self.recl_trk.log(self.logger, step, split)

	def get_ave_loss(self, key=None):
		if key is None: key = 'total_loss' if self.args.with_kn_loss else 'lm_loss'
		loss = self.loss_trk.losses[key]
		return sum(loss)/len(loss)

	def log_custom(self, name, step, split, value):
		self.logger.add_scalar(f'{split}/ {name}', value, step)


##############################################################
#### Evaluation and Inference  ###############################
##############################################################
re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')

def normalize_answer(s):
	"""
	From Parlai, lower text and remove punctuation, articles and extra whitespace.
	"""
	s = s.lower()
	s = re_punc.sub(' ', s)
	s = re_art.sub(' ', s)
	s = s.strip()
	s = ' '.join(s.split())
	return s


def f1_score(gold_str, pred_str):
	"""From parlai"""
	g_tokens = normalize_answer(gold_str).split()
	p_tokens = normalize_answer(pred_str).split()
	common = Counter(g_tokens) & Counter(p_tokens)
	num_same = sum(common.values())
	if num_same == 0:
		return 0
	precision = 1.0 * num_same / len(p_tokens)
	recall = 1.0 * num_same / len(g_tokens)
	f1 = (2 * precision * recall) / (precision + recall)
	return f1


def eval_epoch(args, model, datasets, epoch, trackers):
	model.eval()
	print('Validation ...')
	for data_name, data_loader in datasets.items():
		trackers.reset()
		for batch in tqdm(data_loader): 
			for k in batch: 
				if k!= 'last_utterance_index': batch[k]=batch[k].to(args.device)
			output = model(**batch)
			trackers.update(output, batch['candidate_labels'])
		avg_loss = trackers.get_ave_loss()
		print(f"{data_name} Loss :", avg_loss)
		trackers.log(epoch, data_name)
	return avg_loss




def inference(args, model_wr, datasets, epoch, trackers):
	device = args.device
	model_wr.eval()
	model = model_wr if not args.distributed else model_wr.module
	tokenizer = model.tokenizer
	for data_name, data_loader in datasets.items():
		gen_responses , gold_responses, f1_scores, pred_kn, gold_kn = [], [], [], [], []
		trackers.reset()
		for batch in tqdm(data_loader): 
			for k in batch: 
				if k!= 'last_utterance_index': batch[k]=batch[k].to(args.device)
			output = model_wr(**batch, with_encoder_outputs=True)
			
			trackers.update(output, batch['candidate_labels'])
			pred_kn.append(output['kn_probs'].argmax(1))
			gold_kn.append(batch['candidate_labels'])
			gold_response_ids = batch['decoder_input_ids'][0]
			gold_response = tokenizer.decode(gold_response_ids, skip_special_tokens=True)
			gold_response = normalize_answer(gold_response)
			# inference 
			encoder_hidden_states = output['encoder_hidden_states']
			past_key_values = None
			new_token = tokenizer.convert_tokens_to_ids(['<agent>'])
			new_token = torch.tensor(new_token).unsqueeze(0).to(device)
			gen_response = new_token.clone()
			while (new_token[0] != tokenizer.eos_token_id) and (gen_response.shape[1] < args.generation_max_length):
				decoder_outputs = model.core.model.decoder(input_ids=new_token, encoder_hidden_states=encoder_hidden_states, past_key_values=past_key_values, use_cache=True)
				lm_logits = model.core.lm_head(decoder_outputs[0])
				lm_logits = lm_logits[0,-1,:]/args.temperature
				lm_logits = top_filtering(lm_logits, top_k=args.top_k, top_p=args.top_p)
				probs = F.softmax(lm_logits, dim=-1)
				new_token = torch.multinomial(probs, 1).unsqueeze(-1) if args.do_sample else torch.topk(probs, 1)[1].unsqueeze(-1)  
				gen_response = torch.cat([gen_response, new_token], dim=1)	
				past_key_values = decoder_outputs[1]	
			
			gen_response = tokenizer.decode(gen_response[0], skip_special_tokens=True)
			gen_response = normalize_answer(gen_response)
			f1 = f1_score(gold_response, gen_response)
			gen_responses.append(gen_response)
			gold_responses.append(gold_response)
			f1_scores.append(f1)
		
		trackers.log(epoch, data_name)
		trackers.log_custom('F1', epoch, data_name, sum(f1_scores)/len(f1_scores))
		pred_kn = torch.cat(pred_kn).squeeze().cpu().numpy().tolist()
		gold_kn = torch.cat(gold_kn).cpu().numpy().tolist()
		results = pd.DataFrame({'Gold_Response':gold_responses, 'Generated_Response':gen_responses, 'Gold_candidate_idx':gold_kn, 'Predicted_kn_idx':pred_kn})
		results.to_csv(f"{args.generation_path}{args.logger_id}_{data_name}_ep{epoch}.csv")
		