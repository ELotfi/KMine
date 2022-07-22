import torch
import entmax
import shutil, os
from torch import nn
from torch.nn import CrossEntropyLoss
from data import ATTR_TO_SPECIAL_TOKEN_BART
from transformers import BartForConditionalGeneration, AutoTokenizer



class Fusion(nn.Module):
	def __init__(self, h_size, max_function):
		super().__init__()
		self.clf = nn.Linear(h_size*2, 1)
		self.max_f = nn.Softmax(dim=1) if max_function == 'softmax' else entmax.Entmax15(dim=1)


	def forward(self, enc_outputs, masks, lu_indx):
		candidate_pool = enc_outputs[:, :, 0, :]
		last_utterance = torch.stack([enc_outputs[i , :, lu_indx[i][0]:lu_indx[i][1] , :].mean(1) for i in range(len(lu_indx))], dim=0) 
		candidate_pool = torch.cat([candidate_pool, last_utterance], -1)
		logits = self.clf(candidate_pool)		
		probs = self.max_f(logits)
		fused_candidates = (probs.unsqueeze(-1)*masks.unsqueeze(-1)*enc_outputs).sum(1)
		return probs, logits, fused_candidates




class KMine(nn.Module):
	def __init__(self, args):
		super().__init__()
		self.args = args
		self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path)
		self.core = BartForConditionalGeneration.from_pretrained(args.model_name_or_path)
		self.fuse = Fusion(self.core.config.d_model, args.max_function)
		self.loss_fc = CrossEntropyLoss()
		self.add_new_tokens()



	def add_new_tokens(self):
		orig_num_tokens = len(self.tokenizer)
		num_added_tokens = self.tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN_BART)
		if num_added_tokens > 0 :
			self.core.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)



	def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, 
				labels=None, candidate_labels=None, last_utterance_index=None, with_encoder_outputs=False):
	
		b, n_c, l=input_ids.shape
		input_ids, attention_mask = input_ids.reshape(b*n_c,l), attention_mask.reshape(b*n_c,l)
		enc_output = self.core.model.encoder(input_ids, attention_mask=attention_mask)[0].view(b, n_c, l, -1)
		candidate_probs, candidate_logits, enc_output = self.fuse(enc_output, attention_mask.view(b, n_c, l), last_utterance_index)
		dec_output = self.core.model.decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, encoder_hidden_states=enc_output)[0]
		lm_logits = self.core.lm_head(dec_output)
		kn_loss = self.loss_fc(candidate_logits, candidate_labels) if self.args.with_kn_loss else None
		lm_loss = self.loss_fc(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1)) if labels is not None else None
		total_loss = lm_loss if kn_loss is None else lm_loss + self.args.kn_loss_coef*kn_loss
		
		return {
			"total_loss":total_loss,
			"lm_loss":lm_loss,
			"lm_logits":lm_logits,
			"kn_probs":candidate_probs,
			"kn_loss":kn_loss,
			"encoder_hidden_states":enc_output if with_encoder_outputs else None}

	
	def load_from_checkpoint(self):
		print(f"Loading model from checkpoint at {self.args.checkpoint_path}  ...")
		self.core = BartForConditionalGeneration.from_pretrained(self.args.checkpoint_path)
		self.fuse.load_state_dict(torch.load(f"{self.args.checkpoint_path}/fuse.pth"))



	def save_model(self, epoch):
		if epoch==1: shutil.rmtree(self.args.output_path, ignore_errors=True)
		path = os.path.join(self.args.output_path, f"epoch_{epoch}")
		print(f"Saving checkpoint at {path}  ....")
		os.makedirs(path, exist_ok=True)
		self.core.save_pretrained(path)
		torch.save(self.fuse.state_dict(), f'{path}/fuse.pth')




