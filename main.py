from numpy import False_
from tqdm import tqdm
import torch
import os
import argparse
from torch.cuda.amp import GradScaler, autocast
from transformers import AutoConfig, set_seed , AdamW,  get_linear_schedule_with_warmup
from data import get_wow_loaders
from model import KMine
from utils import Trackers, eval_epoch, inference
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from datetime import timedelta


def train_epoch(args, model, train_data, global_step, schdlr, opt, scaler, trackers):
	model.train()
	model.zero_grad()
	trackers.reset() 
	for i, batch in enumerate(tqdm(train_data, disable=not args.is_master)):
		for k in batch: 
			if k!= 'last_utterance_index': batch[k]=batch[k].to(args.device)
		if args.fp16:
			with autocast(): output = model(**batch)
		else: output = model(**batch)
		loss = output['total_loss']/args.accumulate_grad 
		if args.fp16: scaler.scale(loss).backward()
		else: loss.backward()
		trackers.update(output, batch['candidate_labels'])

		if (i+1)%args.accumulate_grad==0:
			if args.fp16: scaler.unscale_(opt)
			torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
			if args.fp16:
				scaler.step(opt)
				scaler.update()
			else: opt.step()
			opt.zero_grad()
			schdlr.step()
			if args.is_master: trackers.log(global_step, 'Train')
			global_step += 1
			trackers.reset()
	return global_step




def main(args):
	set_seed(args.seed)
	delta = timedelta(minutes=120)

	if args.distributed:
		torch.distributed.init_process_group(backend='nccl', init_method='env://', timeout=delta)

	model = KMine(args)
	if args.load_from_checkpoint: model.load_from_checkpoint()
	model.to(args.device)

	train_loader, valid_loaders, test_loaders = get_wow_loaders(args)

	if args.do_train:
		if args.distributed:
			model = DistributedDataParallel(model, device_ids=[args.local_rank])
		sel_params = model.fuse.parameters() if not args.distributed else model.module.fuse.parameters()
		gen_params = model.core.parameters() if not args.distributed else model.module.core.parameters()
		opt = AdamW([
			{"params": sel_params, "lr": args.lr_sel},
			{"params": gen_params, "lr":args.lr_gen}], lr=1.e-5)	
		s_total = len(train_loader) // args.accumulate_grad * args.epochs
		schdlr = get_linear_schedule_with_warmup(opt, num_warmup_steps=args.warmup_steps, num_training_steps=s_total)
		scaler = GradScaler() if args.fp16 else None

	best_value, global_step = 100, 0
	trackers = Trackers(args)
	for epoch in range(1,args.epochs+1):
		if args.do_train:
			global_step = train_epoch(args, model, train_loader, global_step, schdlr, opt, scaler, trackers) 
		if args.do_evaluate and args.is_master:
			with torch.no_grad():
				avg_val_loss = eval_epoch(args, model, valid_loaders, epoch, trackers)
				if args.do_train and args.save_best and avg_val_loss < best_value:
					model.save_model(epoch) if not args.distributed else model.module.save_model(epoch)
					best_value = avg_val_loss
		if args.do_inference and epoch in args.epochs_predict and args.is_master:
			with torch.no_grad():
				inference(args, model, test_loaders, epoch, trackers)
		

	if args.distributed: dist.destroy_process_group()




if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_name_or_path", default='facebook/bart-base')
	parser.add_argument("--load_from_checkpoint", default=False)
	parser.add_argument("--checkpoint_path", default='outputs/bart-base_unsup_all/epoch_8')
	parser.add_argument("--output_path", default='outputs/')
	parser.add_argument("--generation_path", default='outputs/results/')
	parser.add_argument("--data_path", default='data/WOW_duke/')
	parser.add_argument('--num_history', default= 3, type=int)
	parser.add_argument('--candidate_max_length', default= 32, type=int)
	parser.add_argument('--utterance_max_length', default= 48, type=int)
	parser.add_argument('--generation_max_length', default= 48, type=int)
	parser.add_argument('--num_candidates', default= 56, type=int)	
	parser.add_argument('--data_mode', default= 'all', choices=['all', 'only_with_kn'])	
	parser.add_argument('--num_workers', type=int, default=8)

	parser.add_argument("--save_best", default=True)
	parser.add_argument("--single_valid", default=True)
	parser.add_argument('--max_train_samples', default= None)
	parser.add_argument('--max_eval_samples', default= None)
	parser.add_argument('--max_infer_samples', default= None)
	parser.add_argument("--do_train", default=True)
	parser.add_argument("--do_evaluate", default=True)
	parser.add_argument("--do_sanity_check", default=False)

	parser.add_argument("--do_inference", default=True)
	parser.add_argument("--do_sample", default=False)
	parser.add_argument('--temperature', default= 1., type=float)
	parser.add_argument('--top_k', default= .0, type=float)
	parser.add_argument('--top_p', default= 1., type=float)
	parser.add_argument('--epochs_predict', default= [1,5,8])

	parser.add_argument("--with_kn_loss", default=False)
	parser.add_argument('--kn_loss_coef', default= .5, type=float)
	parser.add_argument('--max_function', default= 'softmax')	

	parser.add_argument('--epochs', default= 8, type=int)
	parser.add_argument('--warmup_steps', default= 50, type=int)
	parser.add_argument('--train_batch_size', default= 2, type=int)
	parser.add_argument('--eval_batch_size', default= 12, type=int)
	parser.add_argument('--infer_batch_size', default= 1, type=int)
	parser.add_argument('--accumulate_grad', default=8, type=int)   # acc_grad * b_size * n_gpu = 64
	parser.add_argument('--max_grad_norm', default= 1., type=float)
	parser.add_argument('--ratn', default= [1,5])
	parser.add_argument("--fp16", default=True)
	parser.add_argument('--lr_sel', type=float, default=5e-4)
	parser.add_argument('--lr_gen', type=float, default=2e-5)	
	parser.add_argument('--seed', default= 42, type=int)
	parser.add_argument("--device", default='cuda')
	parser.add_argument("--distributed", action="store_true")
	args = parser.parse_args()

	args.local_rank = int(os.environ["LOCAL_RANK"])
	args.distributed = True if args.local_rank >= 0 else False
	device = f"cuda:{args.local_rank}" if args.distributed else args.device
	args.device = torch.device(device)
	args.is_master = is_master = args.local_rank in [-1, 0]

	args.logger_id = f"{args.model_name_or_path.split('/')[-1]}_{'un'*(not args.with_kn_loss)}sup_{'all' if args.data_mode=='all' else 'wkn'}"
	args.output_path = f"{args.output_path}{args.logger_id}/"
	if args.do_sanity_check:
		if args.is_master: print("Running a sanity check...")
		args.do_train = args.do_evaluate = args.do_inference = True
		args.max_train_samples, args.max_eval_samples, args.max_infer_samples, args.epochs_predict  = 500, 50, 10, [1]
	if not args.do_train and not args.do_evaluate and args.do_inference: 
		args.epochs = 1
		args.load_from_checkpoint = True
	if args.epochs_predict is None: args.epochs_predict = [args.epochs]
	main(args)