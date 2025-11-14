import os
import time
import torch
import pickle
import argparse
from torch.utils.tensorboard import SummaryWriter

from model import EnrichedTiSASRec
from tqdm import tqdm
from utils import *

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Books', type=str,
                    help='The preprocess data file name, e.g. the Books.txt file will be Books')
parser.add_argument('--train_dir', default='default', type=str,
                    help='The directory to save the trained model. The directory will be named as: dataset_train_dir')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=100, type=int)
parser.add_argument('--hidden_units', default=64, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--tensorboard_log_dir', default="final_run", type=str)
parser.add_argument('--time_span', default=256, type=int)

args = parser.parse_args()
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

writer = SummaryWriter(args.tensorboard_log_dir)

args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    dataset = data_partition(args.dataset)
    [user_train, user_valid, user_test, usernum, itemnum, timenum, catnum] = dataset
    
    num_batch = len(user_train) // args.batch_size
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    
    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
    
    try:
        relation_matrix = pickle.load(open('data/relation_matrix_%s_%d_%d.pickle'%(args.dataset, args.maxlen, args.time_span),'rb'))
    except:
        relation_matrix = Relation(user_train, usernum, args.maxlen, args.time_span)
        pickle.dump(relation_matrix, open('data/relation_matrix_%s_%d_%d.pickle'%(args.dataset, args.maxlen, args.time_span),'wb'))
    

    sampler = WarpSampler(user_train, usernum, itemnum, relation_matrix, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    model = EnrichedTiSASRec(usernum, itemnum, itemnum, catnum, args).to(args.device)
    
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_uniform_(param.data)
        except:
            pass # just ignore those failed init layers
    
    model.train() # enable model training
    
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
    
    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args)
        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))

    lr=args.lr
    bce_criterion = torch.nn.BCEWithLogitsLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr, betas=(0.9, 0.98))
    
    ndcg10_last = 0.0
    hr10_last = 0.0
    
    T = 0.0
    t0 = time.time()
    
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break # just to decrease identition
        for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, rat_seq, time_seq, time_matrix, cat_seq, pos, neg = sampler.next_batch() # tuples to ndarray
            u, seq, rat_seq, cat_seq, pos, neg = np.array(u), np.array(seq), np.array(rat_seq), np.array(cat_seq), np.array(pos), np.array(neg)
            time_seq, time_matrix = np.array(time_seq), np.array(time_matrix)
            
            pos_logits, neg_logits = model(u, seq, rat_seq, cat_seq, time_matrix, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
            # print("\neye ball check raw_logits:")
            # print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            for param in model.abs_pos_K_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            for param in model.abs_pos_V_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            for param in model.rat_K_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            for param in model.rat_V_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            for param in model.cat_K_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            for param in model.cat_V_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            for param in model.time_matrix_K_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            for param in model.time_matrix_V_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            
            writer.add_scalar("Loss/Train", loss, epoch)
            loss.backward()
            adam_optimizer.step()
            if step == (num_batch - 1):
                print(time.strftime("%m-%d %H:%M:%S") + " [%2d] Lr: %5f, Training: %.2f%%, " % (epoch, lr, 100 * ((step + 1) / num_batch)) + " Loss: %.5f \r" % (loss.item())) 
    
        if epoch % 10 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            t_test = evaluate(model, dataset, args)
            t_valid = evaluate_valid(model, dataset, args)
            print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f, NDCG@20: %.4f, HR@20: %.4f), test (NDCG@10: %.4f, HR@10: %.4f, NDCG@20: %.4f, HR@20: %.4f)'
                    % (epoch, T, t_valid[0], t_valid[1], t_valid[2], t_valid[3], t_test[0], t_test[1], t_test[2], t_test[3]))
            
            f.write(str(t_valid) + ' ' + str(t_test) + '\n')
            f.flush()
            
             # log to tensorboard
            writer.add_scalar("NDCG@10_validation", t_valid[0], epoch)
            writer.add_scalar("HR@10_validation", t_valid[1], epoch)
            writer.add_scalar("NDCG@20_validation", t_valid[2], epoch)
            writer.add_scalar("HR@20_validation", t_valid[3], epoch)
            writer.add_scalar("NDCG@10_test", t_test[0], epoch)
            writer.add_scalar("HR@10_test", t_test[1], epoch)
            writer.add_scalar("NDCG@20_test", t_test[2], epoch)
            writer.add_scalar("HR@20_test", t_test[3], epoch)
            t0 = time.time()
            model.train()
            
            if t_valid[0] > ndcg10_last and t_valid[1] > hr10_last:
                print("Found best ndcg and hr score on validation dataset, Saving model ...........")
                folder = args.dataset + '_' + args.train_dir
                fname = '{}_EnrichedTiSASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
                fname = fname.format(args.dataset, args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
                torch.save(model.state_dict(), os.path.join(folder, fname))
                ndcg_last = t_valid[0]
                hr_last = t_valid[1]
                f.write(str(t_valid) + ' ' + str(t_test) + '\n')
                f.flush()
    
        if epoch == args.num_epochs:
            folder = args.dataset + '_' + args.train_dir
            fname = '{}_EnrichedTiSASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(args.dataset, args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            torch.save(model.state_dict(), os.path.join(folder, fname))
    
    f.close()
    sampler.close()
    writer.flush()
    writer.close()
    print("Done")
