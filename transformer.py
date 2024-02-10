import torch
from torch import nn,Tensor
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchtext.data.utils import get_tokenizer
from torch.utils.data import dataset
import pickle
import os




class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

        
    def forward(self, x: Tensor) -> Tensor:
        
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]
        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output

    
tokenizer = get_tokenizer(None)


module_dir = os.path.dirname(__file__)
file_path = os.path.join(module_dir, 'models','transformer_vocab.pickle')
with open(file_path,'rb') as f:
    vocab = pickle.load(f)
    
params = {
            'ntokens' : len(vocab),
            'emsize' : 300,
            'd_hid' : 800,
            'nlayers' : 4,
            'nhead' : 4,
            'dropout' : 0.05,
         }


def try_gpu(i=0):
    if torch.cuda.device_count() >= i+1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

device = try_gpu(0)

model = TransformerModel(params['ntokens'],params['emsize'],params['nhead'], params['d_hid'],params['nlayers'], params['dropout']).to(device)

model_path = os.path.join(module_dir, 'models','transformer_model.pt')
model.load_state_dict(torch.load(model_path))

def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
    
    """
    Converts raw text into a flat Tensor.
    
    """
    
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long)
            for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


def preprocess(sentence,device = device):
    
    '''
    Convert sentence into tensor of indices taken from vocabulary
    
    '''
    
    st_i = data_process(sentence)
    st_i = st_i.unsqueeze(1).to(device)    
    return st_i



def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

lnsoftmax = nn.LogSoftmax(dim=2)



bptt = 35
def probability(sent: Tensor):
    model.eval()
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    
    prob = 0
#     for i in range(sent.shape[0]-1):
#         print('i:', i)
#         batch_size = i+1
    batch_size = min(bptt,len(sent))
    if batch_size != bptt:
        src_mask_ = src_mask[:batch_size, :batch_size]
    else:
        src_mask_ = src_mask[:,:]
#         output_softmax = lnsoftmax(model(sent[:i+1,:], src_mask_))
    output_softmax = lnsoftmax(model(sent, src_mask_))
    minimum_tensor,_ = torch.min(output_softmax, dim = 2,keepdim=True)

#     print(output_softmax.shape)
    dummy = torch.cat((output_softmax,minimum_tensor),dim = 2)
#     print(dummy.shape)
#         print(output_softmax_permuted,output_softmax_permuted[0,i,sent[i+1,0]],output_softmax_permuted.max())
    #Index for maximum probability word
#         indices = torch.argmax(output_softmax_permuted, dim=2)

    #Max probability word
#         print('next word: ', [vocab.lookup_tokens(list(index))
#                                   for index in indices][0][-1])
#     print(output_softmax.shape)

    l = sent[1:,:].squeeze(1)
#    print(l)
    for n,i in enumerate(l):
        if i == 0:
            l[n] = torch.tensor(output_softmax.shape[2])
    out_probab = torch.index_select(dummy,2,l)
#     print(out_probab.shape)
    for i in range(len(out_probab)-1):
        prob += float(out_probab[i,0,i])
#         prob+= float(output_softmax[i,0,sent[i+1,0]])
                         
    return prob



def probability_list(sents: Tensor):
    model.eval()
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    
    prob = torch.zeros(sents.shape[1])
    batch_size = min(bptt,sents.shape[0])
#     bs = min(len(sents),64)
    if batch_size != bptt:
        src_mask_ = src_mask[:batch_size, :batch_size]
    else:
        src_mask_ = src_mask[:,:]
#         output_softmax = lnsoftmax(model(sent[:i+1,:], src_mask_))

#     sents = sents.permute(1,0,2)
    output_softmax = lnsoftmax(model(sents.to(device), src_mask_))
    minimum_tensor,_ = torch.min(output_softmax, dim = 2,keepdim=True)

#     print(output_softmax.shape)
    dummy = torch.cat((output_softmax,minimum_tensor),dim = 2)
#     print(dummy.shape)

    l = sents[1:,:]
#     print("Printing..")
#     print(l)
    
    for l_temp in l:
        for n,i in enumerate(l_temp):
            if i == 0:
                l_temp[n] = torch.tensor(output_softmax.shape[2])
#     print(l.T.shape)
    
#     print(dummy.shape)
    out_probab = torch.stack([torch.index_select(dummy,2,l_temp.to(device)) for l_temp in l.T])
#     print(out_probab)

#     print(out_probab.shape)
#     print(out_probab.shape)
    for i in range(len(out_probab)):
        for j in range(dummy.shape[0]-1):
            prob[i] += out_probab[i,j,i,j].to('cpu')
                         
    return prob



b = 64
def probability_list2(sents: Tensor):
    model.eval()
    src_mask = generate_square_subsequent_mask(bptt)
    
    
    batch_size = min(bptt,sents.shape[0])
    bs = min(sents.shape[1],b)
    prob_src = torch.zeros(b)
    prob_ = torch.zeros(1).to(device)
    
    if sents.shape[1]%bs == 0:
        loops = sents.shape[1]//bs
#         print(loops)
    else:
        
        loops = sents.shape[1]//bs+1
#         print(loops)
    
    
    with torch.no_grad():
        for loop in range(loops):
            torch.cuda.empty_cache()
            if batch_size != bptt:
                src_mask_ = src_mask[:batch_size, :batch_size]
            else:
                src_mask_ = src_mask[:,:]
                
            if loop != loops-1:
                s = sents[:,loop*bs:(loop+1)*bs]
                output_softmax = lnsoftmax(model(s.to(device), src_mask_.to(device)))
                l = s[1:].to(device)
                prob = torch.zeros(bs).to(device)
            else:
                s  = sents[:,loop*bs:]
                output_softmax = lnsoftmax(model(s.to(device), src_mask_.to(device)))
                l = s[1:].to(device)
                prob = torch.zeros(len(l[0])).to(device)

            minimum_tensor,_ = torch.min(output_softmax, dim = 2,keepdim=True)

        #     print(output_softmax.shape)
            dummy = torch.cat((output_softmax,minimum_tensor),dim = 2)
        #     print(dummy.shape)


        #     print("Printing..")
        #     print(l)

#             for l_temp in l:
#                 for n,i in enumerate(l_temp):
#                     if i == 0:
#                         l_temp[n] = torch.tensor(output_softmax.shape[2])
                        
            l[l==0] = output_softmax.shape[2]
        #     print(l.T.shape)

        #     print(dummy.shape)
            out_probab = torch.stack([torch.index_select(dummy,2,l_temp) for l_temp in l.T])
        #     print(out_probab)

        #     print(out_probab.shape)
#             print(out_probab.shape)
            for i in range(len(out_probab)):
                for j in range(dummy.shape[0]-1):
                    prob[i] += out_probab[i,j,i,j]
            prob_ = torch.cat((prob_,prob),dim = 0)
        
        
    del s,dummy,minimum_tensor,out_probab
                         
    return prob_[1:]



def probability_list3(sents: Tensor):
    
    
    model.eval()
    src_mask = generate_square_subsequent_mask(bptt)
    
    
    batch_size = min(bptt,sents.shape[0])
    bs = min(sents.shape[1],b)
#     prob_src = torch.zeros(b)
    prob_ = torch.zeros(1).to(device)
    
    if sents.shape[1]%bs == 0:
        loops = sents.shape[1]//bs
#         print(loops)
    else:
        
        loops = sents.shape[1]//bs+1
#         print(loops)
    
    
    with torch.no_grad():
        for loop in range(loops):
            torch.cuda.empty_cache()
            if batch_size != bptt:
                src_mask_ = src_mask[:batch_size, :batch_size]
            else:
                src_mask_ = src_mask[:,:]

        #         output_softmax = lnsoftmax(model(sent[:i+1,:], src_mask_))

        #     sents = sents.permute(1,0,2)
            if loop != loops-1:
                s = sents[:,loop*bs:(loop+1)*bs]
                output_softmax = lnsoftmax(model(s.to(device), src_mask_.to(device)))
                l = s[1:].to(device)
#                 prob = torch.zeros(bs)
            else:
                s  = sents[:,loop*bs:]
                output_softmax = lnsoftmax(model(s.to(device), src_mask_.to(device)))
                l = s[1:].to(device)
#                 prob = torch.zeros(len(l[0]))

            minimum_tensor,_ = torch.min(output_softmax, dim = 2,keepdim=True)

        #     print(output_softmax.shape)
            dummy = torch.cat((output_softmax,minimum_tensor),dim = 2)
        #     print(dummy.shape)


        #     print("Printing..")
        #     print(l)
#             print(l)
#             for l_temp in l:
#                 for n,i in enumerate(l_temp):
#                     if i == 0:
#                         l_temp[n] = torch.tensor(output_softmax.shape[2])
                        
            l[l==0] = output_softmax.shape[2]
                        
#             print(l)
        #     print(l.T.shape)

        #     print(dummy.shape)
            out_probab = torch.stack([torch.index_select(dummy,2,l_temp) for l_temp in l.T])
        #     print(out_probab)

#             print(out_probab.shape)
#             print(out_probab)
#             for i in range(len(out_probab)):
#                 for j in range(dummy.shape[0]-1):
#                     prob[i] += out_probab[i,j,i,j].to('cpu')
#                     print("Up:",prob[i])
                    
#                 prob[i] = sum(torch.diagonal(out_probab[i,:-1,i,:].squeeze(0).squeeze(1)))
            m = torch.diagonal(torch.sum(torch.diagonal(out_probab[:,:-1,:,:],dim1=1,dim2=3),dim=2)).to(device)
#             print("m",m.shape,m)
        
#                 print(prob[i])
#             prob_ = torch.cat((prob_,prob),dim = 0)
            prob_ = torch.cat((prob_,m),dim = 0)
        
        
    del s,dummy,minimum_tensor,out_probab
                         
    return prob_[1:]

def transformer_probab(sentence,device = device):
    '''
    Returns Log probability of sentence using transformer inference
    
    '''
    
    sentence_tensor = preprocess(sentence,device = device) 
#     print(sentence_tensor)
    return probability(sentence_tensor)



def preprocess_list(sentences,device = device):
    '''
    Convert sentence into tensor of indices taken from vocabulary
    
    '''
    length = len(sentences)
#     st_i = torch.stack([data_process(sent) for sent in sentences]).to(device) 
    st_i = data_process(sentences).reshape(length,-1).T
#     st_i = st_i.squeeze(0)   
    return st_i



def transformer_probab_list(sentence_list,device = device):
    
    
    sentence_tensor = preprocess_list(sentence_list,device = device)
    
    return probability_list3(sentence_tensor)


def transformer_probab_final_word(sentence_list, candidates=[], device = device):
    
    sentence_tensor = preprocess_list(sentence_list,device = device)[:-1].to(device)
#     print("Sentence tensor:",sentence_tensor)
    
    candidate_sti = preprocess_list([' '.join(candidates[0])]).squeeze(1)
#     print(candidate_sti)
    
    model.eval()
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    
    prob = 0
    batch_size = min(bptt,len(sentence_tensor))
    if batch_size != bptt:
        src_mask_ = src_mask[:batch_size, :batch_size]
    else:
        src_mask_ = src_mask[:,:]
    output_softmax = lnsoftmax(model(sentence_tensor, src_mask_))
    #print(output_softmax,output_softmax.shape,output_softmax.shape[0])
    prediction_token = output_softmax[-1].squeeze(0)
    #print(prediction_token[candidate_sti])
    
    
    #print(candidate_sti)
    if 0 in candidate_sti:
        #print('true')
        prediction_token[0] = torch.min(prediction_token)
    
    
    selected_candidates_log_probab = prediction_token[candidate_sti]
    #print(torch.argsort(prediction_token))
    
    sorted_indices_desc = torch.argsort(selected_candidates_log_probab, descending=True)
    #print(sorted_indices_desc)
    
    #print(vocab.lookup_token(7752),vocab.lookup_token(4418),vocab.lookup_token(2177),vocab.lookup_token(2067))
    
    return_candidates = [candidates[0][i] for i in sorted_indices_desc]
#     print(return_candidates)
    
    return selected_candidates_log_probab.to('cpu')

    
    
    
    
    

    







bptt = 35
def probability(sent: Tensor):
    model.eval()
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    
    prob = 0
#     for i in range(sent.shape[0]-1):
#         print('i:', i)
#         batch_size = i+1
    batch_size = min(bptt,len(sent))
    if batch_size != bptt:
        src_mask_ = src_mask[:batch_size, :batch_size]
    else:
        src_mask_ = src_mask[:,:]
#         output_softmax = lnsoftmax(model(sent[:i+1,:], src_mask_))
    output_softmax = lnsoftmax(model(sent, src_mask_))
    minimum_tensor,_ = torch.min(output_softmax, dim = 2,keepdim=True)

#     print(output_softmax.shape)
    dummy = torch.cat((output_softmax,minimum_tensor),dim = 2)
#     print(dummy.shape)
#         print(output_softmax_permuted,output_softmax_permuted[0,i,sent[i+1,0]],output_softmax_permuted.max())
    #Index for maximum probability word
#         indices = torch.argmax(output_softmax_permuted, dim=2)

    #Max probability word
#         print('next word: ', [vocab.lookup_tokens(list(index))
#                                   for index in indices][0][-1])
#     print(output_softmax.shape)

    l = sent[1:,:].squeeze(1)
#    print(l)
    for n,i in enumerate(l):
        if i == 0:
            l[n] = torch.tensor(output_softmax.shape[2])
    out_probab = torch.index_select(dummy,2,l)
#     print(out_probab.shape)
    for i in range(len(out_probab)-1):
        prob += float(out_probab[i,0,i])
#         prob+= float(output_softmax[i,0,sent[i+1,0]])
                         
    return prob



def probability_list(sents: Tensor):
    model.eval()
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    
    prob = torch.zeros(sents.shape[1])
    batch_size = min(bptt,sents.shape[0])
#     bs = min(len(sents),64)
    if batch_size != bptt:
        src_mask_ = src_mask[:batch_size, :batch_size]
    else:
        src_mask_ = src_mask[:,:]
#         output_softmax = lnsoftmax(model(sent[:i+1,:], src_mask_))

#     sents = sents.permute(1,0,2)
    output_softmax = lnsoftmax(model(sents.to(device), src_mask_))
    minimum_tensor,_ = torch.min(output_softmax, dim = 2,keepdim=True)

#     print(output_softmax.shape)
    dummy = torch.cat((output_softmax,minimum_tensor),dim = 2)
#     print(dummy.shape)

    l = sents[1:,:]
#     print("Printing..")
#     print(l)
    
    for l_temp in l:
        for n,i in enumerate(l_temp):
            if i == 0:
                l_temp[n] = torch.tensor(output_softmax.shape[2])
#     print(l.T.shape)
    
#     print(dummy.shape)
    out_probab = torch.stack([torch.index_select(dummy,2,l_temp.to(device)) for l_temp in l.T])
#     print(out_probab)

#     print(out_probab.shape)
#     print(out_probab.shape)
    for i in range(len(out_probab)):
        for j in range(dummy.shape[0]-1):
            prob[i] += out_probab[i,j,i,j].to('cpu')
                         
    return prob



b = 64
def probability_list2(sents: Tensor):
    model.eval()
    src_mask = generate_square_subsequent_mask(bptt)
    
    
    batch_size = min(bptt,sents.shape[0])
    bs = min(sents.shape[1],b)
    prob_src = torch.zeros(b)
    prob_ = torch.zeros(1).to(device)
    
    if sents.shape[1]%bs == 0:
        loops = sents.shape[1]//bs
#         print(loops)
    else:
        
        loops = sents.shape[1]//bs+1
#         print(loops)
    
    
    with torch.no_grad():
        for loop in range(loops):
            torch.cuda.empty_cache()
            if batch_size != bptt:
                src_mask_ = src_mask[:batch_size, :batch_size]
            else:
                src_mask_ = src_mask[:,:]
                
            if loop != loops-1:
                s = sents[:,loop*bs:(loop+1)*bs]
                output_softmax = lnsoftmax(model(s.to(device), src_mask_.to(device)))
                l = s[1:].to(device)
                prob = torch.zeros(bs).to(device)
            else:
                s  = sents[:,loop*bs:]
                output_softmax = lnsoftmax(model(s.to(device), src_mask_.to(device)))
                l = s[1:].to(device)
                prob = torch.zeros(len(l[0])).to(device)

            minimum_tensor,_ = torch.min(output_softmax, dim = 2,keepdim=True)

        #     print(output_softmax.shape)
            dummy = torch.cat((output_softmax,minimum_tensor),dim = 2)
        #     print(dummy.shape)


        #     print("Printing..")
        #     print(l)

#             for l_temp in l:
#                 for n,i in enumerate(l_temp):
#                     if i == 0:
#                         l_temp[n] = torch.tensor(output_softmax.shape[2])
                        
            l[l==0] = output_softmax.shape[2]
        #     print(l.T.shape)

        #     print(dummy.shape)
            out_probab = torch.stack([torch.index_select(dummy,2,l_temp) for l_temp in l.T])
        #     print(out_probab)

        #     print(out_probab.shape)
#             print(out_probab.shape)
            for i in range(len(out_probab)):
                for j in range(dummy.shape[0]-1):
                    prob[i] += out_probab[i,j,i,j]
            prob_ = torch.cat((prob_,prob),dim = 0)
        
        
    del s,dummy,minimum_tensor,out_probab
                         
    return prob_[1:]



def probability_list3(sents: Tensor):
    
    
    model.eval()
    src_mask = generate_square_subsequent_mask(bptt)
    
    
    batch_size = min(bptt,sents.shape[0])
    bs = min(sents.shape[1],b)
#     prob_src = torch.zeros(b)
    prob_ = torch.zeros(1).to(device)
    
    if sents.shape[1]%bs == 0:
        loops = sents.shape[1]//bs
#         print(loops)
    else:
        
        loops = sents.shape[1]//bs+1
#         print(loops)
    
    
    with torch.no_grad():
        for loop in range(loops):
            torch.cuda.empty_cache()
            if batch_size != bptt:
                src_mask_ = src_mask[:batch_size, :batch_size]
            else:
                src_mask_ = src_mask[:,:]

        #         output_softmax = lnsoftmax(model(sent[:i+1,:], src_mask_))

        #     sents = sents.permute(1,0,2)
            if loop != loops-1:
                s = sents[:,loop*bs:(loop+1)*bs]
                output_softmax = lnsoftmax(model(s.to(device), src_mask_.to(device)))
                l = s[1:].to(device)
#                 prob = torch.zeros(bs)
            else:
                s  = sents[:,loop*bs:]
                output_softmax = lnsoftmax(model(s.to(device), src_mask_.to(device)))
                l = s[1:].to(device)
#                 prob = torch.zeros(len(l[0]))

            minimum_tensor,_ = torch.min(output_softmax, dim = 2,keepdim=True)

        #     print(output_softmax.shape)
            dummy = torch.cat((output_softmax,minimum_tensor),dim = 2)
        #     print(dummy.shape)


        #     print("Printing..")
        #     print(l)
#             print(l)
#             for l_temp in l:
#                 for n,i in enumerate(l_temp):
#                     if i == 0:
#                         l_temp[n] = torch.tensor(output_softmax.shape[2])
                        
            l[l==0] = output_softmax.shape[2]
                        
#             print(l)
        #     print(l.T.shape)

        #     print(dummy.shape)
            out_probab = torch.stack([torch.index_select(dummy,2,l_temp) for l_temp in l.T])
        #     print(out_probab)

#             print(out_probab.shape)
#             print(out_probab)
#             for i in range(len(out_probab)):
#                 for j in range(dummy.shape[0]-1):
#                     prob[i] += out_probab[i,j,i,j].to('cpu')
#                     print("Up:",prob[i])
                    
#                 prob[i] = sum(torch.diagonal(out_probab[i,:-1,i,:].squeeze(0).squeeze(1)))
            m = torch.diagonal(torch.sum(torch.diagonal(out_probab[:,:-1,:,:],dim1=1,dim2=3),dim=2)).to(device)
#             print("m",m.shape,m)
        
#                 print(prob[i])
#             prob_ = torch.cat((prob_,prob),dim = 0)
            prob_ = torch.cat((prob_,m),dim = 0)
        
        
    del s,dummy,minimum_tensor,out_probab
                         
    return prob_[1:]

def transformer_probab(sentence,device = device):
    '''
    Returns Log probability of sentence using transformer inference
    
    '''
    
    sentence_tensor = preprocess(sentence,device = device) 
#     print(sentence_tensor)
    return probability(sentence_tensor)



def preprocess_list(sentences,device = device):
    '''
    Convert sentence into tensor of indices taken from vocabulary
    
    '''
    length = len(sentences)
#     st_i = torch.stack([data_process(sent) for sent in sentences]).to(device) 
    st_i = data_process(sentences).reshape(length,-1).T
#     st_i = st_i.squeeze(0)   
    return st_i



def transformer_probab_list(sentence_list,device = device):
    
    
    sentence_tensor = preprocess_list(sentence_list,device = device)
    
    return probability_list3(sentence_tensor)


def transformer_probab_final_word(sentence_list, candidates=[], device = device):
    
    sentence_tensor = preprocess_list(sentence_list,device = device)[:-1].to(device)
#     print("Sentence tensor:",sentence_tensor)
    
    candidate_sti = preprocess_list([' '.join(candidates[0])]).squeeze(1)
#     print(candidate_sti)
    
    model.eval()
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    
    prob = 0
    batch_size = min(bptt,len(sentence_tensor))
    if batch_size != bptt:
        src_mask_ = src_mask[:batch_size, :batch_size]
    else:
        src_mask_ = src_mask[:,:]
    output_softmax = lnsoftmax(model(sentence_tensor, src_mask_))
    #print(output_softmax,output_softmax.shape,output_softmax.shape[0])
    prediction_token = output_softmax[-1].squeeze(0)
    #print(prediction_token[candidate_sti])
    
    
    #print(candidate_sti)
    if 0 in candidate_sti:
        #print('true')
        prediction_token[0] = torch.min(prediction_token)
    
    
    selected_candidates_log_probab = prediction_token[candidate_sti]
    #print(torch.argsort(prediction_token))
    
    sorted_indices_desc = torch.argsort(selected_candidates_log_probab, descending=True)
    #print(sorted_indices_desc)
    
    #print(vocab.lookup_token(7752),vocab.lookup_token(4418),vocab.lookup_token(2177),vocab.lookup_token(2067))
    
    return_candidates = [candidates[0][i] for i in sorted_indices_desc]
#     print(return_candidates)
    
    return selected_candidates_log_probab.to('cpu')

    
    
    
    
    

    







    
    
    

    






