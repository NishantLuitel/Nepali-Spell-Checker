import torch
from torch import nn,Tensor
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchtext.data.utils import get_tokenizer
from torch.utils.data import dataset
import pickle



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

with open('models/transformer_vocab.pickle','rb') as f:
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

model.load_state_dict(torch.load('models/transformer_model.pt'))

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
    for i in range(sent.shape[0]-1):
#         print('i:', i)
        batch_size = i+1
        if batch_size != bptt:
            src_mask_ = src_mask[:batch_size, :batch_size]
        else:
            src_mask_ = src_mask[:,:]
        output_softmax = lnsoftmax(model(sent[:i+1,:], src_mask_))

#         print(output_softmax_permuted,output_softmax_permuted[0,i,sent[i+1,0]],output_softmax_permuted.max())
        #Index for maximum probability word
#         indices = torch.argmax(output_softmax_permuted, dim=2)
        
        #Max probability word
#         print('next word: ', [vocab.lookup_tokens(list(index))
#                                   for index in indices][0][-1])
        
        prob+= float(output_softmax[i,0,sent[i+1,0]])
    return prob


def transformer_probab(sentence,device = device):
    '''
    Returns Log probability of sentence using transformer inference
    
    '''
    
    sentence_tensor = preprocess(sentence,device = device) 
#     print(sentence_tensor)
    return probability(sentence_tensor)








