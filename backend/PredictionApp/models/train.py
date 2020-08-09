import spacy
import torchtext
import torchtext.data
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtext.vocab import Vocab
from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization

nlp = spacy.load('en')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, 
                filter_sizes, output_dim,  dropout, pad_idx):
        super().__init__()
                
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, n_filters, kernel_size = (fs, embedding_dim)) 
            for fs
            in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        #text = [sent len, batch size]
        #text = text.permute(1, 0)  
        #text = [batch size, sent len]
        embedded = self.embedding(text)
        #embedded = [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1)
        #embedded = [batch size, 1, sent len, emb dim]
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]   
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        #pooled_n = [batch size, n_filters]
        cat = self.dropout(torch.cat(pooled, dim = 1))
        #cat = [batch size, n_filters * len(filter_sizes)]
        return self.fc(cat)

    def forward_with_sigmoid(input):
        return torch.sigmoid(model(input))


# Loads pretrained model and sets the model to eval mode.
model = torch.load('models/imdb-model-cnn.pt')
model.eval()
model = model.to(device)

# Load a small subset of test data using torchtext from IMDB dataset.
TEXT = torchtext.data.Field(lower=True, tokenize='spacy')
Label = torchtext.data.LabelField(dtype = torch.float)

train, test = torchtext.datasets.IMDB.splits(
    text_field=TEXT,
    label_field=Label,
    train='train',
    test='test',
    path='data/aclImdb'
)
test, _ = test.split(split_ratio = 0.04)


# Loading and setting up vocabulary for word embeddings using torchtext.
from torchtext import vocab

#loaded_vectors = vocab.GloVe(name='6B', dim=50)
loaded_vectors = torchtext.vocab.Vectors('data/glove.6B.50d.txt')
TEXT.build_vocab(train, vectors=loaded_vectors, max_size=len(loaded_vectors.stoi))
    
TEXT.vocab.set_vectors(stoi=loaded_vectors.stoi, vectors=loaded_vectors.vectors, dim=loaded_vectors.dim)
Label.build_vocab(train)


print('Vocabulary Size: ', len(TEXT.vocab))


# create a reference (aka baseline) for the sentences and its constituent parts, tokens.
# TokenReferenceBase which allows us to generate a reference for each input text using the number of tokens in the text and a reference token index.
# We need to provide a reference_token_idx which is padding in this case

PAD_IND = TEXT.vocab.stoi['pad']

token_reference = TokenReferenceBase(reference_token_idx=PAD_IND)


# Let's create an instance of `LayerIntegratedGradients` using forward function of our model and the embedding layer.
# This instance of layer integrated gradients will be used to interpret movie rating review.
# Layer Integrated Gradients will allow us to assign an attribution score to each word/token embedding tensor in the movie review text.
# We will ultimately sum the attribution scores across all embedding dimensions for each word/token in order to attain a word/token level attribution score.
lig = LayerIntegratedGradients(model, model.embedding)

# Function that generates attributions for each movie rating and stores them in a list using `VisualizationDataRecord` class. This will ultimately be used for visualization purposes.
vis_data_records = []

def interpret_sentence(model, sentence, min_len = 7):
    text = [tok.text for tok in nlp.tokenizer(sentence)]
    if len(text) < min_len:
        text += ['pad'] * (min_len - len(text))
    indexed = [TEXT.vocab.stoi[t] for t in text]

    model.zero_grad()

    input_indices = torch.tensor(indexed, device=device)
    input_indices = input_indices.unsqueeze(0)
    
    # input_indices dim: [sequence_length]
    seq_length = min_len

    # predict
    pred = forward_with_sigmoid(input_indices).item()
    pred_ind = round(pred)

    label = round(Label.vocab.itos[pred_ind])

    # generate reference indices for each sample
    reference_indices = token_reference.generate_reference(seq_length, device=device).unsqueeze(0)

    # compute attributions and approximation delta using layer integrated gradients
    attributions_ig, delta = lig.attribute(input_indices, reference_indices, n_steps=500, return_convergence_delta=True)

    print('pred: ', Label.vocab.itos[pred_ind], '(', '%.2f'%pred, ')', ', delta: ', abs(delta))

    add_attributions_to_visualizer(attributions_ig, text, pred, pred_ind, label, delta, vis_data_records_ig)
    
def add_attributions_to_visualizer(attributions, text, pred, pred_ind, label, delta, vis_data_records):
    attributions = attributions.sum(dim=2).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.cpu().detach().numpy()

    # storing couple samples in an array for visualization purposes
    vis_data_records.append(
        visualization.VisualizationDataRecord(
            attributions,
            pred,
            Label.vocab.itos[pred_ind],
            Label.vocab.itos[label],
            Label.vocab.itos[1],
            attributions.sum(),
            text,
            delta
        )
    )

interpret_sentence(model, 'It was a fantastic performance !')
interpret_sentence(model, 'Best film ever')
interpret_sentence(model, 'Such a great show!')
interpret_sentence(model, 'It was a horrible movie')
interpret_sentence(model, 'I\'ve never watched something as bad')
interpret_sentence(model, 'It is a disgusting movie!')

visualization.visualize_text(vis_data_records)