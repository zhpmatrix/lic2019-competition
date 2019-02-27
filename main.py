# -*- coding: utf-8 -*-

import torch
from torch import optim
import os
from preprocess import *
from model import *
from train_eval import *

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

if __name__ == '__main__':
    
    # Running mode
    eval_ = True
    
    # Load data
    #corpus_name = "cornell-movie-dialogs-corpus"
    #corpus = os.path.join("../../public_data", corpus_name)
    #datafile = os.path.join(corpus, "formatted_movie_lines.txt")
    
    corpus_name = "lic2019"
    corpus = os.path.join("../../public_data", corpus_name)
    datafile = os.path.join(corpus, "formatted_train_part.txt")
    # Load/Assemble voc and pairs
    save_dir = 'model/'
    voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir)
    
    # Trim voc and pairs
    pairs = trimRareWords(voc, pairs, MIN_COUNT)

    # Configure models
    model_name = ''
    
    attn_model = 'dot'
    #attn_model = 'general'
    #attn_model = 'concat'
    
    hidden_size = 500
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.1
    batch_size = 64

    # Set checkpoint to load from; set to None if starting from scratch
    checkpoint = None
    loadFilename = None

    checkpoint_iter = 1000
    loadFilename = os.path.join(save_dir, model_name, corpus_name,
                                '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
                               '{}_checkpoint.tar'.format(checkpoint_iter))


    # Load model if a loadFilename is provided
    if loadFilename:
        # If loading on same machine the model was trained on
        checkpoint = torch.load(loadFilename)
        # If loading a model trained on GPU to CPU
        #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        voc.__dict__ = checkpoint['voc_dict']


    print('Building encoder and decoder ...')
    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, hidden_size)
    if loadFilename:
        embedding.load_state_dict(embedding_sd)
    # Initialize encoder & decoder models
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
    if loadFilename:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print('Models built and ready to go!')

    if eval_ == False:

        # Configure training/optimization
        clip = 50.0
        teacher_forcing_ratio = 1.0
        learning_rate = 0.0001
        decoder_learning_ratio = 5.0
        n_iteration = 12000
        print_every = 1
        save_every = 500

        # Ensure dropout layers are in train mode
        encoder.train()
        decoder.train()

        # Initialize optimizers
        print('Building optimizers ...')
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
        if loadFilename:
            encoder_optimizer.load_state_dict(encoder_optimizer_sd)
            decoder_optimizer.load_state_dict(decoder_optimizer_sd)

        # Run training iterations
        print("Starting Training!")
        trainIters(model_name, checkpoint, device, teacher_forcing_ratio, hidden_size, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
                   embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
                   print_every, save_every, clip, corpus_name, loadFilename)
    else:
        # Set dropout layers to eval mode
        encoder.eval()
        decoder.eval()

        # Initialize search module
        searcher = GreedySearchDecoder(encoder, decoder, device)

        # Begin chatting (uncomment and run the following line to begin)
        evaluateInput(encoder, decoder, searcher, voc, device)
