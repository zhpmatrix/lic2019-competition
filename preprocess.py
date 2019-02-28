import re
import os
import torch
import itertools
import random
import unicodedata

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

MAX_LENGTH = 20  # Maximum sentence length to consider
MIN_COUNT = 5    # Minimum word count threshold for trimming

class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3 # Count default tokens

        for word in keep_words:
            self.addWord(word)

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

# Read query/response pairs and return a voc object
def readVocs(datafile, corpus_name):
    print("Reading lines...")
    # Read the file and split into lines
    lines = open(datafile, encoding='utf-8').\
        read().strip().split('\n')
    
    # Split every line into pairs and normalize
    #pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    #pairs = [[s for s in l.split('\t')] for l in lines]
    #pairs = [[s for s in l.split('|')[0].split('\t')] for l in lines]
    pairs = [l.split('|') for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs

# Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
def filterPair(p):
    # Input sequences need to preserve the last word for EOS token
    p_ = p[0].split('\t')
    return len(p_[0].split(' ')) < MAX_LENGTH and len(p_[1].split(' ')) < MAX_LENGTH

# Filter pairs using filterPair condition
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

# Using the functions defined above, return a populated voc object and pairs list
def loadPrepareData(corpus, corpus_name, datafile, save_dir):
    print("Start preparing training data ...")
    voc, pairs = readVocs(datafile, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for elem0 in pairs:
        for elem1 in elem0:
            for elem2 in elem1.split('\t'):
                voc.addSentence(elem2)
    print("Counted words:", voc.num_words)
    return voc, pairs

def trimRareWords(voc, pairs, MIN_COUNT):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        pair_ = pair[0].split('\t')
        input_sentence = pair_[0]
        output_sentence = pair_[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs



def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]


def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

def inputVar_(l, voc):
    padVar = []
    lengths = []
    for l_ in l:
        indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l_]
        lengths.append(torch.tensor([len(indexes) for indexes in indexes_batch]))
        padVar.append(torch.LongTensor( zeroPadding(indexes_batch) ) )
    return padVar, lengths

# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

# Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split('\t')[0].split(' ')), reverse=True)
    conversation_input_batch, goal_input_batch, knowledge_input_batch, output_batch = [], [],[],[]
    for pair in pair_batch:
        conversation_input_batch.append(pair[0].split('\t')[0])
        output_batch.append(pair[0].split('\t')[1])
        goal_input_batch.append(pair[1].split('\t'))
        knowledge_input_batch.append(pair[2].split('\t'))
    
    conversation_inp, conversation_lengths = inputVar(conversation_input_batch, voc)
    goal_inp, goal_lengths = inputVar_(goal_input_batch, voc)
    knowledge_inp, knowledge_lengths = inputVar_(knowledge_input_batch, voc)
    
    output, mask, max_target_len = outputVar(output_batch, voc)
    return conversation_inp, conversation_lengths, goal_inp, goal_lengths, knowledge_inp, knowledge_lengths, output, mask, max_target_len

if __name__ == '__main__':
    
    #corpus_name = "cornell-movie-dialogs-corpus"
    #corpus = os.path.join("../../public_data", corpus_name)
    #datafile = os.path.join(corpus, "formatted_movie_lines.txt")
    
    corpus_name = "lic2019"
    corpus = os.path.join("../../public_data", corpus_name)
    datafile = os.path.join(corpus, "formatted_train_part.txt")
    
    # Load/Assemble voc and pairs
    save_dir = os.path.join("data", "save")
    voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir)
    
    # Trim voc and pairs
    #pairs = trimRareWords(voc, pairs, MIN_COUNT)
    
    # Example for validation
    small_batch_size = 5
    batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
    pass
