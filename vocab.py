class Vocabulary:
    def __init__(self, max_size=10000):
        self.word2index = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.index2word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.n_words = 4  # Het aantal woorden in de vocabulaire (inclusief speciale tokens)
        self.max_size = max_size
    
    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)
    
    def add_word(self, word):
        if word not in self.word2index:
            if self.n_words < self.max_size:
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                self.n_words += 1
    
    def sentence_to_tensor(self, sentence):
        return [self.word2index.get(word, self.word2index['<UNK>']) for word in sentence.split()]
