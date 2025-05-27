
with open('tolkien.txt', 'r') as f:
    raw_data = f.read()

alphabet = sorted(list(set(raw_data)))
vocabulary_size = len(alphabet)
sym_to_num = {sym: num for num, sym in enumerate(alphabet)}
num_to_sym = {num: sym for num, sym in enumerate(alphabet)}
encode = lambda s: [sym_to_num[c] for c in s]
decode = lambda l: ''.join([num_to_sym[i] for i in l])

