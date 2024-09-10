import torch
import re
from torch.nn.utils.rnn import pad_sequence

# DNA sequence to integer encoding
alphabet = 'NACGT'
dna2int = {a: i for i, a in enumerate(alphabet, 1)}
dna2int.update({"pad": 0})  # Adding padding as 0

# Converts DNA sequence to integer sequence
def preprocess_dna_sequence(seq: str):
    seq = seq.upper()
    int_seq = [dna2int[nuc] for nuc in seq if nuc in dna2int]
    return torch.tensor(int_seq, dtype=torch.long).unsqueeze(0)

# Count consecutive 'CG' using regex
def count_cpg(seq: str) -> int:
    return len(re.findall(r'CG', seq.upper()))

# Padding sequences for variable length processing
class PadSequence:
    def __call__(self, batch):
        # Sort the batch by sequence length in descending order
        batch.sort(key=lambda x: len(x[0]), reverse=True)
        sequences, labels = zip(*batch)

        # Pad sequences and get their original lengths
        sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
        lengths = torch.LongTensor([len(seq) for seq in sequences])

        # Convert labels to tensor
        labels = torch.FloatTensor(labels)

        return sequences_padded, lengths, labels
