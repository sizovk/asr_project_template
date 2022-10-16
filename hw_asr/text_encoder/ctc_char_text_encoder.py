# partially inspired from seminar
from collections import defaultdict
from typing import List, NamedTuple

import torch

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"
    EMPTY_IND = 0

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
        last_ind = self.EMPTY_IND
        decoded_output = []
        for ind in inds:
            ind = ind.item() if torch.is_tensor(ind) else ind
            if ind == last_ind:
                continue
            if ind != self.EMPTY_IND:
                decoded_output.append(self.ind2char[ind])
            last_ind = ind
        return "".join(decoded_output)
    
    def ctc_beam_search(self, probs: torch.tensor, probs_length, beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        _, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        paths = {('', self.EMPTY_TOK): 1.0}
        for next_char_probs in probs:
            paths = self.extend_and_merge(next_char_probs, paths)
            paths = self.truncate_beam(paths, beam_size)
        
        return [Hypothesis(prefix, score) for (prefix, _), score in sorted(paths.items(), key=lambda x : -x[1])]

    def extend_and_merge(self, next_char_probs, src_paths):
        new_paths = defaultdict(float)
        for next_char_ind, next_char_prob in enumerate(next_char_probs):
            next_char = self.ind2char[next_char_ind]
            
            for (text, last_char), path_prob in src_paths.items():
                new_prefix = text if next_char == last_char else (text + next_char)
                new_prefix = new_prefix.replace(self.EMPTY_TOK, '')
                new_paths[(new_prefix, next_char)] += path_prob * next_char_prob
        return new_paths

    def truncate_beam(self, paths, beam_size):
        return dict(sorted(paths.items(), key=lambda x : x[1])[-beam_size:])
