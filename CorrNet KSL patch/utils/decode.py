import torch
import numpy as np
from itertools import groupby

try:
    from pyctcdecode import build_ctcdecoder
    _has_pyctc = True
    print(" Using pyctcdecode for beam search")
except Exception:
    _has_pyctc = False
    print(" pyctcdecode not available. Beam search disabled.")

class Decode(object):
    def __init__(self, gloss_dict, num_classes, search_mode="max", blank_id=0):

        self.i2g = {v[0]: k for k, v in gloss_dict.items()}

        self.num_classes = num_classes
        self.blank = blank_id
        self.search_mode = search_mode.lower()

        self.vocab = [""] + [chr(20000 + i) for i in range(1, num_classes)]

        if _has_pyctc and self.search_mode != "max":
            try:
                self.beam_decoder = build_ctcdecoder(self.vocab)
                print(" Beam decoder initialized")
            except Exception as e:
                print(" Beam init failed:", e)
                self.beam_decoder = None
        else:
            self.beam_decoder = None

    def decode(self, nn_output, vid_lgt, batch_first=True, probs=False):
        if not batch_first:
            nn_output = nn_output.permute(1, 0, 2)

        if self.search_mode == "max" or self.beam_decoder is None:
            return self._greedy(nn_output, vid_lgt)
        else:
            return self._beam(nn_output, vid_lgt, probs)


    def _greedy(self, logits, lengths):
        index = torch.argmax(logits, dim=2)

        results = []
        for b in range(index.size(0)):
            L = int(lengths[b])
            seq = index[b][:L].tolist()
            seq = [s for s, _ in groupby(seq) if s != self.blank]

            sent = [(self.i2g.get(cid, "UNK"), i)
                    for i, cid in enumerate(seq)]

            results.append(sent)

        return results


    def _beam(self, logits, lengths, probs=False):

        # softmax
        if not probs:
            logits = logits.softmax(dim=-1)

        # 关键修复：detach() 后才能 numpy()
        logits = logits.detach().cpu().numpy()

        results = []

        for b in range(logits.shape[0]):
            L = int(lengths[b])
            logit = logits[b][:L]

            try:
                decoded = self.beam_decoder.decode(logit)
            except Exception as e:
                print(" beam error:", e)
                return self._greedy(torch.tensor(logits), lengths)

            # unicode → class_id
            class_ids = [ord(ch) - 20000 for ch in decoded]

            sent = [(self.i2g.get(cid, "UNK"), i)
                    for i, cid in enumerate(class_ids) if cid != self.blank]

            results.append(sent)


        return results