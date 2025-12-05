# llm_lab/core/tokenization/char_tokenizer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict,List,Sequence,Optional

UNK_TOKEN = "<unk>"

@dataclass
class CharTokenizer:
    stoi : Dict[str,int]
    itos : Dict[int,str]
    unk_id : Optional[int] = None

    @classmethod
    def from_text(cls, text : str, add_unk : bool = False ) -> "CharTokenizer":
        """
        Build a character vocabulary from the given text.

        If add_unk=True, include an <unk> token and map unknown chars to it.
        """
        stoi : Dict[str,int] = {}
        itos : Dict[int,str] = {}
        unk_id : Optional[int] = None
        presentToken = 0
        for c in text:
            if c not in stoi :
                stoi[c]=presentToken
                itos[presentToken]=c
                presentToken += 1
        if add_unk == True :
            stoi[UNK_TOKEN]=presentToken
            itos[presentToken] = UNK_TOKEN
            unk_id = presentToken
        return cls(stoi=stoi,itos=itos,unk_id=unk_id)

    def encode (self, s : str) -> List[int]:
        """
        Convert a string into a list of token ids.
        """
        encodedOutput : List[int] = []
        for c in s :
            if c in self.stoi:
                encodedOutput.append(self.stoi[c])
            elif self.unk_id is not None:
                encodedOutput.append(self.unk_id)
            else:
                 raise KeyError ("No Unk token set")
        return encodedOutput

    def decode (self, ids: Sequence[int]) -> str:
        """
        Convert a list of token ids back into a string.
        """
        decodedOutput : str = ""
        for id in ids:
            if id in self.itos:
                decodedOutput +=self.itos[id]
            elif self.unk_id is not None and id ==self.unk_id :
                decodedOutput += UNK_TOKEN
            else:
                raise KeyError (f"unknown token {id}")
        return decodedOutput

                   