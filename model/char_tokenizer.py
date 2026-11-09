import re

OZEL_TOKENLAR = ["<TARIF_BASLA>", "<ISIM>", "<MALZEMELER>", "<YAPILISI>", "<TARIF_BITIR>", "<UNK>"]
# Parantezli desen: re.split bölerken ayırıcıları da listede tutar
DESEN = "(" + "|".join(re.escape(t) for t in OZEL_TOKENLAR) + ")"


class CharTokenizer:
    def __init__(self, metin):
        # 1) Özel token'ları çıkar, kalan metindeki benzersiz karakterleri bul
        duz_metin = re.sub(DESEN, "", metin)
        karakterler = sorted(set(duz_metin))

        # 2) Vocab: önce özel token'lar (id 0-5), sonra karakterler
        vocab = OZEL_TOKENLAR + karakterler
        self.stoi = {tok: i for i, tok in enumerate(vocab)}   # token -> id
        self.itos = {i: tok for tok, i in self.stoi.items()}  # id -> token

    @property
    def vocab_size(self):
        return len(self.stoi)

    def encode(self, metin):
        unk_id = self.stoi["<UNK>"]
        ids = []
        for parca in re.split(DESEN, metin):
            if parca in OZEL_TOKENLAR:
                ids.append(self.stoi[parca])                         # özel token = tek id
            else:
                ids.extend(self.stoi.get(ch, unk_id) for ch in parca)  # bilinmeyen -> <UNK>
        return ids

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)