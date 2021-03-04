# inspired by https://github.com/renatoviolin/next_word_prediction

import torch
import string
import transformers
from transformers import BertTokenizerFast, BertForMaskedLM, AlbertForMaskedLM

transformers.logging.set_verbosity_error()


class Predict:

    def __init__(self, device):
        self.device = device

        self.bert_tokenizer = BertTokenizerFast.from_pretrained('kykim/bert-kor-base')
        self.bert_model = BertForMaskedLM.from_pretrained('kykim/bert-kor-base').eval()
        self.bert_model.device(device)

        self.albert_tokenizer = BertTokenizerFast.from_pretrained('kykim/albert-kor-base')
        self.albert_model = AlbertForMaskedLM.from_pretrained('kykim/albert-kor-base').eval()
        self.albert_model.device(device)

    def decode(tokenizer, pred_idx, top_clean):
        ignore_tokens = string.punctuation + '[PAD][UNK]<pad><unk> '
        tokens = []

        for w in pred_idx:
            token = ''.join(tokenizer.decode(w).split())

            if token not in ignore_tokens:
                tokens.append(token.replace('##', ''))

        return ' / '.join(tokens[:top_clean])

    def encode(tokenizer, text_sentence, add_special_tokens=True, mask_token='[MASK]', mask_token_id=4):
        # mask_token = tokenizer.mask_token
        # mask_token_id = tokenizer.mask_token_id

        text_sentence = text_sentence.replace('<mask>', mask_token)
        # if <mask> is the last token, append a "." so that models dont predict punctuation.
        if mask_token == text_sentence.split()[-1]:
            text_sentence += ' .'

        input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
        mask_idx = torch.where(input_ids == mask_token_id)[1].tolist()[0]

        return input_ids, mask_idx

    def predict(self, text_sentence, top_k=10, top_clean=3):
        if '<mask>' not in text_sentence:
            print('<mask> 를 입력해주세요. 예시: 이거 <mask> 재밌네? ')
            return

        # ========================= BERT =================================
        input_ids, mask_idx = self.encode(self.bert_tokenizer, text_sentence)
        input_ids = input_ids.to(self.device)

        with torch.no_grad():
            predict = self.bert_model(input_ids)[0]

        bert = self.decode(self.bert_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

        # ========================= ALBERT =================================
        input_ids, mask_idx = self.encode(self.albert_tokenizer, text_sentence)
        input_ids = input_ids.to(self.device)

        with torch.no_grad():
            predict = self.albert_model(input_ids)[0]

        albert = self.decode(self.albert_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

        results = {'kykim/bert-kor-base': bert,
                'kykim/albert-kor-base': albert}

        return results
