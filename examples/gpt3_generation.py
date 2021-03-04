import transformers
from transformers import BertTokenizerFast, TFGPT2LMHeadModel, GPT2LMHeadModel
transformers.logging.set_verbosity_error()


class Inference:
    def __init__(self, model_name, device, tf_pt='tf'):
        self.tf_pt = tf_pt
        self.device = device
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)

        if self.tf_pt == 'tf':
            self.model = TFGPT2LMHeadModel.from_pretrained(model_name, pad_token_id=self.tokenizer.eos_token_id)
        else:
            self.model = GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=self.tokenizer.eos_token_id)

        self.model.device(device)

    def __call__(self, text, howmany=1, length=100):
        input_ids = self.tokenizer.encode(text, return_tensors='tf' if self.tf_pt=='tf' else 'pt')

        # input_ids also need to apply gpu device!
        input_ids = input_ids.to(self.device)

        input_ids = input_ids[:, 1:]  # remove cls token

        min_length = len(input_ids.tolist()[0])
        length = length if length > 0 else 1
        length += min_length

        outputs = self.model.generate(
            input_ids,
            min_length=min_length,
            max_length=length,
            do_sample=True,
            top_k=10,
            top_p=0.95,
            no_repeat_ngram_size=2,
            num_return_sequences=howmany
        )

        result = dict()

        for idx, sample_output in enumerate(outputs):
            result[idx] = self.tokenizer.decode(sample_output.tolist(), skip_special_tokens=True)

        return result

