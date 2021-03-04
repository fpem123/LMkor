import transformers
import torch
from transformers import BertTokenizerFast, EncoderDecoderModel
transformers.logging.set_verbosity_error()


class Summarize:
    def __init__(self, model_name):
        transformers.logging.set_verbosity_error()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)

        self.device = torch.device(device)
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.model = EncoderDecoderModel.from_pretrained(model_name)

    def __call__(self, text, samples=1):
        input_ids = self.tokenizer.encode(text, return_tensors='pt')
        input_ids = input_ids.to(self.device)

        sentence_length = len(input_ids[0])
        min_length = max(10, int(0.1*sentence_length))
        max_length = min(128, int(0.3*sentence_length))

        outputs = self.model.generate(
            input_ids,
            min_length=min_length,
            max_length=max_length,
            num_return_sequences=samples
        )

        result = dict()

        for idx, sample_output in enumerate(outputs):
            result[idx] = self.tokenizer.decode(sample_output.tolist(), skip_special_tokens=True)

        return result
