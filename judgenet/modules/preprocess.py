import torch
from transformers import BertModel, BertTokenizer


class SentenceEncoder():
    def __init__(self, device=None):
        self.device = torch.device('cpu') if device is None else device
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased")
        self.model = BertModel.from_pretrained(
            "bert-base-uncased").to(self.device)
    
    def _tokenize_function(self, sentence):
        return self.tokenizer(
            sentence,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
    
    def tokenize(self, sentences):
        batched_tokens = {
            "input_ids": [],
            "attention_mask": []
        }
        for sentence in sentences:
            tokens = self._tokenize_function(sentence)
            for key in ("input_ids", "attention_mask"):
                batched_tokens[key].append(tokens[key])
        for key in ("input_ids", "attention_mask"):
            batched_tokens[key] = torch.cat(batched_tokens[key], dim=0)
        return batched_tokens
    
    def encode_batched_tokens(self, batched_tokens):
        return self.model(**batched_tokens).last_hidden_state[:, 0, :]


if __name__=="__main__":
    se = SentenceEncoder()
    batched_tokens = se.tokenize(["hello my name is Jullian", "How are you today?"])
    sentence_embeddings = se.encode_batched_tokens(batched_tokens)
    print(
        batched_tokens["input_ids"].shape,
        batched_tokens["attention_mask"].shape,
        sentence_embeddings.shape
    )
