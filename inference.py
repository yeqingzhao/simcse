import torch
import torch.nn.functional as F
from transformers import BertModel
from transformers import BertPreTrainedModel
from transformers import BertTokenizer


class SimCseModel(BertPreTrainedModel):
    """自定义模型"""

    def __init__(self, config):
        super(SimCseModel, self).__init__(config)
        self.bert = BertModel(config)  # transformers的写法，方便保存，加载模型
        self.output_way = 'cls'

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # sequence_output, pooler_output, hidden_states = outputs[0], outputs[1], outputs[2]
        if self.output_way == 'cls':
            output = outputs.last_hidden_state[:, 0]
        else:
            output = outputs.pooler_output
            output = torch.tanh(output)

        return output


if __name__ == '__main__':
    sentences = ['******', '******', '******']

    model_path = './simcse'
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = SimCseModel.from_pretrained(model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = tokenizer(sentences, truncation=True, padding=True, max_length=512, return_tensors='pt')
        input_ids, attention_mask, token_type_ids = outputs['input_ids'].to(device), outputs['attention_mask'].to(device), outputs['token_type_ids'].to(device)
        embedding = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sim_score = F.cosine_similarity(embedding.unsqueece(0), embedding.unsqueece(1), dim=2)[0]

        for i, sentence in enumerate(sentences):
            print(f"{sentences[0]}###{sentence}###相似度为：{sim_score[i]:.5f}")