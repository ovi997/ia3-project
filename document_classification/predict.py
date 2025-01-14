from transformers import BertForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
from load_data import load_bert_data
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from utils import get_optimizer_and_scheduler, save_predictions, load_label_mapping

train_loader, dev_loader, test_loader = load_bert_data(16, 1)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=10) 

model.load_state_dict(torch.load('best_model.pth', weights_only=True))

model.eval()
model.to('cuda')
def predict(model, inputs):
    model.eval()
    predictions = []
    ids = []
    labels = []

    with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

    return preds

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

for i, batch in enumerate(test_loader):
    inputs = {
        'input_ids': batch['input_ids'].to('cuda'),
        'attention_mask': batch['attention_mask'].to('cuda')
    }
    
    if i == 0 :
        text = tokenizer.convert_ids_to_tokens(ids=batch['input_ids'][0])
        print(text)
    print(predict(model, inputs))
