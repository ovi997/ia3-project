from transformers import BertForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
from document_classification import load_bert_data
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from deep_fool import DeepFoolAttack
from fmn import FastMinNormAttack
import gc


def test_with_normal_data(model, loader):
    model.eval()
    
    predictions = []
    ids = []
    labels = []

    with torch.no_grad():
        for batch in loader:
            inputs = {
                'input_ids': batch['input_ids'].to(_DEVICE),
                'attention_mask': batch['attention_mask'].to(_DEVICE)
            }
            
            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            ids.extend(batch['ids'].cpu().numpy())
            labels.extend(batch['labels'].cpu().numpy())
    accuracy = np.mean(np.array(predictions) == np.array(labels))
    
    print(f'CLEAN Test Dataset -> Accuracy: {accuracy}')
    
    
def test_with_perturbed_data(model, loader, perturbation, perturb_type='DF'):
    model.eval()
    
    predictions = []
    ids = []
    labels = []

    with torch.no_grad():
        for batch in loader:
            inputs = {
                'input_ids': batch['input_ids'].to(_DEVICE),
                'attention_mask': batch['attention_mask'].to(_DEVICE)
            }
            
            embeddings = model.get_input_embeddings()(input_ids)
            embeddings = embeddings + perturbation
            
            outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            ids.extend(batch['ids'].cpu().numpy())
            labels.extend(batch['labels'].cpu().numpy())
    accuracy = np.mean(np.array(predictions) == np.array(labels))
    
    print(f'PERTURBED({perturb_type}) Test Dataset -> Accuracy: {accuracy}')


if __name__ == "__main__":
    
    _DEVICE = 'cpu'

    train_loader, dev_loader, test_loader = load_bert_data(1, 1)
    
    for i, batch in enumerate(test_loader):
        if i == 3 :
            inputs = {
                'input_ids': batch['input_ids'].to(_DEVICE),
                'attention_mask': batch['attention_mask'].to(_DEVICE),
                'label': batch['labels']
            }
            break
            
            
    input_ids = inputs['input_ids'] 
    attention_mask = inputs['attention_mask']  
    true_label = inputs['label']  

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=10)
    model.load_state_dict(torch.load('./document_classification/model/best_model.pth', map_location=_DEVICE, weights_only=True))
    model.to(_DEVICE)
    
    deep_fool_attack = DeepFoolAttack()
    fmn_attack = FastMinNormAttack()
    
    perturbation_df = deep_fool_attack.get_min_perturbation(model, input_ids, attention_mask, true_label)
    perturbation_fmn = fmn_attack.get_min_perturbation(model=model, inputs_ids=input_ids, attention_mask=attention_mask, label=true_label)
    
    print("Normal model results:")
    test_with_normal_data(model, test_loader)
    test_with_perturbed_data(model, test_loader, perturbation_df)
    test_with_perturbed_data(model, test_loader, perturbation_fmn, perturb_type='FMN')
    
    del model
    del deep_fool_attack
    del fmn_attack
    gc.collect()
    
    print("Student model results:")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=10)
    model.load_state_dict(torch.load('./document_classification/student/student.pth', map_location=_DEVICE, weights_only=True))
    model.to(_DEVICE)
    
    test_with_normal_data(model, test_loader)
    test_with_perturbed_data(model, test_loader, perturbation_df)
    test_with_perturbed_data(model, test_loader, perturbation_fmn, perturb_type='FMN')
 
    del model
    gc.collect()
    
    print("Model trained with unnormalized text results:")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=10)
    model.load_state_dict(torch.load('./document_classification/unsqueezed/unsqueezed_model.pth', map_location=_DEVICE, weights_only=True))
    model.to(_DEVICE)
    
    test_with_normal_data(model, test_loader)
    test_with_perturbed_data(model, test_loader, perturbation_df)
    test_with_perturbed_data(model, test_loader, perturbation_fmn, perturb_type='FMN')
