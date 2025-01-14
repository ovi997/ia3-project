from transformers import BertForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
from document_classification import load_bert_data
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import math

def l2_projection(delta, epsilon):

    l2_norms = torch.clamp(delta.norm(p=2, dim=-1, keepdim=True), min=1e-12)
    delta = torch.mul(input=delta, other=(epsilon.unsqueeze(-1)) / l2_norms)
    delta = torch.clamp(delta, min=-1, max=1)
    
    return delta

def h_func(x_0, x_K, k, K):
    return x_K + ((x_0 - x_K) / 2) * (1 + math.cos((k * math.pi) / K))

class FastMinNormAttack:
    
    def __init__(self, alpha_0: float = 1, gamma_0: float = 0.1, p: int = 2, num_steps: int = 50, targeted: int = -1):
        self.alpha_0 = alpha_0
        self.alpha_K: float = alpha_0 / num_steps
        self.gamma_0 = gamma_0
        self.gamma_K: float = gamma_0 / num_steps
        self.adv_found: bool = False
        self.targeted = targeted
        self.p = p 
        self.num_steps = num_steps
        
    def get_min_perturbation(self, model: torch.nn.Module = None, label: int = None, inputs_ids: torch.tensor = None, attention_mask: torch.tensor = None):
        
        with torch.no_grad():
            embeddings = model.bert.embeddings.word_embeddings(inputs_ids)        
        
        self.delta: torch.tensor = torch.zeros_like(embeddings, requires_grad=True) 
        self.delta_star: torch.tensor = torch.full(self.delta.shape, float('inf'))            
        
        for k in range(1, self.num_steps + 1):
            
            perturbed_embeddings = embeddings + self.delta
            
            outputs = model(inputs_embeds=perturbed_embeddings, attention_mask=attention_mask)
            
            logits = outputs.logits

            loss = F.cross_entropy(logits, torch.tensor([label]))

            model.zero_grad()
            self.delta.retain_grad()
            loss.backward(retain_graph=True)
            
            delta_grad = self.targeted * (self.delta.grad)
            self.gamma_k = h_func(self.gamma_0, self.gamma_K, k, self.num_steps)
            
            current_prediction = torch.argmax(F.softmax(logits, dim=-1)).item()
            if current_prediction == label:
                if not self.adv_found:
                    self.epsilon_k = self.delta.norm(p=self.p, dim=-1) + loss.item() / delta_grad.norm(p=self.p, dim=-1)
                else:
                    self.epsilon_k = self.epsilon_k * (1 + self.gamma_k)
            else:
                self.adv_found = True
                self.delta_star = torch.minimum(self.delta, self.delta_star)
                self.epsilon_k = torch.minimum(self.epsilon_k * (1 - self.gamma_k), self.delta_star.norm(p=self.p, dim=-1))

            self.alpha_k = h_func(self.alpha_0, self.alpha_K, k, self.num_steps)
            
            scaled_gradient = delta_grad / delta_grad.norm(p=self.p, dim=-1, keepdim=True)
            self.delta = self.delta + self.alpha_k * scaled_gradient
                
            self.delta = l2_projection(delta=(self.delta + embeddings), epsilon=self.epsilon_k) - embeddings

            self.delta = torch.clamp((self.delta + embeddings), min=-1, max=1) - embeddings
        
        with torch.no_grad():
            adv_embeddings = embeddings + self.delta_star
            adv_logits = model(inputs_embeds=adv_embeddings, attention_mask=attention_mask).logits
            adv_pred = torch.argmax(F.softmax(adv_logits, dim=-1)).item()
            print(f'Real Prediction: {label}\nAdversarial Prediction: {adv_pred}')
            

        return self.delta_star.cpu()


if __name__ == '__main__':
    _DEVICE = 'cpu'

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=10)
    model.load_state_dict(torch.load('./document_classification/model/best_model.pth', map_location=_DEVICE, weights_only=True))

    train_loader, dev_loader, test_loader = load_bert_data(16, 1)

    for i, batch in enumerate(test_loader):
        if i == len(test_loader) - 1 :
            inputs = {
                'input_ids': batch['input_ids'].to(_DEVICE),
                'attention_mask': batch['attention_mask'].to(_DEVICE),
                'label': batch['labels'].item()
            }
            break
            

    input_ids = inputs['input_ids']  

    attention_mask = inputs['attention_mask']  
    true_label = inputs['label'] 

    fmn_attack = FastMinNormAttack()

    perturbation = fmn_attack.get_min_perturbation(model=model, inputs_ids=input_ids, attention_mask=attention_mask, label=true_label)

    embeddings = model.get_input_embeddings()(input_ids)
    
    plt.figure()
    plt.title('Word embeddings with and without perturbation')
    plt.plot((embeddings + perturbation).detach().numpy().squeeze()[0, :])
    plt.plot(embeddings.detach().numpy().squeeze()[0, :])
    plt.legend(['Perturbed', 'Original'])
    plt.savefig('fnm_result.png')
    
