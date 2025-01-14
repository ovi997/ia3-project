from transformers import BertForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
from document_classification import load_bert_data
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


class DeepFoolAttack:
    def __init__(self, num_classes: int = 10, num_steps: int = 50):

        self.num_classes = num_classes
        self.num_steps = num_steps

    # def embeddings_to_ids(self, perturbed_embeddings):

    #     embedding_weights = self.model.get_input_embeddings().weight.data
    #     perturbed_ids = []

    #     for embedding in perturbed_embeddings[0]:  # Process batch with a single input
    #         distances = torch.norm(embedding_weights - embedding, dim=1)
    #         closest_id = torch.argmin(distances).item()
    #         perturbed_ids.append(closest_id)

    #     return torch.tensor([perturbed_ids])

    def get_min_perturbation(self, model, input_ids, attention_mask, label):

        embeddings = model.get_input_embeddings()(input_ids)
        perturbed_embeddings = embeddings.clone().detach().requires_grad_(True)
        original_label = label

        for _ in range(self.num_steps):
            outputs = model(inputs_embeds=perturbed_embeddings, attention_mask=attention_mask)
            logits = outputs.logits

            good_output = logits[0, original_label]
            model.zero_grad()
            good_output.backward(retain_graph=True)
            grad_orig = perturbed_embeddings.grad.data.clone()

            min_perturbation = float('inf')
            w = None
            r = None

            for k in range(self.num_classes):
                if k == original_label:
                    continue

                perturbed_embeddings.grad.zero_()
                
                class_output = logits[0, k]
                class_output.backward(retain_graph=True)
                
                grad_k = perturbed_embeddings.grad.data.clone()

                w_k = grad_k - grad_orig
                f_k = logits[0, k] - logits[0, original_label]
                
                perturbation = abs(f_k) / torch.norm(w_k)

                if perturbation < min_perturbation:
                    min_perturbation = perturbation
                    w = w_k
                
                r = perturbation * w / torch.norm(w)

            perturbed_embeddings = perturbed_embeddings + r

            perturbed_embeddings = perturbed_embeddings.detach().requires_grad_(True)
            outputs = model(inputs_embeds=perturbed_embeddings, attention_mask=attention_mask)
            
            new_label = torch.argmax(outputs.logits, dim=1).item()

            if new_label != original_label:
                break
        print(f'Real Prediction: {original_label}\nAdversarial Prediction: {new_label}')

        return r.cpu().detach().numpy()


if __name__ == "__main__":
    
    _DEVICE = 'cpu'
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=10)
    model.load_state_dict(torch.load('./document_classification/model/best_model.pth', map_location=_DEVICE, weights_only=True))
    # model.eval()
    model.to(_DEVICE)
    
    train_loader, dev_loader, test_loader = load_bert_data(16, 1)
    
    for i, batch in enumerate(test_loader):
        if i == 0 :
            inputs = {
                'input_ids': batch['input_ids'].to(_DEVICE),
                'attention_mask': batch['attention_mask'].to(_DEVICE),
                'label': batch['labels']
            }
        break
            
            
    input_ids = inputs['input_ids'] 
    attention_mask = inputs['attention_mask']  
    true_label = inputs['label']  

    deep_fool_attack = DeepFoolAttack()
    perturbation = deep_fool_attack.get_min_perturbation(model, input_ids, attention_mask, true_label)

    embeddings = model.get_input_embeddings()(input_ids).cpu().detach().numpy()
    
    perturbed_embeddings = embeddings + perturbation
    
    plt.figure()
    plt.title('Word embeddings with and without perturbation')
    plt.plot(perturbed_embeddings.squeeze()[2, :])
    plt.plot(embeddings.squeeze()[2, :])
    plt.legend(['Perturbed', 'Original'])
    plt.savefig('deep_fool_result.png')