from utils import get_optimizer_and_scheduler
import torch.nn.functional as F
from finetune_bert_classifier import evaluate_model
from load_data import load_bert_data
from transformers import BertForSequenceClassification
import torch
from tqdm import tqdm

_TEACHER_MODEL = './teacher/teacher.pth'
_STUDENT_MODEL = './student/student.pth'

class Config:
    num_epochs: int = 20
    batch_size: int = 16
    test_batch_size: int = 32
    optimizer_type: str = 'AdamW'
    learning_rate: float = 5e-6
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    scheduler_type: str = 'cosine'
    temperature: int = 5
    DEVICE: str = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def train_student(config, student_model, teacher_model, train_loader, dev_loader, optimizer, scheduler, path_to_save):
    best_score = float('-inf')
    
    teacher_model.to('cuda')
    teacher_model.eval()
    
    batch_targets = []

    for batch in tqdm(train_loader):
        
        with torch.no_grad():
            inputs = {
                'input_ids': batch['input_ids'].to('cuda'),
                'attention_mask': batch['attention_mask'].to('cuda'),
                'labels': batch['labels'].to('cuda')
            }
            
            teacher_out = teacher_model(**inputs)
            teacher_logits = teacher_out.logits / config.temperature         
            
            targets = F.softmax(teacher_logits, dim=-1).to('cpu')
            batch_targets.append(targets)

    teacher_model.to('cpu')
    del teacher_model
        
    student_model.to(config.DEVICE)        
    criterion = torch.nn.KLDivLoss(reduction="batchmean")

    for epoch in range(config.num_epochs):
        student_model.train()
        total_train_loss = 0

        for i, batch in tqdm(enumerate(train_loader), desc=f"Training Epoch {epoch+1}"):
            optimizer.zero_grad()

            inputs = {
                'input_ids': batch['input_ids'].to(config.DEVICE),
                'attention_mask': batch['attention_mask'].to(config.DEVICE),
                'labels': batch['labels'].to(config.DEVICE)
            }
            
            labels = batch['labels'].to(config.DEVICE)

            stud_outputs = student_model(**inputs)
            logits = stud_outputs.logits / config.temperature
            
            outputs = F.log_softmax(logits, dim=-1)
            
            targets = batch_targets[i].to(config.DEVICE)
            
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        eval_loss = evaluate_model(student_model, dev_loader)
        score = -eval_loss
        
        if score > best_score:
            best_score = score
            torch.save(student_model.state_dict(), path_to_save)

        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {eval_loss:.4f}")
        
        
if __name__ == '__main__':
    
    config = Config()
    temperature = 5
    train_loader, dev_loader, test_loader = load_bert_data(config.batch_size, config.test_batch_size)
    
    student_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=10)  
    optimizer, scheduler = get_optimizer_and_scheduler(config, student_model, train_loader)
    
    
    teacher_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=10) 
    teacher_model.load_state_dict(torch.load(_TEACHER_MODEL, weights_only=True))
    
    print("Training model")
    train_student(config, student_model, teacher_model, train_loader, dev_loader, optimizer, scheduler, _STUDENT_MODEL)
    