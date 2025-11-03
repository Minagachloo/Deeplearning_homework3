import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import BertModel, BertTokenizerFast, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import defaultdict
import re
import time

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print("=" * 70)
print("BERT QA MODEL WITH BASELINE COMPARISON")
print("=" * 70)

# Configuration
config = {
    'model_name': 'bert-base-uncased',
    'max_len': 384,
    'max_question_len': 64,
    'max_paragraph_len': 256,
    'doc_stride': 128,
    'batch_size': 16,
    'lr': 2e-5,
    'epochs': 5,
    'warmup_ratio': 0.1,
    'max_train_samples': 10000,
    'max_test_samples': 1000,
    'gradient_accumulation_steps': 2,
    'weight_decay': 0.01,
    'max_grad_norm': 1.0
}

def load_squad_data(file_path, max_samples=None):
    """Load and parse SQuAD format data"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    examples = []
    
    for article in data['data']:
        for para in article['paragraphs']:
            context = para['context'].strip()
            for qa in para['qas']:
                question = qa['question'].strip()
                
                if qa['answers']:
                    for answer in qa['answers']:
                        answer_text = answer['text'].strip()
                        answer_start = answer['answer_start']
                        
                        # Verify answer is actually in context
                        if context[answer_start:answer_start + len(answer_text)] == answer_text:
                            examples.append({
                                'id': qa['id'],
                                'context': context,
                                'question': question,
                                'answer': {
                                    'text': answer_text,
                                    'answer_start': answer_start
                                }
                            })
                            break
                
                if max_samples and len(examples) >= max_samples:
                    return examples
    
    return examples

print("Loading data...")
train_examples = load_squad_data('spoken_train-v1.1.json', config['max_train_samples'])
test_examples = load_squad_data('spoken_test-v1.1.json', config['max_test_samples'])
print(f"Train: {len(train_examples)} samples | Test: {len(test_examples)} samples")

# Tokenizer
tokenizer = BertTokenizerFast.from_pretrained(config['model_name'])

def find_answer_span(context_tokens, answer_text, tokenizer):
    """Find answer span in tokenized context"""
    answer_tokens = tokenizer.tokenize(answer_text)
    
    # Try to find exact match first
    for i in range(len(context_tokens) - len(answer_tokens) + 1):
        if context_tokens[i:i + len(answer_tokens)] == answer_tokens:
            return i, i + len(answer_tokens) - 1
    
    # Fallback: find partial match
    for i in range(len(context_tokens)):
        for j in range(i + 1, min(i + len(answer_tokens) + 3, len(context_tokens))):
            candidate = tokenizer.convert_tokens_to_string(context_tokens[i:j])
            if answer_text.lower() in candidate.lower() or candidate.lower() in answer_text.lower():
                return i, j - 1
    
    return None, None

def create_features_from_examples(examples, tokenizer, config, is_training=True):
    """Create features with sliding windows"""
    features = []
    
    for example_idx, example in enumerate(tqdm(examples, desc="Creating features")):
        context = example['context']
        question = example['question']
        
        # Tokenize
        question_tokens = tokenizer.tokenize(question)
        if len(question_tokens) > config['max_question_len']:
            question_tokens = question_tokens[:config['max_question_len']]
        
        context_tokens = tokenizer.tokenize(context)
        
        # Find answer span in tokens
        answer_start_token = None
        answer_end_token = None
        
        if is_training and example['answer'] is not None:
            answer_text = example['answer']['text']
            answer_start_token, answer_end_token = find_answer_span(context_tokens, answer_text, tokenizer)
        
        # Create sliding windows
        max_context_len = config['max_paragraph_len']
        doc_stride = config['doc_stride']
        
        starts = list(range(0, len(context_tokens), doc_stride))
        if not starts:
            starts = [0]
        
        for start_idx in starts:
            end_idx = min(start_idx + max_context_len, len(context_tokens))
            window_tokens = context_tokens[start_idx:end_idx]
            
            # Build sequence
            tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + window_tokens + ['[SEP]']
            
            # Convert to IDs
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            
            # Ensure within max length
            if len(input_ids) > config['max_len']:
                max_context_in_window = config['max_len'] - len(question_tokens) - 3
                window_tokens = window_tokens[:max_context_in_window]
                tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + window_tokens + ['[SEP]']
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
            
            # Pad to max length
            padding_length = config['max_len'] - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
            
            # Token type IDs
            token_type_ids = ([0] * (len(question_tokens) + 2) + 
                            [1] * (len(window_tokens) + 1) + 
                            [0] * padding_length)
            
            # Attention mask
            attention_mask = [1] * len(tokens) + [0] * padding_length
            
            # Answer positions
            start_position = 0
            end_position = 0
            
            if (is_training and answer_start_token is not None and answer_end_token is not None):
                # Check if answer is in this window
                if (answer_start_token >= start_idx and answer_end_token < end_idx):
                    # Adjust for question prefix
                    start_position = answer_start_token - start_idx + len(question_tokens) + 2
                    end_position = answer_end_token - start_idx + len(question_tokens) + 2
                    
                    # Ensure valid positions
                    if start_position >= len(tokens) or end_position >= len(tokens):
                        start_position = 0
                        end_position = 0
            
            features.append({
                'example_id': example['id'],
                'example_idx': example_idx,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'start_position': start_position,
                'end_position': end_position,
                'window_start': start_idx,
                'tokens': tokens
            })
            
            # For training: if we have the answer in this window, focus on it
            if is_training and start_position > 0:
                break
            
            # Stop if we've covered all context
            if end_idx >= len(context_tokens):
                break
    
    return features

print("\nCreating features...")
train_features = create_features_from_examples(train_examples, tokenizer, config, is_training=True)
test_features = create_features_from_examples(test_examples, tokenizer, config, is_training=False)
print(f"Train features: {len(train_features)} | Test features: {len(test_features)}")

# Dataset
class QADataset(Dataset):
    def __init__(self, features):
        self.features = features
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        return {
            'input_ids': torch.tensor(feature['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(feature['attention_mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(feature['token_type_ids'], dtype=torch.long),
            'start_position': torch.tensor(feature['start_position'], dtype=torch.long),
            'end_position': torch.tensor(feature['end_position'], dtype=torch.long)
        }

train_dataset = QADataset(train_features)
test_dataset = QADataset(test_features)

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])

# Model
class QAModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.qa_outputs = nn.Linear(self.bert.config.hidden_size, 2)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.qa_outputs(sequence_output)
        
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        return start_logits, end_logits

def compute_loss(start_logits, end_logits, start_positions, end_positions, attention_mask):
    """Compute QA loss"""
    # Mask invalid positions
    start_logits = start_logits + (1.0 - attention_mask) * -10000.0
    end_logits = end_logits + (1.0 - attention_mask) * -10000.0
    
    start_loss = nn.CrossEntropyLoss()(start_logits, start_positions)
    end_loss = nn.CrossEntropyLoss()(end_logits, end_positions)
    
    return (start_loss + end_loss) / 2

def compute_f1_em(pred_text, true_text):
    """Compute F1 and Exact Match scores"""
    def normalize_text(text):
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    pred_normalized = normalize_text(pred_text)
    true_normalized = normalize_text(true_text)
    
    # Exact Match
    em = 1.0 if pred_normalized == true_normalized else 0.0
    
    # F1 Score
    if not pred_normalized or not true_normalized:
        return 0.0, em
    
    pred_tokens = pred_normalized.split()
    true_tokens = true_normalized.split()
    
    common_tokens = set(pred_tokens) & set(true_tokens)
    
    if not common_tokens:
        return 0.0, em
    
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(true_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    
    return f1, em

def evaluate_model(model, test_features, test_examples, tokenizer):
    """Evaluate model on test set"""
    model.eval()
    
    example_to_features = defaultdict(list)
    for i, feature in enumerate(test_features):
        example_to_features[feature['example_idx']].append((i, feature))
    
    predictions = []
    all_f1 = []
    all_em = []
    
    with torch.no_grad():
        for example_idx, example in enumerate(tqdm(test_examples, desc="Evaluating")):
            if example_idx not in example_to_features:
                predictions.append("")
                all_f1.append(0.0)
                all_em.append(0.0)
                continue
            
            # Get all windows for this example
            windows = example_to_features[example_idx]
            best_score = float('-inf')
            best_start = 0
            best_end = 0
            best_feature = None
            
            # Score each window
            for feature_idx, feature in windows:
                input_ids = torch.tensor([feature['input_ids']]).to(device)
                attention_mask = torch.tensor([feature['attention_mask']]).to(device)
                token_type_ids = torch.tensor([feature['token_type_ids']]).to(device)
                
                start_logits, end_logits = model(input_ids, attention_mask, token_type_ids)
                
                start_logits = start_logits[0].cpu().numpy()
                end_logits = end_logits[0].cpu().numpy()
                
                # Find best span in this window
                for start_idx in range(len(start_logits)):
                    for end_idx in range(start_idx, min(start_idx + 20, len(end_logits))):
                        if feature['attention_mask'][start_idx] == 0 or feature['attention_mask'][end_idx] == 0:
                            continue
                        score = start_logits[start_idx] + end_logits[end_idx]
                        if score > best_score:
                            best_score = score
                            best_start = start_idx
                            best_end = end_idx
                            best_feature = feature
            
            # Extract prediction
            if best_feature is not None:
                pred_tokens = best_feature['input_ids'][best_start:best_end + 1]
                pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)
            else:
                pred_text = ""
            
            predictions.append(pred_text)
            
            # Compute metrics
            if example['answer'] is not None:
                true_text = example['answer']['text']
                f1, em = compute_f1_em(pred_text, true_text)
                all_f1.append(f1)
                all_em.append(em)
            else:
                all_f1.append(0.0)
                all_em.append(0.0)
    
    return predictions, np.mean(all_f1), np.mean(all_em)

# ============================================================================
# BASELINE EVALUATION - CRITICAL FOR COMPARISON
# ============================================================================

print("\n" + "=" * 70)
print("STEP 1: ESTABLISHING BASELINE PERFORMANCE")
print("=" * 70)
print("Evaluating pre-trained BERT without fine-tuning (zero-shot)...")
print("This is our BASELINE to beat...")
print("\nNote: Pre-trained BERT without fine-tuning typically performs very poorly")
print("on QA tasks because:")
print("  1. The QA head (start/end classifiers) is randomly initialized")
print("  2. BERT was pre-trained on MLM, not question answering")
print("  3. It has no task-specific knowledge of finding answer spans")
print("\nExpect baseline scores to be very low (often near random chance).\n")

# Create baseline model (pre-trained BERT without fine-tuning)
baseline_model = QAModel(config['model_name']).to(device)
baseline_start_time = time.time()

# Evaluate baseline
baseline_predictions, baseline_f1, baseline_em = evaluate_model(
    baseline_model, test_features, test_examples, tokenizer
)
baseline_eval_time = time.time() - baseline_start_time

print(f"\n{'='*50}")
print("BASELINE RESULTS (Pre-trained BERT, No Fine-tuning):")
print(f"{'='*50}")
print(f"F1 Score: {baseline_f1:.4f}")
print(f"Exact Match: {baseline_em:.4f}")
print(f"Evaluation Time: {baseline_eval_time:.2f} seconds")
print(f"{'='*50}")
print(f"\nThese are the scores to beat!")

# Store baseline for comparison
baseline_scores = {
    'f1': baseline_f1,
    'em': baseline_em
}

# ============================================================================
# FINE-TUNED MODEL TRAINING
# ============================================================================

print("\n" + "=" * 70)
print("STEP 2: TRAINING FINE-TUNED MODEL")
print("=" * 70)

# Create fine-tuned model
model = QAModel(config['model_name']).to(device)

# Optimizer and scheduler
optimizer = AdamW(
    model.parameters(), 
    lr=config['lr'], 
    weight_decay=config['weight_decay'],
    eps=1e-8,
    betas=(0.9, 0.999)
)

total_steps = len(train_loader) * config['epochs'] // config['gradient_accumulation_steps']
warmup_steps = int(total_steps * config['warmup_ratio'])
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

def train_epoch(model, loader, optimizer, scheduler, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    start_correct = 0
    end_correct = 0
    total_samples = 0
    optimizer.zero_grad()
    
    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    for step, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        start_positions = batch['start_position'].to(device)
        end_positions = batch['end_position'].to(device)
        
        start_logits, end_logits = model(input_ids, attention_mask, token_type_ids)
        
        loss = compute_loss(start_logits, end_logits, start_positions, end_positions, attention_mask.float())
        loss = loss / config['gradient_accumulation_steps']
        loss.backward()
        
        if (step + 1) % config['gradient_accumulation_steps'] == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Compute accuracy
        start_preds = start_logits.argmax(dim=-1)
        end_preds = end_logits.argmax(dim=-1)
        
        start_correct += (start_preds == start_positions).sum().item()
        end_correct += (end_preds == end_positions).sum().item()
        total_samples += start_positions.size(0)
        total_loss += loss.item() * config['gradient_accumulation_steps']
        
        pbar.set_postfix({
            'loss': f'{loss.item() * config["gradient_accumulation_steps"]:.3f}',
            'start_acc': f'{start_correct/total_samples:.3f}',
            'end_acc': f'{end_correct/total_samples:.3f}'
        })
    
    return total_loss / len(loader), start_correct / total_samples, end_correct / total_samples

# Training with baseline comparison
metrics = {
    'train_loss': [], 
    'train_start_acc': [], 
    'train_end_acc': [],
    'eval_f1': [], 
    'eval_em': [],
    'lr': [],
    'f1_improvement': [],  # New: track improvement over baseline
    'em_improvement': []   # New: track improvement over baseline
}

print(f"\nStarting training for {config['epochs']} epochs...")
print(f"Baseline to beat - F1: {baseline_f1:.4f}, EM: {baseline_em:.4f}\n")

for epoch in range(1, config['epochs'] + 1):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch}/{config['epochs']}")
    print(f"{'='*60}")
    
    # Train
    train_loss, start_acc, end_acc = train_epoch(model, train_loader, optimizer, scheduler, epoch)
    
    # Evaluate
    predictions, eval_f1, eval_em = evaluate_model(model, test_features, test_examples, tokenizer)
    
    # Calculate improvement over baseline (handle edge cases)
    if baseline_f1 > 0:
        f1_improvement = ((eval_f1 - baseline_f1) / baseline_f1) * 100
    else:
        f1_improvement = float('inf') if eval_f1 > 0 else 0
    
    if baseline_em > 0:
        em_improvement = ((eval_em - baseline_em) / baseline_em) * 100
    else:
        em_improvement = float('inf') if eval_em > 0 else 0
    
    # Store metrics
    metrics['train_loss'].append(train_loss)
    metrics['train_start_acc'].append(start_acc)
    metrics['train_end_acc'].append(end_acc)
    metrics['eval_f1'].append(eval_f1)
    metrics['eval_em'].append(eval_em)
    metrics['lr'].append(scheduler.get_last_lr()[0])
    metrics['f1_improvement'].append(f1_improvement)
    metrics['em_improvement'].append(em_improvement)
    
    print(f"\nEpoch {epoch} Results:")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Start Acc: {start_acc:.4f} | End Acc: {end_acc:.4f}")
    print(f"\n  Current Performance:")
    print(f"  F1 Score: {eval_f1:.4f} | EM Score: {eval_em:.4f}")
    print(f"\n  vs. Baseline (Pre-trained BERT):")
    print(f"  Baseline F1: {baseline_f1:.4f} → Current F1: {eval_f1:.4f}", end="")
    if f1_improvement == float('inf'):
        print(f" (∞ improvement - baseline was ~0)")
    elif f1_improvement > 1000:
        print(f" ({eval_f1/baseline_f1:.1f}x better)")
    else:
        print(f" ({f1_improvement:+.1f}%)")
    
    print(f"  Baseline EM: {baseline_em:.4f} → Current EM: {eval_em:.4f}", end="")
    if em_improvement == float('inf'):
        print(f" (∞ improvement - baseline was 0)")
    elif em_improvement > 1000:
        print(f" ({eval_em/baseline_em:.1f}x better)")
    else:
        print(f" ({em_improvement:+.1f}%)")
    
    if eval_f1 > baseline_f1:
        if f1_improvement > 1000 or em_improvement > 1000:
            print(f"\n  ✓ DRAMATICALLY BEATING BASELINE!")
            print(f"    Note: Baseline performs very poorly without fine-tuning")
            print(f"    (Pre-trained BERT has random QA head, no task-specific training)")
        else:
            print(f"\n  ✓ BEATING BASELINE by {f1_improvement:.1f}% (F1) and {em_improvement:.1f}% (EM)!")
    else:
        print(f"\n  ✗ Still below baseline")
    
    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics,
        'baseline_scores': baseline_scores
    }, f'checkpoint_epoch_{epoch}.pt')

# ============================================================================
# FINAL RESULTS AND COMPARISON
# ============================================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETED - FINAL RESULTS")
print("=" * 70)

best_f1_epoch = np.argmax(metrics['eval_f1']) + 1
best_em_epoch = np.argmax(metrics['eval_em']) + 1
best_f1 = max(metrics['eval_f1'])
best_em = max(metrics['eval_em'])

print("\n📊 BASELINE vs FINE-TUNED MODEL COMPARISON:")
print("=" * 70)
print(f"{'Model':<30} {'F1 Score':<15} {'Exact Match':<15}")
print("-" * 60)
print(f"{'Baseline (Pre-trained BERT)':<30} {baseline_f1:<15.4f} {baseline_em:<15.4f}")
print(f"{'Fine-tuned (Your Model)':<30} {best_f1:<15.4f} {best_em:<15.4f}")
print("-" * 60)
print(f"{'IMPROVEMENT':<30} {'+' + str(round((best_f1 - baseline_f1) / baseline_f1 * 100, 1)) + '%':<15} "
      f"{'+' + str(round((best_em - baseline_em) / baseline_em * 100, 1)) + '%':<15}")
print("=" * 70)

if best_f1 > baseline_f1:
    print(f"\n✅ SUCCESS: Your fine-tuned model BEATS the baseline!")
    print(f"   - F1 improved by {(best_f1 - baseline_f1) / baseline_f1 * 100:.1f}%")
    print(f"   - EM improved by {(best_em - baseline_em) / baseline_em * 100:.1f}%")
else:
    print(f"\n❌ Model did not beat baseline")

print(f"\nBest performance achieved at:")
print(f"  - Best F1: {best_f1:.4f} (Epoch {best_f1_epoch})")
print(f"  - Best EM: {best_em:.4f} (Epoch {best_em_epoch})")

# ============================================================================
# VISUALIZATION WITH BASELINE COMPARISON
# ============================================================================

print("\nCreating visualizations with baseline comparison...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('QA Model Performance vs Baseline BERT', fontsize=16, fontweight='bold')

# 1. F1 Score vs Baseline
axes[0, 0].axhline(y=baseline_f1, color='r', linestyle='--', linewidth=2, label=f'Baseline F1: {baseline_f1:.4f}')
axes[0, 0].plot(range(1, len(metrics['eval_f1']) + 1), metrics['eval_f1'], 'b-o', linewidth=2, markersize=6, label='Fine-tuned F1')
axes[0, 0].fill_between(range(1, len(metrics['eval_f1']) + 1), baseline_f1, metrics['eval_f1'], 
                         where=[f > baseline_f1 for f in metrics['eval_f1']], 
                         color='green', alpha=0.3, label='Improvement')
axes[0, 0].set_title('F1 Score vs Baseline', fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('F1 Score')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Exact Match vs Baseline
axes[0, 1].axhline(y=baseline_em, color='r', linestyle='--', linewidth=2, label=f'Baseline EM: {baseline_em:.4f}')
axes[0, 1].plot(range(1, len(metrics['eval_em']) + 1), metrics['eval_em'], 'g-s', linewidth=2, markersize=6, label='Fine-tuned EM')
axes[0, 1].fill_between(range(1, len(metrics['eval_em']) + 1), baseline_em, metrics['eval_em'],
                         where=[e > baseline_em for e in metrics['eval_em']],
                         color='green', alpha=0.3, label='Improvement')
axes[0, 1].set_title('Exact Match vs Baseline', fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Exact Match Score')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Percentage Improvement Over Baseline
axes[0, 2].plot(range(1, len(metrics['f1_improvement']) + 1), metrics['f1_improvement'], 
                'purple', marker='o', linewidth=2, markersize=6, label='F1 Improvement %')
axes[0, 2].plot(range(1, len(metrics['em_improvement']) + 1), metrics['em_improvement'], 
                'orange', marker='s', linewidth=2, markersize=6, label='EM Improvement %')
axes[0, 2].axhline(y=0, color='black', linestyle='-', linewidth=1)
axes[0, 2].fill_between(range(1, len(metrics['f1_improvement']) + 1), 0, metrics['f1_improvement'],
                         where=[i > 0 for i in metrics['f1_improvement']],
                         color='purple', alpha=0.2)
axes[0, 2].set_title('Percentage Improvement Over Baseline', fontweight='bold')
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('Improvement (%)')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# 4. Training Loss
axes[1, 0].plot(range(1, len(metrics['train_loss']) + 1), metrics['train_loss'], 'b-o', linewidth=2, markersize=6)
axes[1, 0].set_title('Training Loss', fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].grid(True, alpha=0.3)

# 5. Training Accuracy
axes[1, 1].plot(range(1, len(metrics['train_start_acc']) + 1), metrics['train_start_acc'], 
                'g-o', label='Start Position', linewidth=2, markersize=6)
axes[1, 1].plot(range(1, len(metrics['train_end_acc']) + 1), metrics['train_end_acc'], 
                'r-s', label='End Position', linewidth=2, markersize=6)
axes[1, 1].set_title('Training Accuracy', fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_ylim([0, 1])

# 6. Comparison Bar Chart
models = ['Baseline\n(Pre-trained)', 'Fine-tuned\n(Best)']
f1_scores = [baseline_f1, best_f1]
em_scores = [baseline_em, best_em]
x = np.arange(len(models))
width = 0.35

bars1 = axes[1, 2].bar(x - width/2, f1_scores, width, label='F1 Score', color='blue', alpha=0.7)
bars2 = axes[1, 2].bar(x + width/2, em_scores, width, label='Exact Match', color='green', alpha=0.7)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    axes[1, 2].annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

for bar in bars2:
    height = bar.get_height()
    axes[1, 2].annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

axes[1, 2].set_title('Final Model Comparison', fontweight='bold')
axes[1, 2].set_ylabel('Score')
axes[1, 2].set_xticks(x)
axes[1, 2].set_xticklabels(models)
axes[1, 2].legend()
axes[1, 2].set_ylim([0, max(max(f1_scores), max(em_scores)) * 1.2])

plt.tight_layout()
plt.savefig('model_vs_baseline_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: model_vs_baseline_comparison.png")

# ============================================================================
# SAVE COMPREHENSIVE RESULTS
# ============================================================================

final_results = {
    'config': config,
    'baseline_performance': {
        'f1': float(baseline_f1),
        'em': float(baseline_em)
    },
    'best_performance': {
        'f1': float(best_f1),
        'f1_epoch': int(best_f1_epoch),
        'em': float(best_em),
        'em_epoch': int(best_em_epoch)
    },
    'improvement_over_baseline': {
        'f1_improvement_percent': float((best_f1 - baseline_f1) / baseline_f1 * 100),
        'em_improvement_percent': float((best_em - baseline_em) / baseline_em * 100),
        'f1_absolute_improvement': float(best_f1 - baseline_f1),
        'em_absolute_improvement': float(best_em - baseline_em)
    },
    'beats_baseline': bool(best_f1 > baseline_f1),
    'epoch_wise_metrics': metrics,
    'training_summary': {
        'total_epochs': config['epochs'],
        'total_train_samples': len(train_features),
        'total_test_samples': len(test_features),
        'model_parameters': sum(p.numel() for p in model.parameters())
    }
}

with open('results_with_baseline.json', 'w') as f:
    json.dump(final_results, f, indent=2)

# Save detailed report
with open('baseline_comparison_report.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("BERT QA MODEL - BASELINE COMPARISON REPORT\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("1. BASELINE ESTABLISHMENT\n")
    f.write("-" * 40 + "\n")
    f.write("Baseline Model: Pre-trained BERT (bert-base-uncased)\n")
    f.write("Baseline Type: Zero-shot (no fine-tuning)\n")
    f.write(f"Baseline F1 Score: {baseline_f1:.4f}\n")
    f.write(f"Baseline Exact Match: {baseline_em:.4f}\n\n")
    
    f.write("2. FINE-TUNED MODEL CONFIGURATION\n")
    f.write("-" * 40 + "\n")
    f.write(f"Model: BERT-base-uncased (fine-tuned)\n")
    f.write(f"Epochs: {config['epochs']}\n")
    f.write(f"Batch Size: {config['batch_size']}\n")
    f.write(f"Learning Rate: {config['lr']}\n")
    f.write(f"Max Sequence Length: {config['max_len']}\n")
    f.write(f"Training Samples: {len(train_features)}\n")
    f.write(f"Test Samples: {len(test_features)}\n\n")
    
    f.write("3. PERFORMANCE COMPARISON\n")
    f.write("-" * 40 + "\n")
    f.write(f"{'Metric':<20} {'Baseline':<15} {'Fine-tuned':<15} {'Improvement':<20}\n")
    f.write("-" * 70 + "\n")
    f.write(f"{'F1 Score':<20} {baseline_f1:<15.4f} {best_f1:<15.4f} "
            f"{'+' + str(round((best_f1 - baseline_f1) / baseline_f1 * 100, 1)) + '%':<20}\n")
    f.write(f"{'Exact Match':<20} {baseline_em:<15.4f} {best_em:<15.4f} "
            f"{'+' + str(round((best_em - baseline_em) / baseline_em * 100, 1)) + '%':<20}\n")
    f.write("-" * 70 + "\n\n")
    
    f.write("4. CONCLUSION\n")
    f.write("-" * 40 + "\n")
    if best_f1 > baseline_f1:
        f.write("✅ SUCCESS: The fine-tuned model BEATS the baseline BERT!\n\n")
        f.write(f"The fine-tuned model shows significant improvement over the baseline:\n")
        f.write(f"- F1 Score improved by {(best_f1 - baseline_f1) / baseline_f1 * 100:.1f}%\n")
        f.write(f"- Exact Match improved by {(best_em - baseline_em) / baseline_em * 100:.1f}%\n\n")
        f.write("This demonstrates that fine-tuning on the specific Spoken-SQuAD dataset\n")
        f.write("provides substantial performance gains over using pre-trained BERT alone.\n")
    else:
        f.write("❌ The model did not beat the baseline.\n")
    
    f.write("\n5. EPOCH-BY-EPOCH PROGRESSION\n")
    f.write("-" * 40 + "\n")
    f.write("Epoch | F1 Score | vs Baseline | EM Score | vs Baseline | Status\n")
    f.write("-" * 70 + "\n")
    for i in range(len(metrics['eval_f1'])):
        status = "✓ Beating" if metrics['eval_f1'][i] > baseline_f1 else "✗ Below"
        f.write(f"{i+1:5d} | {metrics['eval_f1'][i]:8.4f} | "
                f"{metrics['f1_improvement'][i]:+10.1f}% | "
                f"{metrics['eval_em'][i]:8.4f} | "
                f"{metrics['em_improvement'][i]:+10.1f}% | {status}\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("END OF REPORT\n")
    f.write("=" * 80 + "\n")

print(f"\n📁 Files Generated:")
print(f"  - model_vs_baseline_comparison.png (visualization)")
print(f"  - baseline_comparison_report.txt (detailed report)")
print(f"  - results_with_baseline.json (all metrics)")
print(f"  - checkpoint_epoch_X.pt (model checkpoints)")

print(f"\n" + "=" * 70)
print("SUMMARY: BASELINE vs FINE-TUNED MODEL")
print("=" * 70)
print(f"Baseline F1: {baseline_f1:.4f} → Best F1: {best_f1:.4f} "
      f"({'+'}{(best_f1 - baseline_f1) / baseline_f1 * 100:.1f}%)")
print(f"Baseline EM: {baseline_em:.4f} → Best EM: {best_em:.4f} "
      f"({'+'}{(best_em - baseline_em) / baseline_em * 100:.1f}%)")

if best_f1 > baseline_f1:
    print(f"\n🎉 Congratulations! Your fine-tuned model successfully BEATS the baseline BERT!")
    #print(f"   This proves that fine-tuning provides significant performance gains.")
else:
    print(f"\n📊 The model needs more optimization to beat the baseline.")
