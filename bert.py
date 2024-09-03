import nltk
from tqdm import tqdm
#nltk.download('senseval')
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
from nltk.corpus import senseval

# Load instances for the word 'serve'
instances = senseval.instances('serve.pos')

# Print the number of phrases
num_phrases = len(instances)
print(f"Number of phrases: {num_phrases}")

for i in range(10):
    inst = instances[i]
    context = " ".join([word for word, pos in inst.context])
    senses = inst.senses
    print(f"Phrase {i+1}: {context}")
    print(f"Sense(s): {senses}")

# Create a DataFrame
data = []
for i, inst in tqdm(enumerate(instances)):
    inst_context = list(filter(lambda x: len(x) == 2, inst.context))
    context = " ".join([word for word, pos in inst_context])
    label = inst.senses[0] if inst.senses else None
    data.append({'text': context, 'label': label, 'id': i})

df = pd.DataFrame(data)
print(df.head())

from sklearn.model_selection import train_test_split
df = df.sample(frac=1).reset_index(drop=True)
# Encode labels as integers
label_to_id = {label: idx for idx, label in enumerate(df['label'].unique())}
df['label'] = df['label'].apply(lambda x: label_to_id[x])


train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
def tokenize_texts(texts):
    return tokenizer.batch_encode_plus(
        texts,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
        return_attention_mask=True
    )


train_texts = train_df['text'].tolist()
test_texts = test_df['text'].tolist()

train_encodings = tokenize_texts(train_texts)
test_encodings = tokenize_texts(test_texts)
train_labels = torch.tensor(train_df['label'].tolist())
test_labels = torch.tensor(test_df['label'].tolist())


from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], test_labels)
train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=32)
test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=32)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_to_id), output_attentions=False, output_hidden_states=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)


# Training function
def train(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)
        model.zero_grad()
        outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    avg_loss = total_loss / len(dataloader)
    return avg_loss

# Evaluation function
def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            total_loss += loss.item()
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).flatten()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(b_labels.cpu().numpy())
    avg_loss = total_loss / len(dataloader)
    return avg_loss, all_preds, all_labels

# Train and evaluate the model
epochs = 3
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    train_loss = train(model, train_dataloader, optimizer)
    print(f"Training loss: {train_loss}")
    test_loss, test_preds, test_labels = evaluate(model, test_dataloader)
    print(f"Validation loss: {test_loss}")


accuracy = accuracy_score(test_labels, test_preds)
precision, recall, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")

# Plot confusion matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix


def plot_confusions(y_true, predicted, title="", labels=None, normalize=None):
    # Change figure size and increase dpi for better resolution
    fig, ax = plt.subplots(figsize=(8,6), dpi=100)
    conf_matrix = confusion_matrix(y_true, predicted, normalize=normalize)
    display = ConfusionMatrixDisplay(conf_matrix, display_labels=labels)
    # Set the plot title
    ax.set(title=f'Confusion Matrix {title}')

    display.plot(ax=ax)
    return conf_matrix

plot_confusions(test_labels, test_preds)