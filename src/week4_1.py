import random
import evaluate
import numpy as np

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

imdb = load_dataset("nyu-mll/glue", "mnli")
print(imdb)

train_dataset = imdb['train']
print(train_dataset[0])

# [MYOCDE] 라벨의 종류 확인
label_info = train_dataset.features['label']
print(f"라벨의 종류: {label_info.names}")

# [MYCODE] 라벨의 개수 설정
num_labels = len(label_info.names)

# [MYCODE] pre traiend된 distilbert 토큰나이저를 가져옴.
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

# [MYCODE] 가설과 전제를 SEP로 결합하고 max_length, padding을 설정, label 값을 설정.
def preprocess_function(data):
  max_length = 400
  texts = [premise + " [SEP] " + hypothesis for premise, hypothesis in zip(data['premise'], data['hypothesis'])]
  tokenized_output = tokenizer(texts, truncation=True, max_length=max_length, padding=True)

  tokenized_output['labels'] = data['label']
  return tokenized_output

imdb_tokenized = imdb.map(preprocess_function, batched=True)
print(imdb_tokenized['train'][0].keys())

# [MYCODE] Train 데이터와 Validation 데이터의 개수 제한을 만개로 설정.
data_length = 30000
imdb_split = imdb_tokenized['train'].train_test_split(test_size=0.2)
imdb_train, imdb_val = imdb_split['train'].select(range(data_length)), imdb_split['test'].select(range(data_length))
imdb_test = imdb_tokenized['test_matched']
print(f"train len: {len(imdb_train)}, validation len: {len(imdb_val)}, test len: {len(imdb_test)}")

from transformers import BertConfig

# [MYCODE] BertConfig를 설정하고 num_labels의 값을 변경.
config = BertConfig()
config.dropout = 0.1
config.hidden_size = 64
config.intermediate_size = 64
config.num_hidden_layers = 2
config.num_attention_heads = 4
config.num_labels = num_labels

from transformers import TrainingArguments, Trainer

num_epochs = 60
training_args = TrainingArguments(
    run_name='pratice_week4_1',
    output_dir='mnli_transformer',
    num_train_epochs=num_epochs,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    logging_strategy="epoch",
    do_train=True,
    do_eval=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    load_best_model_at_end=True
)

import evaluate
accuracy = evaluate.load("accuracy")

def compute_metrics(pred):
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

from transformers import EarlyStoppingCallback

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased", num_labels=num_labels
)
print(model)

for param in model.distilbert.parameters():
  param.requires_grad = False

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=imdb_train,
    eval_dataset=imdb_val,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

trainer.train()
# trainer.save_model()

# trainer.predict(imdb_test)