import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import pandas as pd
import json
from transformers import BertTokenizer, BertForSequenceClassification


# 加载数据
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        data = [json.loads(line) for line in lines]
    return pd.DataFrame(data)


# 创建自定义数据集

class MathQuestionDataset(Dataset):
    def __init__(self, questions, labels, tokenizer, max_len):
        self.questions = questions
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, item):
        question = str(self.questions[item])
        label = int(self.labels[item])  # 确保标签是整数类型
        encoding = self.tokenizer.encode_plus(
            question,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,  # 考虑替换为 `padding='max_length'`
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True  # 明确启用截断
        )
        return {
            'question_text': question,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# 检测是否有可用的 GPU，如果没有，则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置
MAX_LEN = 512
BATCH_SIZE = 16
EPOCHS = 3
RANDOM_SEED = 6

# 设置随机种子以确保可复现性
torch.manual_seed(RANDOM_SEED)


# 加载并准备数据
df = load_data('math.jsonl')
df_train, df_val = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED)

# 初始化Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')


# 创建数据加载器
def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = MathQuestionDataset(
        questions=df.question.to_numpy(),
        labels=df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=4)


train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)

# 加载BERT模型
model = BertForSequenceClassification.from_pretrained('bert-large-uncased', num_labels=2)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# 训练
# 使用 PyTorch 的 AdamW，不需要 correct_bias 参数
optimizer = AdamW(model.parameters(), lr=2e-5)


def evaluate(model, data_loader):
    model.eval()
    eval_loop = tqdm(data_loader, leave=True)
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for batch in eval_loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            _, predictions = torch.max(outputs.logits, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_true_labels, all_predictions)
    report = classification_report(all_true_labels, all_predictions, target_names=['Non-Math', 'Math'])
    return accuracy, report


# 训练和评估过程
for epoch in range(EPOCHS):
    model.train()
    train_loop = tqdm(train_data_loader, leave=True)
    for batch in train_loop:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        train_loop.set_description(f'Epoch {epoch}')
        train_loop.set_postfix(loss=loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # 在每个epoch结束后进行评估
    accuracy, report = evaluate(model, val_data_loader)
    print(f"Epoch {epoch} - Evaluation Accuracy: {accuracy:.4f}")
    print(report)

# 保存模型
model.save_pretrained('bert_large_maths')
tokenizer.save_pretrained('bert_large_maths')