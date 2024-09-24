from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from transformers import AutoTokenizer, AutoModel
import torch
import json
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm  # 用于显示进度条

# Step 1: 连接 Milvus 数据库
connections.connect("default", host="localhost", port="19530")


# Step 2: 定义 Collection Schema (向量表)
def create_collection():
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000)  # max_length 限制
    ]
    schema = CollectionSchema(fields, "CUAD_法律文本嵌入")
    collection = Collection(name="cuad_legal_texts", schema=schema)

    # 创建索引以加速查询
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 100}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    return collection


collection = create_collection()

# Step 3: 加载预训练模型和分词器
model_name = "D:/pycharm-code/model/all-mpnet-base-v2"  # 确保路径正确
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 将模型移动到 GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# 定义数据集类
class CuadDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


# Step 4: 加载 CUAD 数据集并处理过长的文本
def load_cuad_data(max_length=2000):
    with open('D:/pycharm-code/法律文本查询/CUAD/CUADv1.json', 'r', encoding='utf-8') as f:
        cuad_data = json.load(f)
    texts = []

    # 使用 tqdm 进度条跟踪加载过程
    for document in tqdm(cuad_data['data'], desc="Loading CUAD data"):
        title = document['title']
        for paragraph in document['paragraphs']:
            paragraph_text = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                # 组合标题、段落和问题作为文本输入
                full_text = f"Title: {title}\nParagraph: {paragraph_text}\nQuestion: {question}"

                # 分割过长的文本为多个部分
                for i in range(0, len(full_text), max_length):
                    text_entry = full_text[i:i + max_length]  # 每个 text_entry 不超过 max_length
                    texts.append(text_entry)

    return texts


# Step 5: 向量化 CUAD 数据中的文本
def embed_texts(texts, batch_size=128):  # 增大批处理大小
    all_embeddings = []
    dataset = CuadDataset(texts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 使用 tqdm 进度条显示处理进度
    for batch_texts in tqdm(dataloader, desc="Embedding texts"):
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            embeddings = model(**inputs).last_hidden_state
        embeddings = torch.mean(embeddings, dim=1)
        all_embeddings.append(embeddings.cpu().numpy())  # 将结果移回 CPU

    return np.vstack(all_embeddings)


# Step 6: 插入向量到 Milvus 中
def insert_data(texts):
    embeddings = embed_texts(texts)
    ids = list(range(1, len(texts) + 1))
    entities = [ids, embeddings.tolist(), texts]

    # 插入数据并使用 tqdm 进度条显示插入进度
    for i in tqdm(range(0, len(ids), 100), desc="Inserting data into Milvus"):  # 分批插入
        batch_ids = ids[i:i + 100]
        batch_embeddings = embeddings[i:i + 100]
        batch_texts = texts[i:i + 100]
        entities = [batch_ids, batch_embeddings.tolist(), batch_texts]
        collection.insert(entities)

    collection.load()


# 加载 CUAD 数据并插入
legal_texts = load_cuad_data()
insert_data(legal_texts)


# Step 7: 查询功能
def search(query, top_k=5):
    query_embedding = embed_texts([query])
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

    results = collection.search(
        data=query_embedding.tolist(),
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["text"]
    )

    return [result.entity.get('text') for result in results[0]]


# Step 8: 评估精度
def evaluate(query, reference_texts, top_k=5):
    search_results = search(query, top_k)
    correct = 0
    for result in search_results:
        if result in reference_texts:
            correct += 1

    precision = correct / min(top_k, len(search_results))  # 确保不超过结果数量
    print(f"查询: {query}")
    print(f"准确率: {precision:.2f}")
    print(f"查询到的文本: {search_results}")
    print(f"参考答案: {reference_texts}")


# 示例查询及参考答案
query = "contract clauses related to compensation"
reference_texts = [
    "Compensation clause content 1...",
    "Compensation clause content 2...",
    "Compensation clause content 3..."
]

# 执行评估
evaluate(query, reference_texts)
