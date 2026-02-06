import pyarrow.parquet as pq
import json
import random

table = pq.read_table("dataset/advbench.parquet")
data = table.to_pylist()

with open("advbench.jsonl", "w") as f:
    for row in data:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

# Sample 50 entries to testdata.json
sample = random.sample(data, min(50, len(data)))
with open("testdata.json", "w") as f:
    json.dump(sample, f, ensure_ascii=False, indent=2)



def to_chat_prompt(messages: List[Dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )