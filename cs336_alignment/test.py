from pathlib import Path
import json

data_path =  Path("data/gsm8k/test.jsonl")
examples = []
with data_path.open("r", encoding = 'utf-8') as f:
    for line in f:
        example = json.loads(line)
        question = example['question']
        answer = example['answer']

        if '####' not in answer:
            raise ValueError("参考答案中缺少 #### 分隔符")
        ground_truth = answer.split('####', 1)[1].strip()
        examples.append({'question':question,'ground_truth':ground_truth})
print("题目数量：", len(examples))
print("第一题：", examples[0]["question"])
print("标准答案：", examples[0]["ground_truth"])