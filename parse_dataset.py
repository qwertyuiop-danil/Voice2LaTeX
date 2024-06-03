import json

file = 'data.jsonl'

with open(file, 'r', encoding='utf-8') as f:
    dataset = f.readlines()
unique_lines = sorted(set(dataset), key=dataset.index)
with open(file, 'w', encoding='utf-8') as f:
    f.writelines(unique_lines)

dataset = [json.loads(line) for line in open(file, 'r', encoding='utf-8').readlines()]

with open('dataset.txt', 'w', encoding='utf-8') as f:
    for data in dataset:
        f.write(f'Формула словами: {data["input"]}\nФормула специальными буквами: <s>{data["output"]}</s>\n\n\n')
