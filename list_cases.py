import json

with open('test_dataset.json') as f:
    data = json.load(f)

cases = data['test_cases']
for i, c in enumerate(cases[:12]):
    print(f"{i}: {c['id']} ({c['category']})")
