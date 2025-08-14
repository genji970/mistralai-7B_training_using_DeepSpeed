from datasets import load_dataset
ds = load_dataset("nvidia/Nemotron-Post-Training-Dataset-v1", split="math", streaming=True)
for i, ex in enumerate(ds):
    print(ex)
    if i > 2:
        break



# head -n 2 nvidia__Nemotron-Post-Training-Dataset-v1/spark_out/part-00000*.json | tail -n 1


"""
{
  "question": "문제 텍스트",
  "context": "문제 부가 설명 (없으면 빈 문자열)",
  "output_cleaned": "<think> ... reasoning ... </think> 실제 답변",
  "messages": [
    ["질문", "user", []],
    ["답변", "assistant", []]
  ]
}
"""