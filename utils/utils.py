def inspect_tokenized_dataset(dataset, num_samples=3):
    print(f"\n🔍 Inspecting first {num_samples} samples in tokenized dataset...")

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        print(f"\n=== Sample {i} ===")
        print("input_ids:", sample.get("input_ids"))
        print("attention_mask:", sample.get("attention_mask"))
        print("labels:", sample.get("labels"))

        # 타입 확인
        if not isinstance(sample.get("labels"), list):
            print(f"❌ Sample {i}: labels가 list가 아님 → {type(sample.get('labels'))}")
        elif not all(isinstance(x, int) for x in sample["labels"]):
            print(f"❌ Sample {i}: labels 내부에 int가 아닌 값 존재")
        else:
            print(f"✅ Sample {i}: labels 구조 정상")

    # 길이 비교
    input_lens = [len(s["input_ids"]) for s in dataset[:num_samples]]
    label_lens = [len(s["labels"]) for s in dataset[:num_samples]]

    print("\n📏 input_ids 길이:", input_lens)
    print("📏 labels 길이:   ", label_lens)

def print_label_lengths(dataset):
    lengths = [len(sample["labels"]) for sample in dataset]
    print(f"[디버깅] labels 길이 - min: {min(lengths)}, max: {max(lengths)}, mean: {sum(lengths) / len(lengths):.2f}")
    # 예시: 10개 샘플 실제 길이 직접 확인
    print("[디버깅] 샘플별 labels 길이 (상위 10개):", lengths[:10])

def print_field_lengths(dataset, n=10, stage=""):
    """
    데이터셋에서 input_ids, attention_mask, labels 길이 분포 및 샘플 표시 (상위 n개)
    """
    print(f"\n[디버깅][{stage}] 길이 통계 ================================")
    for key in ["input_ids", "attention_mask", "labels"]:
        try:
            lengths = [len(x[key]) for x in dataset]
            print(f"{key} → min: {min(lengths)}, max: {max(lengths)}, mean: {sum(lengths)/len(lengths):.2f}")
            print(f"{key} 샘플 (상위 {n}개):", lengths[:n])
        except Exception as e:
            print(f"{key}: (존재하지 않거나 에러) {e}")
    print("====================================================\n")
