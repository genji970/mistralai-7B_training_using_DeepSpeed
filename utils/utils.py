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
