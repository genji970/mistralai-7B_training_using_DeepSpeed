data preprocess using pyspark->mistral 7B instruct -> rlhf training with dataset from huggingface.<br>

<img width="400" height="500" alt="Image" src="https://github.com/user-attachments/assets/978e3e16-9e02-43f9-bc7c-90ff7ffc0026" />

calculating cos_similarity between (context,question) , (context, answer) and adding 2 sentences that is highly related to question and answer.

### now ###
Now I'm working on building data load process with pyspark streaming mode. With this, can load nvidia/Nemotron-Post-Training-Dataset-v1[math]

### plan ###
1. vlm by adding vision adapters.
2. reasoning.

### error ###
```python
[nltk_data] Downloading package punkt to
[nltk_data]     C:\Users\qnckd\AppData\Roaming\nltk_data...
[nltk_data]   Package punkt is already up-to-date!
[nltk_data] Downloading package averaged_perceptron_tagger to
[nltk_data]     C:\Users\qnckd\AppData\Roaming\nltk_data...
[nltk_data]   Package averaged_perceptron_tagger is already up-to-
[nltk_data]       date!
[nltk_data] Downloading package stopwords to
[nltk_data]     C:\Users\qnckd\AppData\Roaming\nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
✅ 저장 경로: D:/
✅ 데이터셋 이름: nvidia/Nemotron-Post-Training-Dataset-v1
✅ 이미 폴더 존재: D:/nvidia__Nemotron-Post-Training-Dataset-v1
Resolving data files: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 183/183 [00:00<00:00, 247.11it/s]
Resolving data files: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 159/159 [00:00<00:00, 257.73it/s]
Resolving data files: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 660/660 [00:00<00:00, 1139.57it/s]
Resolving data files: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 183/183 [00:00<00:00, 305.04it/s]
Resolving data files: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 159/159 [00:00<00:00, 644.47it/s]
Resolving data files: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 660/660 [00:00<00:00, 2222.10it/s]
✅ 원본 jsonl 파일 저장 시작: nvidia__Nemotron-Post-Training-Dataset-v1.jsonl
✅ Streaming 저장 완료 (소요: 1502.38초)
샘플: {"uuid":"fb215f9b-9372-4a08-875e-94184764ad51","license":"CC BY 4.0","generator":"DeepSeek-R1-0528","version":"v1","category":"math","reasoning":"on","messages":[{"role":"user","content":"In a regular
지정된 경로를 찾을 수 없습니다.
```
The way hadoop read a path and open(write_path, "wb") seems different.


## 📜 Licenses

This project includes components from:

1. **This project's Custom Code**
   - Author: genji970
   - License: CC BY-NC-ND License

2. **Mistral-7B-Instruct-v0.3**
   - Source: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
   - License: Apache-2.0
  
3. **nvidia/Nemotron-Post-Training-Dataset-v1**
   - Source: [https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1)
   - License: @misc{bercovich2025llamanemotronefficientreasoningmodels,
      title={Llama-Nemotron: Efficient Reasoning Models}, 
      author={Akhiad Bercovich ...},
      year={2025},
      eprint={2505.00949},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.00949}, 
}
