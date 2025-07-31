mistral 7B instruct -> rlhf training with dataset from huggingface.<br>

<img width="1024" height="1536" alt="Image" src="https://github.com/user-attachments/assets/978e3e16-9e02-43f9-bc7c-90ff7ffc0026" />

calculating cos_similarity between (context,question) , (context, answer) and adding 2 sentences that is highly related to question and answer.

1)This pipeline does not include parallel processing(e.g doing for to multiple answers).(yet)<br>
2)Considering using youtube data(under creative commons license) as training dataset if it's legal.<br>
3)Didn't apply optimization in this repo such as using gpt to generate answer pairs/judge such answer's reward score/ etc.<br>
4) paper used:

### Not yet done ###
1) pyspark 
2) deepspeed opt
3) prompt opt such as "Black-Box Prompt Optimization: Aligning Large Language Models without Model Training"

## 📜 Licenses

This project includes components from:

1. **This project's Custom Code**
   - Author: genji970
   - License: CC BY-NC-ND License

2. **Mistral-7B-Instruct-v0.3**
   - Source: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
   - License: Apache-2.0

3. **squad**
   - Source: https://huggingface.co/datasets/rajpurkar/squad
   - License: CC BY‑SA 4.0
   - citation : @inproceedings{rajpurkar-etal-2016-squad,
    title = "{SQ}u{AD}: 100,000+ Questions for Machine Comprehension of Text",
    author = "Rajpurkar, Pranav  and
      Zhang, Jian  and
      Lopyrev, Konstantin  and
      Liang, Percy",
    editor = "Su, Jian  and
      Duh, Kevin  and
      Carreras, Xavier",
    booktitle = "Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2016",
    address = "Austin, Texas",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D16-1264",
    doi = "10.18653/v1/D16-1264",
    pages = "2383--2392",
    eprint={1606.05250},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
}

   
     

