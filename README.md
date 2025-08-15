data preprocess using pyspark->mistral 7B instruct -> rlhf training with dataset from huggingface.<br>

<img width="400" height="500" alt="Image" src="https://github.com/user-attachments/assets/978e3e16-9e02-43f9-bc7c-90ff7ffc0026" />

calculating cos_similarity between (context,question) , (context, answer) and adding 2 sentences that is highly related to question and answer.

pyspark data process 
<img width="800" height="700" alt="image" src="https://github.com/user-attachments/assets/1f890280-ba59-4316-aa68-b26876db8644" />


### now ###
Now I'm working on building data load process with pyspark streaming mode. With this, can load nvidia/Nemotron-Post-Training-Dataset-v1[math]

### plan ###
1. vlm by adding vision adapters.
2. reasoning.

### error & run code ###
```python
nvidia math dataset example

 head -n 2 nvidia__Nemotron-Post-Training-Dataset-v1/spark_out/part-00000*.json | tail -n 1
{"question":"If there is a rectangular garden with an area of 250 square meters next to a building and fenced on three sides, with the building acting as the fourth fence, and if the side of the garden parallel to the building has length \\( x \\) meters, express the amount of fencing required as a function of \\( x \\).","expected_answer":"","output_cleaned":"<think> I have a problem about a rectangular garden next to a building. The garden has an area of 250 square meters and is fenced on three sides, with the building acting as the fourth side. The side parallel to the building is length \\( x \\) meters, and I need to express the amount of fencing required as a function of \\( x \\). First, visualize this. There's a rectangle with one side against a building, so we only need to fence the other three sides. The side parallel to the building is given as \\( x \\). Let me denote the other side as \\( y \\). So, the area is length times width, which is \\( x \\times y = 250 \\) square meters. Now, the fencing is only on three sides. Since the building is acting as one side, probably the side parallel to it doesn't need fencing, but we need to fence the two lengths and the one width opposite the building. I need to be careful. Assume that the building is along one of the lengths. Typically, in such problems, the building is along the width or the length, but here it specifies that the side parallel to the building is \\( x \\). So, let's say the building is along one side, and the side parallel to it is also of length \\( x \\), so that's the opposite side. Then, the two other sides are perpendicular to the building, each of length, say, \\( y \\). So, the fencing required would be for the two sides of length \\( y \\) and the one side of length \\( x \\) that's opposite the building. The side against the building doesn't need fencing. So, total fencing \\( F = x + 2y \\) meters. Now, I know that the area is \\( x \\times y = 250 \\). So, I can express \\( y \\) in terms of \\( x \\): \\( y = \\frac{250}{x} \\). Then, substitute into the fencing equation: \\( F = x + 2 \\times \\frac{250}{x} = x + \\frac{500}{x} \\). Is that it? It seems straightforward. But let me double-check. The problem says \"the side of the garden parallel to the building has length \\( x \\) meters.\" So, yes, \\( x \\) is the length parallel to the building, and since the building is one side, we don't fence that, but we do fence the opposite side, which is also length \\( x \\), and the two ends, each of length \\( y \\). So, fencing is \\( x + 2y \\), and with area \\( xy = 250 \\), so \\( F(x) = x + \\frac{500}{x} \\). I recall that sometimes people might confuse which sides are being fenced, but I think this is correct. Let me denote: let the length parallel to the building be \\( l = x \\), and the width perpendicular to the building be \\( w \\). Area \\( A = l \\times w = x w = 250 \\). Fencing: since building is along one length, we fence the other length (opposite) and the two widths. So yes, fencing \\( F = l + 2w = x + 2w \\). Since \\( w = \\frac{250}{x} \\), so \\( F = x + 2 \\times \\frac{250}{x} = x + \\frac{500}{x} \\). I think that's the function. The problem asks to express the amount of fencing required as a function of \\( x \\), and this seems to be it. Is there any constraint on \\( x \\)? Probably \\( x > 0 \\), but since it's a length, and area is positive, \\( x \\) should be positive. But for the function, we don't need to specify domain unless asked. So, the function is \\( F(x) = x + \\frac{500}{x} \\). I should write it neatly. Sometimes they write it as a single fraction, but it's not necessary. I can write it as \\( F(x) = \\frac{x^2 + 500}{x} \\), but that's more complicated. Better to leave it as \\( x + \\frac{500}{x} \\). Now, the answer should be in a box. So, I think that's it. But let me make sure about the sides. Suppose the building is along the width. But the problem says \"the side parallel to the building has length \\( x \\)\", so if the building is along a width, but typically, the side parallel to it would be the opposite width, but then the lengths would be perpendicular. But in that case, if building is along one width, say of length \\( w \\), then side parallel is the opposite width, also \\( w \\), so \\( x = w \\), and lengths are, say, \\( l \\), area \\( l \\times w = l x = 250 \\). Fencing: if building is along one width, then we need to fence the other width (opposite) and the two lengths. So fencing \\( F = w + 2l = x + 2l \\), but since \\( l = \\frac{250}{x} \\), same as before: \\( F = x + 2 \\times \\frac{250}{x} = x + \\frac{500}{x} \\). Same thing! Whether the building is along the length or the width, as long as \\( x \\) is the dimension parallel to the building, it works out the same. Because in both interpretations, the fencing includes the side opposite the building and the two adjacent sides. So, no issue. Therefore, the function is \\( F(x) = x + \\frac{500}{x} \\). I think that's the answer. </think>The garden is rectangular with an area of 250 square meters, and the building acts as one side, so fencing is required for the other three sides. The side parallel to the building is given as \\(x\\) meters. Let the length parallel to the building be \\(x\\) and the width perpendicular to the building be \\(y\\). The area is given by: \\[ x \\cdot y = 250 \\] Solving for \\(y\\): \\[ y = \\frac{250}{x} \\] The fencing is required for the side opposite the building (length \\(x\\)) and the two widths perpendicular to the building (each of length \\(y\\)). Thus, the total fencing required is: \\[ F = x + 2y \\] Substitute \\(y = \\frac{250}{x}\\): \\[ F = x + 2 \\left( \\frac{250}{x} \\right) = x + \\frac{500}{x} \\] Therefore, the amount of fencing required as a function of \\(x\\) is: \\[ F(x) = x + \\frac{500}{x} \\] \\boxed{F(x) = x + \\dfrac{500}{x}}"}
```
The way hadoop read a path and open(write_path, "wb") seems different.
This does not occur in ubuntu env.

pyspark dataset load & process code
```python
cd /workspace/sllm-mistralai-7B_training_using_pyspark-DeepSpeed/pyspark_data_load

python collect_data.py --save_path /workspace --file_name nvidia --dataset_name nvidia/Nemotron-Post-Training-Dataset-v1 --sample_ratio 0.1
```

java install before running pyspark
```python
java -version

apt-get update
apt-get install -y openjdk-17-jdk


# 설치된 Java 17 경로 찾기
update-alternatives --config java

# 예를 들어 /usr/lib/jvm/java-17-openjdk-amd64 라면
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH


echo "export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64" >> ~/.bashrc
echo "export PATH=\$JAVA_HOME/bin:\$PATH" >> ~/.bashrc

source ~/.bashrc
```

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

