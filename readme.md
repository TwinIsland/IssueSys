### Step 0: 系统要求

1. 内存：32G
2. 显存：8G

### Step 1: 安装依赖

```shell
pipenv --python 3
pipenv install
```

### Step 2: 安装模型

从`huggingface`下载`text2vec-base-multilingual`模型，放入对应文件夹，清华云盘+`huggingface`下载`ChatGLM2`放入对应文件夹

### Step 3: 配置

文件夹下`config.ini`为配置文件，默认可以不用改

```ini
[model]
embedding_path = text2vec-base-multilingual
llm_path = chatglm2-6b
min_sup = 0.8
max_item = 5

[embedding]
chunk_size = 250
sentence_size = 100

[vector_store]
address = ./cache
```

### Step 4: Run

`Jupyter`打开`test.ipynb`，先运行前两个`block`加载模型以及`embedding`类，然后就不用反复运行了。加载完成后运行后两个`block`测试模型。

### 其他

**调整模板**

```markdown
我的问题是：{1}，匹配到的条目为：{2}。请输出有关联条目的序号，只输出数字，并用逗号分割。请判断与我的问题匹配的条目，输出“好的”两个字"，若无匹配，输出”无匹配“
%
总结所有匹配到的条目并成一句话，只生成一句话
```

用`%`分割每次轮查的prompt

**数据**

数据文件为文件夹下`data.bin`，里面有大约2k条脑经急转弯数据（源数据在items文件里）。数据文件用python的pickle库打包，数据类型为：

```python
with open("data.bin", "wb") as f:
    pickle.dump((q, a, {}), f)
```

其中`q, a`是`question, answer`的`List[str]`

**多显卡**

```python
# https://github.com/THUDM/ChatGLM2-6B/blob/main/cli_demo.py
tokenizer = AutoTokenizer.from_pretrained("xxxxxx", trust_remote_code=True)
model = AutoModel.from_pretrained("xxxxxxx", trust_remote_code=True).cuda()
# 多显卡支持，使用下面两行代替上面一行，将num_gpus改为你实际的显卡数量
# from utils import load_model_on_gpus
# model = load_model_on_gpus("xxxxxx", num_gpus=2)
```

**使用缓存**

程序在跑embedding后会缓存储存知识库为向量库从而加速运行效率。如果发现文件夹下面有`cache`文件，即说明以生成缓存，这个时候就能把程序中`rm_cache()`备注掉以便让程序复用缓存。

```python
embedder.rm_cache()  # remove cache if you need, comment it out once embedding data for the first time
```

### 总结

目前这一套流程使用轮查来提高llm回答的准确度，但效果还是不怎么好，具体表现在FP过高，后期可以通过修改模板尝试解决这个问题。
