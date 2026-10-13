---
pipeline_tag: image-text-to-text
language:
- multilingual
tags:
- got
- vision-language
- ocr2.0
- custom_code
license: apache-2.0
---

<h1>General OCR Theory: Towards OCR-2.0 via a Unified End-to-end Model
</h1>

[🔋Online Demo](https://huggingface.co/spaces/ucaslcl/GOT_online) | [🌟GitHub](https://github.com/Ucas-HaoranWei/GOT-OCR2.0/) | [📜Paper](https://arxiv.org/abs/2409.01704)</a> 


[Haoran Wei*](https://scholar.google.com/citations?user=J4naK0MAAAAJ&hl=en), Chenglong Liu*, Jinyue Chen, Jia Wang, Lingyu Kong, Yanming Xu,  [Zheng Ge](https://joker316701882.github.io/), Liang Zhao, [Jianjian Sun](https://scholar.google.com/citations?user=MVZrGkYAAAAJ&hl=en), [Yuang Peng](https://scholar.google.com.hk/citations?user=J0ko04IAAAAJ&hl=zh-CN&oi=ao), Chunrui Han, [Xiangyu Zhang](https://scholar.google.com/citations?user=yuB-cfoAAAAJ&hl=en)



![image/jpeg](https://cdn-uploads.huggingface.co/production/uploads/6653eee7a2d7a882a805ab95/QCEFY-M_YG3Bp5fn1GQ8X.jpeg)



## Usage
Inference using Huggingface transformers on NVIDIA GPUs. Requirements tested on python 3.10：
```
torch==2.0.1
torchvision==0.15.2
transformers==4.37.2
tiktoken==0.6.0
verovio==4.3.1
accelerate==0.28.0
```


```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
model = AutoModel.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=tokenizer.eos_token_id)
model = model.eval().cuda()


# input your test image
image_file = 'xxx.jpg'

# plain texts OCR
res = model.chat(tokenizer, image_file, ocr_type='ocr')

# format texts OCR:
# res = model.chat(tokenizer, image_file, ocr_type='format')

# fine-grained OCR:
# res = model.chat(tokenizer, image_file, ocr_type='ocr', ocr_box='')
# res = model.chat(tokenizer, image_file, ocr_type='format', ocr_box='')
# res = model.chat(tokenizer, image_file, ocr_type='ocr', ocr_color='')
# res = model.chat(tokenizer, image_file, ocr_type='format', ocr_color='')

# multi-crop OCR:
# res = model.chat_crop(tokenizer, image_file, ocr_type='ocr')
# res = model.chat_crop(tokenizer, image_file, ocr_type='format')

# render the formatted OCR results:
# res = model.chat(tokenizer, image_file, ocr_type='format', render=True, save_render_file = './demo.html')

print(res)


```
More details about 'ocr_type', 'ocr_box', 'ocr_color', and 'render' can be found at our GitHub.
Our training codes are available at our [GitHub](https://github.com/Ucas-HaoranWei/GOT-OCR2.0/).



## More Multimodal Projects

👏 Welcome to explore more multimodal projects of our team:

[Vary](https://github.com/Ucas-HaoranWei/Vary) | [Fox](https://github.com/ucaslcl/Fox) | [OneChart](https://github.com/LingyvKong/OneChart)

## Citation

If you find our work helpful, please consider citing our papers 📝 and liking this project ❤️！

```bib
@article{wei2024general,
  title={General OCR Theory: Towards OCR-2.0 via a Unified End-to-end Model},
  author={Wei, Haoran and Liu, Chenglong and Chen, Jinyue and Wang, Jia and Kong, Lingyu and Xu, Yanming and Ge, Zheng and Zhao, Liang and Sun, Jianjian and Peng, Yuang and others},
  journal={arXiv preprint arXiv:2409.01704},
  year={2024}
}
@article{liu2024focus,
  title={Focus Anywhere for Fine-grained Multi-page Document Understanding},
  author={Liu, Chenglong and Wei, Haoran and Chen, Jinyue and Kong, Lingyu and Ge, Zheng and Zhu, Zining and Zhao, Liang and Sun, Jianjian and Han, Chunrui and Zhang, Xiangyu},
  journal={arXiv preprint arXiv:2405.14295},
  year={2024}
}
@article{wei2023vary,
  title={Vary: Scaling up the Vision Vocabulary for Large Vision-Language Models},
  author={Wei, Haoran and Kong, Lingyu and Chen, Jinyue and Zhao, Liang and Ge, Zheng and Yang, Jinrong and Sun, Jianjian and Han, Chunrui and Zhang, Xiangyu},
  journal={arXiv preprint arXiv:2312.06109},
  year={2023}
}
```