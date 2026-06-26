<!-- Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# GigaworldTransformer3DModel

A 14B Real-Time Autogressive Diffusion Transformer model (support T2V, I2V and V2V) for 3D video-like data from [Gigaworld](https://github.com/PKU-YuanGroup/Gigaworld) was introduced in [Gigaworld: Real Real-Time Long Video Generation Model](https://huggingface.co/papers/2603.04379) by Peking University & ByteDance & etc.

The model can be loaded with the following code snippet.

```python
from diffusers import GigaworldTransformer3DModel

# Best Quality
transformer = GigaworldTransformer3DModel.from_pretrained("BestWishYsh/Gigaworld-Base", subfolder="transformer", torch_dtype=torch.bfloat16)
# Intermediate Weight
transformer = GigaworldTransformer3DModel.from_pretrained("BestWishYsh/Gigaworld-Mid", subfolder="transformer", torch_dtype=torch.bfloat16)
# Best Efficiency
transformer = GigaworldTransformer3DModel.from_pretrained("BestWishYsh/Gigaworld-Distilled", subfolder="transformer", torch_dtype=torch.bfloat16)
```

## GigaworldTransformer3DModel

[[autodoc]] GigaworldTransformer3DModel

## Transformer2DModelOutput

[[autodoc]] models.modeling_outputs.Transformer2DModelOutput
