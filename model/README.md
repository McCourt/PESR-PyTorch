When use models, please try
```python
from model import Model
model = Model(name='model_name', 
              mode=mode, 
              checkpoint='checkpoint_dir', 
              train=True, 
              map_location=None, 
              **kwargs)
```
**Please Note:**
* Make sure that the model name and class name must correpond in **models.json** in format of
```json
{
  "upscaler": {
    "pyramid_attention": "PyramidAttentionSR",
    "pyramid_edsr": "PyramidEDSR"
  },
  "downscaler": {
    "bicubic": "BicubicDownSample",
    "encode_decode": "DeepDownScale"
  },
  "discriminator": {
    "vgg": "Discriminator_VGG_128"
  }
}

```
* Parameter **mode** must be either *'upscaler'*, *'downscaler'* or *'discriminator'*
* **kwargs** are placeholders for the parameters passed into the model defined in customized script
