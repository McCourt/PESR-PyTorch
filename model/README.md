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
* Make sure that the model name and file name correponds well in **models.json** in format of
```json
"mode": {
  "model_name_1": "file_name_1",
  "model_name_2": "file_name_2",
  "...": "..."
}
```
* Parameter **mode** must be either *'upscaler'*, *'downscaler'* or *'discriminator'*
* **kwargs** are placeholders for the parameters passed into the model defined in customized script
