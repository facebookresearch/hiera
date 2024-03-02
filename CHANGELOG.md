### **[2024.03.02]** v0.1.4
 - License made more permissive! The license for the code has been changed to Apache 2.0. The license for the model remains as CC BY-NC 4.0, though (nothing we can do about that).

### **[2024.03.01]** v0.1.3
 - Added support to save and load models to the huggingface hub, if huggingface_hub is installed.
 - Most Hiera models have been uploaded to HuggingFace.

### **[2023.07.20]** v0.1.2
 - Released the full model zoo.
 - Added MAE functionality to the video models.

### **[2023.06.12]** v0.1.1
 - Added the ability to specify multiple pretrained checkpoints per architecture (specify with `checkpoint=<ckpt_name>`).
 - Added the ability to pass `strict=False` to a pretrained model so that you can use a different number of classes. **Note:** when changing the number of classes, the head layer will be reset.
 - Released all in1k finetuned models.

### **[2023.06.01]** v0.1.0
 - Initial Release.
