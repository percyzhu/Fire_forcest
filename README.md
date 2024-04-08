**Codes for reproduction of results in paper "Machine Learning-Driven Real-time Identification of Large space Building Fires and Forecast of Temperature Development"**

**1. pretrain.py**
pretrain the model using classical fire dataset
save the best model as best_model_pretrain.h5
save the training history as pretrain.svg

**2. transfer learning.py**
tranfer the best_model_pretrain to be trained on FDS dataset
save the best model as best_model_transfer
save the training history as transfer learning.svg

**3. analysis.py**
evaluate the model performance using the test subset in FDS dataset

**4. test1.py**
evaluate the model performance using fire test 1

**5. test2.py**
locate and repair damaged thermocouples in fire test 2
evaluate the model performance using fire test 2

Note that the dataset for classic fire models is too large for GitHub and can be found in the following Google Drive link:
https://drive.google.com/file/d/1vhYTlVkAu-98n8NTZwoOSrGbPcflBGre/view?usp=drive_link
