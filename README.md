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
