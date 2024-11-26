from model.tfvsn import TrainingFreeVideoSummarizationNetwork


model_path = "Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5"
tf_model = TrainingFreeVideoSummarizationNetwork(model_path)
print("HELLO WORLD!")
