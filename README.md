# 1. Metric Name
Attack Success Rate

# 2. Trustworthiness aspect
Category -> Security / Robustness\
Description -> ASR measures how easily an AI model’s predictions can be changed by adversarial perturbations — in other words, how vulnerable or robust the model is to attacks.

This metric evaluates the robustness of AI models against adversarial perturbations by quantifying the proportion of successful attacks that cause misclassification

# 3. Class of models it applies to
Attack success rate applies broadly, but for practical reasons, you can focus on:\

Neural network models (e.g., PyTorch- or TensorFlow-based classifiers)

---- spec ----\
Deep learning models for tabular, image, or text data.\
Models that expose a .predict() or .predict_proba() function.

# 4. Working assumptions
- The model outputs a class prediction (classification task).
- The inputs can be numerically perturbed (so no categorical-only datasets).
- You have access to the model’s gradients or can simulate perturbations (e.g., adding small Gaussian noise if gradients are unavailable).
- The attack method must be consistent (e.g., same perturbation size for all samples).

# 5. Datasets to test it

- Tabular
- MNIST
- CIFAR

I think that for the project we will only use the data already present in the a4s framework.

# 6. References
- Performance Evaluation of Adversarial Attacks: Discrepancies and Solutions
-> https://arxiv.org/pdf/2104.11103

- Combining Attack Success Rate and Detection Rate for effective Universal Adversarial Attacks
-> https://www.esann.org/sites/default/files/proceedings/2021/ES2021-160.pdf

- How to Estimate the Success Rate of Higher-Order Side-Channel Attacks
-> https://cyber.gouv.fr/sites/default/files/IMG/pdf/How_to_Estimate_the_Success_Rate_of_Higher-Order_-_CHES2014-anssi.pdf

- Understanding Attack Success Rate (ASR) in Adversarial AI
-> https://vtiya.medium.com/understanding-attack-success-rate-asr-in-adversarial-ai-e4a1764c4e49

- Github implementation
-> https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/metrics/metrics.py

