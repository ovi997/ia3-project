This project studies two methods of adversarial attack and defense in the task of document classification. 
## Running The Code
The project can be run by accessing the file _run_project.ipynb_.

## Document Classification
To implement the task of document classification it was used this [repo](https://github.com/apirjani/doc-classification). The project employs BERT, an encoder-only transformer model, augmented with a sequence classification head, to perform a 10-class document classification task. A linear layer is trained to map BERT-generated embeddings to the specified classes. Additionally, confidence thresholding is implemented to manage out-of-distribution document inputs. The model achieves an accuracy of 94.8% on the test set, which comprises 15% of the entire dataset, selected randomly.
## Attacks
### 1. Fast Minimum-Norm (FMN) Attack
The primary goal of the FMN attack is to generate an adversarial example from a clean input that makes a machine learning model give the wrong output while keeping the changes as subtle and hard to notice as possible. 

<picture> ![image](https://github.com/ovi997/assets/fnm.jpg)
 </picture>

Based on this algorithm, an untargeted attack on the embeddings of the text was implemented, using Cross-Entropy as loss function. 
### 2. DeepFool
The primary goal of the DeepFool attack is to compute a minimal perturbation for an input, such that the model’s prediction changes to a different (incorrect) class. The changes are often so small that they are imperceptible to humans.

<picture>![image](https://github.com/ovi997/assets/df.jpg)
 </picture>

DeepFool operates by treating the classifier as a geometric structure in a high-dimensional space and finding the nearest decision boundary.
 
## Defense
### 1. Defensive Distillation
Defensive Distillation leverages the idea of transferring knowledge from one neural network to another by using soft labels—outputs representing class probabilities from a trained network—instead of traditional hard labels, where each data point belongs exclusively to a single class.
Soft labels are computed using the following equation: 

$$y_i={e^{l_i/T}\over∑_je^{j/T}}$$, where _y_<sub>_i_</sub> is is the probability of the _i_-th class, _l_<sub>_i_</sub> is the _i_-th logit (input to the final softmax layer) and _T_ is the temperature. The parameter _T_ controls the softness of the labels:
- A high temperature (_T_ ≫ 1) produces labels that are more uniform across classes, representing a higher degree of uncertainty.
- A low temperature (_T_ → 0+) results in labels that closely resemble one-hot vectors, concentrating the probability on a single class.
- When _T_ = 1, the equation reduces to the standard softmax function.

In defensive distillation, knowledge is transferred by first training an initial network (the teacher network) as usual. Soft labels for the training dataset are then generated using a high temperature. These soft labels are subsequently used to train a second network (the distilled network). During the training of the distilled network, its final layer maintains the same high temperature, but once training is complete, the temperature is reset to _T_ = 1, allowing the network to be used normally.
### 2. Feature Squeezing
Feature Squeezing reduces the complexity of input data by "squeezing" redundant or unnecessary features, making it harder for attackers to embed adversarial perturbations that mislead the model.
There are several mechanisms to implement Feature Squeezing that can be used in the task of Document Classification. 

This project approaches normalizing text inputs to compress the input data representation. Special characters, diacritics and whitespaces were removed and also all text was converted to lowercase.
## References
- Moosavi-Dezfooli, S. M., Fawzi, A., & Frossard, P. (2016). _Deepfool: a simple and accurate method to fool deep neural networks_
- Pintor, M., Roli, F., Brendel, W., & Biggio, B. (2021). _Fast minimum-norm adversarial attacks through adaptive norm constraints_
- Soll, M., Hinz, T., Magg, S., & Wermter, S. (2019). _Evaluating defensive distillation for defending text processing neural networks against adversarial examples_
- Rosenberg, I., Shabtai, A., Elovici, Y., & Rokach, L. (2019). _Defense methods against adversarial examples for recurrent neural networks_



