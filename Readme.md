# Machine Learning to detect Hacking
This repository includes the feature extraction from a database of **Hacking Attacks** using a Honeypot. The features are:
| Parameters   | Type |
| -------- | ------- |
| Tool Attack  | Categorical or class  |
| IP address | Integer   |
| Country  | Categorical or class    |
| Time | Integer    |

The process is summarized as a Multiple Linear Regression with an output of probability for the Type of Attack as shown in the next table:
| Types of Attack    |
| -------- |
| BRUTE FORCE ATTACKS   | 
|  COMMAND INJECTION   |   
| CROSS-SITE SCRIPTING (XSS)  |  
|  DDOS ATTACK     |   
|  DEFACEMENT |       
| SQL INJECTION |        
| THE MAN IN THE MIDDLE| 

## Machine Learning Algorithms
For this research, we have done a comparison of performance with 3 algorithms:
* Artificial Neural Networks
* Decision Trees
* Random Forest

Moreover, an anomaly detection technique was used to predict the type of attacks:
* One Class SVM
* Isolation Forest

The results will be presented in the paper: ** Classification of Cognitive Patterns of Hackers Using Machine Learning** in Lecture Notes of Computer Science in September 2024.

