<h1 align = "center">DA421 Assignment 1</h1>

<p align = "center"><i>Name: Varun Nagpal</i></p>

<p align = "center"><i>Roll No: 210150020</i></p>

## Objective
Implement `situation testing` using k−nearest neighbor algorithm on the `adult and census-income` datasets. Implement both `DiscoveryN` and `PreventionN` algorithms and perform experiments to analyze their impact while also attempting to reproduce the paper <i>k-NN as an Implementation
 of Situation Testing for Discrimination Discovery and Prevention, SIGKDD, 2011</i> as closely as possible.
 
## Implementation

In this section I describe the implementation details necessary to fulfill the object and reproduce the results of the paper as closely as possible.

### Data Preprocessing

The datasets underwent standard pre-processing steps to ensure uniformity and accuracy. We first removed missing values using mean imputation for numerical features and the mode for categorical features. We used one-hot encoding for categorical features, applied Min-Max scaling to normalize all ordinal attributes between 0 and 1 and standardized all the numeric attributes. Finally, the dataset was split into 70% train, 30% test data.

* Discuss specifics for the Adult dataset
* Discuss specifics for the census income dataset

### Custom Distance Function

The custom distance function described in the paper could be closely achieved via the above mentioned data processing steps followed by a modified euclidean distance. I decided to write a custom wrapper that implemented sklearn's KNN with this custom distance metric. This custom distance function for KNN measures the dissimilarity between two tuples, 𝑟 and 𝑠 based on the attributes of the data. The distance is a non-negative real number, close to zero when the tuples are similar, and larger as they differ. It handles three types of domains:

1. `Interval-Scaled Domains`: The values are standardized using z-scores. The distance between two values is the absolute difference of their z-scores. Missing values are handled by assigning a fixed difference of 3 if either value is missing.

2. `Nominal Domains`: A simple binary function is used, where the distance is 0 if the values are equal and 1 if they differ. Missing values are assigned a distance of 1.

3. `Ordinal Domains`: Ranked values are first mapped into interval-scaled values, and then the absolute difference between these mapped values is used as the distance. If a value is missing, a specific formula is used to calculate the distance, or it is set to 1 if both values are missing.

Finally, the total distance between two tuples is the sum of the individual attribute distances, normalized by the number of attributes.

### DiscoveryN Algorithm

The idea of this algorithm is to identify records in the protected group that have been treated unfairly, based on a comparison of similar records from both protected and unprotected groups. the N in discovery stands for a nominal sensitive attribute.

![image](https://github.com/user-attachments/assets/2309d826-6c6e-49da-97ff-f0c1cadd1d3d)

* **Input**: 
  * R is the set of records.
  * t is the threshold for discrimination.
* **Initialization**:
  * L=∅ initializes an empty set for storing records classified with discrimination.
* **Iteration**:
  * For each record r∈P(R) (the protected group in the dataset): It checks if the decision dec(r) for record r belongs to a specific class (likely a negative or unfavorable decision). It also checks if the difference diff(r), which is calculated based on the differences in decisions between the protected and unprotected groups, is greater than or equal to the threshold t.
* **Classification**:
  * If the conditions are met, r[disc] is set to "yes" (indicating discrimination).
Otherwise, it is set to "no." The record r is then added to L, the set of discriminated records.
* **Output**:
  * A classifier is built on the set L to learn a model that can predict discrimination based on the data.

#### Python Code Explanation

* **Input Parameters**:
  * R: The dataset (pandas DataFrame).
  * t: The threshold for discrimination (diff(r)).
  * k: The number of nearest neighbors for comparison.
  * sensitive_attr: The sensitive attribute (like race, gender) that defines the protected group.
  * protected_values: The values of the sensitive attribute that correspond to the protected group.
  * label_attr: The attribute that contains the decision (classification) for each record.
  * negative_labels: The values of the label that are considered unfavorable.

* **Steps**:
  * The algorithm splits the dataset R into the protected group (P(R)) and the unprotected group (U(R)).
  * It then iterates over each record in P(R), checking its neighbors both in the protected group and unprotected group using k-nearest neighbors (with the help of scikit-learn's NearestNeighbors model along with the custom distance metric).
  * For each record, it computes the fraction of nearest neighbors in the protected group (p1) and the unprotected group (p2) that share the same label.
  * The diff(r) is computed as p1−p2, representing the difference in outcomes for similar records from the protected and unprotected groups.
  * If the record’s label is unfavorable (in negative_labels) and the difference diff(r) exceeds the threshold t, the record is marked as discriminated (disc = yes). Otherwise, it is marked as not discriminated (disc = no).
* **Output**:
  * The algorithm returns a DataFrame L containing the records from the protected group, with an additional column (disc) indicating whether each record is considered discriminated.



### PreventionN Algorithm

The PreventionN algorithm is designed to mitigate discrimination in a dataset by modifying decisions for records in the protected group that are deemed discriminatory based on a threshold for differences in outcomes.

![image](https://github.com/user-attachments/assets/9d72b596-0fcd-403b-af9e-13ba6e5673d5)


* **Input**:
  * T: The training set (dataset).
  * V: The validation set (for model evaluation).
  * t: A threshold for the difference in outcomes between the protected and unprotected groups.
  * 
* **Initialization**:
  * T′=∅: Initialize an empty modified training set (T′) where decisions will be modified to mitigate discrimination.

* **Iteration**:
  * For each record r∈T: Set r′ = r (a copy of the record).
  * Check if the decision dec(r) is unfavorable (belonging to a class marked as ⊕), if the record is in the protected group, and if the difference diff(r) between the protected and unprotected groups exceeds the threshold t.
  * If these conditions hold, the decision of r′ is modified to a favorable one (changing from ⊕ to another class). The modified record r′ is added to T′.
* **Classifiers**:
  * Build classifiers on both the original training set (T) and the modified set (T′).
  * Compare the performance of these classifiers on the validation set V, likely evaluating both accuracy and fairness.

#### Python Code Explanation

* **Input Parameters**:

  * T: Training dataset (pandas DataFrame).
  * V: Validation dataset (pandas DataFrame).
  * t: Threshold for discrimination (diff(r)).
  * k: Number of nearest neighbors.
  * sensitive_attr: Sensitive attribute in the dataset (e.g., race, gender).
  * protected_values: The values of the sensitive attribute identifying the protected group.
  * label_attr: Decision or outcome attribute in the dataset.
  * negative_labels: List of values in label_attr representing negative or unfavorable outcomes.
  * classifierName: Optional, specifies the classifier to use (e.g., Decision Tree, Logistic Regression).

* **Steps Explanation**:

  * The algorithm creates a copy of the training set (T_prime) to store modified records.
  * It splits the training set into two groups: the protected group (P_T) and the unprotected group (U_T), based on the sensitive_attr.
  * Using nearest neighbors, it computes diff(r) for each record in the protected group by comparing outcomes (labels) between similar records in the protected and unprotected groups.
  * If a record belongs to the protected group, has an unfavorable outcome, and the computed difference diff(r) exceeds the threshold t, its outcome is modified in T_prime to a favorable one.
    
* **Classifiers**:
  * The algorithm trains a classifier on the original dataset (T) and a separate classifier on the modified dataset (T_prime).

The algorithm evaluates both classifiers on the validation dataset (V) to compare their performance, using metrics like accuracy and discrimination (to check if fairness has improved). It evaluates how modifying decisions to improve fairness (i.e., reducing discrimination) might impact (reduce) the overall accuracy of the model. By comparing classifiers trained on the original and modified datasets, it helps assess the trade-off between these two objectives.

## Experimental Setup

For adult dataset, we use the protected attribute as race. For census-income dataset, we use the protected attribute as race in one experiment and in another experiment use marital status.

We implement the `DiscoveryN` algorithm on the Adult dataset with race as the protected attributes (non-whites) to obtain the `t-labelled` dataset. Then we use a DecisionTree classifier for 'disc' label and analyze performance metrics such as accuracy, precision, recall, F1-score on the test set (70:30 split)

On the original and `t-corrected` data which is obtained via the `PreventionN` algorithm, we implement Decision Tree, Naive Bayes and Logistic regression classifiers and report accuracy obtained, t = 0.10 discrimination of the classifier predictions.

## Results

Here, we provide the results of our experimentation along with the corresponding results made available in the reference research paper.

## Discrimination Discovery

<table><thead>
  <tr>
    <th></th>
    <th>Accuracy </th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1-Score</th>
  </tr></thead>
<tbody>
  <tr>
    <td>Adult (Race)</td>
    <td>89.12%</td>
    <td>48.73%</td>
    <td>51.33%</td>
    <td>50.00%</td>
  </tr>
  <tr>
    <td>Census-Income (Race)</td>
    <td>92.09%</td>
    <td>32.39%</td>
    <td>29.87%</td>
    <td>31.08%</td>
  </tr>
  <tr>
    <td>Census-Income (Marital Status)</td>
    <td>89.63%</td>
    <td>22.05%</td>
    <td>20.54%</td>
    <td>21.27%</td>
  </tr>
</tbody>
</table>

## Discrimnation Prevention

### Research Paper Results (Adult Dataset)
<table class="tg"><thead>
  <tr>
    <th class="tg-c3ow"></th>
    <th class="tg-c3ow" colspan="2">No preprocessing</th>
    <th class="tg-c3ow" colspan="2">0.1 correction</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-0pky">Classifier</td>
    <td class="tg-0pky">Accuracy </td>
    <td class="tg-0pky">0.1 disc</td>
    <td class="tg-0pky">Accuracy</td>
    <td class="tg-0pky">0.1 disc</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Decision Tree</td>
    <td class="tg-0pky">85.60%</td>
    <td class="tg-0pky"> 4.24%</td>
    <td class="tg-0pky">84.94%</td>
    <td class="tg-0pky"> 1.07%</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Naive Bayes</td>
    <td class="tg-0pky">82.46%</td>
    <td class="tg-0pky"> 4.06%</td>
    <td class="tg-0pky">82.33%</td>
    <td class="tg-0pky"> 2.23%</td>
  </tr>
  <tr>
    <td class="tg-baqh">Logistic Regression</td>
    <td class="tg-0lax">85.28%</td>
    <td class="tg-0lax"> 6.61%</td>
    <td class="tg-0lax">84.70%</td>
    <td class="tg-0lax"> 0.61%</td>
  </tr>
</tbody></table>

### Our Implementation (Adult Dataset)
<table class="tg"><thead>
  <tr>
    <th class="tg-c3ow"></th>
    <th class="tg-c3ow" colspan="2">No preprocessing</th>
    <th class="tg-c3ow" colspan="2">0.1 correction</th>
    <th class="tg-c3ow" colspan="2">0.05 correction</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-0pky">Classifier</td>
    <td class="tg-0pky">Accuracy </td>
    <td class="tg-0pky">0.1 disc</td>
    <td class="tg-0pky">Accuracy</td>
    <td class="tg-0pky">0.1 disc</td>
    <td class="tg-0pky">Accuracy</td>
    <td class="tg-0pky">0.05 disc</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Decision Tree</td>
    <td class="tg-0pky">81.90%</td>
    <td class="tg-0pky"> 7.82%</td>
    <td class="tg-0pky">81.21%</td>
    <td class="tg-0pky"> 6.82%</td>
    <td class="tg-0pky">79.75%</td>
    <td class="tg-0pky"> 3.85%</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Naive Bayes</td>
    <td class="tg-0pky">79.31%</td>
    <td class="tg-0pky">15.34%</td>
    <td class="tg-0pky">77.32%</td>
    <td class="tg-0pky">12.58%</td>
    <td class="tg-0pky">75.15%</td>
    <td class="tg-0pky"> 7.59%</td>
  </tr>
  <tr>
    <td class="tg-baqh">Logistic Regression</td>
    <td class="tg-0lax">85.20%</td>
    <td class="tg-0lax"> 8.46%</td>
    <td class="tg-0lax">85.03%</td>
    <td class="tg-0lax"> 6.65%</td>
    <td class="tg-0lax">84.71%</td>
    <td class="tg-0lax"> 4.20%</td>
  </tr>
</tbody></table>

### Our Implementation (Census Dataset, Race Sensitive Attribute)
<table class="tg"><thead>
  <tr>
    <th class="tg-c3ow"></th>
    <th class="tg-c3ow" colspan="2">No preprocessing</th>
    <th class="tg-c3ow" colspan="2">0.1 correction</th>
    <th class="tg-c3ow" colspan="2">0.05 correction</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-0pky">Classifier</td>
    <td class="tg-0pky">Accuracy </td>
    <td class="tg-0pky">0.1 disc</td>
    <td class="tg-0pky">Accuracy</td>
    <td class="tg-0pky">0.1 disc</td>
    <td class="tg-0pky">Accuracy</td>
    <td class="tg-0pky">0.05 disc</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Decision Tree</td>
    <td class="tg-0pky">92.53%</td>
    <td class="tg-0pky"> 4.06%</td>
    <td class="tg-0pky">91.70%</td>
    <td class="tg-0pky"> 1.72%</td>
    <td class="tg-0pky">90.07%</td>
    <td class="tg-0pky">-1.07%</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Naive Bayes</td>
    <td class="tg-0pky">88.82%</td>
    <td class="tg-0pky">11.71%</td>
    <td class="tg-0pky">87.90%</td>
    <td class="tg-0pky"> 9.46%</td>
    <td class="tg-0pky">83.44%</td>
    <td class="tg-0pky"> 7.56%</td>
  </tr>
  <tr>
    <td class="tg-baqh">Logistic Regression</td>
    <td class="tg-0lax">94.86%</td>
    <td class="tg-0lax"> 2.08%</td>
    <td class="tg-0lax">93.46%</td>
    <td class="tg-0lax">-1.79%</td>
    <td class="tg-0lax">93.44%</td>
    <td class="tg-0lax">-9.62%</td>
  </tr>
</tbody></table>

### Our Implementation (Census Dataset, Marital Status Sensitive Attribute)
<table class="tg"><thead>
  <tr>
    <th class="tg-c3ow"></th>
    <th class="tg-c3ow" colspan="2">No preprocessing</th>
    <th class="tg-c3ow" colspan="2">0.1 correction</th>
    <th class="tg-c3ow" colspan="2">0.05 correction</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-0pky">Classifier</td>
    <td class="tg-0pky">Accuracy </td>
    <td class="tg-0pky">0.1 disc</td>
    <td class="tg-0pky">Accuracy</td>
    <td class="tg-0pky">0.1 disc</td>
    <td class="tg-0pky">Accuracy</td>
    <td class="tg-0pky">0.05 disc</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Decision Tree</td>
    <td class="tg-0pky">92.46%</td>
    <td class="tg-0pky"> 3.07%</td>
    <td class="tg-0pky">91.13%</td>
    <td class="tg-0pky"> 1.86%</td>
    <td class="tg-0pky">90.69%</td>
    <td class="tg-0pky">-3.85%</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Naive Bayes</td>
    <td class="tg-0pky">88.82%</td>
    <td class="tg-0pky">-3.34%</td>
    <td class="tg-0pky">87.30%</td>
    <td class="tg-0pky">-4.58%</td>
    <td class="tg-0pky">82.72%</td>
    <td class="tg-0pky">-6.59%</td>
  </tr>
  <tr>
    <td class="tg-baqh">Logistic Regression</td>
    <td class="tg-0lax">94.86%</td>
    <td class="tg-0lax"> 0.93%</td>
    <td class="tg-0lax">93.70%</td>
    <td class="tg-0lax">-1.65%</td>
    <td class="tg-0lax">90.53%</td>
    <td class="tg-0lax">-3.20%</td>
  </tr>
</tbody></table>

