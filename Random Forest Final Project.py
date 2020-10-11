#!/usr/bin/env python
# coding: utf-8

# # Kolektif Öğrenme Yöntemleri 
# ## Final Projesi
# 
# * Env: Python3
# * Author: **MOGHAM NJIKAM MOHAMED NOURDINE** || ÖĞRENCİ NUMARASI: **198229001004**
# * Function：Random Forest（RF)
# 
# * DATA SOURCE: UCI. wine[DB/OL].https://archive.ics.uci.edu/ml/machine-learning-databases/wine.

# # Import Libraries
# 
# **Import the usual libraries for Pandas, Numpy, sklearn, joblib and the other python mathematical libaries. You can import sklearn later on if you want.**

# In[13]:


# Pandas is used for data manipulation
import pandas as pd
import numpy as np
import random
import math
import collections
from sklearn.datasets import load_digits
from joblib import Parallel, delayed


# In[35]:


class Tree(object):
    # Define a decision tree
    def __init__(self):
        self.split_feature = None
        self.split_value = None
        self.leaf_value = None
        self.tree_left = None
        self.tree_right = None

    def calc_predict_value(self, dataset):
        # Find the leaf node to which the sample belongs through a recursive decision tree
        if self.leaf_value is not None:
            return self.leaf_value
        elif dataset[self.split_feature] <= self.split_value:
            return self.tree_left.calc_predict_value(dataset)
        else:
            return self.tree_right.calc_predict_value(dataset)

    def describe_tree(self):
        # Print decision tree in json format for easy viewing of tree structure
        if not self.tree_left and not self.tree_right:
            leaf_info = "{leaf_value:" + str(self.leaf_value) + "}"
            return leaf_info
        left_info = self.tree_left.describe_tree()
        right_info = self.tree_right.describe_tree()
        tree_structure = "{'split_feature':" + str(self.split_feature) + ",'split_value':" + str(self.split_value) +                          ",'left_tree':" + left_info +                          ",'right_tree':" + right_info + "}"
        return tree_structure


#  
#    ### Random forest parameters
#     ----------
#     n_estimators:      Number of trees
#     max_depth:         Tree depth, -1 means unlimited depth
#     min_samples_split: The minimum number of samples required for node splitting is less than this value
#     min_samples_leaf:  The minimum sample number of leaf nodes, less than this value leaves are merged
#     min_split_gain:    The minimum gain required for splitting is less than this value
#     colsample_bytree:  Column sampling settings can be selected [sqrt, log2]. sqrt means randomly select 
#                        sqrt (n_features) features，log2 means randomly select log (n_features) features, 
#                        set to other, no column sampling
#     subsample:         Line sampling ratio
#     random_state:      Random seeds, the n_estimators sample set generated each time after setting will 
#                        not change, to ensure that the experiment can be repeated
#                        
# 

# In[36]:


class RandomForestClassifier(object):
    def __init__(self, n_estimators=10, max_depth=-1, min_samples_split=2, min_samples_leaf=1,
                 min_split_gain=0.0, colsample_bytree=None, subsample=0.8, random_state=None):

        self.n_estimators = n_estimators
        self.max_depth = max_depth if max_depth != -1 else float('inf')
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_split_gain = min_split_gain
        self.colsample_bytree = colsample_bytree
        self.subsample = subsample
        self.random_state = random_state
        self.trees = None
        self.feature_importances_ = dict()

    def fit(self, dataset, targets):
        print("Model training entrance")
        assert targets.unique().__len__() == 2, "There must be two class for targets!"
        targets = targets.to_frame(name='label')

        if self.random_state:
            random.seed(self.random_state)
        random_state_stages = random.sample(range(self.n_estimators), self.n_estimators)

        # Two column sampling methods
        if self.colsample_bytree == "sqrt":
            self.colsample_bytree = int(len(dataset.columns) ** 0.5)
        elif self.colsample_bytree == "log2":
            self.colsample_bytree = int(math.log(len(dataset.columns)))
        else:
            self.colsample_bytree = len(dataset.columns)

        # Building multiple decision trees in parallel
        self.trees = Parallel(n_jobs=-1, verbose=0, backend="threading")(
            delayed(self._parallel_build_trees)(dataset, targets, random_state)
                for random_state in random_state_stages)
        
    def _parallel_build_trees(self, dataset, targets, random_state):
        """Bootstrap has put back sampling to generate training sample set and build decision tree"""
        subcol_index = random.sample(dataset.columns.tolist(), self.colsample_bytree)
        dataset_stage = dataset.sample(n=int(self.subsample * len(dataset)), replace=True, 
                                        random_state=random_state).reset_index(drop=True)
        dataset_stage = dataset_stage.loc[:, subcol_index]
        targets_stage = targets.sample(n=int(self.subsample * len(dataset)), replace=True, 
                                        random_state=random_state).reset_index(drop=True)

        tree = self._build_single_tree(dataset_stage, targets_stage, depth=0)
        print(tree.describe_tree())
        return tree

    def _build_single_tree(self, dataset, targets, depth):
        """Recursively build a decision tree"""
        # If the categories of the node are all the same / samples are smaller than the minimum 
        # number of samples required for splitting, the category with the most occurrences is selected. 
        # Stop splitting
        if len(targets['label'].unique()) <= 1 or dataset.__len__() <= self.min_samples_split:
            tree = Tree()
            tree.leaf_value = self.calc_leaf_value(targets['label'])
            return tree

        if depth < self.max_depth:
            best_split_feature, best_split_value, best_split_gain = self.choose_best_feature(dataset, targets)
            left_dataset, right_dataset, left_targets, right_targets =                 self.split_dataset(dataset, targets, best_split_feature, best_split_value)

            tree = Tree()
            # If the sample of the left leaf node / right leaf node is smaller than the set minimum 
            # sample number of leaf nodes after the split
            if left_dataset.__len__() <= self.min_samples_leaf or                     right_dataset.__len__() <= self.min_samples_leaf or                     best_split_gain <= self.min_split_gain:
                tree.leaf_value = self.calc_leaf_value(targets['label'])
                return tree
            else:
                # If the feature is used during splitting, the import of the feature is increased by 1.
                self.feature_importances_[best_split_feature] =                     self.feature_importances_.get(best_split_feature, 0) + 1

                tree.split_feature = best_split_feature
                tree.split_value = best_split_value
                tree.tree_left = self._build_single_tree(left_dataset, left_targets, depth+1)
                tree.tree_right = self._build_single_tree(right_dataset, right_targets, depth+1)
                return tree
        # If the depth of the tree exceeds the preset value, the split is terminated
        else:
            tree = Tree()
            tree.leaf_value = self.calc_leaf_value(targets['label'])
            return tree

    def choose_best_feature(self, dataset, targets):
        """Find the best way to divide the data set，Find the optimal split feature,Split threshold,Split gain"""
        best_split_gain = 1
        best_split_feature = None
        best_split_value = None

        for feature in dataset.columns:
            if dataset[feature].unique().__len__() <= 100:
                unique_values = sorted(dataset[feature].unique().tolist())
            # If the dimension feature has too many values, select 100 percentile values 
            # as the candidate split threshold
            else:
                unique_values = np.unique([np.percentile(dataset[feature], x)
                                           for x in np.linspace(0, 100, 100)])

            # Find the splitting gain for the possible splitting threshold and 
            # select the threshold with the largest gain
            for split_value in unique_values:
                left_targets = targets[dataset[feature] <= split_value]
                right_targets = targets[dataset[feature] > split_value]
                split_gain = self.calc_gini(left_targets['label'], right_targets['label'])

                if split_gain < best_split_gain:
                    best_split_feature = feature
                    best_split_value = split_value
                    best_split_gain = split_gain
        return best_split_feature, best_split_value, best_split_gain

    @staticmethod
    def calc_leaf_value(targets):
        """Select the category with the most occurrences in the sample as the value of the leaf node"""
        label_counts = collections.Counter(targets)
        major_label = max(zip(label_counts.values(), label_counts.keys()))
        return major_label[1]

    @staticmethod
    def calc_gini(left_targets, right_targets):
        """The classification tree uses the Gini index as an indicator to select the optimal split point"""
        split_gain = 0
        for targets in [left_targets, right_targets]:
            gini = 1
            # Count how many samples are in each category, and then calculate gini
            label_counts = collections.Counter(targets)
            for key in label_counts:
                prob = label_counts[key] * 1.0 / len(targets)
                gini -= prob ** 2
            split_gain += len(targets) * 1.0 / (len(left_targets) + len(right_targets)) * gini
        return split_gain

    @staticmethod
    def split_dataset(dataset, targets, split_feature, split_value):
        """Divide the sample into left and right according to characteristics and threshold, 
         the left is less than or equal to the threshold, the right is greater than the threshold"""
        left_dataset = dataset[dataset[split_feature] <= split_value]
        left_targets = targets[dataset[split_feature] <= split_value]
        right_dataset = dataset[dataset[split_feature] > split_value]
        right_targets = targets[dataset[split_feature] > split_value]
        return left_dataset, right_dataset, left_targets, right_targets

    def predict(self, dataset):
        """Enter a sample to predict the category"""
        res = []
        for _, row in dataset.iterrows():
            pred_list = []
            # Count the prediction results of each tree, and select the result with 
            # the most occurrences as the final category
            for tree in self.trees:
                pred_list.append(tree.calc_predict_value(row))

            pred_label_counts = collections.Counter(pred_list)
            pred_label = max(zip(pred_label_counts.values(), pred_label_counts.keys()))
            res.append(pred_label[1])
        return np.array(res)


# In[37]:


# Read in data and display first 5 rows
df = pd.read_csv("source/wine.txt")
df.head(5)


# In[38]:


#check labels
print(df.label.unique())


# In[39]:


print('The shape of our features is:', df.shape)


# ####  1.Relevant Information:
# 
#    The attributes are:
#    
#  	1) Alcohol
#  	2) Malic acid
#  	3) Ash
# 	 4) Alcalinity of ash  
#  	5) Magnesium
# 	 6) Total phenols
#  	7) Flavanoids
#  	8) Nonflavanoid phenols
#  	9) Proanthocyanins
# 	 10)Color intensity
#  	11)Hue
#  	12)OD280/OD315 of diluted wines
#  	13)Proline            
# 
# #### 2. Number of Instances
# 
#     class 1 59
# 	class 2 71
# 	class 3 48
# 
# #### 3. Number of Attributes 
# 	
# 	13
# 
# ##### 4. For Each Attribute:
# 
# 	All attributes are continuous
# 	
# 	No statistics available, but suggest to standardise
# 	variables for certain uses (e.g. for us with classifiers
# 	which are NOT scale invariant)
# 
# 	NOTE: 1st attribute is class identifier (1-3)
# 
# ##### 5. Missing Attribute Values:
# 
# 	None
# 
# ##### 6. Class Distribution: number of instances per class
# 
#     class 1 59
# 	class 2 71
# 	class 3 48
#     
#  #### Details information can be found in the official website
#      
#      DATA SOURCE: UCI.wine[DB/OL].
#      
#      https://archive.ics.uci.edu/ml/machine-learning-databases/wine
#      

# In[40]:


df.describe()


# In[41]:


from matplotlib import pyplot as plt

plt.hist(df.label)

plt.show()


# In[42]:


# df = pd.read_csv("source/wine.txt")
# labels = np.array(df['label'])
# df = df.drop('Proline', axis = 1)
# df_list = list(df.columns)
# df = np.array(df)

# from sklearn.model_selection import train_test_split

# train_features, test_features, train_labels, test_labels = train_test_split(df, labels, test_size = .35, random_state = 66)

# print('Training Features Shape:', train_features.shape)
# print('Training Labels Shape:', train_labels.shape)
# print('Testing Features Shape', test_features.shape)
# print('Testing Labels Shape', test_labels.shape)


# There are not any data points that immediately appear as anomalous and no zeros in any of the measurement columns.

# In[43]:


df = df[df['label'].isin([1, 2])].sample(frac=1, random_state=66).reset_index(drop=True)
clf = RandomForestClassifier(n_estimators=10,
                             max_depth=5,
                             min_samples_split=6,
                             min_samples_leaf=2,
                             min_split_gain=0.0,
                             colsample_bytree="sqrt",
                             subsample=0.8,
                             random_state=66)
train_count = int(0.7 * len(df))
feature_list = ["Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols", 
                "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", 
                "OD280/OD315 of diluted wines", "Proline"]

train_features = df.loc[:train_count, feature_list]
train_labels = df.loc[:train_count, 'label']

clf.fit(train_features , train_labels)

# Use the forest's predict method on the test data
predictions = clf.predict(train_features)

# Calculate the absolute errors
errors = abs(predictions - train_labels)

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

from sklearn import metrics
print(metrics.accuracy_score(df.loc[:train_count, 'label'], clf.predict(df.loc[:train_count, feature_list])))
print(metrics.accuracy_score(df.loc[train_count:, 'label'], clf.predict(df.loc[train_count:, feature_list])))


# In[ ]:




