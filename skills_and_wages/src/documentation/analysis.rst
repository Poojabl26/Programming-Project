.. _analysis:

************************************
Main model estimations / simulations
************************************

Documentation of the code in *src.analysis*. This is the core of the project. 


Regression
=================

.. automodule:: src.analysis.reg_tree
    :members:

We’re all familiar with the idea of linear regression as a way of making quantitative predictions. In simple linear regression, a real-valued dependent variable Y is modeled as a linear function of a real-valued independent variable X plus noise. 
In multiple regression, we let there be multiple independent variables. We perform OLS regression with the variables generated in the first step of Data Management. 

We run the regression using statsmodel python package. 

First we investigate the returns to earnings by  using the basic Mincer Equation which used 3 variables : Years of education, Years of Experience and Square of years of experience. 
 We will then expand the basic specification by adding the cognitive skills : Fluency and Symbol test score which we standardised. 
 We then introduce only non cognitive skills in the basic specification : Openness, Conscientiousness, Extraversion , Agreeableness and Neuroticism. 

We then add both skills (Cognitive and Non-Cognitive)to the specification. 

We define our independent variables as X and dependent variables as Y, For each specification, we generate a new matrix. We fit our linear model using the OLS module offered by the sm library and then print the summary which contains the regression output for all the defined model. 

After the first 4 regression, we perform regression for different occupational groups by defining a loop for the values in the our column occupation which contains : 1,2,3,4 thus generates results for 4 different outputs for all categories. 


Decision Tree
=================


Linear regression is a global model, where there is a single predictive formula holding over the entire data-space.
An alternative approach is to sub-divide, or partition, the space into smaller regions, where the interactions are more manageable. We then partition the sub-divisions again this is called recursive partitioning  until finally we get to chunks of the space which are so tame that we can fit simple models to them. The global model thus has two parts: one is just the recursive partition, the other is a simple model for each cell of the partition.

Prediction trees use the tree to represent the recursive partition. Each of the terminal nodes, or leaves, of the tree represents a cell of the partition, and has attached to it a simple model which applies in that cell only. A point x belongs to a leaf if x falls in the corresponding cell of the partition. To figure out which cell we are in, we start at the root node of the tree, and ask a sequence of questions about the features. The interior nodes are labeled with questions, and the edges or branches between them labeled by the answers.
For classic regression trees, the model in each cell is just a constant estimate of
Y.
Advantages of Decision Trees: 

* Making predictions is fast (no complicated calculations, just looking up constants in the tree)

* It’s easy to understand what variables are important in making the pre- diction (look at the tree)

* If some data is missing, we might not be able to go all the way down the tree to a leaf, but we can still make a prediction by averaging all the leaves in the sub-tree we do reach

* The model gives a jagged response, so it can work when the true regression surface is not smooth. If it is smooth, though, the piecewise-constant surface can approximate it arbitrarily closely (with enough leaves)

* There are fast, reliable algorithms to learn these trees


The basic regression-tree-growing algorithm then is as follows:
#. Start with a single node containing all points.Calculate the prediction for leaf  and Total Sum of Squares.

#. If all the points in the node have the same value for all the independent variables, stop. Otherwise, search over all binary splits of all variables for the one which will reduce S as much as possible. If the largest decrease in S would be less than some threshold δ, or one of the resulting nodes would contain less than q points, stop. Otherwise, take that split, creating two new nodes.

#. In each new node, go back to step 1.

We use the Top-Down Induction of Decision Trees.
Main: 
#. **let** T := Node := a decision tree consisting of an empty root node
#. **return** TDIT (E,Atts,T,Node)      E: set of examples, Atts: set of attributes

Discussing functions: 

TDIDT gain: 


*. Entropy: 
		
A decision tree is built top-down from a root node and involves partitioning the data into subsets that contain instances with similar values (homogenous). TDIDT algorithm uses entropy to calculate the homogeneity of a sample. If the sample is completely homogeneous the entropy is zero and if the sample is an equally divided it has entropy of one.The dataset is then split on the different attributes. The entropy for each branch is calculated. Then it is added proportionally, to get total entropy for the split. The resulting entropy is subtracted from the entropy before the split. The result is the Information Gain, or decrease in entropy. WE choose attribute with the largest information gain as the decision node, divide the dataset by its branches and repeat the same process on every branch.

*. Get Information Gain: 
The information gain is based on the decrease in entropy after a dataset is split on an attribute. Constructing a decision tree is all about finding attribute that returns the highest information gain (i.e., the most homogeneous branches).We calculate the entropy using the defined entropy function. The dataset is then split on the different attributes. The entropy for each branch is calculated. Then it is added proportionally, to get total entropy for the split. The resulting entropy is subtracted from the entropy before the split. The result is the Information Gain, or decrease in entropy.


Number of positives: 



Create tree TDIDT: TDIDT(E, Atts, T, Node)


Classify : : Choose attribute with the largest information gain as the decision node, divide the dataset by its branches and repeat the same process on every branch.A branch with entropy of 0 is a leaf node. A branch with entropy more than the threshold needs further splitting. 

Test Data Output: Predicting the values using results generated from the decision tree and calculating the accuracy/percentage of match the validate our results. 

Export tree node

Update to Dot File



