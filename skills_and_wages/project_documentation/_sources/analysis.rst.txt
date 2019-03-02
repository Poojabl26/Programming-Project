.. _analysis:

************************************
Main model estimations / simulations
************************************

Documentation of the code in *src.analysis*. This is the core of the project. 


Regression
=================

.. automodule:: src.analysis.reg_tree
    :members:


We run the regression using statsmodel python package. 

First we investigate the returns to earnings by  using the basic Mincer Equation which used 3 variables : Years of education, Years of Experience and Square of years of experience. 
 We will then expand the basic specification by adding the cognitive skills : Fluency and Symbol test score which we standardised. 
 We then introduce only non cognitive skills in the basic specification : Openness, Conscientiousness, Extraversion , Agreeableness and Neuroticism. 

We then add both skills (Cognitive and Non-Cognitive)to the specification. 

We define our independent variables as X and dependent variables as Y, For each specification, we generate a new matrix. We fit our linear model using the OLS module offered by the sm library and then print the summary which contains the regression output for all the defined model. 

After the first 4 regression, we perform regression for different occupational groups by defining a loop for the values in the our column occupation which contains : 1,2,3,4 thus generates results for 4 different outputs for all categories, 

