.. _analysis:

************************************
Main model estimations / simulations
************************************

Documentation of the code in *src.analysis*. This is the core of the project. 


Regression
=================

.. automodule:: src.analysis.reg_tree
    :members:


First we investigate the returns to earnings by  using the basic Mincer Equation which used 3 variables : Years of education, Years of Experience and Square of years of experience. 
 We will then expand the basic specification by adding the cognitive skills : Fluency and Symbol test score which we standardised. 
 We then introduce only non cognitive skills in the basic specification : Openness, Conscientiousness, Extraverion , Neuroticism 