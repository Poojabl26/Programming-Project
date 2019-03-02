.. _data_management:

***************
Data management
***************


Documentation of the code in *src.data_management*.

.. automodule:: src.data_management.get_skill_data
    :members:

We start with merging all the relevant data files. SOEP has provided with the data for two different waves using different questionnaires. We first merge all of the datasets and then keep only the useful variables. 
Now, all our data for skill variables, be it cognitive or non-cognitive was categorical. So we first converted the categorical figures into numeric to perform mathematical operations on them. We then rename the important variables to be used later for our convenience. 
Then we restrict our data with individuals of age between 20 and 60 and drop individuals with occupations not useful in our analysis. After this, we dropped all the missing values by first replacing them nan. 
The variables for Cognitive abilities were standardised using the sklearn library. 
For personality we had 15 variables to be reduced to 5 be taking averages of 3 variables corresponding to a particular personality trait. Out of those 15, 4 had to be reversed since they represented the opposite quality corresponding the respective trait. We then created 5 variables by taking the average and standardising them. 
We then generated the experience and experience squared variables for the Mincer equation. 
We then took a log of wages to improve our regression model. 
For occupations, we combined similar categories and sorted them in order. 

vgfghdjtrddyt
