# Proclivity_Propagation
Novel predictor for making making predictions of attribute values in social networks

## What does this project do?
This repository proposes a new method for prediction of missing attributes in friendship networks by using a state-of-the-art metric called PROclivity index for attributed NEtworks ( [ProNe](https://link.springer.com/chapter/10.1007%2F978-3-319-57454-7_18) ).  This method presents the first successful integration of the ProNe correlation metric with attribute prediction techniques. This new method is able achieve **prediction accuracies as high as 85%, 78%, 75% and 69% for attributes like class year, dormitory, status and gender** respectively. Alongside the high prediction accuracies, **Proclivity_Propagation** can also give **confidence intervals on the predicted accuracies**. **Proclivity Propagation** gives better values of prediction accuracies than established methods like Support Vector Machines, Low Rank Matrix Completion and Label Propagation.

## Applications of project?
Chief among the intended applications are the following:

* **Missing-data imputation:** Missing data is known to create many problems in statisti- cal analysis of datasets and hence missing-data imputation is an extremely worthwhile goal to be desired in a project.

* **Marketing:** Attribute prediction in social networks can help networks in more ap- propriate predictions for e-shopping for users, thereby making their online shopping experience more fast, relevant and worthwhile. 

* **Privacy considerations:** Accurate prediction of attributes can help to reveal which attributes or combinations of attributes can reveal sensitive information about users in a network. 


A visualization of one of the datasets used in this project ( *American75* ) can be found below. The plot given below uses modularity statistics from [Gephi](https://gephi.org/) to detect communities in the network. The plot uses different colors to show different communities in the said social network.

<img src="https://user-images.githubusercontent.com/26308648/51215393-cd783100-18ee-11e9-88db-b58df3719fbd.png" width="600">


## How to get started with this project?
```
$ git clone https://github.com/sidd0529/Proclivity_Propagation.git
$ cd Proclivity_Propagation
```

You will need to download **faceboook100** dataset to run this project. You can get it from [here.](https://drive.google.com/drive/folders/1wPF1eSdj-44O3snk03N4w1yk8TCi09Mc?usp=sharing)


## How to get run this project?
This code takes commandline input for a few parameters (filenames, ProNe matrix types and attribute numbers). This is the general format for running the code.
```
$ python proclivity.py -file Filename -prtype ProNe_Type -att Attribute_Num
```

Following is a specific example of a syntax for running the code. For Filename='American75', prtype='ProNel' and att=5, run the code using:
```
$ python proclivity.py -file American75 -prtype ProNel -att 5
```



## Where can you get help with this project?
I will be very happy to help in case you have any questions regarding this project. You can find me at siddharthsatpathy.ss@gmail.com .

