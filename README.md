# NBA Fantasy Draft Projector 2023-24
Projection model for NBA players' stats in the 2023-24 season, for use when drafting an NBA fantasy team.

Steps in the model:
1. normalise data across seasons
2. find 20 most similar players seasons historically
3. rank and weight each of those 20 players seasons stats depending on how similar they are
4. look at the 20 players' following season stats
5. use weighted averages to predict current players next season

Repeat for each player in 2023-24

## Installation
conda install python

conda create --name venv

conda activate venv

conda install pandas

conda install numpy

conda install matplotlib


## CSV data files
Obtained from https://www.kaggle.com/