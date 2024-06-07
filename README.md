# E2E Solution for Flight Delay Prediction

Our repository is broken up into 4 main components:

- `pipeline/`: Contains logging, source code, and configurations for our models.
- `web-app/`: Contains logging, source code, configurations, and docker files for our web application.
- `lambda/`: Contains AWS Lambda scripts used for processing data and deploying ECS clusters.
- `deep-learning/`: Contains deep learning models for predicting both flight cancelation and delay. 

Each individual component contains its own readme with greater detail regarding how the application was built and how it was deployed. 
