# ML Ops Major Assignment  
**Name:** Sunandita Babu  
**Roll No:** G24AI2048 


## Public Links (Required for Submission)
**GitHub Repository:** https://github.com/nita967/G24AI2048_mlops_Major  
**Docker Hub Image:** https://hub.docker.com/repository/docker/nita967/olivetti-classifier/


## Branch Structure (As per Assignment Requirement)
`main` - initial project setup, no merges into it  
`dev` - model development, training, and testing  
`docker_cicd` - Flask app, Docker, CI/CD pipeline & Kubernetes manifests  


## Project Overview
This project implements an end-to-end ML Deployment Pipeline using the **Olivetti Faces dataset** with:
  Training a **DecisionTreeClassifier**,
  Saving model as `savedmodel.pth`,
  Running automated tests via **GitHub Actions**,
  Building a Docker image and hosting on Docker Hub,
  Deploying the model-serving Flask app on **Kubernetes** using Deployment & Service,
  Providing screenshots for each step in the submission PDF.

## Model Summary
  **Dataset:** Olivetti Faces (from `sklearn.datasets`)
  **Algorithm:** Decision Tree Classifier  
  **Train/Test Split:** 70% train / 30% test  
  **Saved Model:** `model/savedmodel.pth`  
  **Preprocessing:** Flattening 64×64 grayscale images  
  **Evaluation Metric:** Accuracy  



