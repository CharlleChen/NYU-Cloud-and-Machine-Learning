# Assignment 5: Kubernetes Practices
This assignment goes through the training and deployment of deep learning applications on IBM Kubernetes Services. Kubernetes is one of the most popular containerization orchestration frameworks. It allows quick deployment and scaling, thanks to the containerization technology. This experiment uses docker as the containers, Pytorch and Flask as programming frameworks.
## Quickstart
```bash
# Persistent Volume Claim
kubectl apply -f mypvc.yaml
```
Use `kubectl get pvc` to check status. Wait till the Persistent Volume Claim finishes. It should be fairly quick.
```bash
# Training
kubectl apply -f train-cloud.yaml
```

Use `kubectl get pods` to check status. Run the following after training is complete. It should take one to two minutes because of the large size of the docker image.
```bash
# Inferring
kubectl apply -f deployment-cloud.yaml
```
Use `kubectl get pods` to check status. Wait till the inference point is created.  
Find the endpoint: Kubernetes Dashboard => Services => infer-end => External Endpoints
```bash
curl --request POST -F image=@<local path to image> <external endpoint url:port>/inference
```

## Folders:
- train: used to build train docker image
- inference: used to build inference docker image

## YAML file
- mypvc.yaml: Persistent Volume Claim to store models
- train-cloud.yaml: train models as Job
- deployment-cloud.yaml: run inference point as Deployment and register load balancer as Service
