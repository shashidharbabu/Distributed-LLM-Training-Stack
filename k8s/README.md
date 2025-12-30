# Kubernetes deployment

This folder contains a minimal multi-node configuration:

- `service-headless.yaml`: headless service to provide stable DNS for rendezvous.
- `job-worker.yaml`: StatefulSet with 2 replicas (nodes) and 4 GPUs each. Adjust `replicas`, `GPUS_PER_NODE`, and `NUM_NODES` to match your cluster.
- `pvc.yaml`: shared PVC for checkpoints and logs.

## Usage

```bash
kubectl apply -f pvc.yaml
kubectl apply -f service-headless.yaml
kubectl apply -f job-worker.yaml
```

Workers compute `NODE_RANK` from the StatefulSet ordinal and launch `llmtrain run` with rendezvous via the headless service. Checkpoints and metrics are written to `/checkpoints` mounted from the PVC.
