## Plan: FedGAN PoC Implementation with MNIST

Proof-of-concept implementation of Federated GAN based on Algorithm 1 from the paper, using TensorFlow with MNIST dataset split across 5 agents in a non-IID manner. Single-process simulation with periodic parameter synchronization.

### Steps

1. **Create project structure** under `examples/fedgan_poc/` with `train.py`, `models.py`, `data.py`, `utils.py`, `requirements.txt`, and output directories (`outputs/images/`, `outputs/checkpoints/`, `outputs/metrics/`)

2. **Implement data partitioning** in `data.py` to split MNIST non-IID across B=5 agents (Agent 1: classes 0-1, Agent 2: classes 2-3, etc.), calculate weights p_i = |R_i| / sum(|R_j|), normalize images to [-1, 1]

3. **Build ACGAN-style models** in `models.py` following paper's architecture: Generator (100-dim noise → 1024 → 128 → 4x4 Conv → 4x4 Conv → 28x28x1) and Discriminator (28x28x1 → Conv → Conv → 1024 → binary output), with BatchNorm and LeakyReLU

4. **Implement FedGAN training loop** in `train.py` following Algorithm 1: create B local (G, D) pairs, run K=20 local gradient steps per agent in parallel, synchronize via weighted averaging every K iterations, track per-epoch generator/discriminator losses

5. **Add visualization outputs**: generate GIF from fixed noise across epochs showing generator evolution, plot combined G/D loss graph per epoch as PNG, save checkpoints at synchronization points, include Algorithm 1 step comments in code

### Further Considerations

1. **Hyperparameters to confirm**: Total training iterations N (suggest 10,000 for PoC), batch size (64 vs 128), learning rate (1e-4 with Adam beta_1=0.5), noise dimension (100 vs 62 from paper)?

2. **Synchronization frequency**: Start with K=20 per paper, but should we test K=[1, 5, 10] to demonstrate communication efficiency trade-offs?

3. **Evaluation approach**: Beyond visual GIF and loss curves, should we implement FID score calculation or compare against centralized GAN baseline?

### Process Sequence Diagram

```mermaid
sequenceDiagram
    participant I as Intermediary/Server
    participant A1 as Agent 1 (G1, D1)
    participant A2 as Agent 2 (G2, D2)
    participant A3 as Agent 3 (G3, D3)
    participant A4 as Agent 4 (G4, D4)
    participant A5 as Agent 5 (G5, D5)

    Note over I,A5: Initialization Phase
    I->>A1: Initialize w1_0=w_hat, theta1_0=theta_hat
    I->>A2: Initialize w2_0=w_hat, theta2_0=theta_hat
    I->>A3: Initialize w3_0=w_hat, theta3_0=theta_hat
    I->>A4: Initialize w4_0=w_hat, theta4_0=theta_hat
    I->>A5: Initialize w5_0=w_hat, theta5_0=theta_hat
    
    Note over I,A5: Training Loop (n=1 to N-1)
    
    loop For n=1 to N-1
        
        Note over A1,A5: Step 1 & 2: Local Training (Parallel)
        
        par Local Updates at Each Agent
            A1->>A1: Sample batch from R1
            A1->>A1: Generate fake data with G1
            A1->>A1: Compute g_tilde_1(theta_n_1, w_n_1)
            A1->>A1: Compute h_tilde_1(theta_n_1, w_n_1)
            A1->>A1: w_n_1 = w_(n-1)_1 + a(n-1) * g_tilde_1
            A1->>A1: theta_n_1 = theta_(n-1)_1 + b(n-1) * h_tilde_1
        and
            A2->>A2: Sample batch from R2
            A2->>A2: Generate fake data with G2
            A2->>A2: Compute g_tilde_2(theta_n_2, w_n_2)
            A2->>A2: Compute h_tilde_2(theta_n_2, w_n_2)
            A2->>A2: w_n_2 = w_(n-1)_2 + a(n-1) * g_tilde_2
            A2->>A2: theta_n_2 = theta_(n-1)_2 + b(n-1) * h_tilde_2
        and
            A3->>A3: Sample batch from R3
            A3->>A3: Generate fake data with G3
            A3->>A3: Compute g_tilde_3(theta_n_3, w_n_3)
            A3->>A3: Compute h_tilde_3(theta_n_3, w_n_3)
            A3->>A3: w_n_3 = w_(n-1)_3 + a(n-1) * g_tilde_3
            A3->>A3: theta_n_3 = theta_(n-1)_3 + b(n-1) * h_tilde_3
        and
            A4->>A4: Sample batch from R4
            A4->>A4: Generate fake data with G4
            A4->>A4: Compute g_tilde_4(theta_n_4, w_n_4)
            A4->>A4: Compute h_tilde_4(theta_n_4, w_n_4)
            A4->>A4: w_n_4 = w_(n-1)_4 + a(n-1) * g_tilde_4
            A4->>A4: theta_n_4 = theta_(n-1)_4 + b(n-1) * h_tilde_4
        and
            A5->>A5: Sample batch from R5
            A5->>A5: Generate fake data with G5
            A5->>A5: Compute g_tilde_5(theta_n_5, w_n_5)
            A5->>A5: Compute h_tilde_5(theta_n_5, w_n_5)
            A5->>A5: w_n_5 = w_(n-1)_5 + a(n-1) * g_tilde_5
            A5->>A5: theta_n_5 = theta_(n-1)_5 + b(n-1) * h_tilde_5
        end
        
        alt if n mod K == 0 (Synchronization Point)
            Note over I,A5: Step 3: Synchronization
            
            A1->>I: Send w_n_1, theta_n_1
            A2->>I: Send w_n_2, theta_n_2
            A3->>I: Send w_n_3, theta_n_3
            A4->>I: Send w_n_4, theta_n_4
            A5->>I: Send w_n_5, theta_n_5
            
            I->>I: Calculate w_n = sum(p_j * w_n_j)
            I->>I: Calculate theta_n = sum(p_j * theta_n_j)
            Note over I: p_j = |Rj| / sum(|Ri|)
            
            I->>A1: Broadcast w_n, theta_n
            I->>A2: Broadcast w_n, theta_n
            I->>A3: Broadcast w_n, theta_n
            I->>A4: Broadcast w_n, theta_n
            I->>A5: Broadcast w_n, theta_n
            
            A1->>A1: w_n_1 = w_n, theta_n_1 = theta_n
            A2->>A2: w_n_2 = w_n, theta_n_2 = theta_n
            A3->>A3: w_n_3 = w_n, theta_n_3 = theta_n
            A4->>A4: w_n_4 = w_n, theta_n_4 = theta_n
            A5->>A5: w_n_5 = w_n, theta_n_5 = theta_n
            
            Note over I,A5: Generate evaluation images
        else No Synchronization
            Note over A1,A5: Continue local training
        end
        
    end
    
    Note over I,A5: Training Complete
```