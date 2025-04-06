Absolutely! Here's a **simpler version** of the `README.md` — clean, concise, and ready to copy into your repo:

---

```markdown
# Heterogeneous Homomorphic Encryption on Privacy-Preserving Federated Learning

This project implements federated learning enhanced with heterogeneous homomorphic encryption and Shamir's Secret Sharing to ensure privacy and secure model aggregation.

## Features

- Homomorphic encryption for secure model updates
- Shamir's Secret Sharing for decryption and aggregation
- Federated learning with multiple simulated clients

## Requirements

- Python 3.8+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## How to Run

1. Start the server:
   ```bash
   python server.py
   ```

2. Run each client in a separate terminal:
   ```bash
   python client.py
   ```
   or
   ```bash
   python DLClient.py
   ```

## Files

- `client.py`, `DLClient.py`: Client-side logic
- `server.py`: Server for aggregation
- `FeatureEngineering.py`: Data preprocessing
- `ShamirSecret.py`: Secret sharing for secure computation
```
---

