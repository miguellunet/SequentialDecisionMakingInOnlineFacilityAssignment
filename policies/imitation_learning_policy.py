import numpy as np
import torch
from pathlib import Path

from policies.base_policy import BasePolicy
from policies.auxiliaries.imitation_learning_model import ImitationLearningNet

TRAIN_DIR = Path(__file__).resolve().parent.parent / "training"

class ImitationLearningPolicy(BasePolicy):
    """Neural net trained to imitate a reference policy. Builds the same flattened
    feature vector the model was trained on (per-warehouse normalized distance/capacity),
    then masks out depleted warehouses. Ported from Keras/TensorFlow to PyTorch: a raw
    torch module call has much lower per-inference dispatch overhead than Keras'
    predict_on_batch (~120us vs ~17us observed here), which matters because this runs
    once per order in a tight loop, not batched."""

    def __init__(self, env, num_warehouses, num_customers, capacity_distribution):
        super().__init__(env)
        self.num_warehouses = num_warehouses
        self.model_nn = ImitationLearningNet(num_warehouses)
        state_dict = torch.load(
            TRAIN_DIR / "imitation_learning_training" / "imitation_learning_models" / f"model_nn_w_{num_warehouses}_c_50_d_{capacity_distribution}.pt",
            map_location="cpu",
        )
        self.model_nn.load_state_dict(state_dict)
        self.model_nn.eval()

        # first forward pass pays PyTorch's one-time thread-pool/backend init cost;
        # warm it up here so it doesn't land inside the first timed act() call
        with torch.no_grad():
            self.model_nn(torch.zeros(1, 2 * num_warehouses, dtype=torch.float32))

    def act(self, state):
        
        warehouses_distance = state["warehouses_distance"]
        warehouses_capacity = state["warehouses_capacity"]

        features = []
        for i in range(self.num_warehouses):
            features.append(warehouses_distance[i] / 212.13)
            features.append(warehouses_capacity[i] / state['static_info']['warehouses_initial_capacity'][i])

        features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        # raw logits, not softmax probabilities - argmax is unaffected, and softmax
        # would be wasted work here since we only need the arg-best action
        with torch.no_grad():
            logits = self.model_nn(features)[0].numpy()

        for i in range(len(logits)):
            if warehouses_capacity[i] == 0:
                logits[i] = -np.inf

        return int(np.argmax(logits))
