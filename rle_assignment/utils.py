from typing import Tuple, Dict
import numpy as np

class LinearSchedule(object):
    def __init__(self, initial_value: float, final_value: float, schedule_steps: int) -> None:
        super().__init__()
        self._initial_value = initial_value
        self._final_value = final_value
        self._schedule_steps = schedule_steps

    def value(self, t: int):
        interpolation = min(t / self._schedule_steps, 1.0)
        return self._initial_value + interpolation * (self._final_value - self._initial_value)


class RingBuffer(object):
    def __init__(self, size, specs: Dict[str, Tuple[Tuple, np.dtype]]):
        self.size = size
        self.specs = specs
        self.buffers = {k: np.empty((size,) + tuple(shape), dtype) for k, (shape, dtype) in specs.items()}
        self.next_idx = 0
        self.num_in_buffer = 0

    def __len__(self):
        return self.num_in_buffer

    def put(self, samples: Dict[str, np.ndarray]) -> None:
        num_samples = next(iter(samples.values())).shape[0]
        for key, buffer in self.buffers.items():
            features = samples[key]
            assert features.shape[0] == num_samples
            if self.next_idx+num_samples > self.size:
                buffer[self.next_idx:] = features[:self.next_idx+num_samples-self.size]
                buffer[:(self.next_idx + num_samples) % self.size] = features[self.next_idx+num_samples-self.size:]
            else:
                buffer[self.next_idx:self.next_idx+num_samples] = features
        self.next_idx = (self.next_idx + num_samples) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + num_samples)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        idx = np.random.randint(0, self.num_in_buffer, batch_size)
        return {
            key: buffer[idx]
            for key, buffer in self.buffers.items()
        }

class PrioritizedRingBuffer(RingBuffer):
    def __init__(self, size, specs: Dict[str, Tuple[Tuple, np.dtype]], alpha=0.6):
        super().__init__(size, specs)
        self.priorities = np.zeros((size,), dtype=np.float32)  # Initialize priorities
        self.alpha = alpha  # Exponent alpha determines how much prioritization is used

    def put(self, samples: Dict[str, np.ndarray]) -> None:
        max_priority = self.priorities.max() if self.num_in_buffer > 0 else 1.0  # Set max priority if not empty
        super().put(samples)
        self.priorities[self.next_idx - 1] = max_priority  # Set priority for new samples at maximum

    def sample(self, batch_size: int, beta=0.4) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        if self.num_in_buffer == self.size:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.next_idx]

        # Convert priorities into sampling probabilities
        sampling_probabilities = priorities ** self.alpha
        sampling_probabilities /= sampling_probabilities.sum()

        # Sample indices using the probabilities
        indices = np.random.choice(len(priorities), batch_size, p=sampling_probabilities)

        # Compute importance sampling weights
        weights = (len(priorities) * sampling_probabilities[indices]) ** -beta
        weights /= weights.max()  # Normalize weights

        samples = {
            key: buffer[indices]
            for key, buffer in self.buffers.items()
        }

        return samples, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        self.priorities[indices] = priorities
