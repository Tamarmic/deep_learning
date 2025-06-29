import torch
import attention
import math


def test_attention_scores():
    # Two batches (B=2), sequence length N=2, embedding dim D=2
    a = torch.tensor([
        [[1.0, 0.0], [0.0, 1.0]],  # Batch 0: orthogonal basis
        [[1.0, 2.0], [3.0, 4.0]]  # Batch 1: non-orthogonal
    ])  # shape: (2, 2, 2)

    b = torch.tensor([
        [[1.0, 0.0], [0.0, 1.0]],  # Batch 0: identity vectors
        [[5.0, 6.0], [7.0, 8.0]]  # Batch 1: richer test case
    ])  # shape: (2, 2, 2)

    # Expected output
    scale = math.sqrt(2)

    # Manually compute expected outputs for each batch
    # Batch 0: identity matrix scaled
    expected_0 = torch.tensor([[1.0, 0.0],
                               [0.0, 1.0]]) / scale

    # Batch 1: dot product manually
    # b[1,0] · a[1,0] = 5*1 + 6*2 = 17
    # b[1,0] · a[1,1] = 5*3 + 6*4 = 39
    # b[1,1] · a[1,0] = 7*1 + 8*2 = 23
    # b[1,1] · a[1,1] = 7*3 + 8*4 = 53
    expected_1 = torch.tensor([[17.0, 39.0],
                               [23.0, 53.0]]) / scale

    expected_output = torch.stack([expected_0, expected_1], dim=0)

    # Run actual attention scores
    A = attention.attention_scores(a, b)

    assert torch.allclose(A, expected_output, atol=1e-5)
    print("test_attention_scores with batches passed.")


# Run the test
test_attention_scores()

