from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
print(_Tokenizer().encode(text="tench"))

"""

import clip
import torch

#device = "cuda" if torch.cuda.is_available() else "cpu"
device="cuda"
model, preprocess = clip.load("ViT-B/32", device=device)

#print(model.encode_text)
model.token_embedding=torch.nn.Embedding(49000, 512)
print(model.token_embedding)
#print(model.encode_text.positional_embedding)

"""
"""
N_WAY = 5  # Number of classes in a task
N_SHOT = 5  # Number of images per class in the support set
N_QUERY = 10  # Number of images per class in the query set
N_EVALUATION_TASKS = 100

# The sampler needs a dataset with a "get_labels" method. Check the code if you have any doubt!
test_set.get_labels = lambda: [
    instance[1] for instance in test_set._flat_character_images
]
test_sampler = TaskSampler(
    test_set, n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_EVALUATION_TASKS
)

test_loader = DataLoader(
    test_set,
    batch_sampler=test_sampler,
    num_workers=12,
    pin_memory=True,
    collate_fn=test_sampler.episodic_collate_fn,
)
"""