from functionary.train.custom_datasets import create_mask_from_lengths


a = create_mask_from_lengths([5], 5, float("-inf"))
print(a)