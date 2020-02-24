import numpy as np


def generate_transforms(n, k):
    num_pos = n ** k
    base_np = np.reshape(np.arange(num_pos), [n] * k)
    # produce all 'one-step' transforms
    transforms = []
    for dim in range(k):
        # produce base flipped through axis dim
        transforms.append(np.flip(base_np, dim))
        for dim_ in range(k):
            # produce base rotated in the plane given by dim and dim_
            if dim != dim_:
                transforms.append(
                    np.rot90(base_np, k=1, axes=(dim, dim_)))  # no need to use other k, can compose with self
    transforms = [np.reshape(arr, num_pos) for arr in transforms]
    collection = [[i for i in range(num_pos)]]
    collection_grew = True
    while collection_grew:
        collection_grew = False
        for transform in collection:
            for base_transform in transforms:
                temp_tf = apply_transform(transform, base_transform, num_pos)
                if temp_tf not in collection:
                    collection.append(temp_tf)
                    collection_grew = True
    return collection


def apply_transform(base, transform, num_pos):
    """
    :param base: 1-D array to be transformed
    :param transform: 1-D transform to apply
    """
    return [base[transform[i]] for i in range(num_pos)]

print(len(generate_transforms(2, 5)))
# print("Applying stuff")
# print(apply([0, 1, 2], [2, 1, 0], 3))  # reverse base array for 3*1
# print(apply([2, 1, 0], [2, 1, 0], 3))  # reverse reveresed array for 3*1
# print(apply([1,0,3,2], [2,0,3,1], 4))  # reverese
