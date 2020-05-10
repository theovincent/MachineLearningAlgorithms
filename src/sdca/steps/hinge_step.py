

def hinge_step(samples, label, idx, kernel, param_kernel, former_step, box):
    to_compare = (1 - label * former_step @ kernel(samples[idx], samples, param_kernel)) / box
    min_value = min(1, to_compare / (kernel(samples[idx], samples[idx], param_kernel) ** 2 + 10 ** -16) + former_step[idx] * label)
    max_value = max(0, min_value)

    return label * max_value - former_step[idx]
