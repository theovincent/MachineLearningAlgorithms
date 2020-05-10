

def square_step(samples, label, idx, kernel, param_kernel, former_step, box):
    above = label - former_step @ kernel(samples[idx], samples, param_kernel) - 0.5 * former_step[idx]
    underneath = 0.5 + kernel(samples[idx], samples[idx], param_kernel) * box

    return above / underneath
