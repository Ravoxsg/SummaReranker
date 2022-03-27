import numpy as np



def rank_array(t):
    y = np.copy(t)
    y.sort()
    y = y[::-1]
    ranks = np.zeros(len(t))
    flagged = np.zeros(len(t))
    for i in range(len(t)):
        el = t[i]
        for j in range(len(t)):
            if el == y[j] and flagged[j] == 0:
                ranks[i] = j
                flagged[j] = 1
                break

    return ranks


def nested_detach(tensors):
    "Detach `tensors` (even if it's a nested list/tuple of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_detach(t) for t in tensors)

    return tensors.detach()



if __name__ == '__main__':
    t = np.random.randint(0, 5, 10)
    print(t)
    ranks = rank_array(t)
    print(ranks)


