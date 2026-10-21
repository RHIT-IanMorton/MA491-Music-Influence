import numpy as np

death = np.array([0.280692308, 0.925615385, 0.226815385, 107.5200769, -6.482346154, 1, 7, 0.000259423, 0.289719615, 0.159226923, 0.126769231, 298886.1538])
mozart = np.array([0.353346375, 0.137869382, 0.330528998, 108.6043401, -20.17425693, 1, 7, 0.961572495, 0.508880787, 0.188827825, 0.068484861, 329702.9243])
acdc = np.array([0.46912, 0.850753333, 0.495099333, 128.7065667, -4.80952, 1, 9, 0.030241769, 0.227545577, 0.291534, 0.07204, 263588.7333])
aerosmith = np.array([0.440015152, 0.790477273, 0.536305303, 127.8730152, -7.660575758, 1, 9, 0.078510877, 0.033241413, 0.263157576, 0.04770303, 266323.1364])
motley_crue = np.array([0.471, 0.865964706, 0.486822353, 123.3003647, -5.864388235, 1, 0, 0.03286194, 0.10451767, 0.234508235, 0.063287059, 237349.6118])
one = np.ones(12)

def stdize(a):
    a = np.log10(np.abs(a))
    return (a - np.mean(a)) / np.std(a)

def euclid_mag(a):
    return np.sqrt(np.dot(a, a))
    
def taxi_mag(a):
    return sum(list(a))

def chess_mag(a):
    return max(list(a))

def cosine_sim(a, b, mag):
    return np.dot(a, b) / (mag(a) * mag(b))

def log_cos_sim(a, b, mag):
    return -np.log10(1 - cosine_sim(a, b, mag))

# print(death)
print(stdize(death))

# print(mozart)
print(stdize(mozart))

print(cosine_sim(stdize(death), stdize(mozart), euclid_mag))
print(cosine_sim(stdize(acdc), stdize(aerosmith), euclid_mag))
print()
print(cosine_sim(stdize(death), stdize(mozart), taxi_mag))
print(cosine_sim(stdize(acdc), stdize(aerosmith), taxi_mag))
print()
print(cosine_sim(stdize(death), stdize(mozart), chess_mag))
print(cosine_sim(stdize(acdc), stdize(aerosmith), chess_mag))

# sim = cosine_sim
# mag = euclid_mag
# print(sim(death, mozart, mag))
# print(sim(acdc, aerosmith, mag))
# print(sim(death, one, mag))
