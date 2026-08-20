#!/usr/bin/env python3
"""Verify the dual addition law before implementing it in CUDA.

Claim (twisted Edwards, a = -1):
    y(P + Q) = (x1*y1 - x2*y2) / (x1*y2 - y1*x2)
    y(P - Q) = (x1*y1 + x2*y2) / (x1*y2 + y1*x2)

Independent of the curve constant d, and yields y with no projective
coordinates for the result.
"""
import random
P = 2**255 - 19
D = (-121665 * pow(121666, P-2, P)) % P

def inv(a): return pow(a, P-2, P)
def on_curve(pt):
    x, y = pt
    return (-x*x + y*y - 1 - D*x*x*y*y) % P == 0
def add(p1, p2):
    x1,y1 = p1; x2,y2 = p2
    k = D*x1*x2*y1*y2 % P
    return ((x1*y2 + y1*x2) * inv(1+k) % P, (y1*y2 + x1*x2) * inv(1-k) % P)
def neg(p): return ((P - p[0]) % P, p[1])
def mul(pt, n):
    r = (0,1)
    while n:
        if n & 1: r = add(r, pt)
        pt = add(pt, pt); n >>= 1
    return r
def recover_x(y, sign):
    xx = (y*y-1) * inv(D*y*y+1) % P
    x = pow(xx, (P+3)//8, P)
    if (x*x-xx) % P: x = x*pow(2,(P-1)//4,P) % P
    if x % 2 != sign: x = P-x
    return x

BY = 4*inv(5) % P
B = (recover_x(BY,0), BY)
assert on_curve(B)

random.seed(7)
ok = 0
for trial in range(150):
    a = random.randrange(1, 2**252); b = random.randrange(1, 2**252)
    Pp = mul(B, a); Qq = mul(B, b)
    x1,y1 = Pp; x2,y2 = Qq
    n_plus  = (x1*y1 - x2*y2) % P; d_plus  = (x1*y2 - y1*x2) % P
    n_minus = (x1*y1 + x2*y2) % P; d_minus = (x1*y2 + y1*x2) % P
    if d_plus == 0 or d_minus == 0:
        continue
    assert n_plus  * inv(d_plus)  % P == add(Pp, Qq)[1],      f"P+Q mismatch, trial {trial}"
    assert n_minus * inv(d_minus) % P == add(Pp, neg(Qq))[1], f"P-Q mismatch, trial {trial}"
    ok += 1

print(f"dual addition law verified on {ok} random point pairs")
print("  y(P+Q) = (x1*y1 - x2*y2) / (x1*y2 - y1*x2)")
print("  y(P-Q) = (x1*y1 + x2*y2) / (x1*y2 + y1*x2)")
print("  no d, no projective coordinates -> 2 muls give TWO candidates")

# How often does a denominator vanish? It must be handled, not assumed away.
zero = 0
for _ in range(2000):
    a = random.randrange(1, 2**252); b = random.randrange(1, 2**252)
    Pp = mul(B, a); Qq = mul(B, b)
    x1,y1 = Pp; x2,y2 = Qq
    if (x1*y2 - y1*x2) % P == 0 or (x1*y2 + y1*x2) % P == 0: zero += 1
print(f"vanishing denominators in 2000 random pairs: {zero}")
