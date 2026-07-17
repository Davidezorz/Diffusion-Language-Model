# Goal
In standard masking diffusion process:
$$
p_{mask}(t)=t
$$
While for us $p_{mask}(t,l)$ so both diffusion time $t$ and position $l$.
We want:
- left positions to stay "clean" longer
- right positions to get masked earlier

To create a right-to-left noising process.

The weights are given by:
$$
w_t=1+\gamma (p_l-0.5)
$$
so that left positions get weights below 1 while right postions get weights above 1.

The probability of a token to "survive" masking is:
$$
\alpha_{t,l}=(1-t)^{w_l}
$$

For example:
- $t=0.5$
- $1-t=0.5$
- $w=[0.5,1,1.5]$
- $\alpha=[0.707, 0.5, 0.354]$ (left side _survives_ more)
and the masking probability is given by:
$$
p_{mask}(t,l)=1-\alpha_{t,l}=1-(1-t)^{w_l}
$$
