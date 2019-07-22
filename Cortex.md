# Cortex M4 power consumption estimate

A high-level back of the envelope estimate of the cost of performing the computation modelled here on a Cortex processor with super-highly optimized code. In reality the results are likely to be higher by a factor.

## Signal properties
---
- Bandwidth of signal = B
- Computation rate = 2*B

## Compute cost
---
- Number of clock cycles for computation of a 128 point FFT ~25000
- Compute operations for an 50 node RNN  
-- 1 input layer - 2 multiply and 1 addition.
-- Input to hidden = 50 multiply + 49 add.
-- 50 node hidden layer - 50*50 multiply - 50*49 add.
-- 1 output node - 50 multiply + 49 add
-- Roughly - 2552 multiply and 2500 add
- Assuming 26 clocks per multiply and 1 clock per add.
- Clocks required per second on ARM M0 = 30*(2602 * 26 + 2549 clocks) = 2,106,030.
--On a 20 MHz clock, it takes roughly - 1/10th of a second to do this.
- Energy consumed by ARM Cortex thumb(?) ~ 10 uW/MHz @ 20 MHz ~ 200 uW

The compute cost of 50 node RNN model is 20 uW approximately.

## Memory cost
---
- Assuming 16 bit weights and SRAM.
- Minimal amount of memory reads of weights = 30*(50*50 + 50 + 50) = 78000 * 16 bits.
- Write back of states = 78000 * 16 bits
- Roughly 2 pJ/bit for read gives us approx 4 uW memory access cost.
(http://www.pitt.edu/~juy9/papers/ping_iccad_318.pdf)


Total energy cost ~ 24 uW.
