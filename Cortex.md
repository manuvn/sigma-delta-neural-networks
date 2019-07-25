# Cortex M4 power consumption estimate

A high-level back of the envelope estimate of the cost of performing the computation modelled here on a Cortex processor with super-highly optimized code. In reality the results are likely to be higher by a factor.

## Signal properties
---
- Bandwidth of signal = B
- Computation rate = 2*B

## Memory cost
---
- Assuming 16 bit weights and DRAM.
- Minimal amount of memory reads of weights = 30*(50*50 + 50 + 50) = 78000 * 16 bits ~ 1.2Mb.. too much for cache.
- Write back of states = 78000 * 16 bits
- Roughly 2 pJ/bit average for read+write gives us approx 4 uW memory access cost.
(http://www.pitt.edu/~juy9/papers/ping_iccad_318.pdf)
- With DRAM, it maybe 6-25pJ/bit, increasing energy cost to 15uW-60uW.
https://www.micron.com/support/tools-and-utilities/power-calc

Assuming 60 clock of memory access time and 128 bit memory line, clocks required = 78000 * 16 * 2 * 60/ 128 = 19500 * 60 = 1,170,000
https://medium.com/mythic-ai/mythic-hot-chips-2018-637dfb9e38b7

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
- Assuming very good caching mechnaism, the memory access delay can be hidden entirely.
--On a 20 MHz clock, it takes roughly - 1/10th of a second to do this.
- Energy consumed by ARM Cortex thumb(?) ~ 10 uW/MHz @ 20 MHz ~ 200 uW

The compute cost of 50 node RNN model is 20 uW approximately.

## Other energy cost
---
- ADC - 12-bit resolution at 2*B rate ~ 30*3pJ = 90 pJ ~ 90 pW

## Total energy cost 20uW + 15-60uW (DRAM) =35-80uW

## Compute cost growth 
- For Cortex = O(n^2)
- For a in-memory neuromorphic system, 
-- Energy cost grows as O(n) for neurons. 
-- For memory.
--- Current model analog memory access is very cheap. 
--- A single "bit" read with a microsecond pulse costs = 1e-9 * 1e-6 ~ 1fJ.
--- Matrix multiplication cost is also covered by the memory read cost.
--- Therefore, there is a lot of headroom.

