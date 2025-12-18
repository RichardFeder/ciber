# Beam Correction Logic for CIBER Lensing QE

## The Problem

There are multiple places where beam corrections can be applied:
1. In `obs_auto` definition: C_obs = B² × C + N
2. In QE normalization (fB_ell parameter)
3. In proc_clkg (B_ell division)
4. In kcorr factor

This leads to confusion about which corrections to apply and where.

## The Physics

### Data Model
- True intensity: I(x)
- Observed: I^obs = B * I + noise
- Power spectra: C^obs = B² × C + N

### QE Formula (from colleague)
κ_L = (∫ F I^obs I^obs) / (∫ F f^κ B_ℓ B_{L-ℓ})

Where:
- F = filter function using C/C_obs
- f^κ = lensing response function  
- B_ℓ B_{L-ℓ} = beam factors at ℓ and L-ℓ

## Current Implementation

### In calc_filters_and_corrections:
```python
# Unlensed (no beam)
cib_unlensed_auto(ell) = C^{II}

# Observed (with beam)
obs_auto(ell) = B² × C^{II} + N

# Weight function
W_ell(ell) = C^{II} / C^obs = C^{II} / (B² × C^{II} + N)
```

### In QE normalization (flat_map.py):
When fB_ell is passed:
```python
# C map term
C_term = fC0²  × fB_ell² / fCtot = C² × B² / C_obs

# WF term  
WF_term = fC0 × fB_ell / fCtot = C × B / C_obs
```

The normalization integral: N^{-1} = ∫ F × f^κ × (implicit B factors from above)

### In proc_clkg:
```python
clkg /= B_ell²  # If B_ell provided
clkg /= kcorr   # Additional correction factor
```

## The Conflict

**Scenario 1**: Pass fB_ell to normalization + set kcorr=1.0
- QE norm includes: B² in numerator → normalization has 1/B² built in
- Result: κ estimate is already "beam-corrected"
- Then proc_clkg divides by B² AGAIN → double removal!
- **Outcome**: Estimates too low, roll-off at high ℓ

**Scenario 2**: Don't pass fB_ell + use kcorr with beam
- QE norm uses: C²/C_obs where C_obs has B² 
- Result: κ estimate has beam suppression from C_obs
- Then proc_clkg divides by kcorr (includes B²)
- **Outcome**: Should be correct, but was overcorrecting before

**Scenario 3**: Pass fB_ell + don't divide by B² in proc_clkg  
- QE norm includes B factors explicitly
- proc_clkg only divides by kcorr=1.0 (no beam)
- **Question**: Is this the right approach?

## The Solution

The key question: **Where does the normalization integral correction end up?**

Looking at the QE formula:
- Numerator: ∫ F I^obs I^obs (applied to data)
- Denominator: ∫ F f^κ B_ℓ B_{L-ℓ} (the normalization)

The result κ̂_L = Numerator / Denominator

If we compute the normalization correctly with B factors, the output κ̂ should already be:
- **Properly normalized** 
- **Beam effects handled** through the filter F and explicit B factors

So after the QE:
- NO need to divide by B² again
- NO need for kcorr if beam is in normalization

## Recommended Configuration

**Option A**: Beam in normalization (fB_ell) 
```python
b_ell_use = clf.bl  # Pass to normalization
kcorr = 1.0  # No additional correction
# In proc_clkg: B_ell = None (don't divide by beam)
```

**Option B**: Beam in post-processing
```python
b_ell_use = None  # Don't pass to normalization  
kcorr = <computed with beam>  # Apply beam correction here
# In proc_clkg: divide by kcorr
```

**NOT BOTH!**

## Current Bug

The code is doing:
1. Passing fB_ell to normalization ✓
2. Setting kcorr=1.0 ✓  
3. **Still dividing by B² in proc_clkg** ✗ ← THIS IS THE BUG!

Fix: When fB_ell is passed to normalization, do NOT pass B_ell to proc_clkg.
