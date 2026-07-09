// Based on C code from
// https://github.com/imneme/pcg-c/blob/master/include/pcg_variants.h
// Uses 32-bit state, and 16-bit value precision

/*
 * PCG Random Number Generation for C.
 *
 * Copyright 2014 Melissa O'Neill <oneill@pcg-random.org>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * For additional information about the PCG random number generation scheme,
 * including its license and other licensing options, visit
 *
 *     http://www.pcg-random.org
 */

const uint PCG_DEFAULT_MULTIPLIER_32 = 747796405U;
const uint PCG_DEFAULT_INCREMENT_32 = 2891336453U;
const float PCG_FLOAT_SCALE = 1.0 / pow(2.0, 16.0);

struct Pcg32State
{
    uint state;
};

// Don't use these internal methods preceded by underscore

void _pcg_step(inout Pcg32State rng)
{
    rng.state = rng.state * PCG_DEFAULT_MULTIPLIER_32
            + PCG_DEFAULT_INCREMENT_32;
}

uint _pcg_rotr_16(in uint value, in uint rot)
{
    value &= uint(0xffff); // value is 16-bits
    uint result = (value >> rot) | (value << ((-int(rot)) & 15));
    return result & uint(0xffff);
}

uint _pcg_output_xsh_rr_32_16(in uint state)
{
    return _pcg_rotr_16(((state >> 10u) ^ state) >> 12u, state >> 28u);
}

uint _pcg_advance_lcg_32(in uint state, in uint delta, in uint cur_mult,
    in uint cur_plus)
{
    uint acc_mult = 1u;
    uint acc_plus = 0u;
    while (delta > 0u) {
        if ((delta & 1u) == 1u) {
            acc_mult *= cur_mult;
            acc_plus = acc_plus * cur_mult + cur_plus;
        }
        cur_plus = (cur_mult + 1u) * cur_plus;
        cur_mult *= cur_mult;
        delta /= 2;
    }
    return acc_mult * state + acc_plus;
}

// The three methods below seed, random, and discard, are the public API

// Initialize a new deterministic random generator using a seed value.
Pcg32State pcg_srandom(in uint seed)
{
    Pcg32State rng = Pcg32State(0U);
    _pcg_step(rng);
    rng.state += seed;
    _pcg_step(rng);
    return rng;
}

// Return the next random value (in the range 0-65535 (16-bit))
// and advance the random number generator
uint pcg_random(inout Pcg32State rng) {
    uint oldstate = rng.state;
    _pcg_step(rng);
    return _pcg_output_xsh_rr_32_16(oldstate);
}

// Efficiently skip the random generator ahead a certain amount.
// Useful for simulating linear random generation along textures
// or screens in a fragment shader.
void pcg_discard(inout Pcg32State rng, in uint delta) {
    rng.state = _pcg_advance_lcg_32(
            rng.state, delta, PCG_DEFAULT_MULTIPLIER_32, PCG_DEFAULT_INCREMENT_32);
}

// Returns random value in the range [0, 1)
float pcg_random_float(inout Pcg32State rng) {
    return pcg_random(rng) * PCG_FLOAT_SCALE;
}

// Returns vec2 with components in the range [0, 1)
vec2 pcg_random_vec2(inout Pcg32State rng) {
    return vec2(pcg_random_float(rng), pcg_random_float(rng));
}

// Returns vec3 with components in the range [0, 1)
vec3 pcg_random_vec3(inout Pcg32State rng) {
    return vec3(pcg_random_float(rng), pcg_random_float(rng), pcg_random_float(rng));
}
