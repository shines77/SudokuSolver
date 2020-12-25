
#ifndef JSTD_BITVEC_H
#define JSTD_BITVEC_H

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
#pragma once
#endif

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <memory.h>
#include <assert.h>

#include <cstdint>
#include <cstddef>
#include <bitset>
#include <cstring>          // For std::memset()
#include <initializer_list>
#include <type_traits>

#define __SSE2__

// For SSE2, SSE3, SSSE3, SSE 4.1, AVX, AVX2
#if defined(_MSC_VER)
#include "msvc_x86intrin.h"
#else
#include <x86intrin.h>
#endif // _MSC_VER

namespace jmSudoku {

union alignas(16) IntVec128 {
    int8_t   i8[16];
    uint8_t  u8[16];
    int16_t  i16[8];
    uint16_t u16[8];
    int32_t  i32[4];
    uint32_t u32[4];
    int64_t  i64[2];
    uint64_t u64[2];
};

union alignas(32) IntVec256 {
    int8_t   i8[32];
    uint8_t  u8[32];
    int16_t  i16[16];
    uint16_t u16[16];
    int32_t  i32[8];
    uint32_t u32[8];
    int64_t  i64[4];
    uint64_t u64[4];
};

#if defined(__SSE2__) || defined(__SSE3__) || defined(__SSSE3__) || defined(__SSE4A__) || defined(__SSE4a__) \
 || defined(__SSE4_1__) || defined(__SSE4_2__) || defined(__AVX__) || defined(__AVX2__) || defined(__AVX512VL__)

struct BitVec16x08 {
    __m128i xmm0;

    BitVec16x08() noexcept {}
    // non-explicit conversions intended
    BitVec16x08(const __m128i & m128i) noexcept : xmm0(m128i) {}
    BitVec16x08(const BitVec16x08 & src) noexcept : xmm0(src.xmm0) {}

    BitVec16x08(uint8_t c00, uint8_t c01, uint8_t c02, uint8_t c03,
                uint8_t c04, uint8_t c05, uint8_t c06, uint8_t c07,
                uint8_t c08, uint8_t c09, uint8_t c10, uint8_t c11,
                uint8_t c12, uint8_t c13, uint8_t c14, uint8_t c15) :
            xmm0(_mm_setr_epi8(c00, c01, c02, c03, c04, c05, c06, c07,
                               c08, c09, c10, c11, c12, c13, c14, c15)) {}

    BitVec16x08(uint16_t w00, uint16_t w01, uint16_t w02, uint16_t w03,
                uint16_t w04, uint16_t w05, uint16_t w06, uint16_t w07) :
            xmm0(_mm_setr_epi16(w00, w01, w02, w03, w04, w05, w06, w07)) {}

    BitVec16x08(uint32_t i00, uint32_t i01, uint32_t i02, uint32_t i03) :
            xmm0(_mm_setr_epi32(i00, i01, i02, i03)) {}

    BitVec16x08 & operator = (const BitVec16x08 & other) {
        this->xmm0 = other.xmm0;
        return *this;
    }

    BitVec16x08 & operator = (const __m128i & xmm1) {
        this->xmm0 = xmm1;
        return *this;
    }

    bool operator == (const BitVec16x08 & other) {
        this->xor(other);
        return this->isAllZeros();
    }

    // Logical operation
    BitVec16x08 & operator & (const BitVec16x08 & vec) {
        this->and(vec.xmm0);
        return *this;
    }

    BitVec16x08 & operator | (const BitVec16x08 & vec) {
        this->or(vec.xmm0);
        return *this;
    }

    BitVec16x08 & operator ^ (const BitVec16x08 & vec) {
        this->xor(vec.xmm0);
        return *this;
    }

    BitVec16x08 & operator ! () {
        this->not();
        return *this;
    }

    // Logical operation
    BitVec16x08 & operator &= (const BitVec16x08 & vec) {
        this->and(vec.xmm0);
        return *this;
    }

    BitVec16x08 & operator |= (const BitVec16x08 & vec) {
        this->or(vec.xmm0);
        return *this;
    }

    BitVec16x08 & operator ^= (const BitVec16x08 & vec) {
        this->xor(vec.xmm0);
        return *this;
    }

    // Logical operation
    void and(const BitVec16x08 & vec) {
        this->xmm0 = _mm_and_si128(this->xmm0, vec.xmm0);
    }

    void and_not(const BitVec16x08 & vec) {
        this->xmm0 = _mm_andnot_si128(this->xmm0, vec.xmm0);
    }

    void or(const BitVec16x08 & vec) {
        this->xmm0 = _mm_or_si128(this->xmm0, vec.xmm0);
    }

    void xor(const BitVec16x08 & vec) {
        this->xmm0 = _mm_xor_si128(this->xmm0, vec.xmm0);
    }

    // Logical operation
    void and(__m128i xmm1) {
        this->xmm0 = _mm_and_si128(this->xmm0, xmm1);
    }

    void and_not(__m128i xmm1) {
        this->xmm0 = _mm_andnot_si128(this->xmm0, xmm1);
    }

    void or(__m128i xmm1) {
        this->xmm0 = _mm_or_si128(this->xmm0, xmm1);
    }

    void xor(__m128i xmm1) {
        this->xmm0 = _mm_xor_si128(this->xmm0, xmm1);
    }

    // Logical not: !
    void not() {
        __m128i zero = _mm_setzero_si128();
        this->xmm0 = _mm_andnot_si128(zero, this->xmm0);
    }

    // fill
    void fill_u8(uint8_t value) {
        this->xmm0 = _mm_set1_epi8(value);       // SSE2
    }

    void fill_u16(uint16_t value) {
        this->xmm0 = _mm_set1_epi16(value);      // SSE2
    }

    void fill_u32(uint32_t value) {
        this->xmm0 = _mm_set1_epi32(value);      // SSE2
    }

    void fill_u64(uint64_t value) {
        this->xmm0 = _mm_set1_epi64x(value);     // SSE2
    }

    // All zeros or all ones
    void setAllZeros() {
        this->xmm0 = _mm_setzero_si128();        // SSE2
    }

    void setAllOnes() {
        this->xmm0 = _mm_andnot_si128(this->xmm0, this->xmm0);
    }

    bool isAllZeros() const {
#ifdef __SSE4_1__
        return (_mm_test_all_zeros(this->xmm0, this->xmm0) != 0);
#else
        return (_mm_movemask_epi8(whichIsEqual(_mm_setzero_si128()).xmm0) == 0xffff);
#endif
    }

    bool isAllOnes() const {
#ifdef __SSE4_1__
        return (_mm_test_all_ones(this->xmm0) != 0);
#else
        BitVec16x08 ones;
        ones.setAllOnes;
        BitVec16x08 compare_mask = whichIsEqual(ones);
        return (_mm_movemask_epi8(compare_mask.xmm0) == 0xffff);
#endif
    }

    BitVec16x08 whichIsEqual(const BitVec16x08 & other) const {
        return _mm_cmpeq_epi16(this->xmm0, other.xmm0);
    }

    BitVec16x08 whichIsNonZero() const {
        return _mm_cmpeq_epi16(this->xmm0, _mm_setzero_si128());
    }

    BitVec16x08 whichIsAllOnes() const {
        __m128i ones;
        return _mm_cmpeq_epi16(this->xmm0, _mm_andnot_si128(ones, ones));
    }
};

#endif // >= SSE2

#if !defined(__AVX2__)

struct BitVec16x16 {
    BitVec16x08 low;
    BitVec16x08 high;

    BitVec16x16() noexcept : low(), high() {}

    // non-explicit conversions intended
    BitVec16x16(const BitVec16x16 & src) noexcept = default;

    BitVec16x16(const BitVec16x08 & _low, const BitVec16x08 & _high) noexcept : low(_low), high(_high) {}

    BitVec16x16(uint8_t c00, uint8_t c01, uint8_t c02, uint8_t c03,
                uint8_t c04, uint8_t c05, uint8_t c06, uint8_t c07,
                uint8_t c08, uint8_t c09, uint8_t c10, uint8_t c11,
                uint8_t c12, uint8_t c13, uint8_t c14, uint8_t c15,
                uint8_t c16, uint8_t c17, uint8_t c18, uint8_t c19,
                uint8_t c20, uint8_t c21, uint8_t c22, uint8_t c23,
                uint8_t c24, uint8_t c25, uint8_t c26, uint8_t c27,
                uint8_t c28, uint8_t c29, uint8_t c30, uint8_t c31) :
            low(c00, c01, c02, c03, c04, c05, c06, c07,
                c08, c09, c10, c11, c12, c13, c14, c15),
            high(c16, c17, c18, c19, c20, c21, c22, c23,
                 c24, c25, c26, c27, c28, c29, c30, c31) {}

    BitVec16x16(uint16_t w00, uint16_t w01, uint16_t w02, uint16_t w03,
                uint16_t w04, uint16_t w05, uint16_t w06, uint16_t w07,
                uint16_t w08, uint16_t w09, uint16_t w10, uint16_t w11,
                uint16_t w12, uint16_t w13, uint16_t w14, uint16_t w15) :
            low(w00, w01, w02, w03, w04, w05, w06, w07),
            high(w08, w09, w10, w11, w12, w13, w14, w15) {}

    BitVec16x16(uint32_t i00, uint32_t i01, uint32_t i02, uint32_t i03,
                uint32_t i04, uint32_t i05, uint32_t i06, uint32_t i07) :
            low(i00, i01, i02, i03), high(i04, i05, i06, i07)  {}

    BitVec16x16 & operator = (const BitVec16x16 & other) {
        this->low = other.low;
        this->high = other.high;
        return *this;
    }

    bool operator == (const BitVec16x16 & other) {
        this->xor(other);
        return (this->isAllZeros());
    }

    // Logical operation
    BitVec16x16 & operator & (const BitVec16x16 & vec) {
        this->and(vec);
        return *this;
    }

    BitVec16x16 & operator | (const BitVec16x16 & vec) {
        this->or(vec);
        return *this;
    }

    BitVec16x16 & operator ^ (const BitVec16x16 & vec) {
        this->xor(vec);
        return *this;
    }

    BitVec16x16 & operator ! () {
        this->not();
        return *this;
    }

    // Logical operation
    BitVec16x16 & operator &= (const BitVec16x16 & vec) {
        this->and(vec);
        return *this;
    }

    BitVec16x16 & operator |= (const BitVec16x16 & vec) {
        this->or(vec);
        return *this;
    }

    BitVec16x16 & operator ^= (const BitVec16x16 & vec) {
        this->xor(vec);
        return *this;
    }

    // Logical operation
    void and(const BitVec16x16 & vec) {
        this->low.and(vec.low);
        this->high.and(vec.low);
    }

    void and_not(const BitVec16x16 & vec) {
        this->low.and_not(vec.low);
        this->high.and_not(vec.low);
    }

    void or(const BitVec16x16 & vec) {
        this->low.or(vec.low);
        this->high.or(vec.low);
    }

    void xor(const BitVec16x16 & vec) {
        this->low.xor(vec.low);
        this->high.xor(vec.low);
    }

    // Logical not: !
    void not() {
        this->low.not();
        this->high.not();
    }

    // fill
    void fill_u8(uint8_t value) {
        this->low.fill_u8(value);
        this->high.fill_u8(value);
    }

    void fill_u16(uint16_t value) {
        this->low.fill_u16(value);
        this->high.fill_u16(value);
    }

    void fill_u32(uint32_t value) {
        this->low.fill_u32(value);
        this->high.fill_u32(value);
    }

    void fill_u64(uint64_t value) {
        this->low.fill_u64(value);
        this->high.fill_u64(value);
    }

    // All zeros or all ones
    void setAllZeros() {
        this->low.setAllZeros();
        this->high.setAllZeros();
    }

    void setAllOnes() {
        this->low.setAllOnes();
        this->high.setAllOnes();
    }

    bool isAllZeros() const {
        return (this->low.isAllZeros() && this->high.isAllZeros());
    }

    bool isAllOnes() const {
        return (this->low.isAllOnes() && this->high.isAllOnes());
    }

    BitVec16x16 whichIsEqual(const BitVec16x16 & other) const {
        return BitVec16x16(this->low.whichIsEqual(other.low), this->high.whichIsEqual(other.high));
    }

    BitVec16x16 whichIsNonZero() const {
        return BitVec16x16(this->low.whichIsNonZero(), this->high.whichIsNonZero());
    }

    BitVec16x16 whichIsAllOnes() const {
        return BitVec16x16(this->low.whichIsAllOnes(), this->high.whichIsAllOnes());
    }
};

#endif // !__AVX2__

} // namespace jmSudoku

#endif // JSTD_BITVEC_H
