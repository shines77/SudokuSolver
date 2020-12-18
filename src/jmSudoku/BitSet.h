
#ifndef JSTD_BITSET_H
#define JSTD_BITSET_H

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
#include <algorithm>        // For std::min()
#include <cassert>

#include "BitUtils.h"

namespace jstd {

struct dont_init_t {};

template <size_t Bits>
class SmallBitSet {
public:
    typedef typename std::conditional<
                (Bits <= sizeof(uint32_t) * 8), uint32_t, size_t
            >::type  unit_type;
    typedef SmallBitSet<Bits> this_type;

    static const size_t kUnitBytes = sizeof(unit_type);
    static const size_t kUnitBits  = 8 * kUnitBytes;
    static const size_t kUnits = (Bits + kUnitBits - 1) / kUnitBits;
    static const size_t kBytes = kUnits * kUnitBytes;
    static const size_t kBits  = kUnits * kUnitBits;
    static const size_t kRestBits = (Bits % kUnitBits);
    static const unit_type kFullMask = unit_type(-1);
    static const unit_type kTrimMask = (kRestBits != 0) ? (unit_type(size_t(1) << kRestBits) - 1) : kFullMask;

private:
    unit_type array_[kUnits];

public:
    SmallBitSet() noexcept {
        static_assert((Bits != 0), "SmallBitSet<Bits>: Bits can not be 0 size.");
        this->reset();
    }

    SmallBitSet(dont_init_t & dont_init) noexcept {
        static_assert((Bits != 0), "SmallBitSet<Bits>: Bits can not be 0 size.");
        /* Here we don't need initialize for optimize sometimes. */
    }

    SmallBitSet(const this_type & src) noexcept {
        for (size_t i = 0; i < kUnits; i++) {
            this->array_[i] = src.array(i);
        }
    }

    template <size_t UBits>
    SmallBitSet(const SmallBitSet<UBits> & src) noexcept {
        typedef SmallBitSet<UBits> SourceBitMap;
        static const size_t copyUnits = std::min(kUnits, SourceBitMap::kUnits);
        for (size_t i = 0; i < copyUnits; i++) {
            this->array_[i] = src.array(i);
        }
        if (kRestBits != 0) {
            this->trim();
        }
    }

    SmallBitSet(unit_type value) noexcept {
        if (kRestBits == 0)
            this->array_[0] = value;
        else
            this->array_[0] = value & kTrimMask;
    }

    SmallBitSet(std::initializer_list<unit_type> init_list) noexcept {
        if (init_list.size() <= kUnits) {
            size_t i = 0;
            for (auto iter : init_list) {
                this->array_[i++] = *iter;
            }
            if (kRestBits != 0) {
                this->trim();
            }
        }
        else {
            size_t i = 0;
            for (auto iter : init_list) {
                this->array_[i++] = *iter;
                if (i >= kUnits) {
                    break;
                }
            }
            if (kRestBits != 0) {
                this->trim();
            }
        }
    }

    ~SmallBitSet() = default;

    size_t size() const        { return Bits; }

          char * data()        { return (char *)      this->array_; }
    const char * data() const  { return (const char *)this->array_; }

    size_t total_bytes() const { return kBytes; }
    size_t unit_size() const { return kUnits; }
    size_t per_unit_bytes() const { return sizeof(unit_type); }

    unit_type array(size_t index) const {
        assert(index < kUnits);
        return this->array_[index];
    }

    this_type & init(std::initializer_list<unit_type> init_list) noexcept {
        if (init_list.size() <= kUnits) {
            size_t i = 0;
            for (auto iter : init_list) {
                this->array_[i++] = *iter;
            }
            if (kRestBits != 0) {
                this->trim();
            }
        }
        else {
            size_t i = 0;
            for (auto iter : init_list) {
                this->array_[i++] = *iter;
                if (i >= kUnits) {
                    break;
                }
            }
            if (kRestBits != 0) {
                this->trim();
            }
        }
        return (*this);
    }

    class reference {
    private:
        this_type * bitset_;    // pointer to the bitmap
        size_t pos_;            // position of element in bitset

        // proxy for an element
        friend class SmallBitSet<Bits>;

    public:
        ~reference() noexcept {
            // destroy the object
        }

        reference & operator = (bool value) noexcept {
            // assign Boolean to element
            this->bitset_->set(pos_, value);
            return (*this);
        }

        reference & operator = (const reference & right) noexcept {
            // assign reference to element
            this->bitset_->set(pos_, bool(right));
            return (*this);
        }

        reference & flip() noexcept {
            // complement stored element
            this->bitset_->flip(pos_);
            return (*this);
        }

        bool operator ~ () const noexcept {
            // return complemented element
            return (!this->bitset_->test(pos_));
        }

        bool operator ! () const noexcept {
            // return complemented element
            return (!this->bitset_->test(pos_));
        }

        operator bool () const noexcept {
            // return element
            return (this->bitset_->test(pos_));
        }

    private:
        reference() noexcept
            : bitset_(nullptr), pos_(0) {
            // default construct
        }

        reference(this_type & bitsets, size_t pos) noexcept
            : bitset_(&bitsets), pos_(pos) {
            // construct from bitmap reference and position
        }
    };

    this_type & operator = (const this_type & right) noexcept {
        for (size_t i = 0; i < kUnits; i++) {
            this->array_[i] = right.array(i);
        }
        return (*this);
    }

    constexpr bool operator [] (size_t pos) const {
        assert(pos < Bits);
        return this->test(pos);
    }

	reference operator [] (size_t pos) {
        assert(pos < Bits);
        return reference(*this, pos);
    }

    this_type & operator & (unit_type value) noexcept {
        if (kRestBits == 0)
            this->array_[0] &= value;
        else
            this->array_[0] &= value & kTrimMask;
        return (*this);
    }

    this_type & operator | (unit_type value) noexcept {
        if (kRestBits == 0)
            this->array_[0] |= value;
        else
            this->array_[0] |= value & kTrimMask;
        return (*this);
    }

    this_type & operator ^ (unit_type value) noexcept {
        if (kRestBits == 0)
            this->array_[0] ^= value;
        else
            this->array_[0] ^= value & kTrimMask;
        return (*this);
    }

    this_type & operator &= (unit_type value) noexcept {
        if (kRestBits == 0)
            this->array_[0] &= value;
        else
            this->array_[0] &= value & kTrimMask;
        return (*this);
    }

    this_type & operator |= (unit_type value) noexcept {
        if (kRestBits == 0)
            this->array_[0] |= value;
        else
            this->array_[0] |= value & kTrimMask;
        return (*this);
    }

    this_type & operator ^= (unit_type value) noexcept {
        if (kRestBits == 0)
            this->array_[0] ^= value;
        else
            this->array_[0] ^= value & kTrimMask;
        return (*this);
    }

    this_type & operator &= (const this_type & right) noexcept {
        for (size_t i = 0; i < kUnits; i++) {
            this->array_[i] &= right.array(i);
        }
        return (*this);
    }

    this_type & operator |= (const this_type & right) noexcept {
        for (size_t i = 0; i < kUnits; i++) {
            this->array_[i] |= right.array(i);
        }
        return (*this);
    }

    this_type & operator ^= (const this_type & right) noexcept {
        for (size_t i = 0; i < kUnits; i++) {
            this->array_[i] ^= right.array(i);
        }
        return (*this);
    }

	this_type operator ~ () const noexcept {
        // Flip all bits
		return (this_type(*this).flip());
    }

	this_type operator ! () const noexcept {
        // Flip all bits
		return (this_type(*this).flip());
    }

    bool operator == (const this_type & right) noexcept {
        for (size_t i = 0; i < kUnits; i++) {
            if (this->array_[i] != right.array(i)) {
                return false;
            }
        }
        return true;
    }

    bool operator != (const this_type & right) noexcept {
        for (size_t i = 0; i < kUnits; i++) {
            if (this->array_[i] == right.array(i)) {
                return false;
            }
        }
        return true;
    }

    this_type & fill(size_t value) noexcept {
        if (kRestBits != 0) {
            size_t i = 0;
            for (; i < kUnits - 1; i++) {
                this->array_[i] = (unit_type)value;
            }
            this->array_[i] = value & kTrimMask;
        }
        else {
            for (size_t i = 0; i < kUnits; i++) {
                this->array_[i] = (unit_type)value;
            }
        }
        return (*this);
    }

    this_type & set() noexcept {
        if (kRestBits != 0) {
            if (kUnits <= 8) {
                size_t i = 0;
                for (; i < kUnits - 1; i++) {
                    this->array_[i] = kFullMask;
                }
                this->array_[i] = kTrimMask;
            }
            else {
                std::memset(this->array_, (kFullMask & 0xFF), (kUnits - 1) * sizeof(unit_type));
                this->array_[kUnits - 1] = kTrimMask;
            }
        }
        else {
            if (kUnits <= 8) {
                for (size_t i = 0; i < kUnits; i++) {
                    this->array_[i] = kFullMask;
                }
            }
            else {
                std::memset(this->array_, (kFullMask & 0xFF), kUnits * sizeof(unit_type));
            }
        }
        return (*this);
    }

    this_type & set(size_t pos) {
        assert(pos < Bits);
        if (Bits <= kUnitBits) {
            this->array_[0] |= unit_type(size_t(1) << pos);
        }
        else {
            size_t index = pos / kUnitBits;
            size_t shift = pos % kUnitBits;
            this->array_[index] |= unit_type(size_t(1) << shift);
        }
        return (*this);
    }

    this_type & set(size_t pos, bool value) {
        if (value)
            this->set(pos);
        else
            this->reset(pos);
        return (*this);
    }

    this_type & reset() noexcept {
        if (kUnits <= 8) {
            for (size_t i = 0; i < kUnits; i++) {
                this->array_[i] = 0;
            }
        }
        else {
            std::memset(this->array_, 0, kUnits * sizeof(unit_type));
        }
        return (*this);
    }

#if 1
    this_type & reset(size_t pos) {
        assert(pos < Bits);
        if (Bits <= kUnitBits) {
            this->array_[0] ^= unit_type(size_t(1) << pos);
        }
        else {
            size_t index = pos / kUnitBits;
            size_t shift = pos % kUnitBits;
            this->array_[index] ^= unit_type(size_t(1) << shift);
        }
        return (*this);
    }
#else
    this_type & reset(size_t pos) {
        assert(pos < Bits);
        if (Bits <= kUnitBits) {
            this->array_[0] &= unit_type(~(size_t(1) << pos));
        }
        else {
            size_t index = pos / kUnitBits;
            size_t shift = pos % kUnitBits;
            this->array_[index] &= unit_type(~(size_t(1) << shift));
        }
        return (*this);
    }
#endif

    this_type & flip() noexcept {
        if (kRestBits != 0) {
            size_t i = 0;
            for (; i < kUnits - 1; i++) {
                this->array_[i] ^= kFullMask;
            }
            this->array_[i] ^= kTrimMask;
        }
        else {
            for (size_t i = 0; i < kUnits; i++) {
                this->array_[i] ^= kFullMask;
            }
        }
        return (*this);
    }

    this_type & flip(size_t pos) {
        assert(pos < Bits);
        if (Bits <= kUnitBits) {
            this->array_[0] ^= unit_type(~(size_t(1) << pos));
        }
        else {
            size_t index = pos / kUnitBits;
            size_t shift = pos % kUnitBits;
            this->array_[index] ^= unit_type(~(size_t(1) << shift));
        }
        return (*this);
    }

    this_type & trim() noexcept {
        if (kRestBits != 0) {
		    this->array_[kUnits - 1] &= kTrimMask;
        }
        return (*this);
    }

    bool test(size_t pos) const {
        assert(pos < Bits);
        if (Bits <= kUnitBits) {
            return ((this->array_[0] & unit_type(size_t(1) << pos)) != 0);
        }
        else {
            size_t index = pos / kUnitBits;
            size_t shift = pos % kUnitBits;
            return ((this->array_[index] & unit_type(size_t(1) << shift)) != 0);
        }
    }

    size_t value(size_t pos) const {
        assert(pos < Bits);
        if (Bits <= kUnitBits) {
            return (this->array_[0]);
        }
        else {
            size_t index = pos / kUnitBits;
            return (this->array_[index]);
        }
    }

    bool any() const noexcept {
        for (size_t i = 0; i < kUnits - 1; i++) {
            size_t unit = this->array_[i];
            if (unit != 0) {
                return true;
            }
        }
        return (this->array_[kUnits - 1] != 0);
    }

    bool none() const noexcept {
#if 1
        return !(this->any());
#else
        for (size_t i = 0; i < kUnits - 1; i++) {
            size_t unit = this->array_[i];
            if (unit != 0) {
                return false;
            }
        }
        return (this->array_[kUnits - 1] == 0);
#endif
    }

    bool all() const noexcept {
        for (size_t i = 0; i < kUnits - 1; i++) {
            size_t unit = this->array_[i];
            if (unit != kFullMask) {
                return false;
            }
        }
        if (kRestBits != 0) {
            size_t unit = this->array_[kUnits - 1] & kTrimMask;
            return (unit == kTrimMask);
        }
        else {
            return (this->array_[kUnits - 1] == kFullMask);
        }
    }

    size_t bsf() const noexcept {
        for (size_t i = 0; i < kUnits; i++) {
            size_t unit = this->array_[i];
            if (unit != 0) {
                unsigned int index = jstd::BitUtils::bsf(unit);
                return (i * kUnitBits + index);
            }
        }
        return 0;
    }

    size_t bsr() const noexcept {
        for (ptrdiff_t i = kUnits - 1; i >= 0; i--) {
            size_t unit = this->array_[i];
            if (unit != 0) {
                unsigned int index = jstd::BitUtils::bsr(unit);
                return size_t(i * kUnitBits + index);
            }
        }
        return Bits;
    }

    size_t count() const noexcept {
        size_t total_popcnt = 0;
        for (size_t i = 0; i < kUnits; i++) {
            size_t unit = this->array_[i];
            unsigned int popcnt = jstd::BitUtils::popcnt(unit);
            total_popcnt += popcnt;
        }
        return total_popcnt;
    }

    unsigned long to_ulong() const {
        if (Bits <= sizeof(uint32_t) * 8) {
            return this->array_[0];
        }
        else {
            return static_cast<unsigned long>(this->array_[0]);
        }
    }

    uint64_t to_ullong() const {
        if (Bits <= sizeof(uint32_t) * 8) {
            return static_cast<uint64_t>(this->array_[0]);
        }
        else {
            return this->array_[0];
        }
    }

    unit_type value() const {
        return this->array_[0];
    }

    size_t value_sz() const {
        if (Bits <= sizeof(uint32_t) * 8) {
            return static_cast<size_t>(this->array_[0]);
        }
        else {
            return this->array_[0];
        }
    }
};

template <size_t Bits>
inline
SmallBitSet<Bits> operator & (const SmallBitSet<Bits> & left,
                              const SmallBitSet<Bits> & right) noexcept {
    // left And right
    SmallBitSet<Bits> answer = left;
    return (answer &= right);
}

template <size_t Bits>
inline
SmallBitSet<Bits> operator | (const SmallBitSet<Bits> & left,
                              const SmallBitSet<Bits> & right) noexcept {
    // left Or right
    SmallBitSet<Bits> answer = left;
    return (answer |= right);
}

template <size_t Bits>
inline
SmallBitSet<Bits> operator ^ (const SmallBitSet<Bits> & left,
                              const SmallBitSet<Bits> & right) noexcept {
    // left Xor right
    SmallBitSet<Bits> answer = left;
    return (answer ^= right);
}

template <size_t Rows, size_t Cols, typename TBitSet = std::bitset<Cols>>
class SmallBitMatrix {
public:
    typedef TBitSet bitset_type;

private:
    size_t rows_;
    bitset_type array_[Rows];

public:
    SmallBitMatrix() : rows_(Rows) {}
    ~SmallBitMatrix() {}

    size_t rows() const { return this->rows_; }
    size_t cols() const { return Cols; }

    size_t size() const { return Rows; }
    size_t total_size() const { return (Rows * Cols); }

    void setRows(size_t rows) {
        this->rows_ = rows;
    }

    bool test(size_t row, size_t col) {
        assert(row < Rows);
        return this->array_[row].test(col);
    }

    size_t value(size_t row, size_t col) {
        assert(row < Rows);
        return (size_t)(this->array_[row].test(col));
    }

    void set() {
        for (size_t row = 0; row < Rows; row++) {
            this->array_[row].set();
        }
    }

    void reset() {
        for (size_t row = 0; row < Rows; row++) {
            this->array_[row].reset();
        }
    }

    void flip() {
        for (size_t row = 0; row < Rows; row++) {
            this->array_[row].flip();
        }
    }

    bitset_type & operator [] (size_t pos) {
        assert(pos < Rows);
        return this->array_[pos];
    }

    const bitset_type & operator [] (size_t pos) const {
        assert(pos < Rows);
        return this->array_[pos];
    }
};

template <size_t Rows, size_t Cols, typename TBitSet = std::bitset<Cols>>
class SmallBitMatrix2 {
public:
    typedef TBitSet                                 bitset_type;
    typedef SmallBitMatrix2<Rows, Cols, TBitSet>    this_type;

private:
    bitset_type array_[Rows];

public:
    SmallBitMatrix2() = default;
    ~SmallBitMatrix2() = default;

    size_t rows() const { return Rows; }
    size_t cols() const { return Cols; }

    size_t size() const { return Rows; }
    size_t total_size() const { return (Rows * Cols); }

    bool test(size_t row, size_t col) {
        assert(row < Rows);
        return this->array_[row].test(col);
    }

    this_type & fill(size_t value) noexcept {
        for (size_t row = 0; row < Rows; row++) {
            this->array_[row].fill(value);
        }
        return (*this);
    }

    this_type & set() noexcept {
        for (size_t row = 0; row < Rows; row++) {
            this->array_[row].set();
        }
        return (*this);
    }

    this_type & reset() noexcept {
        for (size_t row = 0; row < Rows; row++) {
            this->array_[row].reset();
        }
        return (*this);
    }

    this_type & flip() noexcept {
        for (size_t row = 0; row < Rows; row++) {
            this->array_[row].flip();
        }
        return (*this);
    }

    bitset_type & operator [] (size_t pos) {
        assert(pos < Rows);
        return this->array_[pos];
    }

    const bitset_type & operator [] (size_t pos) const {
        assert(pos < Rows);
        return this->array_[pos];
    }
};

template <size_t Depths, size_t Rows, size_t Cols,
          typename TSmallBitMatrix2 = SmallBitMatrix2<Rows, Cols, std::bitset<Cols>>>
class SmallBitMatrix3 {
public:
    typedef TSmallBitMatrix2                                        matrix_type;
    typedef typename TSmallBitMatrix2::bitset_type                  bitset_type;
    typedef SmallBitMatrix3<Depths, Rows, Cols, TSmallBitMatrix2>   this_type;

private:
    matrix_type matrix_[Depths];

public:
    SmallBitMatrix3() = default;
    ~SmallBitMatrix3() = default;

    size_t depths() const { return Depths; }
    size_t rows() const { return Rows; }
    size_t cols() const { return Cols; }

    size_t size() const { return Depths; }
    size_t matrix2d_size() const { return (Rows * Cols); }
    size_t total_size() const { return (Depths * Rows * Cols); }

    bool test(size_t depth, size_t row, size_t col) {
        assert(depth < Depths);
        return this->matrix_[depth][row].test(col);
    }

    this_type & fill(size_t value) {
        for (size_t depth = 0; depth < Depths; depth++) {
            this->matrix_[depth].fill(value);
        }
        return (*this);
    }

    this_type & set() {
        for (size_t depth = 0; depth < Depths; depth++) {
            this->matrix_[depth].set();
        }
        return (*this);
    }

    this_type & reset() {
        for (size_t depth = 0; depth < Depths; depth++) {
            this->matrix_[depth].reset();
        }
        return (*this);
    }

    this_type & flip() {
        for (size_t depth = 0; depth < Depths; depth++) {
            this->matrix_[depth].flip();
        }
        return (*this);
    }

    matrix_type & operator [] (size_t pos) {
        assert(pos < Depths);
        return this->matrix_[pos];
    }

    const matrix_type & operator [] (size_t pos) const {
        assert(pos < Depths);
        return this->matrix_[pos];
    }
};

template <size_t Rows, size_t Cols>
using SmallBitSet2D = SmallBitMatrix2<Rows, Cols, SmallBitSet<Cols>>;

template <size_t Depths, size_t Rows, size_t Cols>
using SmallBitSet3D = SmallBitMatrix3<Depths, Rows, Cols, SmallBitMatrix2<Rows, Cols, SmallBitSet<Cols>>>;

template <size_t Rows, size_t Cols, typename TBitSet = std::bitset<Cols>>
class BitMatrix2 {
public:
    typedef TBitSet                             bitset_type;
    typedef BitMatrix2<Rows, Cols, TBitSet>     this_type;

private:
    std::vector<bitset_type> array_;

public:
    BitMatrix2() {
        this->array_.resize(Rows);
    }

    BitMatrix2(const BitMatrix2 & src) {
        this->array_.reserve(Rows);
        for (size_t row = 0; row < Rows; row++) {
            this->array_.push_back(src[row]);
        }
    }

    BitMatrix2(const SmallBitMatrix2<Rows, Cols, TBitSet> & src) {
        this->array_.reserve(Rows);
        for (size_t row = 0; row < Rows; row++) {
            this->array_.push_back(src[row]);
        }
    }

    ~BitMatrix2() = default;

    this_type & operator = (const this_type & rhs) {
        if (&rhs != this) {
            for (size_t row = 0; row < Rows; row++) {
                this->array_[row] = rhs[row];
            }
        }
    }

    this_type & operator = (const SmallBitMatrix2<Rows, Cols, TBitSet> & rhs) {
        for (size_t row = 0; row < Rows; row++) {
            this->array_[row] = rhs[row];
        }
    }

    size_t rows() const { return Rows; }
    size_t cols() const { return Cols; }

    size_t size() const { return Rows; }
    size_t total_size() const { return (Rows * Cols); }

    bool test(size_t row, size_t col) {
        assert(row < Rows);
        return this->array_[row].test(col);
    }

    this_type & fill(size_t value) {
        for (size_t row = 0; row < Rows; row++) {
            this->array_[row].fill(value);
        }
        return (*this);
    }

    this_type & set() {
        for (size_t row = 0; row < Rows; row++) {
            this->array_[row].set();
        }
        return (*this);
    }

    this_type & reset() {
        for (size_t row = 0; row < Rows; row++) {
            this->array_[row].reset();
        }
        return (*this);
    }

    this_type & flip() {
        for (size_t row = 0; row < Rows; row++) {
            this->array_[row].flip();
        }
        return (*this);
    }

    bitset_type & operator [] (size_t pos) {
        assert(pos < Rows);
        return this->array_[pos];
    }

    const bitset_type & operator [] (size_t pos) const {
        assert(pos < Rows);
        return this->array_[pos];
    }
};

template <size_t Depths, size_t Rows, size_t Cols,
          typename TBitMatrix2 = BitMatrix2<Rows, Cols>>
class BitMatrix3 {
public:
    typedef TBitMatrix2                                 matrix_type;
    typedef typename TBitMatrix2::bitset_type           bitset_type;
    typedef BitMatrix3<Depths, Rows, Cols, TBitMatrix2> this_type;

private:
    std::vector<matrix_type> matrix_;

public:
    BitMatrix3() {
        this->matrix_.resize(Depths);
    }

    BitMatrix3(const this_type & src) {
        this->matrix_.reserve(Depths);
        for (size_t depth = 0; depth < Depths; depth++) {
            this->matrix_.push_back(src[depth]);
        }
    }

    BitMatrix3(const SmallBitMatrix3<Depths, Rows, Cols,
                     SmallBitMatrix2<Rows, Cols, bitset_type>> & src) {
        this->matrix_.reserve(Depths);
        for (size_t depth = 0; depth < Depths; depth++) {
            this->matrix_.push_back(src[depth]);
        }
    }

    ~BitMatrix3() = default;

    BitMatrix3 & operator = (const this_type & rhs) {
        if (&rhs != this) {
            for (size_t depth = 0; depth < Depths; depth++) {
                this->matrix_[depth] = rhs[depth];
            }
        }
    }

    BitMatrix3 & operator = (const SmallBitMatrix3<Depths, Rows, Cols,
                                   SmallBitMatrix2<Rows, Cols, bitset_type>> & rhs) {
        for (size_t depth = 0; depth < Depths; depth++) {
            this->matrix_[depth] = rhs[depth];
        }
    }

    size_t depths() const { return Depths; }
    size_t rows() const { return Rows; }
    size_t cols() const { return Cols; }

    size_t size() const { return Depths; }
    size_t matrix2d_size() const { return (Rows * Cols); }
    size_t total_size() const { return (Depths * Rows * Cols); }

    bool test(size_t depth, size_t row, size_t col) {
        assert(depth < Depths);
        return this->matrix_[depth][row].test(col);
    }

    this_type & fill(size_t value) {
        for (size_t depth = 0; depth < Depths; depth++) {
            this->matrix_[depth].fill(value);
        }
        return (*this);
    }

    this_type & set() {
        for (size_t depth = 0; depth < Depths; depth++) {
            this->matrix_[depth].set();
        }
        return (*this);
    }

    this_type & reset() {
        for (size_t depth = 0; depth < Depths; depth++) {
            this->matrix_[depth].reset();
        }
        return (*this);
    }

    this_type & flip() {
        for (size_t depth = 0; depth < Depths; depth++) {
            this->matrix_[depth].flip();
        }
        return (*this);
    }

    matrix_type & operator [] (size_t pos) {
        assert(pos < Depths);
        return this->matrix_[pos];
    }

    const matrix_type & operator [] (size_t pos) const {
        assert(pos < Depths);
        return this->matrix_[pos];
    }
};

template <size_t Rows, size_t Cols, typename TBitSet = std::bitset<Cols>>
static void matrix2_copy(SmallBitMatrix2<Rows, Cols, TBitSet> & dest,
                         const BitMatrix2<Rows, Cols, TBitSet> & src)
{
    for (size_t row = 0; row < Rows; row++) {
        dest[row] = src[row];
    }
}

template <size_t Rows, size_t Cols, typename TBitSet = std::bitset<Cols>>
static void matrix2_copy(BitMatrix2<Rows, Cols, TBitSet> & dest,
                         const SmallBitMatrix2<Rows, Cols, TBitSet> & src)
{
    for (size_t row = 0; row < Rows; row++) {
        dest[row] = src[row];
    }
}

template <size_t Depths, size_t Rows, size_t Cols,
          typename TSmallBitMatrix2 = SmallBitMatrix2<Rows, Cols>,
          typename TBitMatrix2 = BitMatrix2<Rows, Cols>>
static void matrix3_copy(SmallBitMatrix3<Depths, Rows, Cols, TSmallBitMatrix2> & dest,
                         const BitMatrix3<Depths, Rows, Cols, TBitMatrix2> & src)
{
    for (size_t depth = 0; depth < Depths; depth++) {
        for (size_t row = 0; row < Rows; row++) {
            dest[depth][row] = src[depth][row];
        }
    }
}

template <size_t Depths, size_t Rows, size_t Cols,
          typename TSmallBitMatrix2 = SmallBitMatrix2<Rows, Cols>,
          typename TBitMatrix2 = BitMatrix2<Rows, Cols>>
static void matrix3_copy(BitMatrix3<Depths, Rows, Cols, TBitMatrix2> & dest,
                         const SmallBitMatrix3<Depths, Rows, Cols, TSmallBitMatrix2> & src)
{
    for (size_t depth = 0; depth < Depths; depth++) {
        for (size_t row = 0; row < Rows; row++) {
            dest[depth][row] = src[depth][row];
        }
    }
}

} // namespace jstd

#endif // JSTD_BITSET_H
