
#ifndef JSTD_BITMATRIX_H
#define JSTD_BITMATRIX_H

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
#include <type_traits>
#include <algorithm>        // For std::min()

#include "BitSet.h"
#include "PackedBitSet.h"

namespace jmSudoku {

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
    SmallBitMatrix2(const this_type & src) {
        for (size_t row = 0; row < Rows; row++) {
            this->array_[row] = src[row];
        }
    }
    ~SmallBitMatrix2() = default;

    this_type & operator = (const this_type & right) noexcept {
        for (size_t row = 0; row < Rows; row++) {
            this->array_[row] = right[row];
        }
        return (*this);
    }

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

    this_type & operator & (const this_type & right) noexcept {
        for (size_t row = 0; row < Rows; row++) {
            this->array_[row] &= right[row];
        }
        return (*this);
    }

    this_type & operator | (const this_type & right) noexcept {
        for (size_t row = 0; row < Rows; row++) {
            this->array_[row] |= right[row];
        }
        return (*this);
    }

    this_type & operator ^ (const this_type & right) noexcept {
        for (size_t row = 0; row < Rows; row++) {
            this->array_[row] ^= right[row];
        }
        return (*this);
    }

    this_type & operator &= (const this_type & right) noexcept {
        for (size_t row = 0; row < Rows; row++) {
            this->array_[row] &= right[row];
        }
        return (*this);
    }

    this_type & operator |= (const this_type & right) noexcept {
        for (size_t row = 0; row < Rows; row++) {
            this->array_[row] |= right[row];
        }
        return (*this);
    }

    this_type & operator ^= (const this_type & right) noexcept {
        for (size_t row = 0; row < Rows; row++) {
            this->array_[row] ^= right[row];
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
        for (size_t row = 0; row < Rows; row++) {
            if (this->array_[row] != right[row]) {
                return false;
            }
        }
        return true;
    }

    bool operator != (const this_type & right) noexcept {
        for (size_t row = 0; row < Rows; row++) {
            if (this->array_[row] != right[row]) {
                return true;
            }
        }
        return false;
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
    SmallBitMatrix3(const this_type & src) {
        for (size_t depth = 0; depth < Depths; depth++) {
            this->matrix_[depth] = src[depth];
        }
    }
    ~SmallBitMatrix3() = default;

    this_type & operator = (const this_type & right) noexcept {
        for (size_t depth = 0; depth < Depths; depth++) {
            this->matrix_[depth] = right[depth];
        }
        return (*this);
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

    this_type & operator & (const this_type & right) noexcept {
        for (size_t depth = 0; depth < Depths; depth++) {
            this->matrix_[depth] &= right[depth];
        }
        return (*this);
    }

    this_type & operator | (const this_type & right) noexcept {
        for (size_t depth = 0; depth < Depths; depth++) {
            this->matrix_[depth] |= right[depth];
        }
        return (*this);
    }

    this_type & operator ^ (const this_type & right) noexcept {
        for (size_t depth = 0; depth < Depths; depth++) {
            this->matrix_[depth] ^= right[depth];
        }
        return (*this);
    }

    this_type & operator &= (const this_type & right) noexcept {
        for (size_t depth = 0; depth < Depths; depth++) {
            this->matrix_[depth] &= right[depth];
        }
        return (*this);
    }

    this_type & operator |= (const this_type & right) noexcept {
        for (size_t depth = 0; depth < Depths; depth++) {
            this->matrix_[depth] |= right[depth];
        }
        return (*this);
    }

    this_type & operator ^= (const this_type & right) noexcept {
        for (size_t depth = 0; depth < Depths; depth++) {
            this->matrix_[depth] ^= right[depth];
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
        for (size_t depth = 0; depth < Depths; depth++) {
            if (this->matrix_[depth] != right[depth]) {
                return false;
            }
        }
        return true;
    }

    bool operator != (const this_type & right) noexcept {
        for (size_t depth = 0; depth < Depths; depth++) {
            if (this->matrix_[depth] != right[depth]) {
                return true;
            }
        }
        return false;
    }
};

//
// Matrix of SmallBitSet
//
template <size_t Rows, size_t Cols>
using SmallBitSet2D = SmallBitMatrix2<Rows, Cols, SmallBitSet<Cols>>;

template <size_t Depths, size_t Rows, size_t Cols>
using SmallBitSet3D = SmallBitMatrix3<Depths, Rows, Cols, SmallBitMatrix2<Rows, Cols, SmallBitSet<Cols>>>;

//
// Matrix of PackedBitSet
//
template <size_t Rows, size_t Cols>
using PackedBitSet2D = SmallBitMatrix2<Rows, Cols, PackedBitSet<Cols>>;

template <size_t Depths, size_t Rows, size_t Cols>
using PackedBitSet3D = SmallBitMatrix3<Depths, Rows, Cols, SmallBitMatrix2<Rows, Cols, PackedBitSet<Cols>>>;

// BitMatrix2
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

} // namespace jmSudoku

#endif // JSTD_BITMATRIX_H
