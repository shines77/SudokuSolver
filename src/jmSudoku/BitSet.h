
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
#include <cstring>      // For std::memset()
#include <initializer_list>
#include <type_traits>

#include "BitUtils.h"

namespace jstd {

template <size_t Bits, bool NeedTrim = true>
class SmallBitSet {
public:
    typedef typename std::conditional<
                (Bits <= sizeof(uint32_t) * 8), uint32_t, size_t
            >::type  unit_type;
    typedef SmallBitSet<Bits, NeedTrim> this_type;

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
    SmallBitSet() noexcept = default;

    SmallBitSet(const SmallBitSet<Bits, NeedTrim> & src) noexcept {
        for (size_t i = 0; i < kUnits; i++) {
            this->array_[i] = src.array(i);
        }
    }

    template <size_t UBits, bool UNeedTrim>
    SmallBitSet(const SmallBitSet<UBits, UNeedTrim> & src) noexcept {
        typedef SmallBitSet<UBits, UNeedTrim> SourceBitMap;
        size_t copyUnits = std::min(kUnits, SourceBitMap::kUnits);
        for (size_t i = 0; i < copyUnits; i++) {
            this->array_[i] = src.array(i);
        }
    }

    SmallBitSet(std::initializer_list<unit_type> init_list) noexcept {
        if (init_list.size() <= kUnits) {
            size_t i = 0;
            for (auto iter : init_list) {
                this->array_[i++] = *iter;
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
        }
    }

    ~SmallBitSet() = default;

    size_t size() const        { return Bits; }

          char * data()        { return (char *)      this->array_; }
    const char * data() const  { return (const char *)this->array_; }

    size_t total_bytes() const { return kBytes; }
    size_t unit_size() const { return kUnits; }
    size_t per_unit_bytes() const { return sizeof(unit_type); }

    bool need_trim() const { return NeedTrim; }

    size_t array(size_t index) const {
        assert(index < kUnits);
        return this->array_[index];
    }

private:
    template <size_t Pos>
    inline bool tail_is_any() const noexcept {
        if (need_trim() && (kRestBits != 0)) {
            size_t unit = this->array_[Pos] & kTrimMask;
            return (unit != 0);
        }
        else {
            return (this->array_[Pos] != 0);
        }
    }

    template <size_t Pos>
    inline bool tail_is_none() const noexcept {
        if (need_trim() && (kRestBits != 0)) {
            size_t unit = this->array_[Pos] & kTrimMask;
            return (unit == 0);
        }
        else {
            return (this->array_[Pos] == 0);
        }
    }

    template <size_t Pos>
    inline bool tail_is_all() const noexcept {
        if (need_trim() && (kRestBits != 0)) {
            size_t unit = this->array_[Pos] & kTrimMask;
            return (unit == kTrimMask);
        }
        else {
            return (this->array_[Pos] == kTrimMask);
        }
    }

public:
    this_type & init(std::initializer_list<unit_type> init_list) noexcept {
        if (init_list.size() <= kUnits) {
            size_t i = 0;
            for (auto iter : init_list) {
                this->array_[i++] = *iter;
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
        }
        return (*this);
    }

    class reference {
    private:
        this_type * bitmap_;    // pointer to the bitmap
        size_t pos_;            // position of element in bitset

        // proxy for an element
        friend class SmallBitSet<Bits, NeedTrim>;

    public:
        ~reference() noexcept {
            // destroy the object
        }

        reference & operator = (bool value) noexcept {
            // assign Boolean to element
            this->bitmap_->set(pos_, value);
            return (*this);
        }

        reference & operator = (const reference & right) noexcept {
            // assign reference to element
            this->bitmap_->set(pos_, bool(right));
            return (*this);
        }

        reference & flip() noexcept {
            // complement stored element
            this->bitmap_->flip(pos_);
            return (*this);
        }

        bool operator ~ () const noexcept {
            // return complemented element
            return (!this->bitmap_->test(pos_));
        }

        operator bool () const noexcept {
            // return element
            return (this->bitmap_->test(pos_));
        }

    private:
        reference() noexcept
            : bitmap_(nullptr), pos_(0) {
            // default construct
        }

        reference(this_type & bitmap, size_t pos) noexcept
            : bitmap_(&bitmap), pos_(pos) {
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

	this_type operator ~() const noexcept {
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

    this_type & clear() noexcept {
        std::memset(this->array_, 0, kUnits * sizeof(unit_type));
        return (*this);
    }

    this_type & fill(size_t value) noexcept {
        for (size_t i = 0; i < kUnits; i++) {
            this->array_[i] = (unit_type)value;
        }
        if (need_trim()) {
            this->trim();
        }
        return (*this);
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

    size_t count() const {
        size_t total_popcnt = 0;
        for (size_t i = 0; i < kUnits; i++) {
            size_t unit = this->array_[i];
            unsigned int popcnt = jstd::BitUtils::popcnt(unit);
            total_popcnt += popcnt;
        }
        return total_popcnt;
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
            return (this->array_[0] & unit_type(size_t(1) << pos));
        }
        else {
            size_t index = pos / kUnitBits;
            size_t shift = pos % kUnitBits;
            return (this->array_[index] & unit_type(size_t(1) << shift));
        }
    }

    bool any() const noexcept {
        if (Bits <= kUnitBits) {
            return tail_is_any<0>();
        }
        else {
            for (size_t i = 0; i < kUnits - 1; i++) {
                size_t unit = this->array_[i];
                if (unit != 0) {
                    return true;
                }
            }
            return tail_is_any<kUnits - 1>();
        }
    }

    bool none() const noexcept {
#if 1
        return !(this->any());
#else
        if (Bits <= kUnitBits) {
            return tail_is_none<0>();
        }
        else {
            for (size_t i = 0; i < kUnits - 1; i++) {
                size_t unit = this->array_[i];
                if (unit != 0) {
                    return false;
                }
            }
            return tail_is_none<kUnits - 1>();
        }
#endif
    }

    bool all() const noexcept {
        if (Bits <= kUnitBits) {
            return tail_is_all<0>();
        }
        else {
            for (size_t i = 0; i < kUnits - 1; i++) {
                size_t unit = this->array_[i];
                if (unit != kFullMask) {
                    return false;
                }
            }
            return tail_is_all<kUnits - 1>();
        }
    }

    this_type & set() noexcept {
        for (size_t i = 0; i < kUnits; i++) {
            this->array_[i] = kFullMask;
        }
        if (need_trim()) {
            this->trim();
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
        for (size_t i = 0; i < kUnits; i++) {
            this->array_[i] = 0;
        }
        return (*this);
    }

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

    this_type & flip() noexcept {
        for (size_t i = 0; i < kUnits; i++) {
            this->array_[i] ^= unit_type(-1);
        }
        if (need_trim()) {
            this->trim();
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
};

template <size_t Bits, bool NeedTrim>
inline
SmallBitSet<Bits, NeedTrim> operator & (const SmallBitSet<Bits, NeedTrim> & left,
                                        const SmallBitSet<Bits, NeedTrim> & right) noexcept {
    // left And right
    SmallBitSet<Bits, NeedTrim> answer = left;
    return (answer &= right);
}

template <size_t Bits, bool NeedTrim>
inline
SmallBitSet<Bits, NeedTrim> operator | (const SmallBitSet<Bits, NeedTrim> & left,
                                        const SmallBitSet<Bits, NeedTrim> & right) noexcept {
    // left Or right
    SmallBitSet<Bits, NeedTrim> answer = left;
    return (answer |= right);
}

template <size_t Bits, bool NeedTrim>
inline
SmallBitSet<Bits, NeedTrim> operator ^ (const SmallBitSet<Bits, NeedTrim> & left,
                                        const SmallBitSet<Bits, NeedTrim> & right) noexcept {
    // left Xor right
    SmallBitSet<Bits, NeedTrim> answer = left;
    return (answer ^= right);
}

template <size_t Rows, size_t Cols, typename TBitSet = std::bitset<Cols>>
class SmallBitMatrix {
private:
    typedef TBitSet bitset_type;

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
private:
    typedef TBitSet bitset_type;

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

template <size_t Depths, size_t Rows, size_t Cols,
          typename TSmallBitMatrix2 = SmallBitMatrix2<Rows, Cols>>
class SmallBitMatrix3 {
private:
    typedef TSmallBitMatrix2 matrix_type;

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

    void set() {
        for (size_t depth = 0; depth < Depths; depth++) {
            this->matrix_[depth].set();
        }
    }

    void reset() {
        for (size_t depth = 0; depth < Depths; depth++) {
            this->matrix_[depth].reset();
        }
    }

    void flip() {
        for (size_t depth = 0; depth < Depths; depth++) {
            this->matrix_[depth].flip();
        }
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
class BitMatrix2 {
private:
    typedef TBitSet bitset_type;

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

    BitMatrix2(const SmallBitMatrix2<Rows, Cols> & src) {
        this->array_.reserve(Rows);
        for (size_t row = 0; row < Rows; row++) {
            this->array_.push_back(src[row]);
        }
    }

    ~BitMatrix2() = default;

    BitMatrix2 & operator = (const BitMatrix2 & rhs) {
        if (&rhs != this) {
            for (size_t row = 0; row < Rows; row++) {
                this->array_[row] = rhs[row];
            }
        }
    }

    BitMatrix2 & operator = (const SmallBitMatrix2<Rows, Cols> & rhs) {
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

template <size_t Depths, size_t Rows, size_t Cols,
          typename TBitMatrix2 = BitMatrix2<Rows, Cols>>
class BitMatrix3 {
private:
    typedef TBitMatrix2 matrix_type;

    std::vector<matrix_type> matrix_;

public:
    BitMatrix3() {
        this->matrix_.resize(Depths);
    }

    BitMatrix3(const BitMatrix3 & src) {
        this->matrix_.reserve(Depths);
        for (size_t depth = 0; depth < Depths; depth++) {
            this->matrix_.push_back(src[depth]);
        }
    }

    BitMatrix3(const SmallBitMatrix3<Depths, Rows, Cols> & src) {
        this->matrix_.reserve(Depths);
        for (size_t depth = 0; depth < Depths; depth++) {
            this->matrix_.push_back(src[depth]);
        }
    }

    ~BitMatrix3() = default;

    BitMatrix3 & operator = (const BitMatrix3 & rhs) {
        if (&rhs != this) {
            for (size_t depth = 0; depth < Depths; depth++) {
                this->matrix_[depth] = rhs[depth];
            }
        }
    }

    BitMatrix3 & operator = (const SmallBitMatrix3<Depths, Rows, Cols> & rhs) {
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

    void set() {
        for (size_t depth = 0; depth < Depths; depth++) {
            this->matrix_[depth].set();
        }
    }

    void reset() {
        for (size_t depth = 0; depth < Depths; depth++) {
            this->matrix_[depth].reset();
        }
    }

    void flip() {
        for (size_t depth = 0; depth < Depths; depth++) {
            this->matrix_[depth].flip();
        }
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
