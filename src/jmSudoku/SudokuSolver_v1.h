
#ifndef JM_SUDOKU_SOLVER_V1_H
#define JM_SUDOKU_SOLVER_V1_H

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
#pragma once
#endif

#if defined(_MSC_VER)
#define __MMX__
#define __SSE__
#define __SSE2__
#define __SSE3__
#define __SSSE3__
#define __SSE4A__
#define __SSE4a__
#define __SSE4_1__
#define __SSE4_2__
#define __POPCNT__
#define __LZCNT__
#define __AVX__
#define __AVX2__
#define __3dNOW__
#else
//#undef __SSE4_1__
//#undef __SSE4_2__
#endif

#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <memory.h>
#include <assert.h>

#include <cstdint>
#include <cstddef>
#include <cstring>      // For std::memset(), std::memcpy()
#include <vector>
#include <bitset>
#include <array>        // For std::array<T, Size>

#if defined(_MSC_VER)
#include <emmintrin.h>      // For SSE 2
#include <tmmintrin.h>      // For SSE 3
#include <smmintrin.h>      // For SSE 4.1
#else
#include <x86intrin.h>      // For SSE 4.1
#endif // _MSC_VER

#include "Sudoku.h"
#include "StopWatch.h"
#include "BitUtils.h"

using namespace jstd;

/************************************************

#define SEARCH_MODE_ONE_ANSWER              0
#define SEARCH_MODE_MORE_THAN_ONE_ANSWER    1
#define SEARCH_MODE_ALL_ANSWERS             2

************************************************/

#define V1_SEARCH_MODE          SEARCH_MODE_ONE_ANSWER

#define V1_LITERAL_ORDER_MODE   0
#define V1_USE_STD_BITSET       0

namespace jmSudoku {
namespace v1 {

static const size_t kSearchMode = V1_SEARCH_MODE;

template <typename SudokuTy>
class Solver {
public:
    typedef Solver slover_type;
    typedef typename SudokuTy::NeighborCells    NeighborCells;
    typedef typename SudokuTy::CellInfo         CellInfo;

    static const size_t kAlignment = SudokuTy::kAlignment;
    static const size_t BoxCellsX = SudokuTy::BoxCellsX;      // 3
    static const size_t BoxCellsY = SudokuTy::BoxCellsY;      // 3
    static const size_t BoxCountX = SudokuTy::BoxCountX;      // 3
    static const size_t BoxCountY = SudokuTy::BoxCountY;      // 3
    static const size_t MinNumber = SudokuTy::MinNumber;      // 1
    static const size_t MaxNumber = SudokuTy::MaxNumber;      // 9

    static const size_t Rows = SudokuTy::Rows;
    static const size_t Cols = SudokuTy::Cols;
    static const size_t Boxes = SudokuTy::Boxes;
    static const size_t BoxSize = SudokuTy::BoxSize;
    static const size_t Numbers = SudokuTy::Numbers;

    static const size_t BoardSize = SudokuTy::BoardSize;
    static const size_t TotalSize = SudokuTy::TotalSize;
    static const size_t Neighbors = SudokuTy::Neighbors;

    static const size_t TotalCellLiterals = Rows * Cols;
    static const size_t TotalRowLiterals = Rows * Numbers;
    static const size_t TotalColLiterals = Cols * Numbers;
    static const size_t TotalBoxLiterals = Boxes * Numbers;

    static const size_t TotalLiterals =
        TotalCellLiterals + TotalRowLiterals + TotalColLiterals + TotalBoxLiterals;

#if (V1_LITERAL_ORDER_MODE == 0)
    static const size_t LiteralFirst     = 0;
    static const size_t CellLiteralFirst = LiteralFirst;
    static const size_t RowLiteralFirst  = CellLiteralFirst + TotalCellLiterals;
    static const size_t ColLiteralFirst  = RowLiteralFirst + TotalRowLiterals;
    static const size_t BoxLiteralFirst  = ColLiteralFirst + TotalColLiterals;
    static const size_t LiteralLast      = BoxLiteralFirst + TotalBoxLiterals;

    static const size_t CellLiteralLast  = RowLiteralFirst;
    static const size_t RowLiteralLast   = ColLiteralFirst;
    static const size_t ColLiteralLast   = BoxLiteralFirst;
    static const size_t BoxLiteralLast   = LiteralLast;
#else
    static const size_t LiteralFirst     = 0;
    static const size_t CellLiteralFirst = LiteralFirst;
    static const size_t BoxLiteralFirst  = CellLiteralFirst + TotalCellLiterals;
    static const size_t RowLiteralFirst  = BoxLiteralFirst + TotalBoxLiterals;
    static const size_t ColLiteralFirst  = RowLiteralFirst + TotalRowLiterals;
    static const size_t LiteralLast      = ColLiteralFirst + TotalColLiterals;

    static const size_t CellLiteralLast  = BoxLiteralFirst;
    static const size_t BoxLiteralLast   = RowLiteralFirst;
    static const size_t RowLiteralLast   = ColLiteralFirst;
    static const size_t ColLiteralLast   = LiteralLast;
#endif // (V1_LITERAL_ORDER_MODE == 0)

    static const size_t kAllRowsBit = SudokuTy::kAllRowsBit;
    static const size_t kAllColsBit = SudokuTy::kAllColsBit;
    static const size_t kAllBoxesBit = SudokuTy::kAllBoxesBit;
    static const size_t kAllNumbersBit = SudokuTy::kAllNumbersBit;

    static const int kLiteralCntThreshold = 0;

    static const size_t kEffectListAlignBytes =
        ((Neighbors * sizeof(uint8_t) + kAlignment - 1) / kAlignment) * kAlignment;

    static const size_t kEffectListReserveBytes1 = kEffectListAlignBytes - Neighbors * sizeof(uint8_t);
    static const size_t kEffectListReserveBytes  = (kEffectListReserveBytes1 != 0) ? kEffectListReserveBytes1 : kAlignment;

    static size_t num_guesses;
    static size_t num_unique_candidate;
    static size_t num_failed_return;

private:
#if (V1_LITERAL_ORDER_MODE == 0)
    enum LiteralType {
        CellNums,
        RowNums,
        ColNums,
        BoxNums,
        MaxLiteralType
    };
#else
    enum LiteralType {
        CellNums,
        BoxNums,
        RowNums,
        ColNums,
        MaxLiteralType
    };
#endif // (V1_LITERAL_ORDER_MODE == 0)

#pragma pack(push, 1)

    union literal_info_t {
        struct {
            uint8_t count;
            uint8_t enable;
        };
        uint16_t value;
    };

    // Aligned to sizeof(size_t) for cache friendly
    struct EffectList {
        uint8_t cells[Neighbors];
        uint8_t reserve[kEffectListReserveBytes];
    };

#pragma pack(pop)

#if V1_USE_STD_BITSET
    typedef std::bitset<Numbers>    bitset_type;
    std::bitset<BoardSize>          cell_filled_;

    SmallBitMatrix2<Boxes, Numbers>   boxes_;
    SmallBitMatrix2<Rows, Numbers>    rows_;
    SmallBitMatrix2<Cols, Numbers>    cols_;

    alignas(16) SmallBitMatrix2<BoardSize, Numbers>       cell_nums_;     // [row * 9 + col][num]

    alignas(16) SmallBitMatrix3<Boxes, Numbers, BoxSize>  box_nums_;      // [box][num][box_size]
    alignas(16) SmallBitMatrix3<Rows,  Numbers, Cols>     row_nums_;      // [row][num][col]
    alignas(16) SmallBitMatrix3<Cols,  Numbers, Rows>     col_nums_;      // [col][num][row]
#else
    typedef SmallBitSet<Numbers>    bitset_type;
    SmallBitSet<BoardSize>          cell_filled_;

    SmallBitSet2D<Boxes, Numbers>   boxes_;
    SmallBitSet2D<Rows, Numbers>    rows_;
    SmallBitSet2D<Cols, Numbers>    cols_;

    alignas(16) SmallBitSet2D<BoardSize, Numbers>       cell_nums_;     // [row * 9 + col][num]

    alignas(16) SmallBitSet3D<Boxes, Numbers, BoxSize>  box_nums_;      // [box][num][box_size]
    alignas(16) SmallBitSet3D<Rows,  Numbers, Cols>     row_nums_;      // [row][num][col]
    alignas(16) SmallBitSet3D<Cols,  Numbers, Rows>     col_nums_;      // [col][num][row]
#endif

#if defined(__SSE4_1__)
    alignas(16) literal_info_t literal_info_[TotalLiterals];
#else
    alignas(16) uint8_t literal_count_[TotalLiterals];
    alignas(16) uint8_t literal_enable_[TotalLiterals];
#endif

    size_t empties_;

    std::vector<EffectList>             effect_list_;
    std::vector<SudokuBoard<BoardSize>> answers_;

public:
    Solver() : empties_(0) {
    }
    ~Solver() {}

    static size_t get_num_guesses() { return slover_type::num_guesses; }
    static size_t get_num_unique_candidate() { return slover_type::num_unique_candidate; }
    static size_t get_num_failed_return() { return slover_type::num_failed_return; }

    static size_t get_search_counter() {
        return (slover_type::num_guesses + slover_type::num_unique_candidate + slover_type::num_failed_return);
    }

    static double get_guess_percent() {
        return calc_percent(slover_type::num_guesses, slover_type::get_search_counter());
    }

    static double get_failed_return_percent() {
        return calc_percent(slover_type::num_failed_return, slover_type::get_search_counter());
    }

    static double get_unique_candidate_percent() {
        return calc_percent(slover_type::num_unique_candidate, slover_type::get_search_counter());
    }

private:
    size_t calc_empties(char board[BoardSize]) {
        size_t empties = 0;
        for (size_t pos = 0; pos < BoardSize; pos++) {
            unsigned char val = board[pos];
            if (val == '.') {
                empties++;
            }
        }
        return empties;
    }

    void init_board(char board[BoardSize]) {
        init_literal_info();

        this->cell_filled_.reset();

        this->boxes_.set();
        this->rows_.set();
        this->cols_.set();

        this->cell_nums_.set();

        this->box_nums_.set();
        this->row_nums_.set();
        this->col_nums_.set();

        num_guesses = 0;
        num_unique_candidate = 0;
        num_failed_return = 0;
        if (kSearchMode > SEARCH_MODE_ONE_ANSWER) {
            this->answers_.clear();
        }

        size_t empties = calc_empties(board);
        this->effect_list_.resize(empties + 1);
        this->empties_ = empties;

        size_t pos = 0;
        for (size_t row = 0; row < Rows; row++) {
            size_t box_y = (row / BoxCellsY) * BoxCountX;
            size_t cell_y = (row % BoxCellsY) * BoxCellsX;
            for (size_t col = 0; col < Cols; col++) {
                unsigned char val = board[pos];
                if (val != '.') {
                    size_t box_x = col / BoxCellsX;
                    size_t box = box_y + box_x;
                    size_t cell_x = col % BoxCellsX;
                    size_t cell = cell_y + cell_x;
                    size_t num = val - '1';
                    this->fillNum(pos, row, col, box, cell, num);
                }
                pos++;
            }
        }
    }

    void setup_state(char board[BoardSize]) {
        size_t pos = 0;
        for (size_t row = 0; row < Rows; row++) {
            size_t box_y = (row / BoxCellsY) * BoxCountX;
            for (size_t col = 0; col < Cols; col++) {
                char val = board[pos];
                if (val == '.') {
                    // Get can fill numbers each cell.
                    size_t box_x = col / BoxCellsX;
                    size_t box = box_y + box_x;
                    bitset_type num_bits = getCanFillNums(row, col, box);
                    assert(this->cell_nums_[pos] == num_bits);
                    //this->cell_nums_[pos] = num_bits;
                }
                pos++;
            }
        }

        this->calc_literal_count();
    }

    void calc_literal_count() {
        size_t pos = 0;
        for (size_t row = 0; row < Rows; row++) {
            size_t box_y = (row / BoxCellsY) * BoxCountX;
            for (size_t col = 0; col < Cols; col++) {
                uint8_t cell_nums = (uint8_t)this->cell_nums_[pos].count();
                assert(get_cell_literal_cnt(pos) == cell_nums);
                //set_cell_literal_cnt(pos, cell_nums);
                pos++;

                size_t box_x = col / BoxCellsX;
                size_t box = box_y + box_x;
                for (size_t num = MinNumber - 1; num < MaxNumber; num++) {
                    uint8_t box_nums = (uint8_t)this->box_nums_[box][num].count();
                    uint8_t row_nums = (uint8_t)this->row_nums_[row][num].count();
                    uint8_t col_nums = (uint8_t)this->col_nums_[col][num].count();

                    assert(get_box_literal_cnt(box, num) == box_nums);
                    assert(get_row_literal_cnt(row, num) == row_nums);
                    assert(get_col_literal_cnt(col, num) == col_nums);

                    //set_box_literal_cnt(box, num, box_nums);
                    //set_row_literal_cnt(row, num, row_nums);
                    //set_col_literal_cnt(col, num, col_nums);
                }
            }
        }
    }

    static const bool kAllDimIsSame = ((Numbers == BoxSize) && (Rows == Cols) && (Numbers == Rows));

#if defined(__SSE4_1__)
    static const size_t kLiteralStep = sizeof(size_t) / sizeof(literal_info_t);

 #if defined(WIN64) || defined(_WIN64) || defined(_M_X64) || defined(_M_AMD64) \
  || defined(_M_IA64) || defined(__amd64__) || defined(__x86_64__)
    static const size_t kInitCellLiteral = Numbers | (Numbers << 16U) | (Numbers << 32U) | (Numbers << 48U);
    static const size_t kInitRowLiteral  = Cols | (Cols << 16U) | (Cols << 32U) | (Cols << 48U);
    static const size_t kInitColLiteral  = Rows | (Rows << 16U) | (Rows << 32U) | (Rows << 48U);
    static const size_t kInitBoxLiteral  = BoxSize | (BoxSize << 16U) | (BoxSize << 32U) | (BoxSize << 48U);
    static const size_t kInitLiteral     = kInitCellLiteral;
#else
    static const size_t kInitCellLiteral = Numbers | (Numbers << 16U);
    static const size_t kInitRowLiteral  = Cols | (Cols << 16U);
    static const size_t kInitColLiteral  = Rows | (Rows << 16U);
    static const size_t kInitBoxLiteral  = BoxSize | (BoxSize << 16U);
    static const size_t kInitLiteral     = kInitCellLiteral;
#endif

    inline void init_literal_info() {
        if (kAllDimIsSame)
            init_literal_info_is_same();
        else
            init_literal_info_not_same();
    }

    inline void init_literal_info_is_same() {
        static const size_t kLiteralLimit = (LiteralLast / kLiteralStep) * kLiteralStep;

        // Cell-Literal, Row-Literal, Col-Literal, Box-Literal
        size_t * pinfo = (size_t *)(&this->literal_info_[0]);
        assert((uintptr_t(pinfo) & 0x0FU) == 0);

        size_t * pinfo_limit = (size_t *)(&this->literal_info_[kLiteralLimit]);
        while (pinfo < pinfo_limit) {
            *pinfo = kInitLiteral;
            pinfo++;
        }
        for (size_t i = kLiteralLimit; i < LiteralLast; i++) {
            init_literal_info(i, (uint16_t)Numbers);
        }
    }

    inline void init_literal_info_not_same() {
        size_t * pinfo_first = (size_t *)(&this->literal_info_[0]);
        assert((uintptr_t(pinfo_first) & 0x0FU) == 0);

        size_t i, literalFront, literalLast;

        // Cell-Literal
        literalLast = (CellLiteralLast / kLiteralStep) * kLiteralStep;
        for (i = CellLiteralFirst; i < literalLast; i += kLiteralStep) {
            size_t * pinfo = (size_t *)&this->literal_info_[i];
            *pinfo = kInitCellLiteral;
        }
        for (; i < CellLiteralLast; i++) {
            init_literal_info(i, (uint16_t)Numbers);
        }

        // Row-Literal
        literalFront = RowLiteralFirst + (RowLiteralFirst % kLiteralStep);
        for (i = RowLiteralFirst; i < literalFront; i++) {
            init_literal_info(i, (uint16_t)Cols);
        }
        literalLast = (RowLiteralLast / kLiteralStep) * kLiteralStep;
        for (; i < literalLast; i += kLiteralStep) {
            size_t * pinfo = (size_t *)&this->literal_info_[i];
            *pinfo = kInitRowLiteral;
        }
        for (; i < RowLiteralLast; i++) {
            init_literal_info(i, (uint16_t)Cols);
        }

        // Col-Literal
        literalFront = ColLiteralFirst + (ColLiteralFirst % kLiteralStep);
        for (i = ColLiteralFirst; i < literalFront; i++) {
            init_literal_info(i, (uint16_t)Rows);
        }
        literalLast = (ColLiteralLast / kLiteralStep) * kLiteralStep;
        for (; i < literalLast; i += kLiteralStep) {
            size_t * pinfo = (size_t *)&this->literal_info_[i * kLiteralStep];
            *pinfo = kInitColLiteral;
        }
        for (; i < ColLiteralLast; i++) {
            init_literal_info(i, (uint16_t)Rows);
        }

        // Box-Literal
        literalFront = BoxLiteralFirst + (BoxLiteralFirst % kLiteralStep);
        for (i = BoxLiteralFirst; i < literalFront; i++) {
            init_literal_info(i, (uint16_t)BoxSize);
        }
        literalLast = (BoxLiteralLast / kLiteralStep) * kLiteralStep;
        for (; i < literalLast; i += kLiteralStep) {
            size_t * pinfo = (size_t *)&this->literal_info_[i];
            *pinfo = kInitBoxLiteral;
        }
        for (; i < BoxLiteralLast; i++) {
            init_literal_info(i, (uint16_t)BoxSize);
        }
    }

    inline void init_literal_info_normal() {
#if 0
        std::memset((void *)&this->literal_info_[0], 0, sizeof(this->literal_info_));
#endif
        for (size_t i = CellLiteralFirst; i < CellLiteralLast; i++) {
            init_literal_info(i, (uint16_t)Numbers);
        }
        for (size_t i = RowLiteralFirst; i < RowLiteralLast; i++) {
            init_literal_info(i, (uint16_t)Cols);
        }
        for (size_t i = ColLiteralFirst; i < ColLiteralLast; i++) {
            init_literal_info(i, (uint16_t)Rows);
        }
        for (size_t i = BoxLiteralFirst; i < BoxLiteralLast; i++) {
            init_literal_info(i, (uint16_t)BoxSize);
        }
    }

    inline void init_literal_info(size_t literal, uint16_t info) {
        uint16_t * pinfo = (uint16_t *)&this->literal_info_[literal];
        *pinfo = info;
    }

    inline void enable_literal(size_t literal) {
        this->literal_info_[literal].enable = 0x00;
    }

    inline void disable_literal(size_t literal) {
        this->literal_info_[literal].enable = 0xFF;
    }

    inline uint8_t get_literal_cnt(size_t literal) {
        return this->literal_info_[literal].count;
    }

    inline void set_literal_cnt(size_t literal, uint8_t count) {
        this->literal_info_[literal].count = count;
    }

    inline void inc_literal_cnt(size_t literal) {
        this->literal_info_[literal].count++;
        assert(this->literal_info_[literal].count <= Numbers);
    }

    inline void dec_literal_cnt(size_t literal) {
        assert(this->literal_info_[literal].count > 0);
        this->literal_info_[literal].count--;
    }

#else // !__SSE4_1__

    inline void init_literal_info() {
        if (kAllDimIsSame)
            init_literal_info_is_same();
        else
            init_literal_info_not_same();
    }

    inline void init_literal_info_is_same() {
        std::memset((void *)&this->literal_info_[0], kInitLiteral, sizeof(this->literal_info_));
        std::memset((void *)&this->literal_enable_[0], 0, sizeof(this->literal_enable_));
    }

    inline void init_literal_info_not_same() {
        for (size_t i = CellLiteralFirst; i < CellLiteralLast; i++) {
            init_literal_info(i, (uint8_t)Numbers);
        }
        for (size_t i = RowLiteralFirst; i < RowLiteralLast; i++) {
            init_literal_info(i, (uint8_t)Cols);
        }
        for (size_t i = ColLiteralFirst; i < ColLiteralLast; i++) {
            init_literal_info(i, (uint8_t)Rows);
        }
        for (size_t i = BoxLiteralFirst; i < BoxLiteralLast; i++) {
            init_literal_info(i, (uint8_t)BoardSize);
        }

        std::memset((void *)&this->literal_enable_[0], 0, sizeof(this->literal_enable_));
    }

    inline void init_literal_info(size_t literal, uint8_t count) {
        this->literal_count_[literal] = count;
    }

    inline void enable_literal(size_t literal) {
        this->literal_enable_[literal] = 0x00;
    }

    inline void disable_literal(size_t literal) {
        this->literal_enable_[literal] = 0xF0;
    }

    inline uint8_t get_literal_cnt(size_t literal) {
        return this->literal_count_[literal];
    }

    inline void set_literal_cnt(size_t literal, uint8_t count) {
        this->literal_count_[literal] = count;
    }

    inline void inc_literal_cnt(size_t literal) {
        this->literal_count_[literal]++;
        assert(this->literal_count_[literal] <= Numbers);
    }

    inline void dec_literal_cnt(int literal) {
        assert(this->literal_count_[literal] > 0);
        this->literal_count_[literal]--;
    }

#endif // __SSE4_1__

    // enable_xxxx_literal()
    inline void enable_cell_literal(size_t pos) {
        size_t literal = CellLiteralFirst + pos;
        this->enable_literal(literal);
    }

    inline void enable_box_literal(size_t box, size_t num) {
        size_t literal = BoxLiteralFirst + box * Numbers + num;
        this->enable_literal(literal);
    }

    inline void enable_row_literal(size_t row, size_t num) {
        size_t literal = RowLiteralFirst + row * Numbers + num;
        this->enable_literal(literal);
    }

    inline void enable_col_literal(size_t col, size_t num) {
        size_t literal = ColLiteralFirst + col * Numbers + num;
        this->enable_literal(literal);
    }

    // disable_xxxx_literal()
    inline void disable_cell_literal(size_t pos) {
        size_t literal = CellLiteralFirst + pos;
        this->disable_literal(literal);
    }

    inline void disable_box_literal(size_t box, size_t num) {
        size_t literal = BoxLiteralFirst + box * Numbers + num;
        this->disable_literal(literal);
    }

    inline void disable_row_literal(size_t row, size_t num) {
        size_t literal = RowLiteralFirst + row * Numbers + num;
        this->disable_literal(literal);
    }

    inline void disable_col_literal(size_t col, size_t num) {
        size_t literal = ColLiteralFirst + col * Numbers + num;
        this->disable_literal(literal);
    }

    // get_xxxx_literal_cnt()
    inline uint8_t get_cell_literal_cnt(size_t pos) {
        size_t literal = CellLiteralFirst + pos;
        return this->get_literal_cnt(literal);
    }

    inline uint8_t get_box_literal_cnt(size_t box, size_t num) {
        size_t literal = BoxLiteralFirst + box * Numbers + num;
        return this->get_literal_cnt(literal);
    }

    inline uint8_t get_row_literal_cnt(size_t row, size_t num) {
        size_t literal = RowLiteralFirst + row * Numbers + num;
        return this->get_literal_cnt(literal);
    }

    inline uint8_t get_col_literal_cnt(size_t col, size_t num) {
        size_t literal = ColLiteralFirst + col * Numbers + num;
        return this->get_literal_cnt(literal);
    }

    // set_xxxx_literal_cnt()
    inline void set_cell_literal_cnt(size_t pos, uint8_t count) {
        size_t literal = CellLiteralFirst + pos;
        this->set_literal_cnt(literal, count);
    }

    inline void set_box_literal_cnt(size_t index, uint8_t count) {
        size_t literal = BoxLiteralFirst + index;
        this->set_literal_cnt(literal, count);
    }

    inline void set_box_literal_cnt(size_t box, size_t num, uint8_t count) {
        size_t literal = BoxLiteralFirst + box * Numbers + num;
        this->set_literal_cnt(literal, count);
    }

    inline void set_row_literal_cnt(size_t index, uint8_t count) {
        size_t literal = RowLiteralFirst + index;
        this->set_literal_cnt(literal, count);
    }

    inline void set_row_literal_cnt(size_t row, size_t num, uint8_t count) {
        size_t literal = RowLiteralFirst + row * Numbers + num;
        this->set_literal_cnt(literal, count);
    }

    inline void set_col_literal_cnt(size_t index, uint8_t count) {
        size_t literal = ColLiteralFirst + index;
        this->set_literal_cnt(literal, count);
    }

    inline void set_col_literal_cnt(size_t col, size_t num, uint8_t count) {
        size_t literal = ColLiteralFirst + col * Numbers + num;
        this->set_literal_cnt(literal, count);
    }

    // inc_xxxx_literal_cnt()
    inline void inc_cell_literal_cnt(size_t pos) {
        size_t literal = CellLiteralFirst + pos;
        this->inc_literal_cnt(literal);
    }

    inline void inc_box_literal_cnt(size_t box, size_t num) {
        size_t literal = BoxLiteralFirst + box * Numbers + num;
        this->inc_literal_cnt(literal);
    }

    inline void inc_row_literal_cnt(size_t row, size_t num) {
        size_t literal = RowLiteralFirst + row * Numbers + num;
        this->inc_literal_cnt(literal);
    }

    inline void inc_col_literal_cnt(size_t col, size_t num) {
        size_t literal = ColLiteralFirst + col * Numbers + num;
        this->inc_literal_cnt(literal);
    }

    // dec_xxxx_literal_cnt()
    inline void dec_cell_literal_cnt(size_t pos) {
        size_t literal = CellLiteralFirst + pos;
        this->dec_literal_cnt(literal);
    }

    inline void dec_box_literal_cnt(size_t box, size_t num) {
        size_t literal = BoxLiteralFirst + box * Numbers + num;
        this->dec_literal_cnt(literal);
    }

    inline void dec_row_literal_cnt(size_t row, size_t num) {
        size_t literal = RowLiteralFirst + row * Numbers + num;
        this->dec_literal_cnt(literal);
    }

    inline void dec_col_literal_cnt(size_t col, size_t num) {
        size_t literal = ColLiteralFirst + col * Numbers + num;
        this->dec_literal_cnt(literal);
    }

#if defined(__SSE4_1__)
    int get_min_literal(int & out_min_literal_cnt) const {
        int min_literal_cnt = 254;
        int min_literal_id = -1;
        for (int i = 0; i < TotalLiterals; i++) {
            if (literal_info_[i].enable == 0) {
                int literal_cnt = literal_info_[i].count;
                if (literal_cnt < min_literal_cnt) {
                    assert(literal_cnt >= 0);
                    if (literal_cnt <= kLiteralCntThreshold) {
                        out_min_literal_cnt = literal_cnt;
                        return i;
                    }
                    min_literal_cnt = literal_cnt;
                    min_literal_id = i;
                }
            }
        }
        out_min_literal_cnt = min_literal_cnt;
        return min_literal_id;
    }

    int get_min_literal_simd(int & out_min_literal_cnt) {
        int min_literal_cnt = 254;
        int min_literal_id = -1;
        int index_base = 0;

        const char * pinfo     = (const char *)&literal_info_[0];
        const char * pinfo_end = (const char *)&literal_info_[TotalLiterals];
        while ((pinfo_end - pinfo) >= 64) {
            __m128i xmm0 = _mm_load_si128((const __m128i *)(pinfo + 0));
            __m128i xmm1 = _mm_load_si128((const __m128i *)(pinfo + 16));
            __m128i xmm2 = _mm_load_si128((const __m128i *)(pinfo + 32));
            __m128i xmm3 = _mm_load_si128((const __m128i *)(pinfo + 48));

            __m128i xmm4 = _mm_minpos_epu16(xmm0);      // SSE 4.1
            __m128i xmm5 = _mm_minpos_epu16(xmm1);      // SSE 4.1
            __m128i xmm6 = _mm_minpos_epu16(xmm2);      // SSE 4.1
            __m128i xmm7 = _mm_minpos_epu16(xmm3);      // SSE 4.1

            __m128i xmm5_ = _mm_slli_epi64(xmm5, 32);
            __m128i xmm7_ = _mm_slli_epi64(xmm7, 32);
            __m128i xmm4_5  = _mm_blend_epi16(xmm4, xmm5_, 0b00001100); // SSE 4.1
            __m128i xmm6_7  = _mm_blend_epi16(xmm6, xmm7_, 0b00001100); // SSE 4.1
            __m128i xmm6_7_ = _mm_slli_si128(xmm6_7, 8);
            __m128i comb_0  = _mm_or_si128(xmm4_5, xmm6_7_);
            __m128i __literal_index = comb_0;

            __m128i kLiteralCntMask = _mm_set1_epi32((int)0xFFFF0000L);
            __m128i __literal_cnt = _mm_or_si128(comb_0, kLiteralCntMask);
            __m128i __min_literal_cnt = _mm_minpos_epu16(__literal_cnt);      // SSE 4.1

            uint32_t min_literal_cnt32 = (uint32_t)_mm_cvtsi128_si32(__min_literal_cnt);
            int min_literal_cnt16 = (int)(min_literal_cnt32 & 0x0000FFFFULL);
            if (min_literal_cnt16 < min_literal_cnt) {
                min_literal_cnt = min_literal_cnt16;

                uint32_t min_block_index16 = min_literal_cnt32 >> 17U;
                __m128i __min_literal_id_sr15 = _mm_srli_epi64(__min_literal_cnt, 15);

                __m128i __literal_index_sr16 = _mm_srli_epi32(__literal_index, 16);

                // SSSE3
                __m128i __min_literal_id = _mm_shuffle_epi8(__literal_index_sr16, __min_literal_id_sr15);
                uint32_t min_literal_id32 = (uint32_t)_mm_cvtsi128_si32(__min_literal_id);
                int min_literal_offset = (int)(min_literal_id32 & 0x000000FFUL);
                min_literal_id = index_base + min_block_index16 * 8 + min_literal_offset;

                if (min_literal_cnt <= kLiteralCntThreshold) {
                    out_min_literal_cnt = min_literal_cnt;
                    return min_literal_id;
                }
            }
            index_base += 32;
            pinfo += 64;
        }

        if ((pinfo_end - pinfo) >= 32) {
            __m128i xmm0 = _mm_load_si128((const __m128i *)(pinfo + 0));
            __m128i xmm1 = _mm_load_si128((const __m128i *)(pinfo + 16));

            __m128i xmm2 = _mm_minpos_epu16(xmm0);      // SSE 4.1
            __m128i xmm3 = _mm_minpos_epu16(xmm1);      // SSE 4.1

            __m128i xmm3_ = _mm_slli_epi64(xmm3, 32);
            __m128i comb_0 = _mm_blend_epi16(xmm2, xmm3_, 0b00001100);  // SSE 4.1
            __m128i __literal_index = comb_0;

            __m128i kLiteralCntMask = _mm_set_epi32(0xFFFFFFFFL, 0xFFFFFFFFL, 0xFFFF0000L, 0xFFFF0000L);
            __m128i __literal_cnt = _mm_or_si128(comb_0, kLiteralCntMask);
            __m128i __min_literal_cnt = _mm_minpos_epu16(__literal_cnt);      // SSE 4.1

            uint32_t min_literal_cnt32 = (uint32_t)_mm_cvtsi128_si32(__min_literal_cnt);
            int min_literal_cnt16 = (int)(min_literal_cnt32 & 0x0000FFFFULL);
            if (min_literal_cnt16 < min_literal_cnt) {
                min_literal_cnt = min_literal_cnt16;

                uint32_t min_block_index16 = min_literal_cnt32 >> 17U;
                __m128i __min_literal_id_sr15 = _mm_srli_epi64(__min_literal_cnt, 15);

                __m128i __literal_index_sr16 = _mm_srli_epi32(__literal_index, 16);

                // SSSE3
                __m128i __min_literal_id = _mm_shuffle_epi8(__literal_index_sr16, __min_literal_id_sr15);
                uint32_t min_literal_id32 = (uint32_t)_mm_cvtsi128_si32(__min_literal_id);
                int min_literal_offset = (int)(min_literal_id32 & 0x000000FFUL);
                min_literal_id = index_base + min_block_index16 * 8 + min_literal_offset;

                if (min_literal_cnt <= kLiteralCntThreshold) {
                    out_min_literal_cnt = min_literal_cnt;
                    return min_literal_id;
                }
            }
            index_base += 16;
            pinfo += 32;
        }

        if ((pinfo_end - pinfo) >= 16) {
            __m128i xmm0 = _mm_load_si128((const __m128i *)(pinfo + 0));
            __m128i __min_literal_cnt = _mm_minpos_epu16(xmm0);    // SSE 4.1

            uint32_t min_literal_cnt32 = (uint32_t)_mm_cvtsi128_si32(__min_literal_cnt);
            int min_literal_cnt16 = (int)(min_literal_cnt32 & 0x0000FFFFULL);
            if (min_literal_cnt16 < min_literal_cnt) {
                min_literal_cnt = min_literal_cnt16;

                uint32_t min_literal_offset = min_literal_cnt32 >> 17U;
                min_literal_id = index_base + min_literal_offset;

                if (min_literal_cnt <= kLiteralCntThreshold) {
                    out_min_literal_cnt = min_literal_cnt;
                    return min_literal_id;
                }
            }
            index_base += 8;
            pinfo += 16;
        }

        // Last remain items (less than 8 items)
        while (pinfo < pinfo_end) {
            literal_info_t * pliteral_info = (literal_info_t *)pinfo;
            if (pliteral_info->enable == 0) {
                int literal_cnt = pliteral_info->count;
                if (literal_cnt < min_literal_cnt) {
                    if (literal_cnt <= kLiteralCntThreshold) {
                        out_min_literal_cnt = literal_cnt;
                        return index_base;
                    }
                    min_literal_cnt = literal_cnt;
                    min_literal_id = index_base;
                }
            }
            index_base++;
            pinfo += 2;
        }

        out_min_literal_cnt = min_literal_cnt;
        return min_literal_id;
    }

#elif defined(__SSE2__)

    int get_min_literal(int & out_min_literal_cnt) const {
        int min_literal_cnt = 254;
        int min_literal_id = -1;
        for (int i = 0; i < TotalLiterals; i++) {
            if (literal_enable_[i] == 0) {
                int literal_cnt = literal_count_[i];
                if (literal_cnt < min_literal_cnt) {
                    assert(literal_cnt >= 0);
                    if (literal_cnt <= 1) {
                        if (literal_cnt == 0) {
                            out_min_literal_cnt = 0;
                            return i;
                        }
                        else {
                            out_min_literal_cnt = 1;
                            return i;
                        }
                    }
                    min_literal_cnt = literal_cnt;
                    min_literal_id = i;
                }
            }
        }
        out_min_literal_cnt = min_literal_cnt;
        return min_literal_id;
    }

    //
    // Horizontal minimum and maximum using SSE
    // See: https://stackoverflow.com/questions/22256525/horizontal-minimum-and-maximum-using-sse
    //
    int get_min_literal_simd(int & out_min_literal_cnt) {
        int min_literal_cnt = 254;
        int min_literal_id = 0;
        int index_base = 0;

        const char * pcount     = (const char *)&literal_count_[0];
        const char * pcount_end = (const char *)&literal_count_[TotalLiterals];
        const char * penable    = (const char *)&literal_enable_[0];
        while ((pcount_end - pcount) >= 64) {
            __m128i xmm0 = _mm_load_si128((const __m128i *)(pcount + 0));
            __m128i xmm1 = _mm_load_si128((const __m128i *)(pcount + 16));

            __m128i xmm2 = _mm_load_si128((const __m128i *)(penable + 0));
            __m128i xmm3 = _mm_load_si128((const __m128i *)(penable + 16));

            xmm0 = _mm_or_si128(xmm0, xmm2);
            xmm1 = _mm_or_si128(xmm1, xmm3);

            xmm0 = _mm_min_epu8(xmm0, _mm_shuffle_epi32(xmm0, _MM_SHUFFLE(3, 2, 3, 2)));
            xmm1 = _mm_min_epu8(xmm1, _mm_shuffle_epi32(xmm1, _MM_SHUFFLE(3, 2, 3, 2)));

            xmm0 = _mm_min_epu8(xmm0, _mm_shuffle_epi32(xmm0, _MM_SHUFFLE(1, 1, 1, 1)));
            xmm1 = _mm_min_epu8(xmm1, _mm_shuffle_epi32(xmm1, _MM_SHUFFLE(1, 1, 1, 1)));

            xmm0 = _mm_min_epu8(xmm0, _mm_shufflelo_epi16(xmm0, _MM_SHUFFLE(1, 1, 1, 1)));
            xmm1 = _mm_min_epu8(xmm1, _mm_shufflelo_epi16(xmm1, _MM_SHUFFLE(1, 1, 1, 1)));

            xmm0 = _mm_min_epu8(xmm0, _mm_srli_epi16(xmm0, 8));
            xmm1 = _mm_min_epu8(xmm1, _mm_srli_epi16(xmm1, 8));

            xmm0 = _mm_min_epu8(xmm0, xmm1);

            __m128i xmm4 = _mm_load_si128((const __m128i *)(pcount + 32));
            __m128i xmm5 = _mm_load_si128((const __m128i *)(pcount + 48));

            __m128i xmm6 = _mm_load_si128((const __m128i *)(penable + 32));
            __m128i xmm7 = _mm_load_si128((const __m128i *)(penable + 48));

            xmm4 = _mm_or_si128(xmm4, xmm6);
            xmm5 = _mm_or_si128(xmm5, xmm7);

            xmm4 = _mm_min_epu8(xmm4, _mm_shuffle_epi32(xmm4, _MM_SHUFFLE(3, 2, 3, 2)));
            xmm5 = _mm_min_epu8(xmm5, _mm_shuffle_epi32(xmm5, _MM_SHUFFLE(3, 2, 3, 2)));

            xmm4 = _mm_min_epu8(xmm4, _mm_shuffle_epi32(xmm4, _MM_SHUFFLE(1, 1, 1, 1)));
            xmm5 = _mm_min_epu8(xmm5, _mm_shuffle_epi32(xmm5, _MM_SHUFFLE(1, 1, 1, 1)));

            xmm4 = _mm_min_epu8(xmm4, _mm_shufflelo_epi16(xmm4, _MM_SHUFFLE(1, 1, 1, 1)));
            xmm5 = _mm_min_epu8(xmm5, _mm_shufflelo_epi16(xmm5, _MM_SHUFFLE(1, 1, 1, 1)));

            xmm4 = _mm_min_epu8(xmm4, _mm_srli_epi16(xmm4, 8));
            xmm5 = _mm_min_epu8(xmm5, _mm_srli_epi16(xmm5, 8));

            xmm4 = _mm_min_epu8(xmm4, xmm5);

            // The minimum literal count of per 64 numbers
            __m128i min_count_64 = _mm_min_epu8(xmm0, xmm4);

            int min_literal_cnt16 = _mm_cvtsi128_si32(min_count_64) & 0x000000FFL;
            if (min_literal_cnt16 < min_literal_cnt) {
                min_literal_cnt = min_literal_cnt16;

                __m128i xmm0 = _mm_load_si128((const __m128i *)(pcount + 0));
                __m128i xmm2 = _mm_load_si128((const __m128i *)(penable + 0));

                __m128i min_cmp = _mm_set1_epi8((char)min_literal_cnt);

                xmm0 = _mm_or_si128(xmm0, xmm2);
                xmm0 = _mm_cmpeq_epi8(xmm0, min_cmp);

                int equal_mask = _mm_movemask_epi8(xmm0);
                if (equal_mask == 0) {
                    __m128i xmm1 = _mm_load_si128((const __m128i *)(pcount + 16));
                    __m128i xmm3 = _mm_load_si128((const __m128i *)(penable + 16));
                
                    xmm1 = _mm_or_si128(xmm1, xmm3);  
                    xmm1 = _mm_cmpeq_epi8(xmm1, min_cmp);

                    equal_mask = _mm_movemask_epi8(xmm1);
                    if (equal_mask == 0) {
                        __m128i xmm4 = _mm_load_si128((const __m128i *)(pcount + 32));
                        __m128i xmm6 = _mm_load_si128((const __m128i *)(penable + 32));
                
                        xmm4 = _mm_or_si128(xmm4, xmm6);  
                        xmm4 = _mm_cmpeq_epi8(xmm4, min_cmp);

                        equal_mask = _mm_movemask_epi8(xmm4);
                        if (equal_mask == 0) {
                            __m128i xmm5 = _mm_load_si128((const __m128i *)(pcount + 48));
                            __m128i xmm7 = _mm_load_si128((const __m128i *)(penable + 48));
                
                            xmm5 = _mm_or_si128(xmm5, xmm7);  
                            xmm5 = _mm_cmpeq_epi8(xmm5, min_cmp);

                            equal_mask = _mm_movemask_epi8(xmm5);
                            if (equal_mask == 0) {
                                assert(false);
                            }
                            else {
                                int min_literal_offset = BitUtils::bsf(equal_mask);
                                min_literal_id = index_base + 3 * 16 + min_literal_offset;
                            }
                        }
                        else {
                            int min_literal_offset = BitUtils::bsf(equal_mask);
                            min_literal_id = index_base + 2 * 16 + min_literal_offset;
                        }
                    }
                    else {
                        int min_literal_offset = BitUtils::bsf(equal_mask);
                        min_literal_id = index_base + 1 * 16 + min_literal_offset;
                    }
                }
                else {
                    int min_literal_offset = BitUtils::bsf(equal_mask);
                    min_literal_id = index_base + 0 * 16 + min_literal_offset;
                }

                if (min_literal_cnt == 0) {
                    out_min_literal_cnt = 0;
                    return min_literal_id;
                }
#if 0
                else if (min_literal_cnt == 1) {
                    out_min_literal_cnt = 1;
                    return min_literal_id;
                }
#endif
            }

            index_base += 64;
            penable += 64;
            pcount += 64;
        }

        if ((pcount_end - pcount) >= 32) {
            __m128i xmm0 = _mm_load_si128((const __m128i *)(pcount + 0));
            __m128i xmm1 = _mm_load_si128((const __m128i *)(pcount + 16));

            __m128i xmm2 = _mm_load_si128((const __m128i *)(penable + 0));
            __m128i xmm3 = _mm_load_si128((const __m128i *)(penable + 16));

            xmm0 = _mm_or_si128(xmm0, xmm2);
            xmm1 = _mm_or_si128(xmm1, xmm3);

            xmm0 = _mm_min_epu8(xmm0, _mm_shuffle_epi32(xmm0, _MM_SHUFFLE(3, 2, 3, 2)));
            xmm1 = _mm_min_epu8(xmm1, _mm_shuffle_epi32(xmm1, _MM_SHUFFLE(3, 2, 3, 2)));

            xmm0 = _mm_min_epu8(xmm0, _mm_shuffle_epi32(xmm0, _MM_SHUFFLE(1, 1, 1, 1)));
            xmm1 = _mm_min_epu8(xmm1, _mm_shuffle_epi32(xmm1, _MM_SHUFFLE(1, 1, 1, 1)));

            xmm0 = _mm_min_epu8(xmm0, _mm_shufflelo_epi16(xmm0, _MM_SHUFFLE(1, 1, 1, 1)));
            xmm1 = _mm_min_epu8(xmm1, _mm_shufflelo_epi16(xmm1, _MM_SHUFFLE(1, 1, 1, 1)));

            xmm0 = _mm_min_epu8(xmm0, _mm_srli_epi16(xmm0, 8));
            xmm1 = _mm_min_epu8(xmm1, _mm_srli_epi16(xmm1, 8));

            // The minimum literal count of per 32 numbers
            __m128i min_count_32 = _mm_min_epu8(xmm0, xmm1);

            int min_literal_cnt16 = _mm_cvtsi128_si32(min_count_32) & 0x000000FFL;
            if (min_literal_cnt16 < min_literal_cnt) {
                min_literal_cnt = min_literal_cnt16;

                __m128i xmm0 = _mm_load_si128((const __m128i *)(pcount + 0));
                __m128i xmm2 = _mm_load_si128((const __m128i *)(penable + 0));

                __m128i min_cmp = _mm_set1_epi8((char)min_literal_cnt);

                xmm0 = _mm_or_si128(xmm0, xmm2);
                xmm0 = _mm_cmpeq_epi8(xmm0, min_cmp);

                int equal_mask = _mm_movemask_epi8(xmm0);
                if (equal_mask == 0) {
                    __m128i xmm1 = _mm_load_si128((const __m128i *)(pcount + 16));
                    __m128i xmm3 = _mm_load_si128((const __m128i *)(penable + 16));
                
                    xmm1 = _mm_or_si128(xmm1, xmm3);  
                    xmm1 = _mm_cmpeq_epi8(xmm1, min_cmp);

                    equal_mask = _mm_movemask_epi8(xmm1);
                    if (equal_mask == 0) {
                        assert(false);
                    }
                    else {
                        int min_literal_offset = BitUtils::bsf(equal_mask);
                        min_literal_id = index_base + 1 * 16 + min_literal_offset;
                    }
                }
                else {
                    int min_literal_offset = BitUtils::bsf(equal_mask);
                    min_literal_id = index_base + 0 * 16 + min_literal_offset;
                }

                if (min_literal_cnt == 0) {
                    out_min_literal_cnt = 0;
                    return min_literal_id;
                }
#if 0
                else if (min_literal_cnt == 1) {
                    out_min_literal_cnt = 1;
                    return min_literal_id;
                }
#endif
            }

            index_base += 32;
            penable += 32;
            pcount += 32;
        }

        if ((pcount_end - pcount) >= 16) {
            __m128i xmm0 = _mm_load_si128((const __m128i *)(pcount + 0));
            __m128i xmm2 = _mm_load_si128((const __m128i *)(penable + 0));

            xmm0 = _mm_or_si128(xmm0, xmm2);
            xmm0 = _mm_min_epu8(xmm0, _mm_shuffle_epi32(xmm0, _MM_SHUFFLE(3, 2, 3, 2)));
            xmm0 = _mm_min_epu8(xmm0, _mm_shuffle_epi32(xmm0, _MM_SHUFFLE(1, 1, 1, 1)));
            xmm0 = _mm_min_epu8(xmm0, _mm_shufflelo_epi16(xmm0, _MM_SHUFFLE(1, 1, 1, 1)));

            // The minimum literal count of per 16 numbers
            __m128i min_count_16 = _mm_min_epu8(xmm0, _mm_srli_epi16(xmm0, 8));

            int min_literal_cnt16 = _mm_cvtsi128_si32(min_count_16) & 0x000000FFL;
            if (min_literal_cnt16 < min_literal_cnt) {
                min_literal_cnt = min_literal_cnt16;

                __m128i xmm0 = _mm_load_si128((const __m128i *)(pcount + 0));
                __m128i xmm2 = _mm_load_si128((const __m128i *)(penable + 0));

                __m128i min_cmp = _mm_set1_epi8((char)min_literal_cnt);

                xmm0 = _mm_or_si128(xmm0, xmm2);
                xmm0 = _mm_cmpeq_epi8(xmm0, min_cmp);

                int equal_mask = _mm_movemask_epi8(xmm0);
                if (equal_mask == 0) {
                    assert(false);
                }
                else {
                    int min_literal_offset = BitUtils::bsf(equal_mask);
                    min_literal_id = index_base + min_literal_offset;
                }

                if (min_literal_cnt == 0) {
                    out_min_literal_cnt = 0;
                    return min_literal_id;
                }
#if 0
                else if (min_literal_cnt == 1) {
                    out_min_literal_cnt = 1;
                    return min_literal_id;
                }
#endif
            }

            index_base += 16;
            penable += 16;
            pcount += 16;
        }

        // Last remain items (less than 16 items)
        while (pcount < pcount_end) {
            uint8_t * pcol_enable = (uint8_t *)penable;
            if (*pcol_enable == 0) {
                int literal_cnt = *pcount;
                if (literal_cnt < min_literal_cnt) {
                    if (literal_cnt == 0) {
                        out_min_literal_cnt = 0;
                        return index_base;
                    }
                    min_literal_cnt = literal_cnt;
                    min_literal_id = index_base;
                }
            }
            index_base++;
            penable++;
            pcount++;
        }

        out_min_literal_cnt = min_literal_cnt;
        return min_literal_id;
    }

#else

    int get_min_literal(int & out_min_literal_cnt) const {
        int min_literal_cnt = 254;
        int min_literal_id = -1;
        for (int i = 0; i < TotalLiterals; i++) {
            if (literal_enable_[i] == 0) {
                int literal_cnt = literal_count_[i];
                if (literal_cnt < min_literal_cnt) {
                    assert(literal_cnt >= 0);
                    if (literal_cnt <= 1) {
                        if (literal_cnt == 0) {
                            out_min_literal_cnt = 0;
                            return i;
                        }
                        else {
                            out_min_literal_cnt = 1;
                            return i;
                        }
                    }
                    min_literal_cnt = literal_cnt;
                    min_literal_id = i;
                }
            }
        }
        out_min_literal_cnt = min_literal_cnt;
        return min_literal_id;
    }

#endif // __SSE4_1__

    inline bitset_type getCanFillNums(size_t row, size_t col, size_t box) {
        return (this->rows_[row] & this->cols_[col] & this->boxes_[box]);
    }

    inline size_t fillNum(size_t pos, size_t row, size_t col,
                          size_t box, size_t cell, size_t num) {
        assert(!this->cell_filled_.test(pos));
        assert(this->cell_nums_[pos].test(num));
        bitset_type bits = getCanFillNums(row, col, box);
        assert(this->cell_nums_[pos] == bits);

        assert(this->boxes_[box].test(num));
        assert(this->rows_[row].test(num));
        assert(this->cols_[col].test(num));

        assert(this->box_nums_[box][num].test(cell));
        assert(this->row_nums_[row][num].test(col));
        assert(this->col_nums_[col][num].test(row));

        uint32_t n_num_bit = 1u << num;
        this->boxes_[box] ^= n_num_bit;
        this->rows_[row] ^= n_num_bit;
        this->cols_[col] ^= n_num_bit;

        this->cell_filled_.set(pos);

        disable_cell_literal(pos);
        disable_box_literal(box, num);
        disable_row_literal(row, num);
        disable_col_literal(col, num);

        size_t num_bits = this->cell_nums_[pos].to_ulong();
        while (num_bits != 0) {
            size_t num_bit = BitUtils::ms1b(num_bits);
            size_t _num = BitUtils::bsf(num_bit);

            assert(this->cell_nums_[pos].test(_num));
            assert(this->box_nums_[box][_num].test(cell));
            assert(this->row_nums_[row][_num].test(col));
            assert(this->col_nums_[col][_num].test(row));

            this->cell_nums_[pos].reset(_num);
            this->box_nums_[box][_num].reset(cell);
            this->row_nums_[row][_num].reset(col);
            this->col_nums_[col][_num].reset(row);

            dec_cell_literal_cnt(pos);
            dec_box_literal_cnt(box, _num);
            dec_row_literal_cnt(row, _num);
            dec_col_literal_cnt(col, _num);

            num_bits ^= num_bit;
        }

        size_t effect_count = updateNeighborCellsEffect(0, pos, num);
        return effect_count;
    }

    void remove_box_other_numbers(size_t box, size_t cell) {
        size_t row = 0, col = 0;

        size_t box_bits = this->boxes_[box].value_sz();
        while (box_bits != 0) {
            size_t num_bit = BitUtils::ms1b(box_bits);
            size_t _num = BitUtils::bsf(num_bit);
            if (this->box_nums_[box][_num].test(cell)) {
                this->box_nums_[box][_num].reset(cell);
                dec_box_literal_cnt(box, _num);
            }
            box_bits ^= num_bit;
        }

        size_t row_bits = this->rows_[row].value_sz();
        while (row_bits != 0) {
            size_t num_bit = BitUtils::ms1b(row_bits);
            size_t _num = BitUtils::bsf(num_bit);
            if (this->row_nums_[row][_num].test(col)) {
                this->row_nums_[row][_num].reset(col);
                dec_row_literal_cnt(row, _num);
            }
            row_bits ^= num_bit;
        }

        size_t col_bits = this->cols_[col].value_sz();
        while (col_bits != 0) {
            size_t num_bit = BitUtils::ms1b(col_bits);
            size_t _num = BitUtils::bsf(num_bit);
            if (this->col_nums_[col][_num].test(row)) {
                this->col_nums_[col][_num].reset(row);
                dec_col_literal_cnt(col, _num);
            }
            col_bits ^= num_bit;
        }
    }

    inline size_t doFillNum(size_t empties, size_t pos, size_t row, size_t col,
                            size_t box, size_t cell, size_t num,
                            bitset_type & save_bits) {
        assert(this->cell_nums_[pos].test(num));
        //bitset_type bits = getCanFillNums(row, col, box);
        //assert(this->cell_nums_[pos] == bits);

        assert(this->box_nums_[box][num].test(cell));
        assert(this->row_nums_[row][num].test(col));
        //assert(this->col_nums_[col][num].test(row));

        disable_cell_literal(pos);
        disable_box_literal(box, num);
        disable_row_literal(row, num);
        disable_col_literal(col, num);

        assert(this->box_nums_[box][num].test(cell));
        assert(this->row_nums_[row][num].test(col));
        //assert(this->col_nums_[col][num].test(row));

        this->box_nums_[box][num].reset(cell);
        this->row_nums_[row][num].reset(col);
        this->col_nums_[col][num].reset(row);

        // Save cell num bits
        save_bits = this->cell_nums_[pos];
        this->cell_nums_[pos].reset();

        size_t num_bits = save_bits.to_ulong();
        // Exclude the current number, because it has been processed.
        uint32_t n_num_bit = 1u << num;
        num_bits ^= (size_t)n_num_bit;
        while (num_bits != 0) {
            size_t num_bit = BitUtils::ms1b(num_bits);
            size_t _num = BitUtils::bsf(num_bit);

            assert(this->box_nums_[box][_num].test(cell));
            assert(this->row_nums_[row][_num].test(col));
            assert(this->col_nums_[col][_num].test(row));

            this->box_nums_[box][_num].reset(cell);
            this->row_nums_[row][_num].reset(col);
            this->col_nums_[col][_num].reset(row);

            dec_box_literal_cnt(box, _num);
            dec_row_literal_cnt(row, _num);
            dec_col_literal_cnt(col, _num);

            num_bits ^= num_bit;
        }

        size_t effect_count = updateNeighborCellsEffect(empties, pos, num);
        return effect_count;
    }

    inline void undoFillNum(size_t empties, size_t effect_count,
                            size_t pos, size_t row, size_t col,
                            size_t box, size_t cell, size_t num,
                            bitset_type & save_bits) {
        enable_cell_literal(pos);
        enable_box_literal(box, num);
        enable_row_literal(row, num);
        enable_col_literal(col, num);

        // Restore cell num bits
        this->cell_nums_[pos] = save_bits;

        this->box_nums_[box][num].set(cell);
        this->row_nums_[row][num].set(col);
        this->col_nums_[col][num].set(row);

        // Exclude the current number, because it has been processed.
        save_bits.reset(num);

        size_t num_bits = save_bits.to_ulong();
        while (num_bits != 0) {
            size_t num_bit = BitUtils::ms1b(num_bits);
            size_t _num = BitUtils::bsf(num_bit);

            this->boxes_[box].set(_num);
            this->rows_[row].set(_num);
            this->cols_[col].set(_num);

            this->box_nums_[box][_num].set(cell);
            this->row_nums_[row][_num].set(col);
            this->col_nums_[col][_num].set(row);

            inc_box_literal_cnt(box, _num);
            inc_row_literal_cnt(row, _num);
            inc_col_literal_cnt(col, _num);

            num_bits ^= num_bit;
        }

        restoreNeighborCellsEffect(empties, effect_count, num);
    }

    inline size_t updateNeighborCellsEffect(size_t empties, size_t in_pos, size_t num) {
        EffectList & effect_list = this->effect_list_[empties];
        size_t count = 0;
        const NeighborCells & cellList = SudokuTy::neighbor_cells[in_pos];
        for (size_t index = 0; index < Neighbors; index++) {
            size_t pos = cellList.cells[index];
            if (this->cell_nums_[pos].test(num)) {
                this->cell_nums_[pos].reset(num);
                dec_cell_literal_cnt(pos);

                effect_list.cells[count++] = (uint8_t)pos;

                const CellInfo & cellInfo = SudokuTy::cell_info[pos];

                size_t box = cellInfo.box;
                size_t cell = cellInfo.cell;
                size_t row = cellInfo.row;
                size_t col = cellInfo.col;

                assert(this->box_nums_[box][num].test(cell));
                assert(this->row_nums_[row][num].test(col));
                assert(this->col_nums_[col][num].test(row));

                this->box_nums_[box][num].reset(cell);
                this->row_nums_[row][num].reset(col);
                this->col_nums_[col][num].reset(row);

                dec_box_literal_cnt(box, num);
                dec_row_literal_cnt(row, num);                
                dec_col_literal_cnt(col, num);
            }
        }
        return count;
    }

    inline void restoreNeighborCellsEffect(size_t empties,
                                           size_t effect_count,
                                           size_t num) {
        const EffectList & effect_list = this->effect_list_[empties];
        for (size_t index = 0; index < effect_count; index++) {
            size_t pos = effect_list.cells[index];
            this->cell_nums_[pos].set(num);
            inc_cell_literal_cnt(pos);

            const CellInfo & cellInfo = SudokuTy::cell_info[pos];

            size_t box = cellInfo.box;
            size_t cell = cellInfo.cell;
            size_t row = cellInfo.row;
            size_t col = cellInfo.col;

            this->boxes_[box].set(num);
            this->rows_[row].set(num);
            this->cols_[col].set(num);

            this->box_nums_[box][num].set(cell);
            inc_box_literal_cnt(box, num);

            this->row_nums_[row][num].set(col);
            inc_row_literal_cnt(row, num);

            this->col_nums_[col][num].set(row);
            inc_col_literal_cnt(col, num);
        }
    }

public:
    bool solve(char board[BoardSize], size_t empties) {
        if (empties == 0) {
            if (kSearchMode > SearchMode::OneAnswer) {
                SudokuBoard<BoardSize> answer;
                std::memcpy((void *)&answer.board[0], (const void *)&board[0], BoardSize * sizeof(char));
                this->answers_.push_back(answer);
                if (kSearchMode == SearchMode::MoreThanOneAnswer) {
                    if (this->answers_.size() > 1)
                        return true;
                }
            }
            else {
                return true;
            }
        }
      
        int min_literal_cnt;
#if (defined(__SSE2__) || defined(__SSE4_1__))
        int min_literal_id = get_min_literal_simd(min_literal_cnt);
#else
        int min_literal_id = get_min_literal(min_literal_cnt);
#endif
        assert(min_literal_id < TotalLiterals);

        if (min_literal_cnt > 0) {
            if (min_literal_cnt == 1)
                num_unique_candidate++;
            else
                num_guesses++;

            bitset_type save_bits;
            size_t pos, row, col, box, cell, num;

            int literal_type = min_literal_id / BoardSize;
            assert(literal_type < LiteralType::MaxLiteralType);
            switch (literal_type) {
                case LiteralType::CellNums:
                {
                    pos = (size_t)min_literal_id - CellLiteralFirst;
                    assert(min_literal_id >= CellLiteralFirst);
                    assert(pos < Rows * Cols);
#if 0
                    row = pos / Cols;
                    col = pos % Cols;
                    size_t box_x = col / BoxCellsX;
                    size_t box_y = row / BoxCellsY;
                    box = box_y * BoxCountX + box_x;
                    size_t cell_x = col % BoxCellsX;
                    size_t cell_y = row % BoxCellsY;
                    cell = cell_y * BoxCellsX + cell_x;
#else
                    const CellInfo & cellInfo = SudokuTy::cell_info[pos];
                    row = cellInfo.row;
                    col = cellInfo.col;
                    box = cellInfo.box;
                    cell = cellInfo.cell;
#endif
                    disable_cell_literal(pos);

                    size_t num_bits = this->cell_nums_[pos].to_ulong();
                    assert(this->cell_nums_[pos].count() == get_literal_cnt(min_literal_id));
                    while (num_bits != 0) {
                        size_t num_bit = BitUtils::ms1b(num_bits);
                        num = BitUtils::bsf(num_bit);

                        //this->cell_nums_[pos].reset(num);
                        size_t effect_count = doFillNum(empties, pos, row, col,
                                                        box, cell, num, save_bits);

                        board[pos] = (char)(num + '1');

                        if (this->solve(board, empties - 1)) {
                            if (kSearchMode == SearchMode::OneAnswer) {
                                return true;
                            }
                            else if (kSearchMode == SearchMode::MoreThanOneAnswer) {
                                if (this->answers_.size() > 1)
                                    return true;
                            }
                        }

                        //this->cell_nums_[pos].set(num);
                        undoFillNum(empties, effect_count, pos, row, col,
                                    box, cell, num, save_bits);

                        num_bits ^= num_bit;
                    }

                    enable_cell_literal(pos);
                    break;
                }

                case LiteralType::BoxNums:
                {
                    size_t literal = (size_t)min_literal_id - BoxLiteralFirst;
                    assert(min_literal_id >= BoxLiteralFirst);
                    assert(literal < Boxes * Numbers);
                    box = literal / Numbers;
                    num = literal % Numbers;

                    assert(this->boxes_[box].test(num));

                    disable_box_literal(box, num);

                    size_t cell_bits = this->box_nums_[box][num].to_ulong();
                    assert(this->box_nums_[box][num].count() == get_literal_cnt(min_literal_id));
                    while (cell_bits != 0) {
                        size_t cell_bit = BitUtils::ms1b(cell_bits);
                        cell = BitUtils::bsf(cell_bits);
                        row = (box / BoxCountX) * BoxCellsY + cell / BoxCellsX;
                        col = (box % BoxCountX) * BoxCellsX + cell % BoxCellsX;
                        pos = row * Cols + col;

                        //this->box_nums_[box][num].reset(cell);
                        size_t effect_count = doFillNum(empties, pos, row, col,
                                                        box, cell, num, save_bits);

                        board[pos] = (char)(num + '1');

                        if (this->solve(board, empties - 1)) {
                            if (kSearchMode == SearchMode::OneAnswer) {
                                return true;
                            }
                            else if (kSearchMode == SearchMode::MoreThanOneAnswer) {
                                if (this->answers_.size() > 1)
                                    return true;
                            }
                        }

                        //this->box_nums_[box][num].set(cell);
                        undoFillNum(empties, effect_count, pos, row, col,
                                    box, cell, num, save_bits);

                        cell_bits ^= cell_bit;
                    }

                    enable_box_literal(box, num);
                    break;
                }

                case LiteralType::RowNums:
                {
                    size_t literal = (size_t)min_literal_id - RowLiteralFirst;
                    assert(min_literal_id >= RowLiteralFirst);
                    assert(literal < Rows * Numbers);
                    row = literal / Numbers;
                    num = literal % Numbers;

                    assert(this->rows_[row].test(num));

                    disable_row_literal(row, num);

                    size_t col_bits = this->row_nums_[row][num].to_ulong();
                    assert(this->row_nums_[row][num].count() == get_literal_cnt(min_literal_id));
                    while (col_bits != 0) {
                        size_t col_bit = BitUtils::ms1b(col_bits);
                        col = BitUtils::bsf(col_bits);
                        pos = row * Cols + col;
                        size_t box_x = col / BoxCellsX;
                        size_t box_y = row / BoxCellsY;
                        box = box_y * BoxCountX + box_x;
                        size_t cell_x = col % BoxCellsX;
                        size_t cell_y = row % BoxCellsY;
                        cell = cell_y * BoxCellsX + cell_x;

                        //this->row_nums_[row][num].reset(col);
                        size_t effect_count = doFillNum(empties, pos, row, col,
                                                        box, cell, num, save_bits);

                        board[pos] = (char)(num + '1');

                        if (this->solve(board, empties - 1)) {
                            if (kSearchMode == SearchMode::OneAnswer) {
                                return true;
                            }
                            else if (kSearchMode == SearchMode::MoreThanOneAnswer) {
                                if (this->answers_.size() > 1)
                                    return true;
                            }
                        }

                        //this->row_nums_[row][num].set(col);
                        undoFillNum(empties, effect_count, pos, row, col,
                                    box, cell, num, save_bits);

                        col_bits ^= col_bit;
                    }

                    enable_row_literal(row, num);
                    break;
                }

                case LiteralType::ColNums:
                {
                    size_t literal = (size_t)min_literal_id - ColLiteralFirst;
                    assert(min_literal_id >= ColLiteralFirst);
                    assert(literal < Cols * Numbers);
                    col = literal / Numbers;
                    num = literal % Numbers;

                    assert(this->cols_[col].test(num));

                    disable_col_literal(col, num);

                    size_t row_bits = this->col_nums_[col][num].to_ulong();
                    assert(this->col_nums_[col][num].count() == get_literal_cnt(min_literal_id));
                    while (row_bits != 0) {
                        size_t row_bit = BitUtils::ms1b(row_bits);
                        row = BitUtils::bsf(row_bits);
                        pos = row * Cols + col;
                        size_t box_x = col / BoxCellsX;
                        size_t box_y = row / BoxCellsY;
                        box = box_y * BoxCountX + box_x;
                        size_t cell_x = col % BoxCellsX;
                        size_t cell_y = row % BoxCellsY;
                        cell = cell_y * BoxCellsX + cell_x;

                        //this->col_nums_[col][num].reset(row);
                        size_t effect_count = doFillNum(empties, pos, row, col,
                                                        box, cell, num, save_bits);

                        board[pos] = (char)(num + '1');

                        if (this->solve(board, empties - 1)) {
                            if (kSearchMode == SearchMode::OneAnswer) {
                                return true;
                            }
                            else if (kSearchMode == SearchMode::MoreThanOneAnswer) {
                                if (this->answers_.size() > 1)
                                    return true;
                            }
                        }

                        //this->col_nums_[col][num].set(row);
                        undoFillNum(empties, effect_count, pos, row, col,
                                    box, cell, num, save_bits);

                        row_bits ^= row_bit;
                    }

                    enable_col_literal(col, num);
                    break;
                }

                default:
                    assert(false);
                    break;
            }
        }
        else {
            num_failed_return++;
        }

        return false;
    }

    bool solve(char board[BoardSize],
               double & elapsed_time,
               bool verbose = true) {
        if (verbose) {
            SudokuTy::display_board(board, true);
        }

        jtest::StopWatch sw;
        sw.start();

        this->init_board(board);
        //this->setup_state(board);

        bool success = this->solve(board, this->empties_);

        sw.stop();
        elapsed_time = sw.getElapsedMillisec();

        if (verbose) {
            if (kSearchMode > SearchMode::OneAnswer)
                SudokuTy::display_boards(this->answers_);
            else
                SudokuTy::display_board(board);
            printf("elapsed time: %0.3f ms, recur_counter: %" PRIuPTR "\n\n"
                   "num_guesses: %" PRIuPTR ", num_failed_return: %" PRIuPTR ", num_unique_candidate: %" PRIuPTR "\n"
                   "guess %% = %0.1f %%, failed_return %% = %0.1f %%, unique_candidate %% = %0.1f %%\n\n",
                   elapsed_time, slover_type::get_search_counter(),
                   slover_type::get_num_guesses(),
                   slover_type::get_num_failed_return(),
                   slover_type::get_num_unique_candidate(),
                   slover_type::get_guess_percent(),
                   slover_type::get_failed_return_percent(),
                   slover_type::get_unique_candidate_percent());
        }

        return success;
    }
};

template <typename SudokuTy>
size_t Solver<SudokuTy>::num_guesses = 0;

template <typename SudokuTy>
size_t Solver<SudokuTy>::num_unique_candidate = 0;

template <typename SudokuTy>
size_t Solver<SudokuTy>::num_failed_return = 0;

} // namespace v1
} // namespace jmSudoku

#endif // JM_SUDOKU_SOLVER_V1_H
