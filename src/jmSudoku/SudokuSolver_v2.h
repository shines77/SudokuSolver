
#ifndef JM_SUDOKU_SOLVER_V2_H
#define JM_SUDOKU_SOLVER_V2_H

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
#pragma once
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

#include "BasicSolver.h"
#include "Sudoku.h"
#include "StopWatch.h"
#include "BitUtils.h"
#include "BitSet.h"
#include "BitMatrix.h"

/************************************************

#define SEARCH_MODE_ONE_ANSWER              0
#define SEARCH_MODE_MORE_THAN_ONE_ANSWER    1
#define SEARCH_MODE_ALL_ANSWERS             2

************************************************/

#define V2_SEARCH_MODE          SEARCH_MODE_ONE_ANSWER

#define V2_LITERAL_ORDER_MODE   0

#ifdef _MSC_VER
#define V2_USE_STD_BITSET       0
#else
#define V2_USE_STD_BITSET       0
#endif

namespace jmSudoku {
namespace v2 {

static const size_t kSearchMode = V2_SEARCH_MODE;

template <typename SudokuTy>
class Solver : public BasicSolver<SudokuTy> {
public:
    typedef SudokuTy                            sudoku_t;
    typedef BasicSolver<SudokuTy>               basic_solver_t;
    typedef Solver<SudokuTy>                    solver_type;

    typedef typename basic_solver_t::Board      Board;
    typedef typename sudoku_t::NeighborCells    NeighborCells;
    typedef typename sudoku_t::CellInfo         CellInfo;
    typedef typename sudoku_t::BoxesInfo        BoxesInfo;

    typedef typename sudoku_t::BitMask          BitMask;
    typedef typename sudoku_t::BitMaskTable     BitMaskTable;

    static const size_t kAlignment = sudoku_t::kAlignment;
    static const size_t BoxCellsX = sudoku_t::BoxCellsX;      // 3
    static const size_t BoxCellsY = sudoku_t::BoxCellsY;      // 3
    static const size_t BoxCountX = sudoku_t::BoxCountX;      // 3
    static const size_t BoxCountY = sudoku_t::BoxCountY;      // 3
    static const size_t MinNumber = sudoku_t::MinNumber;      // 1
    static const size_t MaxNumber = sudoku_t::MaxNumber;      // 9

    static const size_t Rows = sudoku_t::Rows;
    static const size_t Cols = sudoku_t::Cols;
    static const size_t Boxes = sudoku_t::Boxes;
    static const size_t BoxSize = sudoku_t::BoxSize;
    static const size_t Numbers = sudoku_t::Numbers;

    static const size_t BoardSize = sudoku_t::BoardSize;
    static const size_t TotalSize = sudoku_t::TotalSize;
    static const size_t Neighbors = sudoku_t::Neighbors;

    static const size_t TotalCellLiterals = Rows * Cols;
    static const size_t TotalRowLiterals = Rows * Numbers;
    static const size_t TotalColLiterals = Cols * Numbers;
    static const size_t TotalBoxLiterals = Boxes * Numbers;

    static const size_t TotalLiterals =
        TotalCellLiterals + TotalRowLiterals + TotalColLiterals + TotalBoxLiterals;

#if (V2_LITERAL_ORDER_MODE == 0)
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
#endif // (V2_LITERAL_ORDER_MODE == 0)

    static const size_t kAllRowBits = sudoku_t::kAllRowBits;
    static const size_t kAllColBits = sudoku_t::kAllColBits;
    static const size_t kAllBoxBits = sudoku_t::kAllBoxBits;
    static const size_t kAllNumberBits = sudoku_t::kAllNumberBits;

    static const bool kAllDimIsSame = sudoku_t::kAllDimIsSame;

    static const int kLiteralCntThreshold = 0;

    static const size_t kEffectListAlignBytes =
        ((Neighbors * sizeof(uint8_t) + kAlignment - 1) / kAlignment) * kAlignment;

    static const size_t kEffectListReserveBytes1 = kEffectListAlignBytes - Neighbors * sizeof(uint8_t);
    static const size_t kEffectListReserveBytes  = (kEffectListReserveBytes1 != 0) ? kEffectListReserveBytes1 : kAlignment;

private:
#if (V2_LITERAL_ORDER_MODE == 0)
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
#endif // (V2_LITERAL_ORDER_MODE == 0)

#pragma pack(push, 1)

    union literal_info_t {
        struct {
            uint8_t count;
            uint8_t enable;
        };
        uint16_t value;
    };

#pragma pack(pop)

#if V2_USE_STD_BITSET
    typedef std::bitset<Numbers>    bitset_type;

    alignas(16) SmallBitMatrix2<BoardSize, Numbers>         cell_nums_;     // [row * Cols + col][num]
    alignas(16) SmallBitMatrix2<Numbers, BoardSize>         num_cells_;     // [num][row * Cols + col]

    alignas(16) SmallBitMatrix2<Numbers * Rows,  Cols>      row_nums_;      // [num * Rows + row][col]
    alignas(16) SmallBitMatrix2<Numbers * Cols,  Rows>      col_nums_;      // [num * Cols + col][row]
    alignas(16) SmallBitMatrix2<Numbers * Boxes, BoxSize>   box_nums_;      // [num * Boxes + box][cell]
#else
    typedef SmallBitSet<Numbers>    bitset_type;

    alignas(16) SmallBitSet2D<BoardSize, Numbers>           cell_nums_;     // [row * Cols + col][num]
    alignas(16) SmallBitSet2D<Numbers, BoardSize>           num_cells_;     // [num][row * Cols + col]

    alignas(16) SmallBitSet2D<Numbers * Rows,  Cols>        row_nums_;      // [num * Rows + row][col]
    alignas(16) SmallBitSet2D<Numbers * Cols,  Rows>        col_nums_;      // [num * Cols + col][row]
    alignas(16) SmallBitSet2D<Numbers * Boxes, BoxSize>     box_nums_;      // [num * Boxes + box][cell]
#endif

#if defined(__SSE4_1__)
    alignas(16) literal_info_t literal_info_[TotalLiterals];
#else
    alignas(16) uint8_t literal_count_[TotalLiterals];
    alignas(16) uint8_t literal_enable_[TotalLiterals];
#endif

public:
    Solver() {
    }
    ~Solver() {}

private:
    void init_board(Board & board) {
        init_literal_info();

        this->num_cells_.set();
        this->cell_nums_.set();
        
        this->row_nums_.set();
        this->col_nums_.set();
        this->box_nums_.set();

        if (kSearchMode > SEARCH_MODE_ONE_ANSWER) {
            this->answers_.clear();
        }

        size_t empties = basic_solver_t::calc_empties(board);
        this->empties_ = empties;

        size_t pos = 0;
        for (size_t row = 0; row < Rows; row++) {
            size_t box_y = (row / BoxCellsY) * BoxCountX;
            size_t cell_y = (row % BoxCellsY) * BoxCellsX;
            for (size_t col = 0; col < Cols; col++) {
                unsigned char val = board.cells[pos];
                if (val != '.') {
                    size_t box_x = col / BoxCellsX;
                    size_t box = box_y + box_x;
                    size_t cell_x = col % BoxCellsX;
                    size_t cell = cell_y + cell_x;
                    size_t num = val - '1';
                    this->fillNum(pos, row, col, box, cell, num);
                    this->updateNeighborCellsEffect(pos, num);
                }
                pos++;
            }
        }
    }

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

#if defined(__SSE4_1__)
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
            size_t * pinfo = (size_t *)&this->literal_info_[i];
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

    inline void init_literal_info(size_t literal, uint16_t value) {
        this->literal_info_[literal].value = value;
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
        std::memset((void *)&this->literal_count_[0],  (int)kInitLiteral, sizeof(this->literal_count_));
        std::memset((void *)&this->literal_enable_[0], 0,                 sizeof(this->literal_enable_));
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

    inline void dec_literal_cnt(size_t literal) {
        assert(this->literal_count_[literal] > 0);
        this->literal_count_[literal]--;
    }

#endif // __SSE4_1__

    // enable_xxxx_literal()
    inline void enable_cell_literal(size_t pos) {
        size_t literal = CellLiteralFirst + pos;
        this->enable_literal(literal);
    }

    inline void enable_row_literal(size_t index) {
        size_t literal = RowLiteralFirst + index;
        this->enable_literal(literal);
    }

    inline void enable_row_literal(size_t num, size_t row) {
        size_t literal = RowLiteralFirst + num * Rows + row;
        this->enable_literal(literal);
    }

    inline void enable_col_literal(size_t index) {
        size_t literal = ColLiteralFirst + index;
        this->enable_literal(literal);
    }

    inline void enable_col_literal(size_t num, size_t col) {
        size_t literal = ColLiteralFirst + num * Cols + col;
        this->enable_literal(literal);
    }

    inline void enable_box_literal(size_t index) {
        size_t literal = BoxLiteralFirst + index;
        this->enable_literal(literal);
    }

    inline void enable_box_literal(size_t num, size_t box) {
        size_t literal = BoxLiteralFirst + num * Boxes + box;
        this->enable_literal(literal);
    }

    // disable_xxxx_literal()
    inline void disable_cell_literal(size_t pos) {
        size_t literal = CellLiteralFirst + pos;
        this->disable_literal(literal);
    }

    inline void disable_row_literal(size_t index) {
        size_t literal = RowLiteralFirst + index;
        this->disable_literal(literal);
    }

    inline void disable_row_literal(size_t num, size_t row) {
        size_t literal = RowLiteralFirst + num * Rows + row;
        this->disable_literal(literal);
    }

    inline void disable_col_literal(size_t index) {
        size_t literal = ColLiteralFirst + index;
        this->disable_literal(literal);
    }

    inline void disable_col_literal(size_t num, size_t col) {
        size_t literal = ColLiteralFirst + num * Cols + col;
        this->disable_literal(literal);
    }

    inline void disable_box_literal(size_t index) {
        size_t literal = BoxLiteralFirst + index;
        this->disable_literal(literal);
    }

    inline void disable_box_literal(size_t num, size_t box) {
        size_t literal = BoxLiteralFirst + num * Boxes + box;
        this->disable_literal(literal);
    }

    // get_xxxx_literal_cnt()
    inline uint8_t get_cell_literal_cnt(size_t pos) {
        size_t literal = CellLiteralFirst + pos;
        return this->get_literal_cnt(literal);
    }

    inline uint8_t get_row_literal_cnt(size_t num, size_t row) {
        size_t literal = RowLiteralFirst + num * Rows + row;
        return this->get_literal_cnt(literal);
    }

    inline uint8_t get_col_literal_cnt(size_t num, size_t col) {
        size_t literal = ColLiteralFirst + num * Cols + col;
        return this->get_literal_cnt(literal);
    }

    inline uint8_t get_box_literal_cnt(size_t num, size_t box) {
        size_t literal = BoxLiteralFirst + num * Boxes + box;
        return this->get_literal_cnt(literal);
    }

    // set_xxxx_literal_cnt()
    inline void set_cell_literal_cnt(size_t pos, uint8_t count) {
        size_t literal = CellLiteralFirst + pos;
        this->set_literal_cnt(literal, count);
    }

    inline void set_row_literal_cnt(size_t index, uint8_t count) {
        size_t literal = RowLiteralFirst + index;
        this->set_literal_cnt(literal, count);
    }

    inline void set_row_literal_cnt(size_t num, size_t row, uint8_t count) {
        size_t literal = RowLiteralFirst + num * Rows + row;
        this->set_literal_cnt(literal, count);
    }

    inline void set_col_literal_cnt(size_t index, uint8_t count) {
        size_t literal = ColLiteralFirst + index;
        this->set_literal_cnt(literal, count);
    }

    inline void set_col_literal_cnt(size_t num, size_t col, uint8_t count) {
        size_t literal = ColLiteralFirst + num * Cols + col;
        this->set_literal_cnt(literal, count);
    }

    inline void set_box_literal_cnt(size_t index, uint8_t count) {
        size_t literal = BoxLiteralFirst + index;
        this->set_literal_cnt(literal, count);
    }

    inline void set_box_literal_cnt(size_t num, size_t box, uint8_t count) {
        size_t literal = BoxLiteralFirst + num * Boxes + box;
        this->set_literal_cnt(literal, count);
    }

    // inc_xxxx_literal_cnt()
    inline void inc_cell_literal_cnt(size_t pos) {
        size_t literal = CellLiteralFirst + pos;
        this->inc_literal_cnt(literal);
    }

    inline void inc_row_literal_cnt(size_t index) {
        size_t literal = RowLiteralFirst + index;
        this->inc_literal_cnt(literal);
    }

    inline void inc_row_literal_cnt(size_t num, size_t row) {
        size_t literal = RowLiteralFirst + num * Rows + row;
        this->inc_literal_cnt(literal);
    }

    inline void inc_col_literal_cnt(size_t index) {
        size_t literal = ColLiteralFirst + index;
        this->inc_literal_cnt(literal);
    }

    inline void inc_col_literal_cnt(size_t num, size_t col) {
        size_t literal = ColLiteralFirst + num * Cols + col;
        this->inc_literal_cnt(literal);
    }

    inline void inc_box_literal_cnt(size_t index) {
        size_t literal = BoxLiteralFirst + index;
        this->inc_literal_cnt(literal);
    }

    inline void inc_box_literal_cnt(size_t num, size_t box) {
        size_t literal = BoxLiteralFirst + num * Boxes + box;
        this->inc_literal_cnt(literal);
    }

    // dec_xxxx_literal_cnt()
    inline void dec_cell_literal_cnt(size_t pos) {
        size_t literal = CellLiteralFirst + pos;
        this->dec_literal_cnt(literal);
    }

    inline void dec_row_literal_cnt(size_t index) {
        size_t literal = RowLiteralFirst + index;
        this->dec_literal_cnt(literal);
    }

    inline void dec_row_literal_cnt(size_t num, size_t row) {
        size_t literal = RowLiteralFirst + num * Rows + row;
        this->dec_literal_cnt(literal);
    }

    inline void dec_col_literal_cnt(size_t index) {
        size_t literal = ColLiteralFirst + index;
        this->dec_literal_cnt(literal);
    }

    inline void dec_col_literal_cnt(size_t num, size_t col) {
        size_t literal = ColLiteralFirst + num * Cols + col;
        this->dec_literal_cnt(literal);
    }

    inline void dec_box_literal_cnt(size_t index) {
        size_t literal = BoxLiteralFirst + index;
        this->dec_literal_cnt(literal);
    }

    inline void dec_box_literal_cnt(size_t num, size_t box) {
        size_t literal = BoxLiteralFirst + num * Boxes + box;
        this->dec_literal_cnt(literal);
    }

#if defined(__SSE4_1__)
    int get_min_literal_normal(int & out_min_literal_cnt) const {
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

    int get_min_literal(int & out_min_literal_cnt) {
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
            __m128i result_minpos = _mm_or_si128(xmm4_5, xmm6_7_);
            __m128i index_minpos = result_minpos;

            __m128i literal_cnt_mask = _mm_set1_epi32((int)0xFFFF0000L);
            __m128i result_minpos_only_cnt = _mm_or_si128(result_minpos, literal_cnt_mask);
            __m128i result_minpos_all = _mm_minpos_epu16(result_minpos_only_cnt);      // SSE 4.1

            uint32_t result_minpos_all32 = (uint32_t)_mm_cvtsi128_si32(result_minpos_all);
            int min_literal_cnt16 = (int)(result_minpos_all32 & 0x0000FFFFULL);
            if (min_literal_cnt16 < min_literal_cnt) {
                min_literal_cnt = min_literal_cnt16;

                uint32_t min_block_index16 = result_minpos_all32 >> 17U;
                __m128i min_literal_id_sr15 = _mm_srli_epi64(result_minpos_all, 15);
                __m128i literal_index_sr16  = _mm_srli_epi32(index_minpos, 16);

                __m128i min_literal_id128 = _mm_shuffle_epi8(literal_index_sr16, min_literal_id_sr15);  // SSSE3
                uint32_t min_literal_id32 = (uint32_t)_mm_cvtsi128_si32(min_literal_id128);
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
            __m128i result_minpos = _mm_blend_epi16(xmm2, xmm3_, 0b00001100);  // SSE 4.1
            __m128i index_minpos = result_minpos;

            __m128i literal_cnt_mask = _mm_set_epi32(0xFFFFFFFFL, 0xFFFFFFFFL, 0xFFFF0000L, 0xFFFF0000L);
            __m128i result_minpos_only_cnt = _mm_or_si128(result_minpos, literal_cnt_mask);
            __m128i result_minpos_all = _mm_minpos_epu16(result_minpos_only_cnt);      // SSE 4.1

            uint32_t result_minpos_all32 = (uint32_t)_mm_cvtsi128_si32(result_minpos_all);
            int min_literal_cnt16 = (int)(result_minpos_all32 & 0x0000FFFFULL);
            if (min_literal_cnt16 < min_literal_cnt) {
                min_literal_cnt = min_literal_cnt16;

                uint32_t min_block_index16 = result_minpos_all32 >> 17U;
                __m128i min_literal_id_sr15 = _mm_srli_epi64(result_minpos_all, 15);
                __m128i literal_index_sr16 = _mm_srli_epi32(index_minpos, 16);

                __m128i min_literal_id128 = _mm_shuffle_epi8(literal_index_sr16, min_literal_id_sr15);  // SSSE3
                uint32_t min_literal_id32 = (uint32_t)_mm_cvtsi128_si32(min_literal_id128);
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
            __m128i result_minpos = _mm_minpos_epu16(xmm0);    // SSE 4.1

            uint32_t min_literal_cnt32 = (uint32_t)_mm_cvtsi128_si32(result_minpos);
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

    int get_min_literal_normal(int & out_min_literal_cnt) const {
        int min_literal_cnt = 254;
        int min_literal_id = -1;
        for (int i = 0; i < TotalLiterals; i++) {
            if (literal_enable_[i] == 0) {
                int literal_cnt = literal_count_[i];
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

    //
    // Horizontal minimum and maximum using SSE
    // See: https://stackoverflow.com/questions/22256525/horizontal-minimum-and-maximum-using-sse
    //
    int get_min_literal(int & out_min_literal_cnt) {
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

            __m128i result_minpos_03 = _mm_min_epu8(xmm0, xmm1);

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

            __m128i result_minpos_47 = _mm_min_epu8(xmm4, xmm5);

            // The minimum literal count of per 64 numbers
            __m128i result_minpos_64 = _mm_min_epu8(result_minpos_03, result_minpos_47);

            int min_literal_cnt16 = _mm_cvtsi128_si32(result_minpos_64) & 0x000000FFL;
            if (min_literal_cnt16 < min_literal_cnt) {
                min_literal_cnt = min_literal_cnt16;

                __m128i xmm0 = _mm_load_si128((const __m128i *)(pcount + 0));
                __m128i xmm1 = _mm_load_si128((const __m128i *)(penable + 0));

                __m128i min_cmp = _mm_set1_epi8((char)min_literal_cnt);

                xmm0 = _mm_or_si128(xmm0, xmm1);
                xmm0 = _mm_cmpeq_epi8(xmm0, min_cmp);

                int equal_mask = _mm_movemask_epi8(xmm0);
                if (equal_mask == 0) {
                    __m128i xmm2 = _mm_load_si128((const __m128i *)(pcount + 16));
                    __m128i xmm3 = _mm_load_si128((const __m128i *)(penable + 16));
                
                    xmm2 = _mm_or_si128(xmm2, xmm3);  
                    xmm2 = _mm_cmpeq_epi8(xmm2, min_cmp);

                    equal_mask = _mm_movemask_epi8(xmm2);
                    if (equal_mask == 0) {
                        __m128i xmm4 = _mm_load_si128((const __m128i *)(pcount + 32));
                        __m128i xmm5 = _mm_load_si128((const __m128i *)(penable + 32));
                
                        xmm4 = _mm_or_si128(xmm4, xmm5);  
                        xmm4 = _mm_cmpeq_epi8(xmm4, min_cmp);

                        equal_mask = _mm_movemask_epi8(xmm4);
                        if (equal_mask == 0) {
                            __m128i xmm6 = _mm_load_si128((const __m128i *)(pcount + 48));
                            __m128i xmm7 = _mm_load_si128((const __m128i *)(penable + 48));
                
                            xmm6 = _mm_or_si128(xmm6, xmm7);  
                            xmm6 = _mm_cmpeq_epi8(xmm6, min_cmp);

                            equal_mask = _mm_movemask_epi8(xmm6);
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

                if (min_literal_cnt <= kLiteralCntThreshold) {
                    out_min_literal_cnt = min_literal_cnt;
                    return min_literal_id;
                }
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
            __m128i result_minpos_32 = _mm_min_epu8(xmm0, xmm1);

            int min_literal_cnt16 = _mm_cvtsi128_si32(result_minpos_32) & 0x000000FFL;
            if (min_literal_cnt16 < min_literal_cnt) {
                min_literal_cnt = min_literal_cnt16;

                __m128i xmm0 = _mm_load_si128((const __m128i *)(pcount + 0));
                __m128i xmm1 = _mm_load_si128((const __m128i *)(penable + 0));

                __m128i min_cmp = _mm_set1_epi8((char)min_literal_cnt);

                xmm0 = _mm_or_si128(xmm0, xmm1);
                xmm0 = _mm_cmpeq_epi8(xmm0, min_cmp);

                int equal_mask = _mm_movemask_epi8(xmm0);
                if (equal_mask == 0) {
                    __m128i xmm2 = _mm_load_si128((const __m128i *)(pcount + 16));
                    __m128i xmm3 = _mm_load_si128((const __m128i *)(penable + 16));
                
                    xmm2 = _mm_or_si128(xmm2, xmm3);  
                    xmm2 = _mm_cmpeq_epi8(xmm2, min_cmp);

                    equal_mask = _mm_movemask_epi8(xmm2);
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

                if (min_literal_cnt <= kLiteralCntThreshold) {
                    out_min_literal_cnt = min_literal_cnt;
                    return min_literal_id;
                }
            }

            index_base += 32;
            penable += 32;
            pcount += 32;
        }

        if ((pcount_end - pcount) >= 16) {
            __m128i xmm0 = _mm_load_si128((const __m128i *)(pcount + 0));
            __m128i xmm1 = _mm_load_si128((const __m128i *)(penable + 0));

            xmm0 = _mm_or_si128(xmm0, xmm1);
            xmm0 = _mm_min_epu8(xmm0, _mm_shuffle_epi32(xmm0, _MM_SHUFFLE(3, 2, 3, 2)));
            xmm0 = _mm_min_epu8(xmm0, _mm_shuffle_epi32(xmm0, _MM_SHUFFLE(1, 1, 1, 1)));
            xmm0 = _mm_min_epu8(xmm0, _mm_shufflelo_epi16(xmm0, _MM_SHUFFLE(1, 1, 1, 1)));

            // The minimum literal count of per 16 numbers
            __m128i result_minpos_16 = _mm_min_epu8(xmm0, _mm_srli_epi16(xmm0, 8));

            int min_literal_cnt16 = _mm_cvtsi128_si32(result_minpos_16) & 0x000000FFL;
            if (min_literal_cnt16 < min_literal_cnt) {
                min_literal_cnt = min_literal_cnt16;

                __m128i xmm0 = _mm_load_si128((const __m128i *)(pcount + 0));
                __m128i xmm1 = _mm_load_si128((const __m128i *)(penable + 0));

                __m128i min_cmp = _mm_set1_epi8((char)min_literal_cnt);

                xmm0 = _mm_or_si128(xmm0, xmm1);
                xmm0 = _mm_cmpeq_epi8(xmm0, min_cmp);

                int equal_mask = _mm_movemask_epi8(xmm0);
                if (equal_mask == 0) {
                    assert(false);
                }
                else {
                    int min_literal_offset = BitUtils::bsf(equal_mask);
                    min_literal_id = index_base + min_literal_offset;
                }

                if (min_literal_cnt <= kLiteralCntThreshold) {
                    out_min_literal_cnt = min_literal_cnt;
                    return min_literal_id;
                }
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

#endif // __SSE4_1__

    inline bitset_type getCanFillNums(size_t row, size_t col, size_t box) {
        //return (this->rows_[row] & this->cols_[col] & this->boxes_[box]);
        return bitset_type{};
    }

    inline void fillNum(size_t pos, size_t row, size_t col,
                          size_t box, size_t cell, size_t num) {
        size_t row_idx = num * Rows + row;
        size_t col_idx = num * Cols + col;
        size_t box_idx = num * Boxes + box;

        assert(this->num_cells_[num].test(pos));
        assert(this->cell_nums_[pos].test(num));
        assert(this->row_nums_[row_idx].test(col));
        assert(this->col_nums_[col_idx].test(row));
        assert(this->box_nums_[box_idx].test(cell));

        disable_cell_literal(pos);
        disable_row_literal(row_idx);
        disable_col_literal(col_idx);
        disable_box_literal(box_idx);

        size_t num_bits = this->cell_nums_[pos].to_ulong();
        while (num_bits != 0) {
            size_t num_bit = BitUtils::ls1b(num_bits);
            size_t _num = BitUtils::bsf(num_bit);

            row_idx = _num * Rows + row;
            col_idx = _num * Cols + col;
            box_idx = _num * Boxes + box;

            assert(this->num_cells_[_num].test(pos));
            assert(this->cell_nums_[pos].test(_num));
            assert(this->row_nums_[row_idx].test(col));
            assert(this->col_nums_[col_idx].test(row));
            assert(this->box_nums_[box_idx].test(cell));

            this->num_cells_[_num].reset(pos);
            this->cell_nums_[pos].reset(_num);

            this->row_nums_[row_idx].reset(col);
            this->col_nums_[col_idx].reset(row);
            this->box_nums_[box_idx].reset(cell);

            dec_cell_literal_cnt(pos);
            dec_row_literal_cnt(row_idx);
            dec_col_literal_cnt(col_idx);
            dec_box_literal_cnt(box_idx);

            num_bits ^= num_bit;
        }
    }

    inline void updateNeighborCellsEffect(size_t fill_pos, size_t num) {
        const BitMask & neighborsMask = sudoku_t::neighbors_mask_tbl[fill_pos];
        BitMask & digitCells = this->num_cells_[num];

        register BitMask effect_cells = digitCells & neighborsMask;
        digitCells ^= effect_cells;

        size_t cell_bit, index;
        while ((cell_bit = effect_cells.ls1b(index)) != 0) {
            size_t bit_pos = BitUtils::bsf(cell_bit);
            effect_cells.reset_bit(index, cell_bit);
            size_t pos = index * effect_cells.unit_bits() + bit_pos;
            if (this->cell_nums_[pos].test(num)) {
                this->cell_nums_[pos].reset(num);
                dec_cell_literal_cnt(pos);

                const CellInfo & cellInfo = sudoku_t::cell_info[pos];

                size_t box = cellInfo.box;
                size_t cell = cellInfo.cell;
                size_t row = cellInfo.row;
                size_t col = cellInfo.col;

                size_t row_idx = num * Rows + row;
                size_t col_idx = num * Cols + col;
                size_t box_idx = num * Boxes + box;

                assert(this->row_nums_[row_idx].test(col));
                assert(this->col_nums_[col_idx].test(row));
                assert(this->box_nums_[box_idx].test(cell));

                this->row_nums_[row_idx].reset(col);
                this->col_nums_[col_idx].reset(row);
                this->box_nums_[box_idx].reset(cell);

                dec_row_literal_cnt(row_idx);
                dec_col_literal_cnt(col_idx);
                dec_box_literal_cnt(box_idx);
            }
        }
    }

    inline void doFillNum(size_t pos, size_t row, size_t col,
                          size_t box, size_t cell, size_t num,
                          bitset_type & save_bits) {
        assert(this->cell_nums_[pos].test(num));

        size_t row_idx = num * Rows + row;
        size_t col_idx = num * Cols + col;
        size_t box_idx = num * Boxes + box;

        disable_cell_literal(pos);
        disable_row_literal(row_idx);
        disable_col_literal(col_idx);
        disable_box_literal(box_idx);

        assert(this->num_cells_[num].test(pos));
        assert(this->cell_nums_[pos].test(num));
        assert(this->row_nums_[row_idx].test(col));
        assert(this->col_nums_[col_idx].test(row));
        assert(this->box_nums_[box_idx].test(cell));

        this->num_cells_[num].reset(pos);
        // Save cell num bits
        save_bits = this->cell_nums_[pos];
        this->cell_nums_[pos].reset();

        this->row_nums_[row_idx].reset(col);
        this->col_nums_[col_idx].reset(row);
        this->box_nums_[box_idx].reset(cell);

        size_t num_bits = save_bits.to_ulong();
        // Exclude the current number, because it has been processed.
        uint32_t n_num_bit = 1u << num;
        num_bits ^= (size_t)n_num_bit;
        while (num_bits != 0) {
            size_t num_bit = BitUtils::ls1b(num_bits);
            size_t _num = BitUtils::bsf(num_bit);

            row_idx = _num * Rows + row;
            col_idx = _num * Cols + col;
            box_idx = _num * Boxes + box;

            assert(this->num_cells_[_num].test(pos));
            assert(this->row_nums_[row_idx].test(col));
            assert(this->col_nums_[col_idx].test(row));
            assert(this->box_nums_[box_idx].test(cell));

            this->num_cells_[_num].reset(pos);

            this->row_nums_[row_idx].reset(col);
            this->col_nums_[col_idx].reset(row);
            this->box_nums_[box_idx].reset(cell);

            dec_row_literal_cnt(row_idx);
            dec_col_literal_cnt(col_idx);
            dec_box_literal_cnt(box_idx);

            num_bits ^= num_bit;
        }
    }

    inline void undoFillNum(size_t pos, size_t row, size_t col,
                            size_t box, size_t cell, size_t num,
                            bitset_type & save_bits) {
        size_t row_idx = num * Rows + row;
        size_t col_idx = num * Cols + col;
        size_t box_idx = num * Boxes + box;

        enable_cell_literal(pos);
        enable_row_literal(row_idx);
        enable_col_literal(col_idx);
        enable_box_literal(box_idx);

        this->num_cells_[num].set(pos);
        // Restore cell num bits
        this->cell_nums_[pos] = save_bits;

        this->row_nums_[row_idx].set(col);
        this->col_nums_[col_idx].set(row);
        this->box_nums_[box_idx].set(cell);

        // Exclude the current number, because it has been processed.
        save_bits.reset(num);

        size_t num_bits = save_bits.to_ulong();
        while (num_bits != 0) {
            size_t num_bit = BitUtils::ls1b(num_bits);
            size_t _num = BitUtils::bsf(num_bit);

            row_idx = _num * Rows + row;
            col_idx = _num * Cols + col;
            box_idx = _num * Boxes + box;

            this->num_cells_[_num].set(pos);
            this->row_nums_[row_idx].set(col);
            this->col_nums_[col_idx].set(row);
            this->box_nums_[box_idx].set(cell);

            inc_row_literal_cnt(row_idx);
            inc_col_literal_cnt(col_idx);
            inc_box_literal_cnt(box_idx);

            num_bits ^= num_bit;
        }
    }

    inline size_t updateNeighborCellsEffect(BitMask & save_effect_cells,
                                            size_t fill_pos, size_t num) {
        const BitMask & neighborsMask = sudoku_t::neighbors_mask_tbl[fill_pos];
        BitMask & digitCells = this->num_cells_[num];

        BitMask effect_cells = digitCells & neighborsMask;
        digitCells ^= effect_cells;
        save_effect_cells = effect_cells;

        size_t count = 0;
        size_t cell_bit, index;
        while ((cell_bit = effect_cells.ls1b(index)) != 0) {
            size_t bit_pos = BitUtils::bsf(cell_bit);
            effect_cells.reset_bit(index, cell_bit);
            size_t pos = index * effect_cells.unit_bits() + bit_pos;
            if (this->cell_nums_[pos].test(num)) {
                this->cell_nums_[pos].reset(num);
                dec_cell_literal_cnt(pos);

                const CellInfo & cellInfo = sudoku_t::cell_info[pos];

                size_t box = cellInfo.box;
                size_t cell = cellInfo.cell;
                size_t row = cellInfo.row;
                size_t col = cellInfo.col;

                size_t row_idx = num * Rows + row;
                size_t col_idx = num * Cols + col;
                size_t box_idx = num * Boxes + box;

                assert(this->row_nums_[row_idx].test(col));
                assert(this->col_nums_[col_idx].test(row));
                assert(this->box_nums_[box_idx].test(cell));

                this->row_nums_[row_idx].reset(col);
                this->col_nums_[col_idx].reset(row);
                this->box_nums_[box_idx].reset(cell);

                dec_row_literal_cnt(row_idx);
                dec_col_literal_cnt(col_idx);
                dec_box_literal_cnt(box_idx);

                count++;
            }
        }

        return count;
    }

    inline size_t restoreNeighborCellsEffect(BitMask & effect_cells, size_t num) {
        size_t count = 0;
        size_t cell_bit, index;
        while ((cell_bit = effect_cells.ls1b(index)) != 0) {
            size_t bit_pos = BitUtils::bsf(cell_bit);
            effect_cells.reset_bit(index, cell_bit);
            size_t pos = index * effect_cells.unit_bits() + bit_pos;
            this->cell_nums_[pos].set(num);
            inc_cell_literal_cnt(pos);

            const CellInfo & cellInfo = sudoku_t::cell_info[pos];

            size_t box = cellInfo.box;
            size_t cell = cellInfo.cell;
            size_t row = cellInfo.row;
            size_t col = cellInfo.col;

            size_t row_idx = num * Rows + row;
            size_t col_idx = num * Cols + col;
            size_t box_idx = num * Boxes + box;

            assert(!this->row_nums_[row_idx].test(col));
            assert(!this->col_nums_[col_idx].test(row));
            assert(!this->box_nums_[box_idx].test(cell));

            this->row_nums_[row_idx].set(col);
            this->col_nums_[col_idx].set(row);
            this->box_nums_[box_idx].set(cell);

            inc_row_literal_cnt(row_idx);
            inc_col_literal_cnt(col_idx);
            inc_box_literal_cnt(box_idx);

            count++;
        }

        return count;
    }

public:
    bool solve(Board & board, size_t empties) {
        if (empties == 0) {
            if (kSearchMode > SearchMode::OneAnswer) {
                this->answers_.push_back(board);
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
        int min_literal_id = get_min_literal(min_literal_cnt);
        assert(min_literal_id < TotalLiterals);
        if (min_literal_cnt > 0) {
            if (min_literal_cnt == 1)
                basic_solver_t::num_unique_candidate++;
            else
                num_guesses++;

            bitset_type save_bits;
            BitMask save_effect_cells;
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
                    const CellInfo & cellInfo = sudoku_t::cell_info[pos];
                    row = cellInfo.row;
                    col = cellInfo.col;
                    box = cellInfo.box;
                    cell = cellInfo.cell;
#endif
                    //disable_cell_literal(pos);

                    size_t num_bits = this->cell_nums_[pos].to_ulong();
                    assert(this->cell_nums_[pos].count() == get_literal_cnt(min_literal_id));
                    while (num_bits != 0) {
                        size_t num_bit = BitUtils::ls1b(num_bits);
                        num = BitUtils::bsf(num_bit);

                        doFillNum(pos, row, col, box, cell, num, save_bits);
                        size_t effect_count = updateNeighborCellsEffect(save_effect_cells, pos, num);

                        board.cells[pos] = (char)(num + '1');

                        if (this->solve(board, empties - 1)) {
                            if (kSearchMode == SearchMode::OneAnswer) {
                                return true;
                            }
                            else if (kSearchMode == SearchMode::MoreThanOneAnswer) {
                                if (this->answers_.size() > 1)
                                    return true;
                            }
                        }

                        undoFillNum(pos, row, col, box, cell, num, save_bits);
                        this->num_cells_[num] |= save_effect_cells;
                        size_t r_effect_count = restoreNeighborCellsEffect(save_effect_cells, num);
                        assert(effect_count == r_effect_count);

                        num_bits ^= num_bit;
                    }

                    //enable_cell_literal(pos);
                    break;
                }

                case LiteralType::RowNums:
                {
                    size_t literal = (size_t)min_literal_id - RowLiteralFirst;
                    assert(min_literal_id >= RowLiteralFirst);
                    assert(literal < Numbers * Rows);
                    num = literal / Rows;
                    row = literal % Rows;

                    //disable_row_literal(num, row);

                    size_t col_bits = this->row_nums_[num * Rows + row].to_ulong();
                    assert(this->row_nums_[num * Rows + row].count() == get_literal_cnt(min_literal_id));
                    while (col_bits != 0) {
                        size_t col_bit = BitUtils::ls1b(col_bits);
                        col = BitUtils::bsf(col_bits);
                        pos = row * Cols + col;
#if 0
                        size_t box_x = col / BoxCellsX;
                        size_t box_y = row / BoxCellsY;
                        box = box_y * BoxCountX + box_x;
                        size_t cell_x = col % BoxCellsX;
                        size_t cell_y = row % BoxCellsY;
                        cell = cell_y * BoxCellsX + cell_x;
#else
                        const CellInfo & cellInfo = sudoku_t::cell_info[pos];
                        box = cellInfo.box;
                        cell = cellInfo.cell;
#endif
                        doFillNum(pos, row, col, box, cell, num, save_bits);
                        size_t effect_count = updateNeighborCellsEffect(save_effect_cells, pos, num);

                        board.cells[pos] = (char)(num + '1');

                        if (this->solve(board, empties - 1)) {
                            if (kSearchMode == SearchMode::OneAnswer) {
                                return true;
                            }
                            else if (kSearchMode == SearchMode::MoreThanOneAnswer) {
                                if (this->answers_.size() > 1)
                                    return true;
                            }
                        }

                        undoFillNum(pos, row, col, box, cell, num, save_bits);
                        this->num_cells_[num] |= save_effect_cells;
                        size_t r_effect_count = restoreNeighborCellsEffect(save_effect_cells, num);
                        assert(effect_count == r_effect_count);

                        col_bits ^= col_bit;
                    }

                    //enable_row_literal(num, row);
                    break;
                }

                case LiteralType::ColNums:
                {
                    size_t literal = (size_t)min_literal_id - ColLiteralFirst;
                    assert(min_literal_id >= ColLiteralFirst);
                    assert(literal < Numbers * Cols);
                    num = literal / Cols;
                    col = literal % Cols;

                    //disable_col_literal(num, col);

                    size_t row_bits = this->col_nums_[num * Cols + col].to_ulong();
                    assert(this->col_nums_[num * Cols + col].count() == get_literal_cnt(min_literal_id));
                    while (row_bits != 0) {
                        size_t row_bit = BitUtils::ls1b(row_bits);
                        row = BitUtils::bsf(row_bits);
                        pos = row * Cols + col;
#if 0
                        size_t box_x = col / BoxCellsX;
                        size_t box_y = row / BoxCellsY;
                        box = box_y * BoxCountX + box_x;
                        size_t cell_x = col % BoxCellsX;
                        size_t cell_y = row % BoxCellsY;
                        cell = cell_y * BoxCellsX + cell_x;
#else
                        const CellInfo & cellInfo = sudoku_t::cell_info[pos];
                        box = cellInfo.box;
                        cell = cellInfo.cell;
#endif
                        doFillNum(pos, row, col, box, cell, num, save_bits);
                        size_t effect_count = updateNeighborCellsEffect(save_effect_cells, pos, num);

                        board.cells[pos] = (char)(num + '1');

                        if (this->solve(board, empties - 1)) {
                            if (kSearchMode == SearchMode::OneAnswer) {
                                return true;
                            }
                            else if (kSearchMode == SearchMode::MoreThanOneAnswer) {
                                if (this->answers_.size() > 1)
                                    return true;
                            }
                        }

                        undoFillNum(pos, row, col, box, cell, num, save_bits);
                        this->num_cells_[num] |= save_effect_cells;
                        size_t r_effect_count = restoreNeighborCellsEffect(save_effect_cells, num);
                        assert(effect_count == r_effect_count);

                        row_bits ^= row_bit;
                    }

                    //enable_col_literal(num, col);
                    break;
                }

               case LiteralType::BoxNums:
                {
                    size_t literal = (size_t)min_literal_id - BoxLiteralFirst;
                    assert(min_literal_id >= BoxLiteralFirst);
                    assert(literal < Numbers * Boxes);
                    num = literal / Boxes;
                    box = literal % Boxes;

                    //disable_box_literal(num, box);

                    size_t cell_bits = this->box_nums_[num * Boxes + box].to_ulong();
                    assert(this->box_nums_[num * Boxes + box].count() == get_literal_cnt(min_literal_id));
                    while (cell_bits != 0) {
                        size_t cell_bit = BitUtils::ls1b(cell_bits);
                        cell = BitUtils::bsf(cell_bits);
#if 0
                        row = (box / BoxCountX) * BoxCellsY + (cell / BoxCellsX);
                        col = (box % BoxCountX) * BoxCellsX + (cell % BoxCellsX);
                        pos = row * Cols + col;
#else
                        const BoxesInfo & boxesInfo = sudoku_t::boxes_info[box * BoxSize + cell];
                        row = boxesInfo.row;
                        col = boxesInfo.col;
                        pos = boxesInfo.pos;
#endif
                        doFillNum(pos, row, col, box, cell, num, save_bits);
                        size_t effect_count = updateNeighborCellsEffect(save_effect_cells, pos, num);

                        board.cells[pos] = (char)(num + '1');

                        if (this->solve(board, empties - 1)) {
                            if (kSearchMode == SearchMode::OneAnswer) {
                                return true;
                            }
                            else if (kSearchMode == SearchMode::MoreThanOneAnswer) {
                                if (this->answers_.size() > 1)
                                    return true;
                            }
                        }

                        undoFillNum(pos, row, col, box, cell, num, save_bits);
                        this->num_cells_[num] |= save_effect_cells;
                        size_t r_effect_count = restoreNeighborCellsEffect(save_effect_cells, num);
                        assert(effect_count == r_effect_count);

                        cell_bits ^= cell_bit;
                    }

                    //enable_box_literal(num, box);
                    break;
                }

                default:
                    assert(false);
                    break;
            }
        }
        else {
            basic_solver_t::num_failed_return++;
        }

        return false;
    }

    bool solve(Board & board) {
        this->init_board(board);
        bool success = this->solve(board, this->empties_);
        return success;
    }

    void display_result(Board & board, double elapsed_time,
                        bool print_answer = true,
                        bool print_all_answers = true) {
        basic_solver_t::display_result<kSearchMode>(board, elapsed_time, print_answer, print_all_answers);
    }
};

} // namespace v2
} // namespace jmSudoku

#endif // JM_SUDOKU_SOLVER_V2_H
