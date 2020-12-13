
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
//#define __MMX__
//#define __SSE__
//#define __SSE2__
//#define __SSE3__
//#define __SSSE3__
//#define __SSE4A__
//#define __SSE4a__
//#define __SSE4_1__
//#define __SSE4_2__
//#define __POPCNT__
//#define __LZCNT__
//#define __AVX__
//#define __AVX2__
//#define __3dNOW__
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
#include <cstring>      // For std::memset()
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

#define V1_SEARCH_MODE      SEARCH_MODE_ONE_ANSWER

namespace jmSudoku {
namespace v1 {

static const size_t kSearchMode = V1_SEARCH_MODE;

class Solver {
public:
    static const size_t Rows = Sudoku::Rows;
    static const size_t Cols = Sudoku::Cols;
    static const size_t Boxes = Sudoku::Boxes;
    static const size_t BoxSize = Sudoku::BoxSize;
    static const size_t Numbers = Sudoku::Numbers;
    static const size_t MinNumber = Sudoku::MinNumber;
    static const size_t MaxNumber = Sudoku::MaxNumber;

    static const size_t BoardSize = Sudoku::BoardSize;
    static const size_t TotalSize = Sudoku::TotalSize;

    static const size_t TotalConditions0 = 0;
    static const size_t TotalConditions1 = Rows * Cols;
    static const size_t TotalConditions2 = Boxes * Numbers;
    static const size_t TotalConditions3 = Rows * Numbers;
    static const size_t TotalConditions4 = Cols * Numbers;

    static const size_t TotalConditions01 = TotalConditions0  + TotalConditions1;
    static const size_t TotalConditions02 = TotalConditions01 + TotalConditions2;
    static const size_t TotalConditions03 = TotalConditions02 + TotalConditions3;
    static const size_t TotalConditions04 = TotalConditions03 + TotalConditions4;

    static const size_t TotalConditions =
        TotalConditions1 + TotalConditions2 + TotalConditions3 + TotalConditions4;

    static const size_t kAllRowsBit = Sudoku::kAllRowsBit;
    static const size_t kAllColsBit = Sudoku::kAllColsBit;
    static const size_t kAllBoxesBit = Sudoku::kAllBoxesBit;
    static const size_t kAllNumbersBit = Sudoku::kAllNumbersBit;

    typedef Solver slover_type;

    static size_t num_guesses;
    static size_t num_unique_candidate;
    static size_t num_early_return;

private:
#pragma pack(push, 1)
    struct literal_info_t {
        uint8_t count;
        uint8_t enable;
    };
#pragma pack(pop)

    SmallBitSet<BoardSize>  cell_filled_;

    SmallBitSet2D<Boxes, Numbers>   boxes_;
    SmallBitSet2D<Cols, Numbers>    rows_;
    SmallBitSet2D<Rows, Numbers>    cols_;

    alignas(16) SmallBitSet2D<BoardSize, Numbers>       cell_nums_;     // [row * 9 + col][num]

    alignas(16) SmallBitSet3D<Boxes, Numbers, BoxSize>  box_nums_;      // [box][num][box_size]
    alignas(16) SmallBitSet3D<Rows,  Numbers, Cols>     row_nums_;      // [row][num][col]
    alignas(16) SmallBitSet3D<Cols,  Numbers, Rows>     col_nums_;      // [col][num][row]

#if defined(__SSE4_1__)
    alignas(16) literal_info_t literal_info_[TotalConditions];
#else
    alignas(16) uint8_t literal_cnt_[TotalConditions];
    alignas(16) uint8_t literal_enable_[TotalConditions];
#endif

    size_t empties_;

    std::vector<SudokuBoard<BoardSize>> answers_;

public:
    Solver() : empties_(0) {
    }
    ~Solver() {}

    static size_t get_num_guesses() { return slover_type::num_guesses; }
    static size_t get_num_unique_candidate() { return slover_type::num_unique_candidate; }
    static size_t get_num_early_return() { return slover_type::num_early_return; }

    static size_t get_search_counter() {
        return (slover_type::num_guesses + slover_type::num_unique_candidate + slover_type::num_early_return);
    }

    static double get_guess_percent() {
        return calc_percent(slover_type::num_guesses, slover_type::get_search_counter());
    }

    static double get_early_return_percent() {
        return calc_percent(slover_type::num_early_return, slover_type::get_search_counter());
    }

    static double get_unique_candidate_percent() {
        return calc_percent(slover_type::num_unique_candidate, slover_type::get_search_counter());
    }

private:
#if defined(__SSE4_1__)

    inline void enable_literal(size_t literal) {
        this->literal_info_[literal].enable = 0x00;
    }

    inline void disable_literal(size_t literal) {
        this->literal_info_[literal].enable = 0xFF;
    }

    inline uint8_t get_literal_cnt(size_t literal) {
        return this->literal_info_[literal].count;
    }

    inline void inc_literal_cnt(size_t literal) {
        this->literal_info_[literal].count++;
    }

    inline void dec_literal_cnt(size_t literal) {
        this->literal_info_[literal].count--;
    }

#else // !__SSE4_1__

    inline void enable_literal(size_t literal) {
        this->literal_enable_[literal] = 0x00;
    }

    inline void disable_literal(size_t literal) {
        this->literal_enable_[literal] = 0xF0;
    }

    inline uint8_t get_literal_cnt(size_t literal) {
        return this->literal_count_[literal];
    }

    inline void inc_literal_cnt(size_t literal) {
        this->literal_count_[literal]++;
    }

    inline void dec_literal_cnt(int literal) {
        this->literal_count_[literal]--;
    }

#endif // __SSE4_1__

    // enable_xxxx_literal()
    inline void enable_cell_literal(size_t pos) {
        size_t literal = TotalConditions0 + pos;
        this->enable_literal(literal);
    }

    inline void enable_box_literal(size_t box, size_t num) {
        size_t literal = TotalConditions01 + box * Numbers + num;
        this->enable_literal(literal);
    }

    inline void enable_row_literal(size_t row, size_t num) {
        size_t literal = TotalConditions02 + row * Numbers + num;
        this->enable_literal(literal);
    }

    inline void enable_col_literal(size_t col, size_t num) {
        size_t literal = TotalConditions03 + col * Numbers + num;
        this->enable_literal(literal);
    }

    // disable_xxxx_literal()
    inline void disable_cell_literal(size_t pos) {
        size_t literal = TotalConditions0 + pos;
        this->disable_literal(literal);
    }

    inline void disable_box_literal(size_t box, size_t num) {
        size_t literal = TotalConditions01 + box * Numbers + num;
        this->disable_literal(literal);
    }

    inline void disable_row_literal(size_t row, size_t num) {
        size_t literal = TotalConditions02 + row * Numbers + num;
        this->disable_literal(literal);
    }

    inline void disable_col_literal(size_t col, size_t num) {
        size_t literal = TotalConditions03 + col * Numbers + num;
        this->disable_literal(literal);
    }

    // inc_xxxx_literal_cnt()
    inline void inc_cell_literal_cnt(size_t pos) {
        size_t literal = TotalConditions0 + pos;
        this->inc_literal_cnt(literal);
    }

    inline void inc_box_literal_cnt(size_t box, size_t num) {
        size_t literal = TotalConditions01 + box * Numbers + num;
        this->inc_literal_cnt(literal);
    }

    inline void inc_row_literal_cnt(size_t row, size_t num) {
        size_t literal = TotalConditions02 + row * Numbers + num;
        this->inc_literal_cnt(literal);
    }

    inline void inc_col_literal_cnt(size_t col, size_t num) {
        size_t literal = TotalConditions03 + col * Numbers + num;
        this->inc_literal_cnt(literal);
    }

    // dec_xxxx_literal_cnt()
    inline void dec_cell_literal_cnt(size_t pos) {
        size_t literal = TotalConditions0 + pos;
        this->dec_literal_cnt(literal);
    }

    inline void dec_box_literal_cnt(size_t box, size_t num) {
        size_t literal = TotalConditions01 + box * Numbers + num;
        this->dec_literal_cnt(literal);
    }

    inline void dec_row_literal_cnt(size_t row, size_t num) {
        size_t literal = TotalConditions02 + row * Numbers + num;
        this->dec_literal_cnt(literal);
    }

    inline void dec_col_literal_cnt(size_t col, size_t num) {
        size_t literal = TotalConditions03 + col * Numbers + num;
        this->dec_literal_cnt(literal);
    }

    inline SmallBitSet<Numbers> getCanFillNums(size_t row, size_t col, size_t box) {
        return (this->rows_[row] & this->cols_[col] & this->boxes_[box]);
    }

    inline void fillNum(size_t row, size_t col, size_t box, size_t box_pos, size_t num) {
        uint32_t num_bit = 1u << num;
        this->boxes_[box] ^= num_bit;
        this->rows_[row] ^= num_bit;
        this->cols_[col] ^= num_bit;

        size_t pos = row * Cols + col;
        this->cell_filled_.set(row * Cols + col);

        for (size_t _num = MinNumber - 1; _num < MaxNumber; _num++) {
            this->box_nums_[box][_num].reset(box_pos);
            this->row_nums_[row][_num].reset(col);
            this->col_nums_[col][_num].reset(row);
        }

        disable_cell_literal(pos);
        disable_box_literal(box, num);
        disable_row_literal(row, num);
        disable_col_literal(col, num);
    }

    inline void doFillNum(size_t row, size_t col, size_t box, size_t box_pos, size_t num) {
        uint32_t num_bit = 1u << num;
        this->boxes_[box] ^= num_bit;
        this->rows_[row] ^= num_bit;
        this->cols_[col] ^= num_bit;

        size_t pos = row * Cols + col;
        this->cell_filled_.set(pos);

        disable_cell_literal(pos);
        disable_box_literal(box, num);
        disable_row_literal(row, num);
        disable_col_literal(col, num);

        size_t box_nums = this->boxes_[box].value_sz();
        while (box_nums != 0) {
            size_t bit_num = BitUtils::ms1b(box_nums);
            size_t _num = BitUtils::bsf(bit_num);
            this->box_nums_[box][_num].reset(box_pos);
            dec_box_literal_cnt(box, _num);
            box_nums ^= bit_num;
        }

        size_t row_nums = this->rows_[row].value_sz();
        while (row_nums != 0) {
            size_t bit_num = BitUtils::ms1b(row_nums);
            size_t _num = BitUtils::bsf(bit_num);
            this->row_nums_[row][_num].reset(col);
            dec_row_literal_cnt(row, _num);
            row_nums ^= bit_num;
        }

        size_t col_nums = this->cols_[col].value_sz();
        while (col_nums != 0) {
            size_t bit_num = BitUtils::ms1b(col_nums);
            size_t _num = BitUtils::bsf(bit_num);
            this->col_nums_[col][_num].reset(row);
            dec_box_literal_cnt(col, _num);
            col_nums ^= bit_num;
        }
    }

    void init_board(char board[BoardSize]) {
        this->cell_filled_.reset();

        this->boxes_.set();
        this->rows_.set();
        this->cols_.set();

        this->cell_nums_.reset();

        this->box_nums_.set();
        this->row_nums_.set();
        this->col_nums_.set();

#if defined(__SSE4_1__)
        std::memset((void *)&this->literal_info_[0], 0, sizeof(this->literal_info_));
#else
        std::memset((void *)&this->literal_enable_[0], 0, sizeof(this->literal_enable_));
#endif

        size_t empties = 0;
        size_t pos = 0;
        for (size_t row = 0; row < Rows; row++) {
            size_t box_base = row / 3 * 3;
            size_t box_cell_y_base = (row % 3) * 3;
            for (size_t col = 0; col < Cols; col++) {
                unsigned char val = board[pos++];
                if (val == '.') {
                    empties++;
                }
                else {
                    size_t box = box_base + col / 3;
                    size_t box_cell_x = (col % 3);
                    size_t box_pos = box_cell_y_base + box_cell_x;
                    size_t num = val - '1';
                    this->fillNum(row, col, box, box_pos, num);
                }
            }
        }

        this->empties_ = empties;
    }

    void setup_state(char board[BoardSize]) {
        size_t pos = 0;
        for (size_t row = 0; row < Rows; row++) {
            size_t box_base = row / 3 * 3;
            for (size_t col = 0; col < Cols; col++) {
                char val = board[pos];
                if (val == '.') {
                    // Get can fill numbers each cell.
                    size_t box = box_base + col / 3;
                    SmallBitSet<Numbers> bitNums = getCanFillNums(row, col, box);
                    this->cell_nums_[pos] = bitNums;
                }
                pos++;
            }
        }

        this->calc_literal_count();
    }

    void calc_literal_count() {
        size_t pos = 0;
        for (size_t row = 0; row < Rows; row++) {
            size_t box_base = row / 3 * 3;
            for (size_t col = 0; col < Cols; col++) {
                uint8_t cell_nums = (uint8_t)this->cell_nums_[pos].count();
#if defined(__SSE4_1__)
                this->literal_info_[TotalConditions0 + pos].count = cell_nums;
#else
                this->literal_count_[TotalConditions0 + pos] = cell_nums;
#endif
                pos++;
                size_t box = box_base + col / 3;
                for (size_t num = MinNumber - 1; num < MaxNumber; num++) {
                    uint8_t box_nums = (uint8_t)this->box_nums_[box][num].count();
                    uint8_t row_nums = (uint8_t)this->row_nums_[row][num].count();
                    uint8_t col_nums = (uint8_t)this->col_nums_[col][num].count();
#if defined(__SSE4_1__)
                    this->literal_info_[TotalConditions01 + box * Numbers + num].count = box_nums;
                    this->literal_info_[TotalConditions02 + row * Numbers + num].count = row_nums;
                    this->literal_info_[TotalConditions03 + col * Numbers + num].count = col_nums;
#else
                    this->literal_count_[TotalConditions01 + box * Numbers + num] = box_nums;
                    this->literal_count_[TotalConditions02 + row * Numbers + num] = row_nums;
                    this->literal_count_[TotalConditions03 + col * Numbers + num] = col_nums;
#endif
                }
            }
        }
    }

public:
    bool solve() {
        return false;
    }

    bool solve(char board[BoardSize],
               double & elapsed_time,
               bool verbose = true) {
        if (verbose) {
            Sudoku::display_board(board, true);
        }

        jtest::StopWatch sw;
        sw.start();

        this->init_board(board);
        this->setup_state(board);

        bool success = this->solve();

        sw.stop();
        elapsed_time = sw.getElapsedMillisec();

        if (verbose) {
            if (kSearchMode > SearchMode::OneAnswer)
                Sudoku::display_boards(this->answers_);
            else
                Sudoku::display_board(board);
            printf("elapsed time: %0.3f ms, recur_counter: %" PRIuPTR "\n\n"
                   "num_guesses: %" PRIuPTR ", num_early_return: %" PRIuPTR ", num_unique_candidate: %" PRIuPTR "\n"
                   "guess %% = %0.1f %%, early_return %% = %0.1f %%, unique_candidate %% = %0.1f %%\n\n",
                   elapsed_time, slover_type::get_search_counter(),
                   slover_type::get_num_guesses(),
                   slover_type::get_num_early_return(),
                   slover_type::get_num_unique_candidate(),
                   slover_type::get_guess_percent(),
                   slover_type::get_early_return_percent(),
                   slover_type::get_unique_candidate_percent());
        }

        return success;
    }
};

size_t Solver::num_guesses = 0;
size_t Solver::num_unique_candidate = 0;
size_t Solver::num_early_return = 0;

} // namespace v1
} // namespace jmSudoku

#endif // JM_SUDOKU_SOLVER_V1_H
