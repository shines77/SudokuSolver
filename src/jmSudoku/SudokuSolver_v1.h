
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
        uint8_t size;
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
    inline SmallBitSet<Numbers> getCanFillNums(size_t row, size_t col, size_t box) {
        return (this->rows_[row] & this->cols_[col] & this->boxes_[box]);
    }

    inline void fillNum(size_t row, size_t col, size_t box, size_t box_pos, size_t num) {
        uint32_t num_bit = 1u << num;
        this->boxes_[box] ^= num_bit;
        this->rows_[row] ^= num_bit;
        this->cols_[col] ^= num_bit;

        this->cell_filled_.set(row * Cols + col);
        this->box_nums_[box][num].reset(box_pos);
        this->row_nums_[row][num].reset(col);
        this->col_nums_[col][num].reset(row);

#if defined(__SSE4_1__)
        this->literal_info_[TotalConditions0  + row * Cols    + col].enable = 0xFF;
        this->literal_info_[TotalConditions01 + box * Numbers + num].enable = 0xFF;
        this->literal_info_[TotalConditions02 + row * Numbers + num].enable = 0xFF;
        this->literal_info_[TotalConditions03 + col * Numbers + num].enable = 0xFF;
#else
        this->literal_enable_[TotalConditions0  + row * Cols    + col] = 0xF0;
        this->literal_enable_[TotalConditions01 + box * Numbers + num] = 0xF0;
        this->literal_enable_[TotalConditions02 + row * Numbers + num] = 0xF0;
        this->literal_enable_[TotalConditions03 + col * Numbers + num] = 0xF0;
#endif
    }

    inline void doFillNum(size_t row, size_t col, size_t box, size_t box_pos, size_t num) {
        uint32_t num_bit = 1u << num;
        this->boxes_[box] ^= num_bit;
        this->rows_[row] ^= num_bit;
        this->cols_[col] ^= num_bit;

        this->cell_filled_.set(row * Cols + col);
        this->box_nums_[box][num].reset(box_pos);
        this->row_nums_[row][num].reset(col);
        this->col_nums_[col][num].reset(row);

#if defined(__SSE4_1__)
        this->literal_info_[TotalConditions0  + row * Cols    + col].enable = 0xFF;
        this->literal_info_[TotalConditions01 + box * Numbers + num].enable = 0xFF;
        this->literal_info_[TotalConditions02 + row * Numbers + num].enable = 0xFF;
        this->literal_info_[TotalConditions03 + col * Numbers + num].enable = 0xFF;
#else
        this->literal_enable_[TotalConditions0  + row * Cols    + col] = 0xF0;
        this->literal_enable_[TotalConditions01 + box * Numbers + num] = 0xF0;
        this->literal_enable_[TotalConditions02 + row * Numbers + num] = 0xF0;
        this->literal_enable_[TotalConditions03 + col * Numbers + num] = 0xF0;
#endif
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
