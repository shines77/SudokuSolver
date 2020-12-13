
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
    static const size_t TotalSize2 = Sudoku::TotalSize2;

    static const size_t kAllRowsBit = Sudoku::kAllRowsBit;
    static const size_t kAllColsBit = Sudoku::kAllColsBit;
    static const size_t kAllBoxesBit = Sudoku::kAllBoxesBit;
    static const size_t kAllNumbersBit = Sudoku::kAllNumbersBit;

    typedef Solver slover_type;

    static size_t num_guesses;
    static size_t num_unique_candidate;
    static size_t num_early_return;

private:
    SmallBitSet<BoardSize>  cell_filled_;

    SmallBitSet2D<Cols, Numbers>    rows_;
    SmallBitSet2D<Rows, Numbers>    cols_;
    SmallBitSet2D<Boxes, Numbers>   boxes_;

    SmallBitSet2D<BoardSize, Numbers>       cell_nums_;     // [row * 9 + col][num]
    SmallBitSet3D<Boxes, BoxSize, Numbers>  box_cells_;     // [box][box_size][num]
    SmallBitSet3D<Boxes, Numbers, BoxSize>  box_nums_;      // [box][num][box_size]
    SmallBitSet2D<Numbers, BoardSize>       nums_pos_;      // [box][row * 9 + col], normal
    SmallBitSet2D<Numbers, BoardSize>       nums_pos_rt_;   // [box][col * 9 + row], rotate

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

    inline void fillNum(size_t row, size_t col, size_t box, uint32_t num_bit) {
        this->cell_filled_.set(row * Cols + col);

        this->rows_[row] ^= num_bit;
        this->cols_[col] ^= num_bit;
        this->boxes_[box] ^= num_bit;
    }

    void init_board(char board[BoardSize]) {
        this->cell_filled_.reset();

        this->rows_.set();
        this->cols_.set();
        this->boxes_.set();

        size_t empties = 0;
        size_t pos = 0;
        for (size_t row = 0; row < Rows; row++) {
            size_t box_base = row / 3 * 3;
            for (size_t col = 0; col < Cols; col++) {
                unsigned char val = board[pos++];
                if (val == '.') {
                    empties++;
                }
                else {
                    size_t box = box_base + col / 3;
                    uint32_t num = val - '1';
                    uint32_t num_bit = 1u << num;
                    this->fillNum(row, col, box, num_bit);
                }
            }
        }

        this->empties_ = empties;
    }

    void setup_state(char board[BoardSize]) {
        size_t pos = 0;
        for (size_t row = 0; row < Rows; row++) {
            size_t box_base = row / 3 * 3;
            size_t box_cell_y_base = (row % 3) * 3;
            for (size_t col = 0; col < Cols; col++) {
                char val = board[pos];
                if (val == '.') {
                    // Get can fill numbers each cell.
                    size_t box = box_base + col / 3;
                    SmallBitSet<Numbers> bitNums = getCanFillNums(row, col, box);
                    this->cell_nums_[pos] = bitNums;

                    // Get can fill positions each number in the same box.
                    size_t box_cell_x = (col % 3);
                    size_t box_pos = box_cell_y_base + box_cell_x;

                    SmallBitSet<Numbers> bitBoxNums = this->boxes_[box];
                    bitBoxNums &= bitNums;
                    this->box_cells_[box][box_pos] = bitBoxNums;

                    for (size_t num = 0; num < Numbers; num++) {
                        bool canFill = bitBoxNums.test(num);
                        if (canFill) {
                            this->box_nums_[box][num].set(box_pos);
                        }
                    }

                    for (size_t num = 0; num < Numbers; num++) {
                        bool canFill = bitNums.test(num);
                        if (canFill) {
                            this->nums_pos_[num].set(pos);
                            this->nums_pos_rt_[num].set(col * Rows + row);
                        }
                    }
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
