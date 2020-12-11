
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
    static const size_t Numbers = Sudoku::Numbers;

    static const size_t TotalSize = Sudoku::TotalSize;
    static const size_t TotalSize2 = Sudoku::TotalSize2;

    typedef Solver slover_type;

    static size_t num_guesses;
    static size_t num_unique_candidate;
    static size_t num_early_return;

private:
    SmallBitSet<81> cell_filled_;

#if 1
    SmallBitMatrix2<Sudoku::Cols, Sudoku::Numbers, SmallBitSet<Sudoku::Numbers>>    rows_;
    SmallBitMatrix2<Sudoku::Rows, Sudoku::Numbers, SmallBitSet<Sudoku::Numbers>>    cols_;
    SmallBitMatrix2<Sudoku::Boxes, Sudoku::Numbers, SmallBitSet<Sudoku::Numbers>>   boxes_;
#else
    std::array<uint32_t, Sudoku::Cols>      rows_;
    std::array<uint32_t, Sudoku::Rows>      cols_;
    std::array<uint32_t, Sudoku::Boxes>     boxes_;
#endif

    size_t empties_;

    std::vector<SudokuBoard<Sudoku::BoardSize>> answers_;

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
    inline void fillNum(size_t row, size_t col, size_t box, uint32_t num_bit) {
        this->cell_filled_.set(row * Sudoku::Cols + col);
        this->rows_[row] ^= num_bit;
        this->cols_[col] ^= num_bit;
        this->boxes_[box] ^= num_bit;
    }

    void init_board(char board[Sudoku::BoardSize]) {
        this->cell_filled_.reset();

        this->rows_.fill(Sudoku::kAllNumbersBit);
        this->cols_.fill(Sudoku::kAllNumbersBit);
        this->boxes_.fill(Sudoku::kAllNumbersBit);

        size_t empties = 0;
        size_t pos = 0;
        for (size_t row = 0; row < Sudoku::Rows; row++) {
            size_t box_base = row / 3 * 3;
            for (size_t col = 0; col < Sudoku::Cols; col++) {
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

public:
    bool solve() {
        return false;
    }

    bool solve(char board[Sudoku::BoardSize],
               double & elapsed_time,
               bool verbose = true) {
        if (verbose) {
            Sudoku::display_board(board, true);
        }

        jtest::StopWatch sw;
        sw.start();

        this->init_board(board);
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
