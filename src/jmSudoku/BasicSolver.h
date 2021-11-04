
#ifndef JM_BASIC_SOLVER_H
#define JM_BASIC_SOLVER_H

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
#pragma once
#endif

#include <stdint.h>
#include <stddef.h>
#include <inttypes.h>

#include <cstdint>
#include <cstddef>
#include <vector>

#include "Sudoku.h"

namespace jmSudoku {

template <typename SudokuTy>
class BasicSolver {
public:
    typedef SudokuTy                            sudoku_t;
    typedef BasicSolver<SudokuTy>               this_type;
    typedef typename sudoku_t::board_type       Board;

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

    static size_t num_guesses;
    static size_t num_unique_candidate;
    static size_t num_failed_return;

protected:
    size_t              empties_;
    std::vector<Board>  answers_;

public:
    BasicSolver() : empties_(0) {
        init_statistics();
    }
    ~BasicSolver() {}

    static size_t get_num_guesses() { return this_type::num_guesses; }
    static size_t get_num_unique_candidate() { return this_type::num_unique_candidate; }
    static size_t get_num_failed_return() { return this_type::num_failed_return; }

    static size_t get_total_search_counter() {
        return (this_type::num_guesses + this_type::num_unique_candidate + this_type::num_failed_return);
    }

    static double get_guess_percent() {
        return calc_percent(this_type::num_guesses, this_type::get_total_search_counter());
    }

    static double get_failed_return_percent() {
        return calc_percent(this_type::num_failed_return, this_type::get_total_search_counter());
    }

    static double get_unique_candidate_percent() {
        return calc_percent(this_type::num_unique_candidate, this_type::get_total_search_counter());
    }

private:
    void init_statistics() {
        num_guesses = 0;
        num_unique_candidate = 0;
        num_failed_return = 0;
    }

public:
    size_t calc_empties(Board & board) {
        size_t empties = 0;
        for (size_t pos = 0; pos < BoardSize; pos++) {
            unsigned char val = board.cells[pos];
            if (val == '.') {
                empties++;
            }
        }
        return empties;
    }

    void display_board(Board & board) {
        sudoku_t::display_board(board, true);
    }

    template <size_t nSearchMode = SearchMode::OneAnswer>
    void display_result_impl(Board & board, double elapsed_time,
                             bool print_answer,
                             bool print_all_answers) {
        if (print_answer) {
            if (nSearchMode == SearchMode::OneAnswer)
                sudoku_t::display_board(board);
            else
                sudoku_t::display_boards(this->answers_);
        }
        printf("elapsed time: %0.3f ms, recur_counter: %" PRIuPTR "\n\n"
               "num_guesses: %" PRIuPTR ", num_failed_return: %" PRIuPTR ", num_unique_candidate: %" PRIuPTR "\n"
               "guess %% = %0.1f %%, failed_return %% = %0.1f %%, unique_candidate %% = %0.1f %%\n\n",
                elapsed_time,
                this_type::get_total_search_counter(),
                this_type::get_num_guesses(),
                this_type::get_num_failed_return(),
                this_type::get_num_unique_candidate(),
                this_type::get_guess_percent(),
                this_type::get_failed_return_percent(),
                this_type::get_unique_candidate_percent());
    }
};

} // namespace jmSudoku

#endif // JM_BASIC_SOLVER_H
