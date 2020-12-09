
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <memory.h>
#include <assert.h>

#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <cstddef>
#include <iostream>
#include <fstream>
#include <cstring>      // For std::memset()
#include <vector>
#include <bitset>

#include "Sudoku.h"
#include "TestCase.h"

#include "SudokuSolver_v1.h"
#include "SudokuSolver_v2.h"
#include "SudokuSolver_v3.h"
//#include "SudokuSolver_v4.h"

#include "CPUWarmUp.h"
#include "StopWatch.h"

using namespace jmSudoku;

static const size_t kEnableV1Solution =   1;
static const size_t kEnableV2Solution =   1;
static const size_t kEnableV3Solution =   1;
static const size_t kEnableV4Solution =   1;

// Index: [0 - 4]
#define TEST_CASE_INDEX         4

void read_sudoku_board(char board[Sudoku::BoardSize], size_t index)
{
    for (size_t row = 0; row < Sudoku::Rows; row++) {
        size_t row_base = row * 9;
        size_t col = 0;
        const char * prows = test_case[index].rows[row];
        char val;
        while ((val = *prows) != '\0') {
            if (val >= '0' && val <= '9') {
                if (val != '0')
                    board[row_base + col] = val;
                else
                    board[row_base + col] = '.';
                col++;
                assert(col <= Sudoku::Cols);
            }
            else if (val == '.') {
                board[row_base + col] = '.';
                col++;
                assert(col <= Sudoku::Cols);
            }
            prows++;  
        }
        assert(col == Sudoku::Cols);
    }
}

void read_sudoku_board(std::vector<std::vector<char>> & board, size_t index)
{
    for (size_t row = 0; row < Sudoku::Rows; row++) {
        std::vector<char> line;
        size_t col = 0;
        const char * prows = test_case[index].rows[row];
        char val;
        while ((val = *prows) != '\0') {
            if (val >= '0' && val <= '9') {
                if (val != '0')
                    line.push_back(val);
                else
                    line.push_back('.');
                col++;
                assert(col <= Sudoku::Cols);
            }
            else if (val == '.') {
                line.push_back('.');
                col++;
                assert(col <= Sudoku::Cols);
            }
            prows++;  
        }
        assert(col == Sudoku::Cols);
        board.push_back(line);
    }
}

size_t read_sudoku_board(char board[Sudoku::BoardSize], char line[256])
{
    char * pline = line;
    // Skip the white spaces
    while (*pline == ' ' || *pline == '\t') {
        pline++;
    }
    // Is a comment ?
    if ((*pline == '#') || ((*pline == '/') && (pline[1] = '/'))) {
        return 0;
    }

    size_t pos = 0;
    char val;
    while ((val = *pline++) != '\0') {
        if (val >= '0' && val <= '9') {
            if (val != '0')
                board[pos] = val;
            else
                board[pos] = '.';
            pos++;
            assert(pos <= Sudoku::BoardSize);
        }
        else if ((val == '.') || (val == ' ') || (val == '-')) {
            board[pos] = '.';
            pos++;
            assert(pos <= Sudoku::BoardSize);
        }
    }
    assert(pos <= Sudoku::BoardSize);
    return pos;
}

size_t read_sudoku_board(std::vector<std::vector<char>> & board, char line[256])
{
    char * pline = line;
    // Skip the white spaces
    while (*pline == ' ' || *pline == '\t') {
        pline++;
    }
    // Is a comment ?
    if ((*pline == '#') || ((*pline == '/') && (pline[1] = '/'))) {
        return 0;
    }
    size_t grid_nums = 0;
    for (size_t row = 0; row < Sudoku::Rows; row++) {
        std::vector<char> line;
        size_t col_valid = 0;
        for (size_t col = 0; col < Sudoku::Cols; col++) {
            char val = *pline;
            if (val >= '0' && val <= '9') {
                if (val != '0')
                    line.push_back(val);
                else
                    line.push_back('.');
                col_valid++;
                assert(col <= Sudoku::Cols);
            }
            else if (val == '.') {
                line.push_back('.');
                col_valid++;
                assert(col <= Sudoku::Cols);
            }
            else if (val == '\0') {
                break;
            }
            pline++;  
        }
        assert(col_valid == Sudoku::Cols || col_valid == 0);
        board.push_back(line);
        grid_nums += col_valid;
    }
    return grid_nums;
}

void run_a_testcase(size_t index)
{
    double elapsed_time = 0.0;
    if (kEnableV1Solution)
    {
        printf("------------------------------------------\n\n");
        printf("jmSudoku: v1::Solver - Dancing Links\n\n");

        char board[Sudoku::BoardSize];
        read_sudoku_board(board, index);

        v1::Solver solver;
        bool success = solver.solve(board, elapsed_time);
    }

    if (kEnableV2Solution)
    {
        printf("------------------------------------------\n\n");
        printf("jmSudoku: v2::Solution - dfs\n\n");

        char board[Sudoku::BoardSize];
        read_sudoku_board(board, index);

        v2::Solver solver;
        bool success = solver.solve(board, elapsed_time);
    }

    if (kEnableV3Solution)
    {
        printf("------------------------------------------\n\n");
        printf("jmSudoku: v3::Solution - dfs\n\n");

        char board[Sudoku::BoardSize];
        read_sudoku_board(board, index);

        v3::Solver solver;
        bool success = solver.solve(board, elapsed_time);
    }

#if 0
#ifdef NDEBUG

    if (kEnableV4Solution)
    {
        printf("------------------------------------------\n\n");
        printf("jmSudoku: v4::Solution - dfs\n\n");

        std::vector<std::vector<char>> board;
        read_sudoku_board(board, index);

        v4::Solver solver;
        bool success = solver.solve(board, elapsed_time);
    }
#endif
#endif

    printf("------------------------------------------\n\n");
}

template <typename SudokuSolver>
void run_sudoku_test(const char * filename, const char * name)
{
    typedef typename SudokuSolver::algorithm algorithm;

    //printf("------------------------------------------\n\n");
    printf("jmSudoku: %s::Solver\n\n", name);

    size_t total_guesses = 0;
    size_t total_unique_candidate = 0;
    size_t total_early_return = 0;
    size_t total_no_guess = 0;

    size_t puzzleCount = 0;
    double total_time = 0.0;

    std::ifstream ifs;
    try {
        ifs.open(filename, std::ios::in);
        if (ifs.good()) {
            while (!ifs.eof()) {
                char line[256];
                std::memset(line, 0, 16);
                ifs.getline(line, sizeof(line) - 1);

                char board[Sudoku::BoardSize];
                size_t num_grids = read_sudoku_board(board, line);
                // Sudoku::BoardSize = 81
                if (num_grids >= Sudoku::BoardSize) {
                    SudokuSolver solver;
                    double elapsed_time;
                    bool success = solver.solve(board, elapsed_time, false);
                    total_time += elapsed_time;
                    if (success) {
                        total_guesses += algorithm::num_guesses;
                        total_unique_candidate += algorithm::num_unique_candidate;
                        total_early_return += algorithm::num_early_return;

                        if (algorithm::num_guesses == 0) {
                            total_no_guess++;
                        }

                        puzzleCount++;
#ifndef NDEBUG
                        if (puzzleCount > 100)
                            break;
#endif
                    }
                }
            }
            ifs.close();
        }
    }
    catch (std::exception & ex) {
        std::cout << "Exception info: " << ex.what() << std::endl << std::endl;
    }

    size_t recur_counter = total_guesses + total_unique_candidate + total_early_return;
    double unique_candidate_percent = calc_percent(total_unique_candidate, recur_counter);
    double early_return_percent = calc_percent(total_early_return, recur_counter);
    double guesses_percent = calc_percent(total_guesses, recur_counter);
    double no_guess_percent = calc_percent(total_no_guess, puzzleCount);

    printf("Total puzzle count = %u, total_no_guess: %" PRIuPTR ", no_guess %% = %0.1f %%\n\n",
           (uint32_t)puzzleCount, total_no_guess, no_guess_percent);
    printf("Total elapsed time: %0.3f ms\n\n", total_time);
    printf("recur_counter: %" PRIuPTR "\n\n"
           "total_guesses: %" PRIuPTR ", total_early_return: %" PRIuPTR ", total_unique_candidate: %" PRIuPTR "\n\n"
           "guess %% = %0.1f %%, early_return %% = %0.1f %%, unique_candidate %% = %0.1f %%\n\n",
           recur_counter,
           total_guesses, total_early_return, total_unique_candidate,
           guesses_percent, early_return_percent, unique_candidate_percent);

    if (puzzleCount != 0) {
        printf("%0.1f usec/puzzle, %0.2f guesses/puzzle, %0.1f puzzles/sec\n\n",
               total_time * 1000.0 / puzzleCount,
               (double)total_guesses / puzzleCount,
               puzzleCount / (total_time / 1000.0));
    }

    printf("------------------------------------------\n\n");
}

int main(int argc, char * argv[])
{
    const char * filename = nullptr;
    const char * out_file = nullptr;
    if (argc > 2) {
        filename = argv[1];
        out_file = argv[2];
    }
    else if (argc > 1) {
        filename = argv[1];
    }

    jtest::CPU::warmup(1000);

    if (1)
    {
        if (filename == nullptr) {
            run_a_testcase(TEST_CASE_INDEX);
        }
    }

    if (1)
    {
        if (filename != nullptr) {
            run_sudoku_test<v1::Solver>(filename, "v1");
            run_sudoku_test<v2::Solver>(filename, "v2");
            run_sudoku_test<v3::Solver>(filename, "v3");
        }
    }

#if !defined(NDEBUG) && defined(_MSC_VER)
    ::system("pause");
#endif

    return 0;
}
