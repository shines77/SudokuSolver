
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
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
//#include "SudokuSolver_v2.h"
//#include "SudokuSolver_v3.h"
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
    size_t grid_nums = 0;
    for (size_t row = 0; row < Sudoku::Rows; row++) {
        size_t row_base = row * 9;
        size_t col_valid = 0;
        for (size_t col = 0; col < Sudoku::Cols; col++) {
            char val = *pline;
            if (val >= '0' && val <= '9') {
                if (val != '0')
                    board[row_base + col_valid] = val;
                else
                    board[row_base + col_valid] = '.';
                col_valid++;
                assert(col <= Sudoku::Cols);
            }
            else if (val == '.') {
                board[row_base + col_valid] = '.';
                col_valid++;
                assert(col <= Sudoku::Cols);
            }
            else if (val == '\0') {
                break;
            }
            pline++;  
        }
        assert(col_valid == Sudoku::Cols || col_valid == 0);
        grid_nums += col_valid;
    }
    return grid_nums;
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

void test_one_case(size_t index)
{
    double elapsed_time = 0.0;
    if (kEnableV1Solution)
    {
        printf("--------------------------------\n\n");
        printf("jmSudoku: v1::Solver - Dancing Links\n\n");

        char board[Sudoku::BoardSize];
        read_sudoku_board(board, index);

        v1::Solver solver;
        bool success = solver.solve(board, elapsed_time);
    }

#if 0
#ifdef NDEBUG
    if (kEnableV2Solution)
    {
        printf("--------------------------------\n\n");
        printf("jmSudoku: v2::Solution - dfs\n\n");

        std::vector<std::vector<char>> board;
        read_sudoku_board(board, index);

        v2::Solver solver;
        bool success = solver.solve(board, elapsed_time);
    }

    if (kEnableV3Solution)
    {
        printf("--------------------------------\n\n");
        printf("jmSudoku: v3::Solution - dfs\n\n");

        std::vector<std::vector<char>> board;
        read_sudoku_board(board, index);

        v3::Solver solver;
        bool success = solver.solve(board, elapsed_time);
    }

    if (kEnableV4Solution)
    {
        printf("--------------------------------\n\n");
        printf("jmSudoku: v4::Solution - dfs\n\n");

        std::vector<std::vector<char>> board;
        read_sudoku_board(board, index);

        v4::Solver solver;
        bool success = solver.solve(board, elapsed_time);
    }
#endif
#endif

    printf("--------------------------------\n\n");
}

template <typename SudokuSolver>
void test_sudoku_files(const char * filename, const char * name)
{
    typedef typename SudokuSolver::algorithm algorithm;

    printf("--------------------------------\n\n");
    printf("jmSudoku: %s::Solver\n\n", name);

    size_t num_guesses = 0;
    size_t num_no_guess = 0;
    size_t num_impossibles = 0;

    size_t puzzleCount = 0;
    double total_time = 0.0;

    std::ifstream ifs;
    try {
        ifs.open(filename, std::ios::in);
        if (ifs.good()) {
            while (!ifs.eof()) {
                char line[128];
                std::memset(line, 0, 16);
                ifs.getline(line, sizeof(line) - 1);

                char board[Sudoku::BoardSize];
                size_t grid_nums = read_sudoku_board(board, line);
                // Sudoku::BoardSize = 81
                if (grid_nums >= Sudoku::BoardSize) {
                    SudokuSolver solver;
                    double elapsed_time;
                    bool success = solver.solve(board, elapsed_time, false);
                    total_time += elapsed_time;
                    if (success) {
                        num_guesses += algorithm::num_guesses;
                        num_no_guess += algorithm::num_no_guess;
                        num_impossibles += algorithm::num_impossibles;

                        puzzleCount++;
#ifndef NDEBUG
                        if (puzzleCount > 100)
                            break;
#endif
                    }
                }
                else break;
            }
            ifs.close();
        }
    }
    catch (std::exception & ex) {
        std::cout << "Exception info: " << ex.what() << std::endl << std::endl;
    }

    size_t recur_counter = num_guesses + num_no_guess + num_impossibles;
    double no_guess_percent = calc_percent(num_no_guess, recur_counter);
    double impossibles_percent = calc_percent(num_impossibles, recur_counter);
    double guesses_percent = calc_percent(num_guesses, recur_counter);

    printf("Total puzzle count = %u\n\n", (uint32_t)puzzleCount);
    printf("Total elapsed time: %0.3f ms\n\n", total_time);
    printf("recur_counter: %u\n\n"
           "num_guesses: %u, num_impossibles: %u, no_guess: %u\n"
           "guess%% = %0.1f %%, impossible%% = %0.1f %%, no_guess%% = %0.1f %%\n\n",
           (uint32_t)recur_counter,
           (uint32_t)num_guesses,
           (uint32_t)num_impossibles,
           (uint32_t)num_no_guess,
           guesses_percent, impossibles_percent, no_guess_percent);

    if (puzzleCount != 0) {
        printf("%0.1f usec/puzzle, %0.2f guesses/puzzle, %0.1f puzzles/sec\n\n",
               total_time * 1000.0 / puzzleCount,
               (double)num_guesses / puzzleCount,
               puzzleCount / (total_time / 1000.0));
    }
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
            test_one_case(TEST_CASE_INDEX);
        }
    }

    if (1)
    {
        if (filename != nullptr) {
            test_sudoku_files<v1::Solver>(filename, "v1");
        }
    }

#if !defined(NDEBUG) && defined(_MSC_VER)
    ::system("pause");
#endif

    return 0;
}
