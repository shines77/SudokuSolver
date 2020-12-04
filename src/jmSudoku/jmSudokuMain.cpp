
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
#include <vector>
#include <bitset>

#include "Sudoku.h"

#include "SudokuSolver_v1.h"
//#include "SudokuSolver_v2.h"
//#include "SudokuSolver_v3.h"
//#include "SudokuSolver_v4.h"

#include "CPUWarmUp.h"
#include "StopWatch.h"

using namespace jmSudoku;

static const bool kEnableV1Solution =   true;
static const bool kEnableV2Solution =   false;
static const bool kEnableV3Solution =   false;
static const bool kEnableV4Solution =   false;

// Index: [0 - 4]
#define TEST_CASE_INDEX         4

struct SudokuBoard {
    const char * rows[9];
};

static SudokuBoard test_case[] = {
    //
    // 0 #
    //
    // Normal (filled = 30)
    // https://leetcode-cn.com/problems/sudoku-solver/
    //
    {
        "53. | .7. | ...",
        "6.. | 195 | ...",
        ".98 | ... | .6.",

        "8.. | .6. | ..3",
        "4.. | 8.3 | ..1",
        "7.. | .2. | ..6",

        ".6. | ... | 28.",
        "... | 419 | ..5",
        "... | .8. | .79",
    },

    //
    // 1 #
    //
    // Advance (filled = 24)
    // https://www.sudoku-cn.com/
    //
    {
        "4.2 | ... | 9..",
        "... | .61 | ...",
        ".19 | ... | ...",

        "7.5 | ... | 6..",
        "2.4 | 7.. | ..5",
        "... | .9. | 7..",

        ".8. | 2.9 | .1.",
        "..7 | ..4 | ...",
        "... | ... | .52",
    },

    //
    // 2 #
    //
    // Hard (filled = 21)
    // https://github.com/tropicalwzc/ice_sudoku.github.io/blob/master/sudoku_solver.c
    //
    {
        "5.. | ... | 3..",
        ".2. | 1.. | .7.",
        "..8 | ... | ..9",

        ".4. | ..7 | ...",
        "... | 821 | ...",
        "... | 6.. | .1.",

        "3.. | ... | 8..",
        ".6. | ..4 | .2.",
        "..9 | ... | ..5",
    },

    // Ice sudoku string: 500000300020100070008000009040007000000821000000600010300000800060004020009000005

    //
    // 3 #
    //
    // Hardcore (filled = 17)
    // http://www.cn.sudokupuzzle.org/play.php
    //
    {
        "5.. | ... | ...",
        ".1. | ... | 32.",
        "... | 84. | ...",

        "... | ... | ...",
        "... | ..3 | 1..",
        "6.8 | 5.. | ...",

        "..7 | ... | .68",
        ".34 | ..1 | ...",
        "... | ... | ...",
    },

    // Ice sudoku string: 500000000010000320000840000000000000000003100608500000007000068034001000000000000

    //
    // 4 #
    //
    // Hardcore (filled = 21)
    // http://news.sohu.com/20130527/n377158555.shtml
    // https://baike.baidu.com/reference/13848819/1bf4HJzRCPCNz9Rypz3HpTtnhc2MpcRr5JMIp0032uiuKPQm4eOMuq2WZWxf77V-UBRjIkyDf9CVZDEjlDeHJBaazlstk30qaDtt
    //
    {
        "8.. | ... | ...",
        "..3 | 6.. | ...",
        ".7. | .9. | 2..",

        ".5. | ..7 | ...",
        "... | .45 | 7..",
        "... | 1.. | .3.",

        "..1 | ... | .68",
        "..8 | 5.. | .1.",
        ".9. | ... | 4..",
    },


    //
    // 5 #, copy from # 4
    //
    // Hardcore (filled = 20)
    //
    {
        "8.. | ... | ...",
        "..3 | 6.. | ...",
        ".7. | .9. | 2..",

        ".5. | ..7 | ...",
        "... | .45 | 7..",
        "... | 1.. | .3.",

        "..1 | ... | ..8",
        "..8 | 5.. | .1.",
        ".9. | ... | 4..",
    },

    //
    // 6 #, copy from /puzzles/sudoku17.txt
    //
    // Hardcore (filled = 16)
    //
    {
        "... | ... | .13",
        ".2. | 5.. | ...",
        "... | ... | ...",
        "1.3 | ... | .7.",
        "... | 8.2 | ...",
        "..4 | ... | ...",
        "... | .4. | 5..",
        "67. | ... | 2..",
        "... | .1. | ...",
    },

    /*************************************************

    // Empty board format (For user custom and copy)
    {
        "... | ... | ...",
        "... | ... | ...",
        "... | ... | ...",

        "... | ... | ...",
        "... | ... | ...",
        "... | ... | ...",

        "... | ... | ...",
        "... | ... | ...",
        "... | ... | ...",
    },

    **************************************************/
};

void make_sudoku_board(std::vector<std::vector<char>> & board, size_t index)
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

size_t make_sudoku_board(std::vector<std::vector<char>> & board, char lines[128])
{
    size_t grid_nums = 0;
    for (size_t row = 0; row < Sudoku::Rows; row++) {
        std::vector<char> line;
        size_t valid = 0;
        const char * prows = &lines[row * 9];
        for (size_t col = 0; col < Sudoku::Cols; col++) {
            char val = *prows;
            if (val >= '0' && val <= '9') {
                if (val != '0')
                    line.push_back(val);
                else
                    line.push_back('.');
                valid++;
                assert(col <= Sudoku::Cols);
            }
            else if (val == '.') {
                line.push_back('.');
                valid++;
                assert(col <= Sudoku::Cols);
            }
            else if (val == '\0') {
                break;
            }
            prows++;  
        }
        assert(valid == Sudoku::Cols || valid == 0);
        board.push_back(line);
        grid_nums += valid;
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

        std::vector<std::vector<char>> board;
        make_sudoku_board(board, index);

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
        make_sudoku_board(board, index);

        v2::Solver solver;
        bool success = solver.solve(board, elapsed_time);
    }

    if (kEnableV3Solution)
    {
        printf("--------------------------------\n\n");
        printf("jmSudoku: v3::Solution - dfs\n\n");

        std::vector<std::vector<char>> board;
        make_sudoku_board(board, index);

        v3::Solver solver;
        bool success = solver.solve(board, elapsed_time);
    }

    if (kEnableV4Solution)
    {
        printf("--------------------------------\n\n");
        printf("jmSudoku: v4::Solution - dfs\n\n");

        std::vector<std::vector<char>> board;
        make_sudoku_board(board, index);

        v4::Solver solver;
        bool success = solver.solve(board, elapsed_time);
    }
#endif
#endif

    printf("--------------------------------\n\n");
}

double calc_percent(size_t num_val, size_t num_total) {
    if (num_total != 0)
        return (num_val * 100.0) / num_total;
    else
        return 0.0;
}

template <typename SudokuSolver>
void test_sudoku_files(const char * filename, const char * name)
{
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

                std::vector<std::vector<char>> board;
                size_t grid_nums = make_sudoku_board(board, line);
                // Sudoku::BoardSize = 81
                if (grid_nums >= Sudoku::BoardSize) {
                    SudokuSolver solver;
                    double elapsed_time;
                    bool success = solver.solve(board, elapsed_time, false);
                    total_time += elapsed_time;
                    if (success) {
                        num_guesses += typename SudokuSolver::algorithm::num_guesses;
                        num_no_guess += typename SudokuSolver::algorithm::num_no_guess;
                        num_impossibles += typename SudokuSolver::algorithm::num_impossibles;

                        puzzleCount++;
#ifndef NDEBUG
                        if (puzzleCount > 1000)
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
