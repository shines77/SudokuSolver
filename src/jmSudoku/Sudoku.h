
#ifndef JM_SUDOKU_H
#define JM_SUDOKU_H

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
#pragma once
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <memory.h>
#include <assert.h>

#include <cstdint>
#include <cstddef>
#include <bitset>
#include <cstring>      // For std::memset()
#include <type_traits>

#include "BitSet.h"

using namespace jstd;

#define SEARCH_MODE_ONE_ANSWER              0
#define SEARCH_MODE_MORE_THAN_ONE_ANSWER    1
#define SEARCH_MODE_ALL_ANSWERS             2

namespace jmSudoku {

enum SearchMode {
    OneAnswer = SEARCH_MODE_ONE_ANSWER,
    MoreThanOneAnswer = SEARCH_MODE_MORE_THAN_ONE_ANSWER,
    AllAnswers = SEARCH_MODE_ALL_ANSWERS,
    SearchModeLast
};

double calc_percent(size_t num_val, size_t num_total) {
    if (num_total != 0)
        return (num_val * 100.0) / num_total;
    else
        return 0.0;
}

struct Tables {
    const unsigned int div3[9] = { 0, 0, 0, 1, 1, 1, 2, 2, 2 };
    const unsigned int mod3[9] = { 0, 1, 2, 0, 1, 2, 0, 1, 2 };

    const unsigned int roundTo3[9] = { 0, 0, 0, 3, 3, 3, 6, 6, 6 };

    const unsigned int div9[81] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3, 3, 3, 3,
        4, 4, 4, 4, 4, 4, 4, 4, 4,
        5, 5, 5, 5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6, 6, 6, 6, 6,
        7, 7, 7, 7, 7, 7, 7, 7, 7,
        8, 8, 8, 8, 8, 8, 8, 8, 8
    };

    const unsigned int mod9[81] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8,
        0, 1, 2, 3, 4, 5, 6, 7, 8,
        0, 1, 2, 3, 4, 5, 6, 7, 8,
        0, 1, 2, 3, 4, 5, 6, 7, 8,
        0, 1, 2, 3, 4, 5, 6, 7, 8,
        0, 1, 2, 3, 4, 5, 6, 7, 8,
        0, 1, 2, 3, 4, 5, 6, 7, 8,
        0, 1, 2, 3, 4, 5, 6, 7, 8,
        0, 1, 2, 3, 4, 5, 6, 7, 8
    };

    const unsigned int box[81] = {
        0, 0, 0, 1, 1, 1, 2, 2, 2,
        0, 0, 0, 1, 1, 1, 2, 2, 2,
        0, 0, 0, 1, 1, 1, 2, 2, 2,
        3, 3, 3, 4, 4, 4, 5, 5, 5,
        3, 3, 3, 4, 4, 4, 5, 5, 5,
        3, 3, 3, 4, 4, 4, 5, 5, 5,
        6, 6, 6, 7, 7, 7, 8, 8, 8,
        6, 6, 6, 7, 7, 7, 8, 8, 8,
        6, 6, 6, 7, 7, 7, 8, 8, 8
    };

    const unsigned int box_X_9[81] = {
        0,   0,  0,  9,  9,  9, 18, 18, 18,
        0,   0,  0,  9,  9,  9, 18, 18, 18,
        0,   0,  0,  9,  9,  9, 18, 18, 18,
        27, 27, 27, 36, 36, 36, 45, 45, 45,
        27, 27, 27, 36, 36, 36, 45, 45, 45,
        27, 27, 27, 36, 36, 36, 45, 45, 45,
        54, 54, 54, 63, 63, 63, 72, 72, 72,
        54, 54, 54, 63, 63, 63, 72, 72, 72,
        54, 54, 54, 63, 63, 63, 72, 72, 72
    };

    const unsigned int box_base[81] = {
        0,   0,  0,  3,  3,  3,  6,  6,  6,
        0,   0,  0,  3,  3,  3,  6,  6,  6,
        0,   0,  0,  3,  3,  3,  6,  6,  6,
        27, 27, 27, 30, 30, 30, 33, 33, 33,
        27, 27, 27, 30, 30, 30, 33, 33, 33,
        27, 27, 27, 30, 30, 30, 33, 33, 33,
        54, 54, 54, 57, 57, 57, 60, 60, 60,
        54, 54, 54, 57, 57, 57, 60, 60, 60,
        54, 54, 54, 57, 57, 57, 60, 60, 60
    };

    Tables() noexcept {
        //
    }
};

static const Tables tables {};

template <size_t TotalSize>
struct SudokuBoard {
    char board[TotalSize];
};

template <size_t nBoxCellsX = 3, size_t nBoxCellsY = 3,
          size_t nBoxCountX = 3, size_t nBoxCountY = 3,
          size_t nMinNumber = 1, size_t nMaxNumber = 9>
struct BasicSudoku {
    static const size_t BoxCellsX = nBoxCellsX;     // 3
    static const size_t BoxCellsY = nBoxCellsY;     // 3
    static const size_t BoxCountX = nBoxCountX;     // 3
    static const size_t BoxCountY = nBoxCountY;     // 3
    static const size_t MinNumber = nMinNumber;     // 1
    static const size_t MaxNumber = nMaxNumber;     // 9

    static const size_t Cols = BoxCellsX * BoxCountX;
    static const size_t Rows = BoxCellsY * BoxCountY;
    static const size_t Boxes = BoxCountX * BoxCountY;
    static const size_t Numbers = (MaxNumber - MinNumber) + 1;

    static const size_t BoxSize = BoxCellsX * BoxCellsY;
    static const size_t BoardSize = Rows * Cols;
    static const size_t TotalSize = BoxSize * Boxes * Numbers;

    static const size_t TotalSize2 = Rows * Cols * Numbers;

    static const size_t TotalConditions0 = 0;
    static const size_t TotalConditions1 = Rows * Cols;
    static const size_t TotalConditions2 = Rows * Numbers;
    static const size_t TotalConditions3 = Cols * Numbers;
    static const size_t TotalConditions4 = Boxes * Numbers;

    static const size_t TotalConditions01 = TotalConditions0  + TotalConditions1;
    static const size_t TotalConditions02 = TotalConditions01 + TotalConditions2;
    static const size_t TotalConditions03 = TotalConditions02 + TotalConditions3;
    static const size_t TotalConditions04 = TotalConditions03 + TotalConditions4;

    static const size_t TotalConditions =
        TotalConditions1 + TotalConditions2 + TotalConditions3 + TotalConditions4;

    static const size_t kAllRowsBit = (size_t(1) << Rows) - 1;
    static const size_t kAllColsBit = (size_t(1) << Cols) - 1;
    static const size_t kAllNumbersBit = (size_t(1) << Numbers) - 1;

    static void clear_board(char board[BasicSudoku::BoardSize]) {
        size_t pos = 0;
        for (size_t row = 0; row < Rows; row++) {
            for (size_t col = 0; col < Cols; col++) {
                board[pos++] = '.';
            }
        }
    }

    static void clear_board(std::vector<std::vector<char>> & board) {
        for (size_t row = 0; row < board.size(); row++) {
            std::vector<char> & line = board[row];
            for (size_t col = 0; col < line.size(); col++) {
                line[col] = '.';
            }
        }
    }

    static void display_board(char board[BasicSudoku::BoardSize],
                              bool is_input = false,
                              int idx = -1) {
        int filled = 0;
        size_t pos = 0;
        for (size_t row = 0; row < Rows; row++) {
            for (size_t col = 0; col < Cols; col++) {
                char val = board[pos++];
                if ((val != '.') && (val != ' ') && (val != '0') && (val != '-')) {
                    filled++;
                }
            }
        }

        if (is_input) {
            printf("The input is: (filled = %d)\n", filled);
        }
        else {
            if (idx == -1)
                printf("The answer is:\n");
            else
                printf("The answer # %d is:\n", idx + 1);
        }
        printf("\n");
        // printf("  ------- ------- -------\n");
        printf(" ");
        for (size_t countX = 0; countX < BoxCountX; countX++) {
            printf(" -------");
        }
        printf("\n");
        pos = 0;
        for (size_t row = 0; row < Rows; row++) {
            printf(" | ");
            for (size_t col = 0; col < Cols; col++) {
                char val = board[pos++];
                if (val == ' ' || val == '0' || val == '-')
                    printf(". ");
                else
                    printf("%c ", val);
                if ((col % BoxCellsX) == (BoxCellsX - 1))
                    printf("| ");
            }
            printf("\n");
            if ((row % BoxCellsY) == (BoxCellsY - 1)) {
                // printf("  ------- ------- -------\n");
                printf(" ");
                for (size_t countX = 0; countX < BoxCountX; countX++) {
                    printf(" -------");
                }
                printf("\n");
            }
        }
        printf("\n");
    }

    static void display_board(const std::vector<std::vector<char>> & board,
                              bool is_input = false,
                              int idx = -1) {
        int filled = 0;
        for (size_t row = 0; row < board.size(); row++) {
            const std::vector<char> & line = board[row];
            for (size_t col = 0; col < line.size(); col++) {
                char val = line[col];
                if ((val != '.') && (val != ' ') && (val != '0') && (val != '-')) {
                    filled++;
                }
            }
        }

        if (is_input) {
            printf("The input is: (filled = %d)\n", filled);
        }
        else {
            if (idx == -1)
                printf("The answer is:\n");
            else
                printf("The answer # %d is:\n", idx + 1);
        }
        printf("\n");
        // printf("  ------- ------- -------\n");
        printf(" ");
        for (size_t countX = 0; countX < BoxCountX; countX++) {
            printf(" -------");
        }
        printf("\n");
        for (size_t row = 0; row < Rows; row++) {
            assert(board.size() >= Rows);
            printf(" | ");
            for (size_t col = 0; col < Cols; col++) {
                assert(board[row].size() >= Cols);
                char val = board[row][col];
                if (val == ' ' || val == '0' || val == '-')
                    printf(". ");
                else
                    printf("%c ", val);
                if ((col % BoxCellsX) == (BoxCellsX - 1))
                    printf("| ");
            }
            printf("\n");
            if ((row % BoxCellsY) == (BoxCellsY - 1)) {
                // printf("  ------- ------- -------\n");
                printf(" ");
                for (size_t countX = 0; countX < BoxCountX; countX++) {
                    printf(" -------");
                }
                printf("\n");
            }
        }
        printf("\n");
    }

    static void display_boards(std::vector<SudokuBoard<BasicSudoku::BoardSize>> & boards) {
        printf("Total answers: %d\n\n", (int)boards.size());
        int i = 0;
        for (auto answer : boards) {
            BasicSudoku::display_board(answer.board, false, i);
            i++;
        }
    }

    static void display_boards(std::vector<std::vector<std::vector<char>>> & boards) {
        printf("Total answers: %d\n\n", (int)boards.size());
        int i = 0;
        for (auto board : boards) {
            BasicSudoku::display_board(board, false, i);
            i++;
        }
    }
};

// Standard sudoku definition
typedef BasicSudoku<3, 3, 3, 3, 1, 9> Sudoku;

} // namespace jmSudoku

#endif // JM_SUDOKU_H
